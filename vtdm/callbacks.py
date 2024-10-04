import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl

from omegaconf import OmegaConf
from PIL import Image
from torchvision.utils import make_grid
from einops import rearrange
from logging import Logger
from typing import Callable, Optional

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from vtdm.util import tensor2vid, export_to_video


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config, os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}), os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                # try:
                    # os.rename(self.logdir, dst)
                # except FileNotFoundError:
                    # pass


class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "image_log", split)
        for k in images:
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}".format(k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            
            if 'video' in k: # log to video
                export_to_video(images[k], path + '.mp4', save_to_gif=False, use_cv2=False, fps=6)
            else:
                grid = torchvision.utils.make_grid(images[k], nrow=4)
                if self.rescale:
                    grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                grid = grid.numpy()
                grid = (grid * 255).astype(np.uint8)
                Image.fromarray(grid).save(path + '.png')

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                if 'video' in k: # log to video
                    images[k] = tensor2vid(images[k])
                else:
                    images[k] = images[k].to(dtype=torch.float32)
                    N = min(images[k].shape[0], self.max_images)
                    images[k] = images[k][:N]
                    if isinstance(images[k], torch.Tensor):
                        images[k] = images[k].detach().cpu()
                        if self.clamp:
                            images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device.index)
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        max_memory = torch.cuda.max_memory_allocated(trainer.strategy.root_device.index) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f} MiB")
        except AttributeError:
            pass


class TextProgressBar(pl.callbacks.ProgressBarBase):

    """A custom ProgressBar to log the training progress."""
    
    def __init__(self, logger: Logger, refresh_rate: int = 50) -> None:
        super().__init__()
        self._logger = logger
        self._refresh_rate = refresh_rate
        self._enabled = True

        # a time flag to indicate the beginning of an epoch
        self._time = 0

    @property
    def refresh_rate(self) -> int:
        return self._refresh_rate

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def is_disabled(self) -> bool:
        return not self.is_enabled

    def disable(self) -> None:
        # No need to disable the ProgressBar on processes with LOCAL_RANK != 1, because the
        # StreamHandler of logging is disabled on these processes.
        self._enabled = True

    def enable(self) -> None:
        self._enabled = True

    @staticmethod
    def _serialize_metrics(progressbar_log_dict: dict, filter_fn: Optional[Callable[[str], bool]] = None) -> str:
        if filter_fn:
            progressbar_log_dict = {k: v for k, v in progressbar_log_dict.items() if filter_fn(k)}
        msg = ''
        for metric, value in progressbar_log_dict.items():
            if type(value) is str:
                msg += f'{metric}: {value:.5f}  '
            elif 'acc' in metric:
                msg += f'{metric}: {value:.3%}  '
            else:
                msg += f'{metric}: {value:f}  '
        return msg

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        self._logger.info(f'Epoch: {trainer.current_epoch}, batch_num: {self.total_train_batches}')
        self._time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        current = self.train_batch_idx
        if self._should_update(current, self.total_train_batches):
            batch_time = (time.time() - self._time) / self.train_batch_idx
            msg = f'[Epoch {trainer.current_epoch}] [Batch {self.train_batch_idx}/{self.total_train_batches} {batch_time:.2f} s/batch] => '
            if current != self.total_train_batches:
                filter_fn = lambda x: not x.startswith('val') and not x.startswith('test') and not x.startswith('global') and not x.endswith('_epoch')
            else:
                filter_fn = lambda x: not x.startswith('val') and not x.startswith('test') and not x.startswith('global')
            msg += self._serialize_metrics(trainer.progress_bar_metrics, filter_fn=filter_fn)
            self._logger.info(msg)

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)
        self._logger.info(f'Training finished.')

    def on_validation_start(self, trainer, pl_module):
        super().on_validation_start(trainer, pl_module)
        self._logger.info('Validation Begins. Epoch: {}, val batch num: {}'.format(trainer.current_epoch, self.total_val_batches))

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        current = self.val_batch_idx
        if self._should_update(current, self.total_val_batches):
            batch_time = (time.time() - self._time) / self.val_batch_idx
            msg = f'[Epoch {trainer.current_epoch}] [Val Batch {self.val_batch_idx}/{self.total_val_batches} {batch_time:.2f} s/batch] => '
            if current != self.total_val_batches:
                filter_fn = lambda x: x.startswith('val') and not x.endswith('_epoch')
            else:
                filter_fn = lambda x: x.startswith('val')
            msg += self._serialize_metrics(trainer.progress_bar_metrics, filter_fn=filter_fn)
            self._logger.info(msg)
    
    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        msg = f'[Epoch {trainer.current_epoch}] [Validation finished] => '
        msg += self._serialize_metrics(trainer.progress_bar_metrics, filter_fn=lambda x: x.startswith('val') and x.endswith('_epoch'))
        self._logger.info(msg)

    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        # don't show the version number
        items.pop("v_num", None)
        return items

    def _should_update(self, current: int, total: int) -> bool:
        return self.is_enabled and (current % self.refresh_rate == 0 or current == total)

    def print(self, *args, sep: str = " ", **kwargs):
        s = sep.join(map(str, args))
        self._logger.info(f"[Progress Print] {s}")

