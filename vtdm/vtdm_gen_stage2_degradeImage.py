import einops
import torch
import torch as th
import torch.nn as nn
import os
from typing import Any, Dict, List, Tuple, Union

from sgm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from sgm.modules.attention import SpatialTransformer
from sgm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, Downsample, ResBlock, AttentionBlock
from sgm.models.diffusion import DiffusionEngine
from sgm.util import log_txt_as_img, exists, instantiate_from_config
from safetensors.torch import load_file as load_safetensors
from .model import load_state_dict
from .degraded_images import DegradedImages

class VideoLDM(DiffusionEngine):
    def __init__(self, num_samples, trained_param_keys=[''], *args, **kwargs):
        self.trained_param_keys = trained_param_keys
        super().__init__(*args, **kwargs)
        self.num_samples = num_samples
        
        # self.warp_model = FirstFrameFlowWarper(freeze=True)
        self.warp_model = DegradedImages(freeze=True)
        # self.warp_model = FirstFrameFlowWarper(freeze=True)

    def init_from_ckpt(
        self,
        path: str,
    ) -> None:
        if path.endswith("ckpt"):
            sd = torch.load(path, map_location="cpu")
            if "state_dict" in sd:
                sd = sd["state_dict"]
        elif path.endswith("pt"):
            sd_raw = torch.load(path, map_location="cpu")
            sd = {}
            for k in sd_raw['module']:
                sd[k[len('module.'):]] = sd_raw['module'][k]
        elif path.endswith("safetensors"):
            sd = load_safetensors(path)
        else:
            raise NotImplementedError

        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
    
    @torch.no_grad()
    def add_custom_cond(self, batch, infer=False):
        batch['num_video_frames'] = self.num_samples
        
        image = batch['video'][:, :, 0]                                     # (1, 3, 16, 1024, 1024)
        batch['cond_frames_without_noise'] = image.half()
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            if not infer:
                video_warp = self.warp_model(image, batch['video'], batch['masks'], batch['kernel1s'], batch['kernel2s'], batch['sinc_kernels'])       # (16, 3, 512, 512)
            else:
                video_warp = rearrange(batch['video'], 'b c t h w -> (b t) c h w')
        
        N = batch['video'].shape[0]
        if not infer:
            cond_aug = ((-3.0) + (0.5) * torch.randn((N,))).exp().cuda().half()
        else:
            cond_aug = torch.full((N, ), 0.02).cuda().half()
        batch['cond_aug'] = cond_aug
        batch['cond_frames'] = (video_warp + rearrange(repeat(cond_aug, "b -> (b t)", t=self.num_samples), 'b -> b 1 1 1') * torch.randn_like(video_warp)).half()
            
        # for dataset without indicator
        if not 'image_only_indicator' in batch:
            batch['image_only_indicator'] = torch.zeros((N, self.num_samples)).cuda().half()
        return batch

    def shared_step(self, batch: Dict) -> Any:
        frames = self.get_input(batch) # b c t h w
        batch = self.add_custom_cond(batch)
        
        frames_reshape = rearrange(frames, 'b c t h w -> (b t) c h w')
        x = self.encode_first_stage(frames_reshape)
        
        batch["global_step"] = self.global_step
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            loss, loss_dict = self(x, batch)
        return loss, loss_dict

    @torch.no_grad()
    def log_images(
        self,
        batch: Dict,
        N: int = 8,
        sample: bool = True,
        ucg_keys: List[str] = None,
        **kwargs,
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        frames = self.get_input(batch)
        batch = self.add_custom_cond(batch, infer=True)
        N = min(frames.shape[0], N)
        frames = frames[:N]
        x = rearrange(frames, 'b c t h w -> (b t) c h w')
        
        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.conditioner.embedders) > 0
            else [],
        )
        
        sampling_kwargs = {}

        aes = c['vector'][:, -256-256]
        caption =  batch['caption'][:N]
        for idx in range(N):
            sub_str = str(aes[idx].item())
            caption[idx] = sub_str + '\n' + caption[idx]
        
        x = x.to(self.device)
        
        z = self.encode_first_stage(x.half())
        x_rec = self.decode_first_stage(z.half())
        log["reconstructions-video"] = rearrange(x_rec, '(b t) c h w -> b c t h w', t=self.num_samples)
        log["conditioning"] = log_txt_as_img((512, 512), caption, size=16)
        x_d = c['concat'][:, :9]                                                  # (16, 9, 64, 64) (0-1)      
        x_d = rearrange(x_d, 'b (c  h0 w0) h w -> b c (h h0) (w w0)', h0=3, w0=3)
        log["depth-video"] = rearrange(x_d.repeat([1, 3, 1, 1]) * 2.0 - 1.0, '(b t) c h w -> b c t h w', t=self.num_samples)
        
        x_cond = self.decode_first_stage(c['concat'][:, 9:].half() * self.scale_factor)
        log["cond-video"] = rearrange(x_cond, '(b t) c h w -> b c t h w', t=self.num_samples)
        
        for k in c:
            if isinstance(c[k], torch.Tensor):
                if k == 'concat':
                    c[k], uc[k] = map(lambda y: y[k][:N * self.num_samples].to(self.device), (c, uc))
                else:
                    c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        additional_model_inputs = {}
        additional_model_inputs["image_only_indicator"] = torch.zeros(
            N * 2, self.num_samples
        ).to(self.device)
        additional_model_inputs["num_video_frames"] = batch["num_video_frames"]
        def denoiser(input, sigma, c):
            return self.denoiser(
                self.model, input, sigma, c, **additional_model_inputs
            )
            
        if sample:
            with self.ema_scope("Plotting"):
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    randn = torch.randn(z.shape, device=self.device)
                    samples = self.sampler(denoiser, randn, cond=c, uc=uc)
            samples = self.decode_first_stage(samples.half())
            log["samples-video"] = rearrange(samples, '(b t) c h w -> b c t h w', t=self.num_samples)
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        # params = list(self.model.parameters())
        names = []
        params = []
        for name, param in self.model.named_parameters():
            flag = False
            for k in self.trained_param_keys:
                if k in name:
                    names += [name]
                    params += [param]
                    flag = True
                if flag:
                    break
            # if not flag:
                # param.requires_grad = False
        print(names)
        
        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                params = params + list(embedder.parameters())
                
        opt = self.instantiate_optimizer_from_config(params, lr, self.optimizer_config)
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt
