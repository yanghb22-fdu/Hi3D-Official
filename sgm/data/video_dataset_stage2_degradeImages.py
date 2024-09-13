import pytorch_lightning as pl
import numpy as np
import torch
import PIL
import os
import random
from skimage.io import imread
import webdataset as wds
import PIL.Image as Image
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path

# from ldm.base_utils import read_pickle, pose_inverse
import torchvision.transforms as transforms
import torchvision
from einops import rearrange

# for the degraded images
import yaml
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
import math

def add_margin(pil_img, color=0, size=256):
    width, height = pil_img.size
    result = Image.new(pil_img.mode, (size, size), color)
    result.paste(pil_img, ((size - width) // 2, (size - height) // 2))
    return result

def prepare_inputs(image_path, elevation_input, crop_size=-1, image_size=256):
    image_input = Image.open(image_path)

    if crop_size!=-1:
        alpha_np = np.asarray(image_input)[:, :, 3]
        coords = np.stack(np.nonzero(alpha_np), 1)[:, (1, 0)]
        min_x, min_y = np.min(coords, 0)
        max_x, max_y = np.max(coords, 0)
        ref_img_ = image_input.crop((min_x, min_y, max_x, max_y))
        h, w = ref_img_.height, ref_img_.width
        scale = crop_size / max(h, w)
        h_, w_ = int(scale * h), int(scale * w)
        ref_img_ = ref_img_.resize((w_, h_), resample=Image.BICUBIC)
        image_input = add_margin(ref_img_, size=image_size)
    else:
        image_input = add_margin(image_input, size=max(image_input.height, image_input.width))
        image_input = image_input.resize((image_size, image_size), resample=Image.BICUBIC)

    image_input = np.asarray(image_input)
    image_input = image_input.astype(np.float32) / 255.0
    ref_mask = image_input[:, :, 3:]
    image_input[:, :, :3] = image_input[:, :, :3] * ref_mask + 1 - ref_mask  # white background
    image_input = image_input[:, :, :3] * 2.0 - 1.0
    image_input = torch.from_numpy(image_input.astype(np.float32))
    elevation_input = torch.from_numpy(np.asarray([np.deg2rad(elevation_input)], np.float32))
    return {"input_image": image_input, "input_elevation": elevation_input}


class VideoTrainDataset(Dataset):
    def __init__(self, base_folder='/data/yanghaibo/datas/OBJAVERSE-LVIS/images', depth_folder="/mnt/drive2/3d/OBJAVERSE-DEPTH/depth256", width=1024, height=576, sample_frames=25):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            channels (int): Number of channels, default is 3 for RGB.
        """
        # Define the path to the folder containing video frames
        self.base_folder = base_folder
        self.depth_folder = depth_folder
        # self.folders1 = os.listdir(self.base_folder)
        # self.folders2 = os.listdir(self.depth_folder)
        # self.folders = list(set(self.folders1).intersection(set(self.folders2)))
        self.folders = os.listdir(self.base_folder)
        self.num_samples = len(self.folders)
        self.channels = 3
        self.width = width
        self.height = height
        self.sample_frames = sample_frames
        self.elevations = [-10, 0, 10, 20, 30, 40]
        
        # for degraded images
        with open('configs/train_realesrnet_x4plus.yml', mode='r') as f:
            opt = yaml.load(f, Loader=yaml.FullLoader)
        self.opt = opt
        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1
        

    def __len__(self):
        return self.num_samples
    
    def load_im(self, path):
        img = imread(path)
        img = img.astype(np.float32) / 255.0
        mask = img[:,:,3:]
        img[:,:,:3] = img[:,:,:3] * mask + 1 - mask # white background
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        return img, mask

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to return.

        Returns:
            dict: A dictionary containing the 'pixel_values' tensor of shape (16, channels, 320, 512).
        """
        # Randomly select a folder (representing a video) from the base folder
        chosen_folder = random.choice(self.folders)
        folder_path = os.path.join(self.base_folder, chosen_folder)
        frames = os.listdir(folder_path)
        # Sort the frames by name
        frames.sort()

        # Ensure the selected folder has at least `sample_frames`` frames
        if len(frames) < self.sample_frames:
            raise ValueError(
                f"The selected folder '{chosen_folder}' contains fewer than `{self.sample_frames}` frames.")

        # Randomly select a start index for frame sequence. Fixed elevation
        start_idx = random.randint(0, len(frames) - 1)
        # start_idx = random.choice([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92])
        range_id = int(start_idx / 16)  # 0, 1, 2, 3, 4, 5
        elevation = self.elevations[range_id]
        selected_frames = []
        
        for frame_idx in range(start_idx, (range_id + 1) * 16):
            selected_frames.append(frames[frame_idx])
        for frame_idx in range((range_id) * 16, start_idx):
            selected_frames.append(frames[frame_idx])
            
        # Initialize a tensor to store the pixel values
        pixel_values = torch.empty((self.sample_frames, self.channels, self.height, self.width))
        masks = []

        # Load and process each frame
        for i, frame_name in enumerate(selected_frames):
            frame_path = os.path.join(folder_path, frame_name)
            img, mask = self.load_im(frame_path)
            mask = mask.squeeze(-1)
            masks.append(mask)
            # Resize the image and convert it to a tensor
            img_resized = img.resize((self.width, self.height))
            img_tensor = torch.from_numpy(np.array(img_resized)).float()

            # Normalize the image by scaling pixel values to [-1, 1]
            img_normalized = img_tensor / 127.5 - 1

            # Rearrange channels if necessary
            if self.channels == 3:
                img_normalized = img_normalized.permute(
                    2, 0, 1)  # For RGB images
            elif self.channels == 1:
                img_normalized = img_normalized.mean(
                    dim=2, keepdim=True)  # For grayscale images

            pixel_values[i] = img_normalized
        
        pixel_values = rearrange(pixel_values, 't c h w -> c t h w')
        masks = torch.from_numpy(np.array(masks))
        caption = chosen_folder
        
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! get the kernels for degraded images
        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernels = []
        kernel2s = []
        sinc_kernels = []
        for i in range(16):
            kernel_size = random.choice(self.kernel_range)
            if np.random.uniform() < self.opt['sinc_prob']:
                # this sinc filter setting is for kernels ranging from [7, 21]
                if kernel_size < 13:
                    omega_c = np.random.uniform(np.pi / 3, np.pi)
                else:
                    omega_c = np.random.uniform(np.pi / 5, np.pi)
                kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            else:
                kernel = random_mixed_kernels(
                    self.kernel_list,
                    self.kernel_prob,
                    kernel_size,
                    self.blur_sigma,
                    self.blur_sigma, [-math.pi, math.pi],
                    self.betag_range,
                    self.betap_range,
                    noise_range=None)
            # pad kernel
            pad_size = (21 - kernel_size) // 2
            kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
            kernels.append(kernel)

            # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
            kernel_size = random.choice(self.kernel_range)
            if np.random.uniform() < self.opt['sinc_prob2']:
                if kernel_size < 13:
                    omega_c = np.random.uniform(np.pi / 3, np.pi)
                else:
                    omega_c = np.random.uniform(np.pi / 5, np.pi)
                kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            else:
                kernel2 = random_mixed_kernels(
                    self.kernel_list2,
                    self.kernel_prob2,
                    kernel_size,
                    self.blur_sigma2,
                    self.blur_sigma2, [-math.pi, math.pi],
                    self.betag_range2,
                    self.betap_range2,
                    noise_range=None)

            # pad kernel
            pad_size = (21 - kernel_size) // 2
            kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))
            kernel2s.append(kernel2)

            # ------------------------------------- the final sinc kernel ------------------------------------- #
            if np.random.uniform() < self.opt['final_sinc_prob']:
                kernel_size = random.choice(self.kernel_range)
                omega_c = np.random.uniform(np.pi / 3, np.pi)
                sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
                sinc_kernel = torch.FloatTensor(sinc_kernel)
            else:
                sinc_kernel = self.pulse_tensor
            sinc_kernels.append(sinc_kernel)
        kernels = np.array(kernels)
        kernel2s = np.array(kernel2s)
        sinc_kernels = torch.stack(sinc_kernels, 0)
        kernels = torch.FloatTensor(kernels)
        kernel2s = torch.FloatTensor(kernel2s)
        return {'video': pixel_values, 'masks': masks, 'elevation': elevation, 'caption': caption, 'kernel1s': kernels, 'kernel2s': kernel2s, 'sinc_kernels': sinc_kernels}           # (16, 3, 512, 512)-> (3, 16, 512, 512)
    
class SyncDreamerEvalData(Dataset):
    def __init__(self, image_dir):
        self.image_size = 512
        self.image_dir = Path(image_dir)
        self.crop_size = 20

        self.fns = []
        for fn in Path(image_dir).iterdir():
            if fn.suffix=='.png':
                self.fns.append(fn)
        print('============= length of dataset %d =============' % len(self.fns))

    def __len__(self):
        return len(self.fns)

    def get_data_for_index(self, index):
        input_img_fn = self.fns[index]
        elevation = 0
        return prepare_inputs(input_img_fn, elevation, 512)

    def __getitem__(self, index):
        return self.get_data_for_index(index)

class VideoDataset(pl.LightningDataModule):
    def __init__(self, base_folder, depth_folder, eval_folder, width, height, sample_frames, batch_size, num_workers=4, seed=0, **kwargs):
        super().__init__()
        self.base_folder = base_folder
        self.depth_folder = depth_folder
        self.eval_folder = eval_folder
        self.width = width
        self.height = height
        self.sample_frames = sample_frames
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.additional_args = kwargs

    def setup(self):
        self.train_dataset = VideoTrainDataset(self.base_folder, self.depth_folder, self.width, self.height, self.sample_frames)
        self.val_dataset = SyncDreamerEvalData(image_dir=self.eval_folder)

    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset, seed=self.seed)
        return wds.WebLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):
        loader = wds.WebLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return loader

    def test_dataloader(self):
        return wds.WebLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)