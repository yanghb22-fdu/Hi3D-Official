import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import re
import cv2
import einops
import numpy as np
import torch
import random
import math
from PIL import Image, ImageDraw, ImageFont
import shutil
import glob
from tqdm import tqdm
import subprocess as sp
import argparse

import imageio
import sys
import json
import datetime
import string
from dataset.opencv_transforms.functional import to_tensor, center_crop

from pytorch_lightning import seed_everything
from sgm.util import append_dims
from sgm.util import autocast, instantiate_from_config
from vtdm.model import create_model, load_state_dict
from vtdm.util import tensor2vid, export_to_video

from einops import rearrange

import yaml
import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.sr_model import SRModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from torch.nn import functional as F
import torch.nn as nn

class DegradedImages(torch.nn.Module):
    def __init__(self, freeze=True):
        super().__init__()
        
        with open('configs/train_realesrnet_x4plus.yml', mode='r') as f:
            opt = yaml.load(f, Loader=yaml.FullLoader)
        self.opt = opt
    
    @autocast
    @torch.no_grad()
    def forward(self, images, videos, masks, kernel1s, kernel2s, sinc_kernels):
        '''
        images: (2, 3, 1024, 1024) [-1, 1]
        videos: (2, 3, 16, 1024, 1024) [-1, 1]
        masks: (2, 16, 1024, 1024)
        kernel1s, kernel2s, sinc_kernels: (2, 16, 21, 21)
        '''
        self.jpeger = DiffJPEG(differentiable=False).cuda()
        B, C, H, W = images.shape
        ori_h, ori_w = videos.size()[3:5]
        videos = videos / 2.0 + 0.5
        videos = rearrange(videos, 'b c t h w -> b t c h w')  #(2, 16, 3, 1024, 1024)

        all_lqs = []
        
        for i in range(B):
            kernel1 = kernel1s[i]
            kernel2 = kernel2s[i]
            sinc_kernel = sinc_kernels[i]

            gt = videos[i]          # (16, 3, 1024, 1024)
            mask = masks[i]         # (16, 1024, 1024)
            
            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(gt, kernel1)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            gray_noise_prob = self.opt['gray_noise_prob']
            if np.random.uniform() < self.opt['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            # blur
            if np.random.uniform() < self.opt['second_blur_prob']:
                out = filter2D(out, kernel2)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range2'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range2'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)
            # add noise
            gray_noise_prob = self.opt['gray_noise_prob2']
            if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range2'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            if np.random.uniform() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, sinc_kernel)

            # clamp and round
            lqs = torch.clamp((out * 255.0).round(), 0, 255) / 255.
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            lqs = F.interpolate(lqs, size=(ori_h, ori_w), mode=mode)      # 16,3,1024,1024

            lqs = rearrange(lqs, 't c h w -> t h w c') # 16, 1024, 1024, 3
            for j in range(16):
                lqs[j][mask[j]==0] = 1.0
            all_lqs.append(lqs)
            
            # import cv2
            # gt1 = gt[0]
            # lq1 = lqs[0]
            # gt1 = rearrange(gt1, 'c h w -> h w c')

            # gt1 = (gt1.cpu().numpy() * 255.).astype('uint8')
            # lq1 = (lq1.cpu().numpy() * 255.).astype('uint8')
            # cv2.imwrite(f'gt{i}.png', gt1)
            # cv2.imwrite(f'lq{i}.png', lq1)

            
        all_lqs = [(f - 0.5) * 2.0 for f in all_lqs]
        all_lqs = torch.stack(all_lqs, 0)   # 2, 16, 1024, 1024, 3
        all_lqs = rearrange(all_lqs, 'b t h w c -> b t c h w')
        for i in range(B):
            all_lqs[i][0] = images[i]
        all_lqs = rearrange(all_lqs, 'b t c h w -> (b t) c h w')
        return all_lqs