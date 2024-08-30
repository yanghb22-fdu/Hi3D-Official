import os
import re
import PIL.Image
import cv2
import einops
import numpy as np
import torch
import random
import math
import PIL
import rembg
from PIL import Image, ImageDraw, ImageFont
import shutil
import glob
from tqdm import tqdm
import subprocess as sp
import argparse
from skimage.io import imread
import imageio
import sys
import json
import datetime
import string
from dataset.opencv_transforms.functional import to_tensor, center_crop

from pytorch_lightning import seed_everything
from sgm.util import append_dims
from vtdm.model import create_model, load_state_dict
from vtdm.util import tensor2vid, export_to_video

models = {}

seed = random.randint(0, 65535)
seed_everything(seed)

import time
stamp = int(time.time())

parser = argparse.ArgumentParser()
parser.add_argument('--denoise_config', type=str, default="configs/inference-v01.yaml")
parser.add_argument('--denoise_checkpoint', type=str, default="ckpts/first_stage.pt")
parser.add_argument('--image_path', type=str, default="demo/15_out.png")
parser.add_argument("--output_dir", type=str, default="outputs/15_out")
parser.add_argument('--elevation', type=int, default=0)
params = parser.parse_args()

denoise_config = params.denoise_config
denoise_checkpoint = params.denoise_checkpoint

denoising_model = create_model(denoise_config).cpu()
denoising_model.init_from_ckpt(denoise_checkpoint)
denoising_model = denoising_model.cuda().half()
                                                 
models['denoising_model'] = denoising_model

def random_name():
    p1 = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    p2 = ''.join(random.choice(string.ascii_lowercase) for i in range(16))
    return p1 + '_' + p2
    
    
def denoising(frames, aes, mv, elevation):
    with torch.no_grad():        
        C, T, H, W = frames.shape
        clip_size = models['denoising_model'].num_samples
        assert T == clip_size
        
        batch = {'video': frames.unsqueeze(0)}
        batch['elevation'] = torch.Tensor([elevation]).to(torch.int64).to(frames.device)
        batch['fps_id'] = torch.Tensor([7]).to(torch.int64).to(frames.device)
        batch['motion_bucket_id'] = torch.Tensor([127]).to(torch.int64).to(frames.device)
        batch = models['denoising_model'].add_custom_cond(batch, infer=True)
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            c, uc = models['denoising_model'].conditioner.get_unconditional_conditioning(
                batch,
                force_uc_zero_embeddings=['cond_frames', 'cond_frames_without_noise']
            )

        additional_model_inputs = {}
        additional_model_inputs["image_only_indicator"] = torch.zeros(
            2, clip_size
        ).to(models['denoising_model'].device)
        additional_model_inputs["num_video_frames"] = batch["num_video_frames"]
        def denoiser(input, sigma, c):
            return models['denoising_model'].denoiser(
                models['denoising_model'].model, input, sigma, c, **additional_model_inputs
            )
            
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            randn = torch.randn([T, 4, H // 8, W // 8], device=models['denoising_model'].device)
            samples = models['denoising_model'].sampler(denoiser, randn, cond=c, uc=uc)
        
        samples = models['denoising_model'].decode_first_stage(samples.half())
        
        samples = einops.rearrange(samples, '(b t) c h w -> b c t h w', t=clip_size)
        
    return tensor2vid(samples)


def video_pipeline(frames, key, args):
    # seed = args['seed']
    num_iter = args['num_iter']
    
    out_list = []
    for it in range(num_iter):
        
        with torch.no_grad():
            results = denoising(frames, args['aes'], args['mv'], args['elevation'])
               
        if len(out_list) == 0:
            out_list = out_list + results
        else:
            out_list = out_list + results[1:]
        img = out_list[-1]
        img = to_tensor(img)
        img = (img - 0.5) * 2.0
        frames[:, 0] = img
    
    prex = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(stamp))
    prex_config = denoise_config.split('/')[-1][:-5]
    prex_ckpt = denoise_checkpoint.split('/')[-2:]
    prex_ckpt = (prex_ckpt[0] + '_' + prex_ckpt[1])[:-3]
    prex = prex + '_' + prex_config + '_' + prex_ckpt + '_seed_' + str(seed)
    
    output_videos_dir = os.path.join(args["output_dir"], "first_step")
    os.makedirs(output_videos_dir, exist_ok=True)
    output_videos_path = os.path.join(output_videos_dir, "first.mp4")
    export_to_video(out_list, output_videos_path, save_to_gif=False, use_cv2=False, fps=8)
            
def process(args, key='image'):
    image_path = args['image_path']

    img = cv2.imread(image_path)                                                   # 2048,2028,3
    frame_list = [img] * args['clip_size']

    h, w = frame_list[0].shape[0:2]
    rate = max(args['input_resolution'][0] * 1.0 / h, args['input_resolution'][1] * 1.0 / w)
    frame_list = [cv2.resize(f, [math.ceil(w * rate), math.ceil(h * rate)]) for f in frame_list]
    frame_list = [center_crop(f, [args['input_resolution'][0], args['input_resolution'][1]]) for f in frame_list]
    frame_list = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frame_list]
    
    frame_list = [to_tensor(f) for f in frame_list]
    frame_list = [(f - 0.5) * 2.0 for f in frame_list]
    frames = torch.stack(frame_list, 1)
    frames = frames.cuda()

    models['denoising_model'].num_samples = args['clip_size']
    models['denoising_model'].image_size = args['input_resolution']
    
    video_pipeline(frames, key, args)

# 1. remove background
rembg_session = rembg.new_session()
image = PIL.Image.open(params.image_path)
image = rembg.remove(image, session=rembg_session)

# 2. save to temp image dir
temp_image_dir = os.path.join(params.output_dir, "temp_image")
os.makedirs(temp_image_dir, exist_ok=True)
# save rgba
temp_image_path = os.path.join(temp_image_dir, "rgba.png")
image.save(temp_image_path)
#save white
white_image_path = os.path.join(temp_image_dir, "white.png")
white_image = Image.new("RGB", image.size, "WHITE")  # WHITE背景
white_image.paste(image, mask=image.split()[3])
white_image.save(white_image_path)

# 3. first step , generate first image, and save in "args.output_dir/first_step/first.mp4"
infer_config = {
        "image_path": white_image_path,
        "clip_size": 16,
        "input_resolution": [
            512,
            512
        ],
        "num_iter": 1,
        "seed": -1,
        "aes": 6.0,
        "mv": [
            0.0,
            0.0,
            0.0,
            10.0
        ],
        "elevation": params.elevation,
        "output_dir": params.output_dir
}

process(infer_config)