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
from copy import deepcopy
import argparse

import imageio
import sys
import os
import json
import datetime
import string
from dataset.opencv_transforms.functional import to_tensor, center_crop

from pytorch_lightning import seed_everything
from sgm.util import append_dims
from vtdm.model import create_model, load_state_dict
from vtdm.util import tensor2vid, export_to_video

seed = random.randint(0, 65535)
# seed = 20
seed_everything(seed)

import time
stamp = int(time.time())

models = {}

parser = argparse.ArgumentParser()
parser.add_argument('--denoise_config', type=str, default="configs/inference-v02.yaml")
parser.add_argument('--denoise_checkpoint', type=str, default="ckpts/second_stage.pt")
parser.add_argument('--image_path', type=str, default="demo/15_out.png")
parser.add_argument("--output_dir", type=str, default="outputs/15_out")
parser.add_argument('--elevation', type=int, default=0)
params = parser.parse_args()

denoising_model = create_model(params.denoise_config).cpu()
denoising_model.init_from_ckpt(params.denoise_checkpoint)
denoising_model = denoising_model.cuda().half()
                                                 
models['denoising_model'] = denoising_model

def remove_white_background(img):
    pic = Image.fromarray(img)
    pic = pic.convert('RGBA') # 转为RGBA模式
    width,height = pic.size
    array = pic.load() # 获取图片像素操作入口
    for i in range(width):
        for j in range(height):
            pos = array[i,j] # 获得某个像素点，格式为(R,G,B,A)元组
            # 如果R G B三者都大于240(很接近白色了，数值可调整)
            isEdit = (sum([1 for x in pos[0:3] if x > 220]) == 3)
            if isEdit:
                # 更改为透明
                array[i,j] = (255,255,255,0)

    # 保存图片
    # pic.save('a.png')
    image = np.array(pic)
    mask = image[:,:,3].astype(np.float32)/255
    return mask
    
def random_name():
    p1 = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    p2 = ''.join(random.choice(string.ascii_lowercase) for i in range(16))
    return p1 + '_' + p2
    
def denoising(frames, masks, aes, mv, elevation):
    with torch.no_grad():        
        C, T, H, W = frames.shape
        clip_size = models['denoising_model'].num_samples
        assert T == clip_size
        
        # skip_steps = 0
        alpha_pow = 40.0
        
        sigmas = models['denoising_model'].sampler.discretization(
            models['denoising_model'].sampler.num_steps, device=models['denoising_model'].sampler.device
        )
        num_sigmas = len(sigmas)

        s_in = frames.new_ones([1 * clip_size])
        
        init_latents = torch.randn([T, 4, H // 8, W // 8], device=models['denoising_model'].device)
        latents = init_latents.clone()
        
        z_list = []
        for t in range(T):
            frame = frames[:, t]
            frame = einops.rearrange(frame, 'c h w -> 1 c h w')
            z = models['denoising_model'].encode_first_stage(frame.half())
            z_list.append(z)
                
        latents *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        
        batch = {}
        batch['video'] = frames.unsqueeze(0)
        batch['masks'] = masks.unsqueeze(0)
        batch['elevation'] = torch.Tensor([elevation]).to(torch.int64).to(frames.device)
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
        
        for i in models['denoising_model'].sampler.get_sigma_gen(num_sigmas):
            alpha = 0.5 * (1 + math.cos(i * 1.0 / models['denoising_model'].sampler.num_steps))
            alpha = math.pow(alpha, alpha_pow)
            print(alpha)
            for t in range(T):
                latents[t:t+1] = latents[t:t+1] * (1 - alpha) + (init_latents[t:t+1] * append_dims(sigmas[i], z.ndim) + z_list[t]) * alpha
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                latents = models['denoising_model'].sampler.step_call(denoiser, latents, i, s_in, sigmas, num_sigmas, c, uc)
        
        samples = models['denoising_model'].decode_first_stage(latents.half())
        
        samples = einops.rearrange(samples, '(b t) c h w -> b c t h w', t=clip_size)
        
    return tensor2vid(samples)


def video_pipeline(frames, masks, key, args):
    # seed = args['seed']
    num_iter = args['num_iter']
    
    out_list = []
    for it in range(num_iter):
        
        with torch.no_grad():
            results = denoising(frames, masks, args['aes'], args['mv'], args['elevation'])
        
        if len(out_list) == 0:
            out_list = out_list + results
        else:
            out_list = out_list + results[1:]
        img = out_list[-1]
        img = to_tensor(img)
        img = (img - 0.5) * 2.0
        frames[:, 0] = img
    return out_list
    
        
def process(args, key='image'):
    image_path = args['image_path']
    video_path = args['video_path']

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_list_raw = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_list_raw.append(frame)

    models['denoising_model'].num_samples = args['clip_size']
    models['denoising_model'].image_size = args['input_resolution']
    
    # cut last frames
    frame_list = deepcopy(frame_list_raw[-args['clip_size']:])
    
    img = cv2.imread(image_path)
    frame_list[0] = img

    frame_list = [cv2.resize(f, [args['input_resolution'][1], args['input_resolution'][0]]) for f in frame_list]
    frame_list = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frame_list]
    
    masks_list = [remove_white_background(f) for f in frame_list]
    masks = torch.from_numpy(np.array(masks_list))
    masks.cuda()
    
    frame_list = [to_tensor(f) for f in frame_list]
    frame_list = [(f - 0.5) * 2.0 for f in frame_list]
    frames = torch.stack(frame_list, 1)
    frames = frames.cuda()
    
    out_list = video_pipeline(frames, masks, key, args)
    
    output_videos_path = args["output_dir"]
    output_videos_path = os.path.join(output_videos_path, "second_step_video")
    os.makedirs(output_videos_path, exist_ok=True)
        
    output_video = os.path.join(output_videos_path, 'second.mp4')
    
    export_to_video(out_list, output_video, save_to_gif=False, use_cv2=False, fps=8)        

# step2: generate high resolution images

temp_image_dir = os.path.join(params.output_dir, "temp_image")
white_image_path = os.path.join(temp_image_dir, "white.png")
first_step_video = os.path.join(params.output_dir, "first_step/first.mp4")

infer_config = {
        "image_path": white_image_path,
        "video_path": first_step_video,
        "clip_size": 16,
        "input_resolution": [
            1024,
            1024
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