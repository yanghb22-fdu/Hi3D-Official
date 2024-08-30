import torch
import cv2
import numpy as np
import imageio
import torchvision
import math

from einops import rearrange
from typing import List

from PIL import Image, ImageDraw, ImageFont

def tensor2vid(video: torch.Tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) -> List[np.ndarray]:
    mean = torch.tensor(mean, device=video.device).reshape(1, -1, 1, 1, 1)  # ncfhw
    std = torch.tensor(std, device=video.device).reshape(1, -1, 1, 1, 1)  # ncfhw
    video = video.mul_(std).add_(mean)  # unnormalize back to [0,1]
    video.clamp_(0, 1)
    images = rearrange(video, 'i c f h w -> (i f) h w c')
    images = images.unbind(dim=0)
    images = [(image.cpu().numpy() * 255).astype('uint8') for image in images]  # f h w c
    return images


def export_to_video(video_frames: List[np.ndarray], output_video_path: str = None, save_to_gif=False, use_cv2=True, fps=8) -> str:
    h, w, c = video_frames[0].shape
    if save_to_gif:
        image_lst = []
        if output_video_path.endswith('mp4'):
            output_video_path = output_video_path[:-3] + 'gif'
        for i in range(len(video_frames)):
            image_lst.append(video_frames[i])
        imageio.mimsave(output_video_path, image_lst, fps=fps)     
        return output_video_path
    else:
        if use_cv2:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
            for i in range(len(video_frames)):
                img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
                video_writer.write(img)
            video_writer.release()
        else:
            duration = math.ceil(len(video_frames) / fps)
            append_num = duration * fps - len(video_frames)
            for k in range(append_num): video_frames.append(video_frames[-1])
            video_stack = np.stack(video_frames, axis=0)
            video_tensor = torch.from_numpy(video_stack)
            # torchvision.io.write_video(output_video_path, video_tensor, fps=fps, options={"crf": "17"})
            torchvision.io.write_video(output_video_path, video_tensor, fps=fps, options={"crf": "17"})
        return output_video_path