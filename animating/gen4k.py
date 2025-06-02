import argparse
import datetime
import logging
import inspect
import math
import os
import random
import gc
import copy
import json

from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms as T
import diffusers
import transformers

from tqdm.auto import tqdm
from PIL import Image

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from diffusers.models import AutoencoderKL, UNetSpatioTemporalConditionModel
from diffusers import DPMSolverMultistepScheduler, DDPMScheduler, EulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, export_to_video
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import AttnProcessor2_0, Attention
from diffusers.models.attention import BasicTransformerBlock
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth import tensor2vid
from diffusers import StableVideoDiffusionPipeline
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import _resize_with_antialiasing


from utils.dataset import VideoJsonDataset, SingleVideoDataset, \
    ImageDataset, VideoFolderDataset, CachedDataset, VideoBLIPDataset
from einops import rearrange, repeat
import imageio


from models.pipeline_sphere import MaskShpericalStableVideoDiffusionPipeline
from utils.lora_handler import LoraHandler, LORA_VERSIONS
from utils.common import read_mask, generate_random_mask, slerp, calculate_motion_score, \
    read_video, calculate_motion_precision, calculate_latent_motion_score, \
    DDPM_forward, DDPM_forward_timesteps, motion_mask_loss
    

def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)



def load_primary_models(pretrained_model_path, eval=False):
    if eval:
        pipeline = StableVideoDiffusionPipeline.from_pretrained(pretrained_model_path, torch_dtype=torch.float16, variant='fp16')
    else:
        pipeline = StableVideoDiffusionPipeline.from_pretrained(pretrained_model_path)
    return pipeline, None, pipeline.feature_extractor, pipeline.scheduler, pipeline.image_processor, \
        pipeline.image_encoder, pipeline.vae, pipeline.unet


def eval(
        pipeline, 
        vae_processor, 
        validation_data, 
        out_file, 
        index, 
        forward_t=25, 
        preview=True, 
        frame_id=0, 
        diff_mode='return_video', 
        resolution=(2048, 4096), 
        switch_to_sliding_window_time=5,
        ref_latents=None,
        start_denoising_step=0,
        window_size=(64, 64),
        stride=(32,32),
):
    vae = pipeline.vae
    device = vae.device
    dtype = vae.dtype
    
    diffusion_scheduler = pipeline.scheduler
    diffusion_scheduler.set_timesteps(validation_data.num_inference_steps, device=device)

    prompt = validation_data.prompt
    pimg = Image.open(validation_data.prompt_image)
    if pimg.mode == "RGBA":
        pimg = pimg.convert("RGB")
    width, height = pimg.size
    scale = math.sqrt(width*height / (validation_data.height*validation_data.width))
    block_size=64
    validation_data.height = round(height/scale/block_size)*block_size
    validation_data.width = round(width/scale/block_size)*block_size

    mask_path = validation_data.prompt_image.split('.')[0] + '_mask.png'
    # cfg_path = 
    if os.path.exists(mask_path):
        mask = Image.open(mask_path)
        mask = mask.resize((validation_data.width, validation_data.height))
        np_mask = np.array(mask)
        if len(np_mask.shape) == 3:
            np_mask = np_mask[:,:,0]
        np_mask[np_mask!=0]=255
    elif os.path.exists(validation_data.mask):
        # logger.info(f"validation_data.mask: {validation_data.mask}")
        mask = Image.open(validation_data.mask)
        mask = mask.resize((validation_data.width, validation_data.height))
        np_mask = np.array(mask)
        if len(np_mask.shape) == 3:
            np_mask = np_mask[:,:,0]
        np_mask[np_mask!=0]=255
    else:
        np_mask = np.ones([validation_data.height, validation_data.width], dtype=np.uint8)*255
    out_mask_path = os.path.splitext(out_file)[0] + "_mask.png"
    Image.fromarray(np_mask).save(out_mask_path)
    motion_mask = pipeline.unet.config.in_channels == 9

    
    hh, ww = resolution
    
    with torch.no_grad():
        if motion_mask:
            h, w = hh//pipeline.vae_scale_factor, ww//pipeline.vae_scale_factor
            initial_latents = torch.randn([1, validation_data.num_frames, 4, h, w], dtype=dtype, device=device)
            mask = T.ToTensor()(np_mask).to(dtype).to(device)
            mask = T.Resize([h, w], antialias=False)(mask)
            if diff_mode == 'decode_latents':
                latents = np.load(out_file.replace('.gif', '.npy'))
                latents = torch.from_numpy(latents).half().cuda()
                video_frames, x_mask = MaskShpericalStableVideoDiffusionPipeline.__call__(
                    pipeline,
                    image=pimg,
                    width=ww, 
                    height=hh,
                    num_frames=validation_data.num_frames,
                    num_inference_steps=validation_data.num_inference_steps,
                    decode_chunk_size=validation_data.decode_chunk_size,
                    fps=validation_data.fps,
                    motion_bucket_id=validation_data.motion_bucket_id,
                    mask=mask,
                    call_mode=diff_mode,
                    pre_latents=latents,
                    window_size=window_size,
                    stride=stride,
                )
            else:
                video_frames, x_mask = MaskShpericalStableVideoDiffusionPipeline.__call__(
                    pipeline,
                    image=pimg,
                    width=ww, 
                    height=hh, 
                    num_frames=validation_data.num_frames,
                    num_inference_steps=validation_data.num_inference_steps,
                    decode_chunk_size=validation_data.decode_chunk_size,
                    fps=validation_data.fps,
                    motion_bucket_id=validation_data.motion_bucket_id,
                    mask=mask,
                    call_mode=diff_mode,
                    switch_to_sliding_window_time=switch_to_sliding_window_time,
                    ref_latents=ref_latents,
                    start_denoising_step=start_denoising_step,
                    window_size=window_size,
                    stride=stride,
                )

            
        else:
            video_frames = pipeline(
                image=pimg,
                width=validation_data.width,
                height=validation_data.height,
                num_frames=validation_data.num_frames,
                num_inference_steps=validation_data.num_inference_steps,
                fps=validation_data.fps,
                decode_chunk_size=validation_data.decode_chunk_size,
                motion_bucket_id=validation_data.motion_bucket_id,
                pers_id=1,
                
            ).frames[0]
    
    return video_frames


def main_eval(
    pretrained_model_path: str,
    validation_data: Dict,
    seed: Optional[int] = None,
    eval_file = None,
    diff_mode='return_video',
    output_dir='temp',
    **kwargs
):
    if seed is not None:
        set_seed(seed)
    pipeline, tokenizer, feature_extractor, train_scheduler, vae_processor, text_encoder, vae, unet = load_primary_models(pretrained_model_path, eval=True)
    device = torch.device("cuda")
    pipeline.to(device)

    if eval_file is not None:
        eval_list = json.load(open(eval_file))
    else:
        eval_list = [[validation_data.prompt_image, validation_data.prompt]]


    config_path = validation_data.config
    with open(config_path, 'r') as fin:
        denoising_config = json.load(fin)


    iters = 5
    for example in eval_list:
        for t in range(iters):
            name, prompt = example
            out_file_dir = f"{output_dir}/{name.split('.')[0]}"
            os.makedirs(out_file_dir, exist_ok=True)
            if diff_mode == 'return_latents':
                suff = '.npy'
            else:
                suff = '.gif'
            out_file = f"{out_file_dir}/{t}{suff}"
            validation_data.prompt_image = name
            validation_data.prompt = prompt
            if diff_mode == 'decode_latents':
                video_frames = eval(
                    pipeline, 
                    vae_processor, 
                    validation_data, 
                    out_file, t, frame_id=t, 
                    diff_mode=diff_mode, 
                    resolution=(2048,4096)
                )
            elif diff_mode == 'return_latents':
                ref_latents = eval(
                    pipeline, 
                    vae_processor, 
                    validation_data, 
                    out_file, t, frame_id=t, 
                    diff_mode=diff_mode, 
                    resolution=denoising_config['lr']['resolution'],
                    switch_to_sliding_window_time=denoising_config['lr']['switch_to_sliding_window_time'],
                    ref_latents=None,
                    start_denoising_step=denoising_config['lr']['start_denoising_step'],
                    window_size=denoising_config['lr']['window_size'],
                    stride=denoising_config['lr']['stride'],
                )
                np.save(out_file.replace('.npy', '_lr.npy'), ref_latents.detach().cpu().numpy())
                video_frames = eval(
                    pipeline, 
                    vae_processor, 
                    validation_data, 
                    out_file, t, frame_id=t, 
                    diff_mode=diff_mode, 
                    resolution=denoising_config['hr']['resolution'],
                    switch_to_sliding_window_time=denoising_config['hr']['switch_to_sliding_window_time'],
                    ref_latents=ref_latents,
                    start_denoising_step=denoising_config['hr']['start_denoising_step'],
                    window_size=denoising_config['hr']['window_size'],
                    stride=denoising_config['hr']['stride'],
                )
                
            if diff_mode == 'return_latents':
                np.save(out_file, video_frames.detach().cpu().numpy())

            else:
                video_frames = video_frames.frames[0]
                imageio.mimwrite(out_file, video_frames, duration=175, loop=0)
                imageio.mimwrite(out_file.replace('.gif', '.mp4'), video_frames, fps=7)
                outfolder = out_file.replace('.gif', '__frames')
                os.makedirs(outfolder, exist_ok=True)
                for k, img_pil in enumerate(video_frames):
                    img_pil.save(os.path.join(outfolder, '''{:02d}.png'''.format(k)))
        
                print("save file", out_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/my_config.yaml")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument('rest', nargs=argparse.REMAINDER)
    parser.add_argument('diff_mode', type=str, default="return_latents")

    args = parser.parse_args()
    args_dict = OmegaConf.load(args.config)
    cli_dict = OmegaConf.from_dotlist(args.rest)
    args_dict = OmegaConf.merge(args_dict, cli_dict)

    main_eval(diff_mode=args.diff_mode,**args_dict)

