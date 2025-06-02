from os import times
from typing import Callable, Dict, List, Optional, Union
import torch
from einops import rearrange, repeat
import PIL
import copy

from diffusers import TextToVideoSDPipeline, StableVideoDiffusionPipeline
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth import tensor2vid, TextToVideoSDPipelineOutput
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import tensor2vid as svd_tensor2vid, StableVideoDiffusionPipelineOutput
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DiffusionPipeline, AutoencoderTiny
from diffusers import AutoencoderKL
import math
from .camera_utils import direction_to_img_coord, img_coord_to_sample_coord, img_coord_to_pano_direction, img_coord_from_hw
from .geo_utils import panorama_to_pers_directions
import torch.nn.functional as F
import imageio
from sklearn.neighbors import NearestNeighbors



def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]

def get_views(panorama_height, panorama_width, window_size=64, stride=16, circular_padding=False):
    if isinstance(window_size, int):
        window_size_h, window_size_w = window_size, window_size
    else:
        window_size_h, window_size_w = window_size[0], window_size[1]
    
    if isinstance(stride, int):
        stride_h, stride_w = stride, stride
    else:
        stride_h, stride_w = stride[0], stride[1]
        

    panorama_height /= 8
    panorama_width /= 8
    num_blocks_height = (panorama_height - window_size_h) // stride_h + 1 if panorama_height > window_size_h else 1
    if circular_padding:
        num_blocks_width = panorama_width // stride_w if panorama_width > window_size_w else 1
    else:
        num_blocks_width = (panorama_width - window_size_w) // stride_w + 1 if panorama_width > window_size_w else 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * stride_h)
        h_end = h_start + window_size_h
        w_start = int((i % num_blocks_width) * stride_w)
        w_end = w_start + window_size_w
        views.append((h_start, h_end, w_start, w_end))
    return views

import inspect


def denoise_using_sliding_window(self, circular_padding, image_latents, mask, latents, views_scheduler_status, step_t, step_i, image_embeddings, added_time_ids, do_classifier_free_guidance):
    count = self.count
    value = self.value
    
    views_batch = self.views_batch
    
    count.zero_()
    value.zero_()

    for j, batch_view in enumerate(views_batch):
        vb_size = len(batch_view)

        if circular_padding:
            latents_for_view = []
            mask_for_view = []
            image_latents_for_view = []
            for h_start, h_end, w_start, w_end in batch_view:
                if w_end > latents.shape[4]:

                    latent_view = torch.cat(
                        (
                            latents[:, :, :, h_start:h_end, w_start:],
                            latents[:, :, :, h_start:h_end, : w_end - latents.shape[4]],
                        ),
                        axis=-1,
                    )
                    image_latent_view = torch.cat(
                        (
                            image_latents[:, :, :, h_start:h_end, w_start:],
                            image_latents[:, :, :, h_start:h_end, : w_end - image_latents.shape[4]],
                        ),
                        axis=-1,
                    )
                    mask_view = torch.cat(
                        (
                            mask[:, :, :, h_start:h_end, w_start:],
                            mask[:, :, :, h_start:h_end, : w_end - mask.shape[4]],
                        ),
                        axis=-1,
                    )
                else:

                    latent_view = latents[:, :, :, h_start:h_end, w_start:w_end]
                    image_latent_view = image_latents[:, :, :, h_start:h_end, w_start:w_end]
                    mask_view = mask[:, :, :, h_start:h_end, w_start:w_end]


                latents_for_view.append(latent_view)
                image_latents_for_view.append(image_latent_view)
                mask_for_view.append(mask_view)
            latents_for_view = torch.cat(latents_for_view)
            image_latents_for_view = torch.cat(image_latents_for_view)
            mask_for_view = torch.cat(mask_for_view)
        else: 
            latents_for_view = torch.cat(
                [
                    latents[:, :, :, h_start:h_end, w_start:w_end]
                    for h_start, h_end, w_start, w_end in batch_view
                ]
            )
            image_latents_for_view = torch.cat(
                [
                    image_latents[:, :, :, h_start:h_end, w_start:w_end]
                    for h_start, h_end, w_start, w_end in batch_view
                ]
            )
            mask_for_view = torch.cat(
                [
                    mask[:, :, :, h_start:h_end, w_start:w_end]
                    for h_start, h_end, w_start, w_end in batch_view
                ]
            )

        self.scheduler.__dict__.update(views_scheduler_status[j])

        latent_model_input = torch.cat([latents_for_view] * 2) if do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, step_t)

        latent_model_input = torch.cat([mask_for_view, latent_model_input, image_latents_for_view], dim=2)

        noise_pred = self.unet(
            latent_model_input,
            step_t,
            encoder_hidden_states=image_embeddings,
            added_time_ids=added_time_ids,
            return_dict=False,
        )[0]

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

        latents_denoised_batch = self.scheduler.step(
            noise_pred, step_t, latents_for_view,
        ).prev_sample

        views_scheduler_status[j] = copy.deepcopy(self.scheduler.__dict__)


        for latents_view_denoised, (h_start, h_end, w_start, w_end) in zip(
            latents_denoised_batch.chunk(vb_size), batch_view
        ):
            if circular_padding and w_end > latents.shape[4]:
                value[:, :, :, h_start:h_end, w_start:] += latents_view_denoised[
                    :, :, :, :, : latents.shape[4] - w_start
                ]
                value[:, :, :, h_start:h_end, : w_end - latents.shape[4]] += latents_view_denoised[
                    :, :, :, :, latents.shape[4] - w_start :
                ]
                count[:, :, :, h_start:h_end, w_start:] += 1
                count[:, :, :, h_start:h_end, : w_end - latents.shape[4]] += 1
            else:
                value[:, :, :, h_start:h_end, w_start:w_end] += latents_view_denoised
                count[:, :, :, h_start:h_end, w_start:w_end] += 1

    latents = torch.where(count > 0, value / count, value)
    return latents



def denoising_using_sphere(
    self, 
    pers_img_latents, 
    pers_mask, 
    pers_latents, 
    views_scheduler_status, 
    step_t, 
    step_i, 
    image_embeddings, 
    added_time_ids, 
    do_classifier_free_guidance, 
    pers_dirs
):
    
    
    
    denoised_pers_latents_l = []
 
    
    for j in range(pers_latents.shape[0]):
        latents_for_view = pers_latents[j]
        mask_for_view = pers_mask[j]
        image_latents_for_view = pers_img_latents[j]
        self.scheduler.__dict__.update(views_scheduler_status[j])

        latent_model_input = torch.stack([latents_for_view] * 2) if do_classifier_free_guidance else latents_for_view
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, step_t)


        latent_model_input = torch.cat([mask_for_view, latent_model_input, image_latents_for_view], dim=2)

        noise_pred = self.unet(
            latent_model_input,
            step_t,
            encoder_hidden_states=image_embeddings,
            added_time_ids=added_time_ids,
            return_dict=False,
        )[0]

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

        latents_denoised_batch = self.scheduler.step(
            noise_pred, step_t, latents_for_view, 
        ).prev_sample

        views_scheduler_status[j] = copy.deepcopy(self.scheduler.__dict__)
        denoised_pers_latents_l.append(latents_denoised_batch)
    
    denoised_pers_latents = torch.cat(denoised_pers_latents_l)
    return denoised_pers_latents



def decode_latents(self, latents, num_frames, decode_chunk_size=14):

    latents = latents.flatten(0, 1)

    latents = 1 / self.vae.config.scaling_factor * latents

    accepts_num_frames = "num_frames" in set(inspect.signature(self.vae.forward).parameters.keys())

    frames = []
    for i in range(0, latents.shape[0], decode_chunk_size):
        num_frames_in = latents[i : i + decode_chunk_size].shape[0]
        decode_kwargs = {}

        frame = self.vae.decode(latents[i : i + decode_chunk_size], **decode_kwargs).sample
        frames.append(frame)
    frames = torch.cat(frames, dim=0)


    frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)

    frames = frames.float()
    return frames


import numpy as np

class SphericalLatent:
    def  __init__(self, num_points1: int, num_points2: int):
        """
        Generate `num_points` approximately uniformly-distributed points
        on the unit sphere using the Fibonacci sphere algorithm.

        :param num_points: Number of points to generate.
        :return: A (num_points, 3) tensor of points on the sphere.
        """
        self.caches = {}
        self.knn_k = 2

        phi = math.pi * (3.0 - math.sqrt(5.0))

        points = []
        cnt = 0

        num_points = num_points1
        for i in range(num_points):
            if cnt % 500000 == 0:
                print(f'{cnt} / {num_points}')
            cnt = cnt + 1

            y = 1.0 - (i / float(num_points - 1)) * 2.0

            radius = math.sqrt(1.0 - y * y)

            theta = phi * i

            x = math.cos(theta) * radius
            z = math.sin(theta) * radius

            if abs(z) >= 0.8:
                points.append([x, y, z])
            
        num_points = num_points2
        for i in range(num_points):
            if cnt % 100000 == 0:
                print(f'{cnt} / {num_points}')
            cnt = cnt + 1

            y = 1.0 - (i / float(num_points - 1)) * 2.0

            radius = math.sqrt(1.0 - y * y)

            theta = phi * i

            x = math.cos(theta) * radius
            z = math.sin(theta) * radius

            if abs(z) < 0.8:
                points.append([x, y, z])
                
        self.pts = np.asarray(points, dtype=np.float32)
        

    def project_from_pers_only_latents(self, dirs, latents):
        # dirs: n h w 3
        n_pers = dirs.shape[0]

        dirs_re = rearrange(dirs, 'n h w c->(n h w) c')
        latents_re = rearrange(latents, 'n l c h w -> (n h w) (l c)')
        sph_features = self.project_from_dir(dirs_re, [latents_re], self.knn_k, cache='from_pers_only_latents')[0]
        return sph_features
        
        
    
    def project_from_pers(self, latents, mask, image_latents, dirs):
        # dirs: n h w 3
        n_pers = dirs.shape[0]

        self.mask_shape = mask.shape
        self.latents_shape = latents.shape
        self.image_latents_shape = image_latents.shape
        
        dirs_re = rearrange(dirs, 'n h w c->(n h w) c')
        mask_re = rearrange(mask, 'n l c h w->(n h w) (l c)')
        latents_re = rearrange(latents, 'n l c h w -> (n h w) (l c)')
        image_latents_re = rearrange(image_latents, 'n l c h w->(n h w) (l c)')
   
        sph_features = self.project_from_dir(dirs_re, [latents_re, mask_re, image_latents_re], self.knn_k)
        return sph_features
        

    def project_to_pers_only_latents(self, dirs, sp_latents, latents_shape):
        lb, ll, lc, lh, lw = latents_shape
        dirs_re = rearrange(dirs, 'n h w c->(n h w) c')
        
        _pers_latents = self.project_to_dir(dirs_re, [sp_latents], self.knn_k, cache='to_pers_only_latents')[0]
        pers_latents = rearrange(_pers_latents, '(n h w) (l c) -> n l c h w', n=lb, l=ll, c=lc, h=lh, w=lw)
     
        return pers_latents
        
        
    def project_to_pers(self, dirs, sp_latents, sp_mask, sp_image_latents):
        mb, ml, mc, mh, mw = self.mask_shape
        lb, ll, lc, lh, lw = self.latents_shape
        ib, il, ic, ih, iw = self.image_latents_shape
        dirs_re = rearrange(dirs, 'n h w c->(n h w) c')
        
        _pers_latents, _pers_mask, _pers_img_latents = self.project_to_dir(dirs_re, [sp_latents, sp_mask, sp_image_latents], self.knn_k)
        pers_latents = rearrange(_pers_latents, '(n h w) (l c) -> n l c h w', n=lb, l=ll, c=lc, h=lh, w=lw)
        pers_mask = rearrange(_pers_mask, '(n h w) (l c) -> n l c h w', n=mb, l=ml, c=mc, h=mh, w=mw)
        pers_image_latents = rearrange(_pers_img_latents, '(n h w) (l c) -> n l c h w', n=ib, l=il, c=ic, h=ih, w=iw)
        
        return pers_latents, pers_mask, pers_image_latents
        
    def project_from_pano(self, pano_latents, dirs, mask=None, image_latents=None):
        dirs_re = rearrange(dirs, 'h w c->(h w) c')
        pano_latents_re = rearrange(pano_latents, 'b l c h w->(h w) (b l c)')
        
        if mask is not None:
            mask_re = rearrange(mask, 'b l c h w->(h w) (b l c)')
            image_latents_re = rearrange(image_latents, 'b l c h w->(h w) (b l c)')
            
        features_to_project = [pano_latents_re]
        if mask is not None:
            features_to_project.append(mask_re)
            features_to_project.append(image_latents_re)
        sph_features = self.project_from_dir(dirs_re, features_to_project, self.knn_k)[0]
        
        if len(sph_features) == 1:
            sph_features = sph_features[0]
        
        return sph_features
    
    def project_to_pano(self, sph_latents, pano_dirs, pano_latents_shape, sph_mask=None, pano_mask_shape=None, sph_image_latents=None, pano_image_latents_shape=None):
        ph, pw, pc = pano_dirs.shape
        b, l, c, _, _ = pano_latents_shape
        
        if sph_mask is not None:
            mb, ml, mc, _, _ = pano_mask_shape
            ib, il, ic, _, _ = pano_image_latents_shape
        
        dirs_re = rearrange(pano_dirs, 'h w c -> (h w) c')
        
        to_be_projected = [sph_latents]
        
        if sph_mask is not None:
            to_be_projected.append(sph_mask)
            to_be_projected.append(sph_image_latents)
            
        _projected_list = self.project_to_dir(dirs_re, to_be_projected, self.knn_k)
        _pano_latents = _projected_list[0]
        pano_latents = rearrange(_pano_latents, '(h w) (b l c) -> b l c h w', b=b, l=l, c=c, h=ph, w=pw)
        
        if sph_mask is not None:
            _pano_mask = _projected_list[1]
            _pano_image_latents = _projected_list[2]
            
            pano_mask = rearrange(_pano_mask, '(h w) (b l c)->b l c h w', b=mb, l=ml, c=mc)
            pano_image_latents = rearrange(_pano_image_latents, '(h w) (b l c)-> b l c h w', b=ib, l=il, c=ic)
            
            return pano_latents, pano_mask, pano_image_latents
        
        return pano_latents
    
        
    def project_from_dir(self, dirs:torch.Tensor, pers_features:list, knn_k: int, cache=None) -> list:

        if isinstance(dirs, torch.Tensor):
            _pts = dirs.detach().cpu().numpy()
            torch_pts = True
        else:
            _pts = dirs
        
        if cache is not None:
            if cache not in self.caches:
                print(f'computing {cache}')
                nbrs = NearestNeighbors(n_neighbors=knn_k, algorithm='auto').fit(_pts)
                distances, indices = nbrs.kneighbors(self.pts)
                self.caches[cache] = (distances, indices)
            else:
                print(f'recovering from cache {cache}')
                distances, indices = self.caches[cache]
        else:
            print(f'computing projection')
            nbrs = NearestNeighbors(n_neighbors=knn_k, algorithm='auto').fit(_pts)
            distances, indices = nbrs.kneighbors(self.pts)

        output_features = []
        
        for feature in pers_features:
            sampled_features = feature[indices]
            torch_feature = False
            if isinstance(sampled_features, torch.Tensor):
                _device = sampled_features.device
                torch_feature = True
                _dtype = sampled_features.dtype
                sampled_features = sampled_features.detach().cpu().numpy()
                
            weights = 1 / distances
            normalized_weights = weights / weights.sum(-1, keepdims=True)
            weighted_feature_sampled = normalized_weights[...,None] * sampled_features
            feature_sampled = weighted_feature_sampled.sum(1)
            if torch_feature:
                feature_sampled = torch.from_numpy(feature_sampled).to(_device).to(_dtype)
            
            output_features.append(feature_sampled)
            
            
            
        return output_features
            
    
    
    def project_to_dir(self, dirs:torch.Tensor, sph_features:list, knn_k: int, cache=None) -> list:
        # dirs: M 3
        # sph_features: each M * C
        _pts = dirs.detach().cpu().numpy()
        
        if cache is not None:
            if cache not in self.caches:
                print(f'computing {cache}')
                nbrs = NearestNeighbors(n_neighbors=knn_k, algorithm='auto').fit(self.pts)
                distances, indices = nbrs.kneighbors(_pts)
                self.caches[cache] = (distances, indices)
            else:
                print(f'recovering from cache {cache}')
                distances, indices = self.caches[cache]
        else:
            print(f'computing projection')
            nbrs = NearestNeighbors(n_neighbors=knn_k, algorithm='auto').fit(self.pts)
            distances, indices = nbrs.kneighbors(_pts)
            
        output_features = []
        
        for feature in sph_features:
            torch_feature = False
            if isinstance(feature, torch.Tensor):
                torch_feature = True
                _dtype = feature.dtype
                _device = feature.device
                feature = feature.detach().cpu().numpy()

            sampled_features = feature[indices]
            weights = 1 / distances
            normalized_weights = weights / weights.sum(-1, keepdims=True)
            weighted_feature_sampled = normalized_weights[...,None] * sampled_features
            feature_sampled = weighted_feature_sampled.sum(1)
            
            if torch_feature:
                feature_sampled = torch.from_numpy(feature_sampled).to(_device).to(_dtype)
            
            output_features.append(feature_sampled)
            
        return output_features
            


def split_latent(latents, mask, image_latents, gen_res):
    pers_dirs, pers_ratios, to_vecs, down_vecs, right_vecs = [], [], [], [], []
    for _ in range(1):

        ratio = 1.5
       
        cur_pers_dirs, cur_pers_ratios, cur_to_vecs, cur_down_vecs, cur_right_vecs = panorama_to_pers_directions(gen_res=gen_res, ratio=ratio, ex_rot=None)
        
        pers_dirs.append(cur_pers_dirs)
        pers_ratios.append(cur_pers_ratios)
        to_vecs.append(cur_to_vecs)
        down_vecs.append(cur_down_vecs)
        right_vecs.append(cur_right_vecs)

    pers_dirs = torch.cat(pers_dirs, 0)
    pers_ratios = torch.cat(pers_ratios, 0)
    to_vecs = torch.cat(to_vecs, 0)
    down_vecs = torch.cat(down_vecs, 0)
    right_vecs = torch.cat(right_vecs, 0)

    fx = torch.linalg.norm(to_vecs, 2, -1, True) / torch.linalg.norm(right_vecs, 2, -1, True) * gen_res * .5
    fy = torch.linalg.norm(to_vecs, 2, -1, True) / torch.linalg.norm(down_vecs, 2, -1, True) * gen_res * .5
    cx = torch.ones_like(fx) * gen_res * .5
    cy = torch.ones_like(fy) * gen_res * .5
    pers_dirs = pers_dirs.cuda()
    pers_ratios = pers_ratios.cuda()
    to_vecs = to_vecs.cuda()
    down_vecs = down_vecs.cuda()
    right_vecs = right_vecs.cuda()
    rot_w2c = torch.stack([right_vecs / torch.linalg.norm(right_vecs, 2, -1, True),
                            down_vecs / torch.linalg.norm(down_vecs, 2, -1, True),
                            to_vecs / torch.linalg.norm(to_vecs, 2, -1, True)],
                            dim=1)
    rot_c2w = torch.linalg.inv(rot_w2c)

    n_pers = len(pers_dirs)
    img_coords = direction_to_img_coord(pers_dirs)
    sample_coords = img_coord_to_sample_coord(img_coords)

    mask_re = rearrange(mask, 'b l c h w -> 1 (b l c) h w').repeat(n_pers, 1, 1, 1)
    img_latents_re = rearrange(image_latents, 'b l c h w -> 1 (b l c) h w').repeat(n_pers, 1, 1, 1)
    latents_re = rearrange(latents, 'b l c h w-> b (l c) h w').repeat(n_pers, 1, 1, 1)
    pers_latents = F.grid_sample(latents_re, sample_coords.half(), padding_mode='border') #reflection [n_pers, 3, gen_res, gen_res]
    pers_mask = F.grid_sample(mask_re, sample_coords.half(), padding_mode='border')
    pers_img_latents = F.grid_sample(img_latents_re, sample_coords.half(), padding_mode='border')

    pers_latents = rearrange(pers_latents, 'b (l c) h w -> b l c h w', c=4)
    pers_mask = rearrange(pers_mask, 'B (b l c) h w -> B b l c h w', b=2, l=14)
    pers_img_latents = rearrange(pers_img_latents, 'B (b l c) h w -> B b l c h w', b=2, l=14)

    pers_coords = sample_coords
    pers_latents_pc = pers_latents[:20]
    pers_dirs_pc = pers_dirs[:20]
    n_pers_pc = len(pers_dirs_pc)

    return pers_latents_pc, pers_mask, pers_img_latents, pers_dirs_pc, n_pers_pc


class MaskShpericalStableVideoDiffusionPipeline(StableVideoDiffusionPipeline):

    # @torch.no_grad()
    def __call__(
        self,
        image,
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: int = 0.02,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
        mask = None,
        call_mode='return_video',
        pre_latents=None,
        switch_to_sliding_window_time=5,
        ref_latents=None,
        start_denoising_step=0,
        window_size=(64, 64),
        stride=(32,32)
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                Image or images to guide image generation. If you provide a tensor, it needs to be compatible with
                [`CLIPImageProcessor`](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json).
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_frames (`int`, *optional*):
                The number of video frames to generate. Defaults to 14 for `stable-video-diffusion-img2vid` and to 25 for `stable-video-diffusion-img2vid-xt`
            num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.
            min_guidance_scale (`float`, *optional*, defaults to 1.0):
                The minimum guidance scale. Used for the classifier free guidance with first frame.
            max_guidance_scale (`float`, *optional*, defaults to 3.0):
                The maximum guidance scale. Used for the classifier free guidance with last frame.
            fps (`int`, *optional*, defaults to 7):
                Frames per second. The rate at which the generated images shall be exported to a video after generation.
                Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
            motion_bucket_id (`int`, *optional*, defaults to 127):
                The motion bucket ID. Used as conditioning for the generation. The higher the number the more motion will be in the video.
            noise_aug_strength (`int`, *optional*, defaults to 0.02):
                The amount of noise added to the init image, the higher it is the less the video will look like the init image. Increase it for more motion.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time. The higher the chunk size, the higher the temporal consistency
                between frames, but also the higher the memory consumption. By default, the decoder will decode all frames at once
                for maximal quality. Reduce `decode_chunk_size` to reduce memory usage.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Returns:
            [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list of list with the generated frames.

        Examples:

        ```py
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.utils import load_image, export_to_video

        pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
        pipe.to("cuda")

        image = load_image("https://lh3.googleusercontent.com/y-iFOHfLTwkuQSUegpwDdgKmOjRSTvPxat63dQLB25xkTs4lhIbRUFeNBWZzYf370g=s1200")
        image = image.resize((1024, 576))

        frames = pipe(image, num_frames=25, decode_chunk_size=8).frames[0]
        export_to_video(frames, "generated.mp4", fps=7)
        ```
        """
        # 0. Default height and width to unet

        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames


        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image, height, width)

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]
        device = self._execution_device

        do_classifier_free_guidance = max_guidance_scale > 1.0

        
        # 3. Encode input image
        image_embeddings = self._encode_image(image, device, num_videos_per_prompt, do_classifier_free_guidance)

        # NOTE: Stable Diffusion Video was conditioned on fps - 1, which
        # is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        fps = fps - 1
        
        # 4. Encode input image using VAE
        image = self.image_processor.preprocess(image, height=height, width=width)
        noise = randn_tensor(image.shape, generator=generator, device=image.device, dtype=image.dtype)
        image = image + noise_aug_strength * noise

        needs_upcasting = True
        # needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        image_latents = self._encode_vae_image(image, device, num_videos_per_prompt, do_classifier_free_guidance)
       
        image_latents = image_latents.to(image_embeddings.dtype)

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        mask = repeat(mask, '1 h w -> 2 f 1 h w', f=num_frames)
        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )

        
        
        if call_mode == 'decode_latents':
            latents=pre_latents
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)

            frames = self.decode_latents(latents, num_frames, decode_chunk_size=1)
            frames = svd_tensor2vid(frames, self.image_processor, output_type=output_type)
            return StableVideoDiffusionPipelineOutput(frames=frames), mask

        # 7. Prepare guidance scale
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        self._guidance_scale = guidance_scale

        print(latents.shape)
        ll_height, ll_width=latents.shape[3], latents.shape[4]

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        view_batch_size = 1

        circular_padding = True
        
        

        print(f'width={width}')
        
        views = get_views(height, width, window_size=window_size, stride=stride, circular_padding=circular_padding)
        views_batch = [views[i : i + view_batch_size] for i in range(0, len(views), view_batch_size)]

        views_scheduler_status = [copy.deepcopy(self.scheduler.__dict__)] * len(views_batch)
        
        self.count = torch.zeros_like(latents)
        self.value = torch.zeros_like(latents)
        self.views_batch = views_batch
        print(f'sampling {len(views)} views with window_size {window_size}, ({ll_height}, {ll_width})')

        logs = []
        
        
        latent_h, latent_w = latents.shape[-2:]

        if switch_to_sliding_window_time >= 1:
            sph_helper = SphericalLatent(200000000, 20000000)
            
            pano_dirs = img_coord_to_pano_direction(img_coord_from_hw(latent_h, latent_w)).cuda()
                   
        
        pers_latents, pers_mask, pers_img_latents, pers_dirs, pers_n = split_latent(latents=latents, mask=mask, image_latents=image_latents,gen_res=128)
        
     
        if ref_latents is not None:
            ref_latents = torch.nn.functional.interpolate(ref_latents[0,...], size=(256, 512), scale_factor=None, mode='bilinear', align_corners=None)[None,...]
            latents = self.scheduler.add_noise(ref_latents, torch.randn_like(ref_latents), timesteps[start_denoising_step:start_denoising_step+1])

        with self.progress_bar(total=num_inference_steps - start_denoising_step) as progress_bar:
            for i, t in enumerate(timesteps[start_denoising_step:]):


                if i < switch_to_sliding_window_time:
                    pers_latents = denoising_using_sphere(
                        self,
                        pers_img_latents,
                        pers_mask,
                        pers_latents,
                        views_scheduler_status,
                        t, 
                        i,
                        image_embeddings,
                        added_time_ids,
                        do_classifier_free_guidance, 
                        pers_dirs
                    )
                    sp_latents = sph_helper.project_from_pers_only_latents(pers_dirs, pers_latents)
                    if i == switch_to_sliding_window_time - 1:
                        latents = sph_helper.project_to_pano(sp_latents, pano_dirs, latents.shape)
                    else:
                        pers_latents = sph_helper.project_to_pers_only_latents(pers_dirs, sp_latents, pers_latents.shape)
                else:
                    latents = denoise_using_sliding_window(
                        self,
                        circular_padding,
                        image_latents,
                        mask,
                        latents,
                        views_scheduler_status,
                        t, 
                        i,
                        image_embeddings,
                        added_time_ids,
                        do_classifier_free_guidance
                    )
                        
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()


        if call_mode == 'return_latents':
            return latents, mask
        elif call_mode != 'return_video':
            raise NotImplementedError()

        torch.cuda.empty_cache()
        if not output_type == "latent":

            if needs_upcasting:
                self.vae.to(dtype=torch.float16)

            
            frames = self.decode_latents(latents, num_frames, decode_chunk_size=decode_chunk_size)
            frames = svd_tensor2vid(frames, self.image_processor, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames), mask