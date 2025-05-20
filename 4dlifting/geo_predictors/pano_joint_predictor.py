from pickletools import optimize
from sys import builtin_module_names
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import numpy as np
import os

# from generate_init_geo import generate_geo_from_gif

from .geo_predictor import GeoPredictor
from .omnidata_predictor import OmnidataPredictor
from .omnidata_normal_predictor import OmnidataNormalPredictor
# from .depth_anything import DepthAnythingPredictor

from fields.networks import VanillaMLP
import tinycudann as tcnn

from utils.geo_utils import panorama_to_pers_directions
from utils.camera_utils import *
import cv2
import imageio
from PIL import Image
from einops import rearrange

# from .depth_anything import DepthAnythingPredictor

def scale_unit(x):
    return (x - x.min()) / (x.max() - x.min())


def save_color_map(depth, path, mask=None):
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
    if mask is not None:
        depth[mask] = 255
    cv2.imwrite(path, depth)

class SphereDistanceField(nn.Module):
    def __init__(self,
                 n_levels=16,
                 log2_hashmap_size=19,
                 base_res=16,
                 fine_res=2048):
        super().__init__()
        per_level_scale = np.exp(np.log(fine_res / base_res) / (n_levels - 1))
        self.hash_grid = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_res,
                "per_level_scale": per_level_scale,
                "interpolation": "Smoothstep",
            }
        )

        self.geo_mlp = VanillaMLP(dim_in=n_levels * 2 + 3,
                                  dim_out=1,
                                  n_neurons=64,
                                  n_hidden_layers=2,
                                  sphere_init=True,
                                  weight_norm=False)

    def forward(self, directions, requires_grad=False):
        if requires_grad:
            if not self.training:
                directions = directions.clone()  # get a copy to enable grad
            directions.requires_grad_(True)

        dir_scaled = directions * 0.49 + 0.49
        selector = ((dir_scaled > 0.0) & (dir_scaled < 1.0)).all(dim=-1).to(torch.float32)
        scene_feat = self.hash_grid(dir_scaled)

        distance = F.softplus(self.geo_mlp(torch.cat([directions, scene_feat], -1))[..., 0] + 1.)


        if requires_grad:
            grad = torch.autograd.grad(
                distance, directions, grad_outputs=torch.ones_like(distance),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]

            return distance, grad
        else:
            return distance



class PanoJointPredictorGIF(GeoPredictor):
    def __init__(self, args):
        super().__init__()
        
        self.depth_predictor = OmnidataPredictor(pred_res=1024)
        
        self.depth_min = args.depth_min
        self.depth_max = args.depth_max

        self.normal_predictor = OmnidataNormalPredictor()
        self.depth_anything_path = args.source_path
        self.depth_vis_path = os.path.join(args.source_path, '__depth_vis')
        self.img_vis_path = os.path.join(args.source_path, '__img_vis')
        os.makedirs(self.depth_vis_path, exist_ok=True)
        os.makedirs(self.img_vis_path, exist_ok=True)


    def grads_to_normal(self, grads):
        grads = grads.cpu()
        height, width, _ = grads.shape
        pano_dirs = img_coord_to_pano_direction(img_coord_from_hw(height, width))
        ortho_a = torch.randn([height, width, 3])
        ortho_b = torch.linalg.cross(pano_dirs, ortho_a)
        ortho_b = ortho_b / torch.linalg.norm(ortho_b, 2, -1, True)
        ortho_a = torch.linalg.cross(ortho_b, pano_dirs)
        ortho_a = ortho_a / torch.linalg.norm(ortho_a, 2, -1, True)

        val_a = (grads * ortho_a).sum(-1, True) * pano_dirs + ortho_a
        val_a = val_a / torch.linalg.norm(val_a, 2, -1, True)
        val_b = (grads * ortho_b).sum(-1, True) * pano_dirs + ortho_b
        val_b = val_b / torch.linalg.norm(val_b, 2, -1, True)

        normals = torch.cross(val_a, val_b)
        normals = normals / torch.linalg.norm(normals, 2, -1, True)
        is_inside = ((normals * pano_dirs).sum(-1, True) < 0.).float()
        normals = normals * is_inside + -normals * (1. - is_inside)
        return normals.cuda()

    def __call__(self, video, mask, gen_res=512,
                 reg_loss_weight=1e-1, normal_loss_weight=1e-2, normal_tv_loss_weight=1e-2,):
        '''
        Do not support batch operation - can only inpaint one image at a time.
        :param img: [H, W, 3]
        :param ref_distance: [H, W] or [H, W, 1]
        :param mask: [H, W] or [H, W, 1], value 1 if need to be inpainted otherwise 0
        :return: inpainted distance [H, W]
        '''

        L, height, width, _ = video.shape
        device = video.device
        # video = video.clone().squeeze().permute(2, 0, 1) # [3, H, W]
        video = video.clone().squeeze().permute(0, 3, 1, 2)

        pers_dirs, pers_ratios, to_vecs, down_vecs, right_vecs = [], [], [], [], []
        
        
        for _ in range(12):
            ratio = 1.5
            cur_pers_dirs, cur_pers_ratios, cur_to_vecs, cur_down_vecs, cur_right_vecs = panorama_to_pers_directions(gen_res=gen_res, ratio=ratio, ex_rot='rand')
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

        pers_dirs = pers_dirs.to(device)
        pers_ratios = pers_ratios.to(device)
        to_vecs = to_vecs.to(device)
        down_vecs = down_vecs.to(device)
        right_vecs = right_vecs.to(device)

        rot_w2c = torch.stack([right_vecs / torch.linalg.norm(right_vecs, 2, -1, True),
                               down_vecs / torch.linalg.norm(down_vecs, 2, -1, True),
                               to_vecs / torch.linalg.norm(to_vecs, 2, -1, True)],
                              dim=1)
        
        rot_c2w = torch.linalg.inv(rot_w2c)

        n_pers = len(pers_dirs)
        img_coords = direction_to_img_coord(pers_dirs)
        sample_coords = img_coord_to_sample_coord(img_coords)

        video = rearrange(video, 'l c h w -> (l c) h w')
        pers_imgs_all = F.grid_sample(video[None].expand(n_pers, -1, -1, -1), sample_coords, padding_mode='border') # [n_pers, 3, gen_res, gen_res]
        pers_imgs_all = rearrange(pers_imgs_all, 'n (l c) h w -> n l c h w', c=3)
        pers_mask_all = F.grid_sample(mask[None,...].expand(n_pers, -1, -1, -1), sample_coords, padding_mode='border') # [n_pers, 3, gen_res, gen_res]
        pers_mask_all = pers_mask_all < 0.5 # non-motion area
        pers_mask_all = pers_mask_all.float()
        
        pers_coords = sample_coords


        pers_imgs_pc = pers_imgs_all[:20]
        pers_dirs_pc = pers_dirs[:20]
        pers_mask_all = pers_mask_all[:20]
        n_pers_pc = len(pers_dirs_pc)


        pred_distances_raw = []
       
        far_dirs = []
        far_colors = []

        for i in range(n_pers_pc):
            with torch.no_grad():
                intri = {
                    'fx': fx[i].item(),
                    'fy': fy[i].item(),
                    'cx': cx[i].item(),
                    'cy': cy[i].item()
                }

                # ********************TODO: here psuedo depth
                psuedo_depth_path = os.path.join(self.depth_anything_path, 'depths', f'image_{i+1}_depth.txt')# f'Depth_Anything/depth_vis_ori/temppano/image_{i+1}_depth.txt'
                
            
                model_pred_depth = self.depth_predictor.predict_depth(pers_imgs_pc[i], intri=intri).cuda().clip(0., None)  # [1, 1, res, res]
                pred_depth = model_pred_depth
                
                pred_depth = pred_depth / (pred_depth.mean() + 1e-5)
                pred_depth_ratio = pred_depth * pers_ratios[i].permute(2, 0, 1)[None]
                
                pred_depth = pred_depth.detach().cpu()
                pred_depth_ratio = pred_depth_ratio.detach().cpu()
                pers_imgs = pers_imgs_pc[i].permute(0, 2, 3, 1).detach().cpu().numpy()
                for i_t in range(pers_imgs.shape[0]):
                    saved_img_path = os.path.join(self.img_vis_path, f'img_vis_{i+1}_{i_t}.png')
                    imageio.imwrite(saved_img_path, (pers_imgs[i_t]*255).astype(np.uint8))

                # save
                for i_t in range(pers_imgs.shape[0]):
                    saved_depth_map = pred_depth[i_t, 0].detach().cpu().numpy()
                    saved_pred_depth_ratio = pred_depth_ratio[i_t, 0].detach().cpu().numpy()
                    save_color_map(saved_depth_map, os.path.join(self.depth_vis_path, f'depth_vis_bbox_{i+1}_{i_t}.png'))
                    save_color_map(saved_pred_depth_ratio, os.path.join(self.depth_vis_path, f'depth_vis_bbox_{i+1}_{i_t}_ratio.png'))

                pred_distances_raw.append(pred_depth_ratio)
               
        
        pred_distances_raw_all = [torch.cat([x[k:k+1,...] for x in pred_distances_raw]) for k in range(pred_distances_raw[0].shape[0])]

        new_distances = self.optimize_depth_field(
            pred_distances_raw_all, pers_mask_all,
            pers_dirs_pc, n_pers_pc, gen_res, 
            height, width, reg_loss_weight,
            normal_loss_weight, normal_tv_loss_weight
        )

        return new_distances, rot_w2c, fx, fy, cx, cy, pers_imgs_all, pers_coords


    def get_cams(self, video, mask, gen_res=512,
                 reg_loss_weight=1e-1, normal_loss_weight=1e-2, normal_tv_loss_weight=1e-2):
        '''
        Do not support batch operation - can only inpaint one image at a time.
        :param img: [H, W, 3]
        :param ref_distance: [H, W] or [H, W, 1]
        :param mask: [H, W] or [H, W, 1], value 1 if need to be inpainted otherwise 0
        :return: inpainted distance [H, W]
        '''

        L, height, width, _ = video.shape
        device = video.device

        
        video = video.clone().squeeze().permute(0, 3, 1, 2)

        pers_dirs, pers_ratios, to_vecs, down_vecs, right_vecs = [], [], [], [], []
        
        
        for _ in range(12):
            ratio = 1.5
            cur_pers_dirs, cur_pers_ratios, cur_to_vecs, cur_down_vecs, cur_right_vecs = panorama_to_pers_directions(gen_res=gen_res, ratio=ratio, ex_rot='rand')
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


        pers_dirs = pers_dirs.to(device)
        pers_ratios = pers_ratios.to(device)
        to_vecs = to_vecs.to(device)
        down_vecs = down_vecs.to(device)
        right_vecs = right_vecs.to(device)

        rot_w2c = torch.stack([right_vecs / torch.linalg.norm(right_vecs, 2, -1, True),
                               down_vecs / torch.linalg.norm(down_vecs, 2, -1, True),
                               to_vecs / torch.linalg.norm(to_vecs, 2, -1, True)],
                              dim=1)
        
    

        rot_c2w = torch.linalg.inv(rot_w2c)

        n_pers = len(pers_dirs)
        img_coords = direction_to_img_coord(pers_dirs)
        sample_coords = img_coord_to_sample_coord(img_coords)

        video = rearrange(video, 'l c h w -> (l c) h w')
        pers_imgs_all = F.grid_sample(video[None].expand(n_pers, -1, -1, -1), sample_coords, padding_mode='border') # [n_pers, 3, gen_res, gen_res]
        pers_imgs_all = rearrange(pers_imgs_all, 'n (l c) h w -> n l c h w', c=3)
        pers_mask_all = F.grid_sample(mask[None,...].expand(n_pers, -1, -1, -1), sample_coords, padding_mode='border') # [n_pers, 3, gen_res, gen_res]
        pers_mask_all = pers_mask_all > 0.5 
        pers_mask_all = pers_mask_all.float()
        
        pers_coords = sample_coords

        return None, rot_w2c, fx, fy, cx, cy, pers_imgs_all, pers_coords, pers_mask_all# pers_imgs, pers_coords



    def optimize_depth_field(
        self, 
        raw_distances_all, 
        pers_mask_all,
        pers_dirs_pc, 
        n_pers_pc, 
        gen_res, 
        height, 
        width,
        reg_loss_weight=1e-1, 
        normal_loss_weight=1e-2, 
        normal_tv_loss_weight=1e-2,
        reg_temporal_weight = 1e-1,
        bias_temporal_weight=1e-1,
    ):
        distances = []
        pers_dirs_pc = pers_dirs_pc.permute(0, 3, 1, 2)
    
        prev_scale = None
        prev_bias = None

        for idx__, pred_distances_raw__ in enumerate(raw_distances_all):
            torch.cuda.empty_cache()
            pred_distances_raw = pred_distances_raw__.cuda()

            sup_infos = torch.cat([pers_dirs_pc, pred_distances_raw, pers_mask_all], dim=1)

            scale_params = torch.zeros([n_pers_pc], requires_grad=True)#.cuda()
            
            bias_params_local_distance  = torch.zeros([n_pers_pc, 1, gen_res, gen_res], requires_grad=True)

            sp_dis_field = SphereDistanceField(fine_res = 2048).cuda()

            # Stage 1: Optimize global parameters
            all_iter_steps = 1500
            # TODO: here
            # all_iter_steps = 5
            lr_alpha = 1e-2
            init_lr = 1e-1
            init_lr_sp = 1e-2
            init_lr_local = 1e-1
            local_batch_size = 256

            optimizer_sp = torch.optim.Adam(sp_dis_field.parameters(), lr=init_lr_sp)
            
            optimizer_global = torch.optim.Adam([scale_params], lr=init_lr)

            optimizer_local = torch.optim.Adam([bias_params_local_distance], lr=init_lr_local)

            for phase in ['global', 'hybrid']:
                ###
                ema_loss_for_log = 0.0
                progress_bar = tqdm(range(1, all_iter_steps + 1), desc="Training progress")
                loss_vis = []
                ###
                #for iter_step in tqdm(range(all_iter_steps)):
                for iter_step in range(1, all_iter_steps + 1):
                    progress = iter_step / all_iter_steps
                    if phase == 'global':
                        progress = progress * .5
                    else:
                        progress = progress * .5 + .5

                    lr_ratio = (np.cos(progress * np.pi) + 1.) * (1. - lr_alpha) + lr_alpha
                    for g in optimizer_global.param_groups:
                        g['lr'] = init_lr * lr_ratio
                    for g in optimizer_local.param_groups:
                        g['lr'] = init_lr_local * lr_ratio
                    for g in optimizer_sp.param_groups:
                        g['lr'] = init_lr_sp * lr_ratio

                    #idx = np.random.randint(low=0, high=n_pers)
                    sample_coords = torch.rand(n_pers_pc, local_batch_size, 1, 2).cuda() * 2. - 1           # [n_pers, local_batch_size, 1, 2] range (-1, +1)
                    cur_sup_info = F.grid_sample(sup_infos, sample_coords, padding_mode='border')        # [n_pers, 7, local_batch_size, 1]
                    distance_bias = F.grid_sample(bias_params_local_distance.cuda(), sample_coords, padding_mode='border')  # [n_pers, 1, local_batch_size, 1]
                    distance_bias = distance_bias[:, :, :, 0].permute(0, 2, 1)  

                    dirs = cur_sup_info[:, :3, :, 0].permute(0, 2, 1)                                    # [n_pers, local_batch_size, 3]
                    dirs = dirs / torch.linalg.norm(dirs, 2, -1, True)

                    ref_pred_distances = cur_sup_info[:, 3: 4, :, 0].permute(0, 2, 1)                         # [n_pers, local_batch_size, 1]
                    ref_pred_distances = ref_pred_distances * F.softplus(scale_params[:, None, None].cuda())  # [n_pers, local_batch_size, 1]
                    ref_pred_distances = ref_pred_distances + distance_bias

                    sample_mask = cur_sup_info[:, 4: 5, :, 0].permute(0, 2, 1)     
                    sample_mask = sample_mask > 0.5

                    pred_distances = sp_dis_field(dirs.reshape(-1, 3), requires_grad=False)
                    pred_distances = pred_distances.reshape(n_pers_pc, local_batch_size, 1)


                    distance_loss = F.smooth_l1_loss(ref_pred_distances, pred_distances, beta=5e-1, reduction='mean')

                    reg_loss = ((F.softplus(scale_params).mean() - 1.)**2).mean()
                    if prev_scale is not None:
                        reg_loss_temporal = ((F.softplus(scale_params)-F.softplus(prev_scale))**2).mean()
                    else:
                        reg_loss_temporal = 0.
                    if phase == 'hybrid':
                        distance_bias_local = bias_params_local_distance
                        distance_bias_tv_loss = F.smooth_l1_loss(distance_bias_local[:, :, 1:, :], distance_bias_local[:, :, :-1, :], beta=1e-2) + \
                                                F.smooth_l1_loss(distance_bias_local[:, :, :, 1:], distance_bias_local[:, :, :, :-1], beta=1e-2)
 
                        if sample_mask.sum() > 0 and prev_bias is not None:
                            sampled_prev_bias = F.grid_sample(prev_bias.cuda(), sample_coords, padding_mode='border')  # [n_pers, 1, local_batch_size, 1]
                            sampled_prev_bias = sampled_prev_bias[:, :, :, 0].permute(0, 2, 1)
                            bias_temporal_loss = ((distance_bias[sample_mask] - sampled_prev_bias[sample_mask])**2).sum() / sample_mask.sum()
                        else:
                            bias_temporal_loss = 0.
                    else:
                        distance_bias_tv_loss = 0.
                        bias_temporal_loss = 0.
     
                    
                    loss = distance_loss +\
                        reg_loss * reg_loss_weight +\
                        distance_bias_tv_loss +\
                        reg_loss_temporal * reg_temporal_weight +\
                        bias_temporal_loss * bias_temporal_weight
                

                    optimizer_global.zero_grad()
                    optimizer_sp.zero_grad()
                    if phase == 'hybrid':
                        optimizer_local.zero_grad()

                    loss.backward()
                    optimizer_global.step()
                    optimizer_sp.step()
                    if phase == 'hybrid':
                        optimizer_local.step()

                    with torch.no_grad():
                        # Progress bar
                        ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                        loss_vis.append(ema_loss_for_log) ###
                        if iter_step % 1 == 0:
                            progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                            progress_bar.update(1)
                        if iter_step == all_iter_steps:
                            progress_bar.close()

            # Get new distance map
            pano_dirs = img_coord_to_pano_direction(img_coord_from_hw(height, width)).cuda()
            new_distances = sp_dis_field(pano_dirs.reshape(-1, 3), requires_grad=False)
            new_distances = new_distances.detach().reshape(height, width, 1)
            distances.append(new_distances.detach().cpu())

            if idx__ == 0:
                prev_scale = scale_params.clone().detach().cpu()        
                prev_bias = bias_params_local_distance.clone().detach().cpu()
        
        return distances 

