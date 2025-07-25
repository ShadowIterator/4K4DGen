#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from re import T
import sys
#sys.path.append('Depth-Anything-TorchVersion')
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, cosine_similarity_loss
from gaussian_renderer import render #, network_gui
from torchmetrics.functional.regression import pearson_corrcoef
import sys
from scenegif4K import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np ###
import matplotlib.pyplot as plt ###
from utils.feature_extractor import get_Feature_from_DinoV2
### midas ###
from utils.depth_utils import estimate_depth
#############
from utils.general_utils import PILtoTorch
from PIL import Image



try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def generate_geo_from_gif(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, api_key, prompt_eng, num_prompt, max_rounds):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, api_key, prompt_eng, num_prompt, max_rounds, generate_geo=True)

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, api_key, prompt_eng, num_prompt, max_rounds):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, api_key, prompt_eng, num_prompt, max_rounds)

    # return
    input_folder = dataset.source_path
    base_folder = '/'.join(input_folder.split('/')[:-1])
    mask_folder = os.path.join(base_folder, '__vis_mask_pers') 
    frame_id = int(input_folder.split('/')[-1])
    if frame_id > 0:
        ref_folder = os.path.join(base_folder, '''{:02d}'''.format(frame_id))
    else:
        ref_folder = input_folder
    
    
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    
    cams = scene.getTrainCameras().copy()
    mask_d = {}
    ref_d = {}
    
    for cam in cams:
        _, rh, rw = cam.original_image.shape
        _res = (rh, rw)
        _name = cam.image_name
        _mask_path = os.path.join(mask_folder, f'mask_' + _name.split('_')[-1] + '.png')
        _ref_path = os.path.join(ref_folder, 'images', f'{_name}.png')
        mask_d[_name] = PILtoTorch(Image.open(_mask_path), _res)[0:1,...]
        ref_d[_name] = PILtoTorch(Image.open(_ref_path).convert('RGB'), _res)
    
    viewpoint_stack = None
    perturbation_viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):   
     
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))


        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background


        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        rendered_depth = render_pkg["depth"] ###
        gt_depth = torch.tensor(viewpoint_cam.depth_image).cuda() ###
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ref_image = ref_d[viewpoint_cam.image_name].cuda()
        mask_image = mask_d[viewpoint_cam.image_name].cuda()
        Lreg = (torch.abs((image - ref_image))*mask_image).mean()
        depth_weight = 0.05  
        loss_depth = depth_weight * (1 - pearson_corrcoef(rendered_depth.reshape(-1, 1), - gt_depth.reshape(-1, 1)))
        
        loss =  (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + depth_weight * loss_depth + 0.1 * Lreg
        loss_feature = torch.tensor(0).cuda() 
        loss_perturbation_depth = torch.tensor(0).cuda() 
        
        if iteration > 5400:
            if iteration > 5400 and iteration <= 6600:
                if not perturbation_viewpoint_stack:
                    perturbation_viewpoint_stack = scene.getPerturbationCameras(stage=1).copy()
                perturbation_viewpoint_cam = perturbation_viewpoint_stack.pop(randint(0, len(perturbation_viewpoint_stack)-1))
            elif iteration > 6600 and iteration <= 7800:
                if not perturbation_viewpoint_stack:
                    perturbation_viewpoint_stack = scene.getPerturbationCameras(stage=2).copy()
                perturbation_viewpoint_cam = perturbation_viewpoint_stack.pop(randint(0, len(perturbation_viewpoint_stack)-1))
            elif iteration <= 9000:
                if not perturbation_viewpoint_stack:
                    perturbation_viewpoint_stack = scene.getPerturbationCameras(stage=3).copy()
                perturbation_viewpoint_cam = perturbation_viewpoint_stack.pop(randint(0, len(perturbation_viewpoint_stack)-1))

            perturbation_render_pkg = render(perturbation_viewpoint_cam, gaussians, pipe, bg)
            perturbation_image, perturbation_rendered_depth= perturbation_render_pkg["render"], perturbation_render_pkg["depth"]

            
            pred_depth = estimate_depth(perturbation_image)
            loss_perturbation_depth = 1 - pearson_corrcoef(perturbation_rendered_depth.reshape(-1, 1), -pred_depth.reshape(-1, 1))

        
            if torch.isnan(loss_perturbation_depth).sum() == 0:

            
                loss += depth_weight * loss_perturbation_depth
            
            ### feature loss
            pred_feature = get_Feature_from_DinoV2(perturbation_image)# (1, 768)
            ref_image = perturbation_viewpoint_cam.original_image.cuda()
            ref_feature = get_Feature_from_DinoV2(ref_image)
            loss_feature = cosine_similarity_loss(pred_feature, ref_feature)
            
            feature_loss_weight = 0.05
            loss += feature_loss_weight * loss_feature 

        # loss_feature = torch.tensor(0).cuda()
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar

            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss_feature, loss_depth, loss, l1_loss, loss_perturbation_depth, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background)) ###
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss_feature, loss_depth, loss, l1_loss, loss_perturbation_depth, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/feature_loss', loss_feature.item(), iteration) ###
        tb_writer.add_scalar('train_loss_patches/depth_loss', loss_depth.item(), iteration) ###
        tb_writer.add_scalar('train_loss_patches/loss_perturbation_depth', loss_perturbation_depth.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 9_000, 10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--prompt_eng", action='store_true', default=False)
    parser.add_argument("--num_prompt", type=int, default = 3)
    parser.add_argument("--max_rounds", type=int, default = 3)
    # parser.add_argument("--depth_estimator", type=str, default = 'MIDAS')
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)


    
    torch.autograd.set_detect_anomaly(args.detect_anomaly)


        
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.api_key, args.prompt_eng, args.num_prompt, args.max_rounds)

    

    # All done
    print("\nTraining complete.")
