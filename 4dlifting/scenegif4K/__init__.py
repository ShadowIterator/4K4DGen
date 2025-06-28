#
# we build this on the top of https://github.com/ShijieZhou-UCLA/DreamScene360
#


from math import dist
import os
import random
import json
from re import T
from utils.system_utils import searchForMaxIteration
from scenegif4K.dataset_readers import sceneLoadTypeCallbacks, CameraInfo, SceneInfo ###
from scenegif4K.gaussian_model import GaussianModel, BasicPointCloud ###
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, img_coord_to_pano_direction, direction_to_img_coord 
from geo_predictors.pano_joint_predictor import *
from utils.utils import read_image
from utils.sh_utils import SH2RGB
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from PIL import Image, ImageSequence
import trimesh
import cv2 as cv
import imageio


from utils.save_data import save_data_k

import sys
import importlib

import time 


sys.path.append('stitch_diffusion/kohya_trainer')




def tensor_to_np(t:torch.Tensor) -> np.array :
    if len(t.shape) == 4:
        t = t[0]
    if len(t.shape) == 2:
        t = torch.stack([t, t, t], dim=0)
        
    if t.shape[0] == 3 or t.shape[0] == 1:
        t = t.permute(1,2,0)
    if t.shape[2] == 1:
        t = torch.cat([t, t, t], dim=2)
    if t.min() < 0:
        t = (t + 1) / 2
    if t.max() < 1.3:
        t = t * 255
    t = t.detach().cpu().numpy().astype(np.uint8)
    return t



def save_color_map(depth, path, mask=None):
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
    if mask is not None:
        depth[mask] = 255
    cv2.imwrite(path, depth)

def pcd_from_depths(pano_img, distances, height, width, source_path):

    pano_dirs = img_coord_to_pano_direction(img_coord_from_hw(height, width)).cuda()
    scale = distances.max().item() * 0.7 #* 0.8 #* 1.05
    distances /= scale
    pts = pano_dirs * distances.squeeze()[..., None].cuda()
    pts = pts.cpu().numpy().reshape(-1, 3)

    return pts

def pcd_from_depths_ls(colors, dirs, distances, height, width, source_path, x_scale):
    if isinstance(distances, torch.Tensor):
        scale = distances.max().item() * 0.7 #* 0.8 #* 1.05
    else:
        scale = distances * 0.7
    distances /= scale
    distances *= x_scale
    pts = dirs * distances
    pts = pts.cpu().numpy().reshape(-1, 3)

    return pts


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def get_info_from_params(source_path, pano_img, distances, rot_w2c, fx, fy, cx, cy, pers_imgs, pts):
        H, W, _ = pano_img.shape
        n_pers, _, h, w = pers_imgs.shape
        cam_infos_unsorted = []
        cam_perturbation_infos_unsorted = [] ###
        cam_perturbation_infos_unsorted_stage2 = [] ###
        cam_perturbation_infos_unsorted_stage3 = [] ###
        for i in range(n_pers):
            with torch.no_grad():
                img = pers_imgs[i].cpu().numpy()
                img = img.transpose(1, 2, 0)
                img = (img*255).astype('uint8')
                img = Image.fromarray(img)
                intri = {
                    'fx': fx[i].item(),
                    'fy': fy[i].item(),
                    'cx': cx[i].item(),
                    'cy': cy[i].item()
                }
                fovx = focal2fov(intri['fx'], w)
                fovy = focal2fov(intri['fy'], h)
                R = np.transpose(np.asarray(rot_w2c[i].cpu()))
                T = np.transpose(np.array( [0, 0, 0]))
                T_perturbation = T + np.random.uniform(-0.05, 0.05, size=(1, 3)) ###
                uid = i
                image_name = 'image' + str(i)
                try:
                    os.mkdir( os.path.join ( source_path, 'images'))
                except Exception as e:
                    pass
                image_path = os.path.join(source_path, 'images', image_name)

                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=fovy, FovX=fovx, image=img,
                              image_path=image_path, image_name=image_name, width=w, height=h)      
                cam_infos_unsorted.append(cam_info)
                ### stage 1 perturbation
                cam_perturbation_info = CameraInfo(uid=uid, R=R, T=T_perturbation, FovY=fovy, FovX=fovx, image=img,
                              image_path=image_path, image_name=image_name, width=w, height=h)
                cam_perturbation_infos_unsorted.append(cam_perturbation_info)
                ### stage 2 perturbation
                T_perturbation_stage2 = T + np.random.uniform(-0.05 * 2, 0.05 * 2, size=(1, 3))
                cam_perturbation_info_stage2  = CameraInfo(uid=uid, R=R, T=T_perturbation_stage2, FovY=fovy, FovX=fovx, image=img,
                              image_path=image_path, image_name=image_name, width=w, height=h)
                cam_perturbation_infos_unsorted_stage2 .append(cam_perturbation_info_stage2)
                ### stage 3 perturbation
                T_perturbation_stage3 = T + np.random.uniform(-0.05 * 4, 0.05 * 4, size=(1, 3))
                cam_perturbation_info_stage3  = CameraInfo(uid=uid, R=R, T=T_perturbation_stage3, FovY=fovy, FovX=fovx, image=img,
                              image_path=image_path, image_name=image_name, width=w, height=h)
                cam_perturbation_infos_unsorted_stage3 .append(cam_perturbation_info_stage3)



        cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
        cam_perturbation_infos = sorted(cam_perturbation_infos_unsorted.copy(), key = lambda x : x.image_name) ###
        cam_perturbation_infos_stage2 = sorted(cam_perturbation_infos_unsorted_stage2.copy(), key = lambda x : x.image_name) ###
        cam_perturbation_infos_stage3 = sorted(cam_perturbation_infos_unsorted_stage3.copy(), key = lambda x : x.image_name) ###
        llffhold = 8
        if eval:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
            perturbation_cam_infos = cam_perturbation_infos ###
        else:
            train_cam_infos = cam_infos
            test_cam_infos = []
            perturbation_cam_infos = cam_perturbation_infos ###
        nerf_normalization = getNerfppNorm(train_cam_infos)

        xyz = pts #pcd_from_depths(pano_img, distances, H, W, source_path)
        vertex_colors = pano_img.reshape(-1, 3).cpu().numpy()
        ply_path = os.path.join(source_path, 'sparse/0/points3D.ply')
        pcd = BasicPointCloud(points = xyz, colors=vertex_colors, normals=np.zeros_like(xyz))
        scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           perturbation_cameras_stage1=perturbation_cam_infos, ###
                           perturbation_cameras_stage2=cam_perturbation_infos_stage2, ###
                           perturbation_cameras_stage3=cam_perturbation_infos_stage3, ###
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)

        return scene_info


class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel,  api_key, prompt_eng, num_prompt, max_rounds, load_iteration=None, shuffle=True, resolution_scales=[1.0], generate_geo=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.perturbation_cameras_stage1 = {} ###
        self.perturbation_cameras_stage2 = {} ###
        self.perturbation_cameras_stage3 = {} ###

        if generate_geo:
            self.generate_from_gif(args, gaussians, api_key, prompt_eng, num_prompt, max_rounds, load_iteration, shuffle, resolution_scales)
        else:
            self.init_scene(args, gaussians, api_key, prompt_eng, num_prompt, max_rounds, load_iteration, shuffle, resolution_scales)
            
    # def __init__(self, args : ModelParams, gaussians : GaussianModel,  api_key, prompt_eng, num_prompt, max_rounds, load_iteration=None, shuffle=True, resolution_scales=[1.0], generate_geo=False):

    def init_scene(self, args : ModelParams, gaussians : GaussianModel,  api_key, prompt_eng, num_prompt, max_rounds, load_iteration=None, shuffle=True, resolution_scales=[1.0], generate_geo=False):
        ## Change loading multi views data to pano ###
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval) 
        if not self.loaded_iter:
            #with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
            #    dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.perturbation_cameras_stage1)  ###
            random.shuffle(scene_info.perturbation_cameras_stage2)  ###
            random.shuffle(scene_info.perturbation_cameras_stage3)  ###

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            print("Loading Perturbation Cameras") ###
            self.perturbation_cameras_stage1[resolution_scale] = cameraList_from_camInfos(scene_info.perturbation_cameras_stage1, resolution_scale, args)
            self.perturbation_cameras_stage2[resolution_scale] = cameraList_from_camInfos(scene_info.perturbation_cameras_stage2, resolution_scale, args)
            self.perturbation_cameras_stage3[resolution_scale] = cameraList_from_camInfos(scene_info.perturbation_cameras_stage3, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

        
    def generate_from_gif(self, args : ModelParams, gaussians : GaussianModel,  api_key, prompt_eng, num_prompt, max_rounds, load_iteration=None, shuffle=True, resolution_scales=[1.0],):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.perturbation_cameras_stage1 = {} ###
        self.perturbation_cameras_stage2 = {} ###
        self.perturbation_cameras_stage3 = {} ###

        assert any(filename.endswith('.gif') or filename.startswith('__gif_frames') for filename in os.listdir(args.source_path))

        im_size = (4096, 2048)
        
        
        files = [f for f in os.listdir(args.source_path) if f.endswith('.gif') or f.startswith('__gif_frames')]

        if files[0].endswith('.gif'):
            img_path = os.path.join(args.source_path, files[0]) ### only 1 pano image in the folder
            motion_mask_path = img_path.replace('.gif','_mask.png')
            
            gif_data = Image.open(img_path)
            frames = []
            for k, frame in enumerate(ImageSequence.Iterator(gif_data)):
                frame = frame.convert('RGB')
                frame = frame.resize((4096, 2048))
                np_curr = np.asarray(frame).astype(np.uint8) # should be H W 3, uint8
                torch_curr = read_image(np_curr, to_torch=True, squeeze=True, pass_img=True).cuda()
                frames.append(torch_curr)

        else:
            img_folder = os.path.join(args.source_path, files[0])
            motion_mask_path = img_folder.replace('__gif_frames.','')+'_mask.png'
            frames_id = os.listdir(img_folder)
            frames_id = [x for x in frames_id if x.endswith('.png')]
            frames_id = sorted(frames_id)
            frames = []
            for frame_id in frames_id:
                input_path = os.path.join(img_folder, frame_id)
                frame = Image.open(input_path)
                frame = frame.convert('RGB')
                frame = frame.resize(im_size)
                
                np_curr = np.asarray(frame).astype(np.uint8) # should be H W 3, uint8
                torch_curr = read_image(np_curr, to_torch=True, squeeze=True, pass_img=True).cuda()
                frames.append(torch_curr)


        # frames = frames[:2]
        frames_torch = torch.stack(frames)

        img = frames_torch[0]
        
        im_gen_res = im_size[1] // 2
        img = cv.resize(img.cpu().numpy(), im_size, cv.INTER_AREA)
        
        img = torch.from_numpy(img).cuda()

        motion_mask = Image.open(motion_mask_path)
        motion_mask = motion_mask.resize(im_size)
        motion_mask = np.asarray(motion_mask)
        motion_mask = (motion_mask != 0)
        motion_mask = torch.from_numpy(motion_mask)
        # motion_mask = motion_mask
        motion_mask = motion_mask.float()
        
        start_time = time.time()
        
        joint_predictor = PanoJointPredictorGIF(args)
        height, width, _ = img.shape
        distances_list, _, _, _, _, _, _, _ = joint_predictor(frames_torch,motion_mask.cuda())

        _, rot_w2c, fx, fy, cx, cy, pers_imgs, pers_coords, pers_mask = joint_predictor.get_cams(frames_torch,motion_mask.cuda(), gen_res=im_gen_res)
        
        diff_vis_folder = os.path.join(args.source_path, '__vis_diff_panodepth')
        os.makedirs(diff_vis_folder,exist_ok=True)  
        
        mask_vis_dir = os.path.join(args.source_path, '__vis_mask_pers')
        os.makedirs(mask_vis_dir,exist_ok=True) 
        
        for k in range(pers_mask.shape[0]):
            _mask = pers_mask[k]
            save_p = os.path.join(mask_vis_dir, f'mask_{k+1}.png') 
            imageio.imwrite(save_p, tensor_to_np(_mask))
            
        for k in range(1, len(distances_list)):
            diff = (distances_list[k] - distances_list[0])[:,:,0]
            diff = torch.abs(diff)
            diff = diff / diff.max()
            diff_vis_path = os.path.join(diff_vis_folder, '''diff_{:02d}.png'''.format(k))
            imageio.imwrite(diff_vis_path, (diff*255).detach().cpu().numpy().astype(np.uint8))

        panodepth_vis_folder = os.path.join(args.source_path, '__vis_panodepth')
        os.makedirs(panodepth_vis_folder, exist_ok=True)
        for k in range(len(distances_list)):
            dist = distances_list[k]
            save_path_dist = os.path.join(panodepth_vis_folder, '''depth_{:02d}.png'''.format(k))
            save_color_map(dist.detach().cpu().numpy(), save_path_dist)
            np.save(save_path_dist.replace('.png', '.npy'), dist.detach().cpu().numpy())

        for k, distances in enumerate(distances_list):
            img = frames_torch[k]
            pts = pcd_from_depths(img, distances, height, width, args.source_path)
            colors = None
            
            print('Saving data for future use...')
            save_data_k(args.source_path, k, img, distances, rot_w2c, fx, fy, cx, cy, pers_imgs, pts, pers_coords, colors)
        
        return 


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getPerturbationCameras(self, stage, scale=1.0): ###
        if stage == 1:
            return self.perturbation_cameras_stage1[scale]
        elif stage == 2:
            return self.perturbation_cameras_stage2[scale]
        elif stage == 3:
            return self.perturbation_cameras_stage3[scale]