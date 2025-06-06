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

from scenegif4K.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
### midas ###
from utils.depth_utils import estimate_depth
#############

# ### depth anything ###
# import sys
# import os
# sys.path.append(os.path.join(os.path.abspath("."), "Depth-Anything-TorchVersion"))
# from depth_anything.dpt import DepthAnything
# from estimate_depth import depth_anything ###
# ######################

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    depth = estimate_depth(gt_image.cuda()).cpu().numpy() ### midas
    # depth = depth_anything(gt_image.cuda(), 'vits', model = model).cpu().numpy()
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device, depth_image=depth)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    # encoder = 'vits'
    # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # DepthAnything_model = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(DEVICE).eval()

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

###
import torch
import numpy as np
from trimesh.creation import icosphere as IcoSphere

from dataclasses import dataclass

from utils.debug_utils import printarr

@dataclass
class Rays:
    o: torch.Tensor  # [..., 3]
    d: torch.Tensor  # [..., 3]

    def __len__(self):
        return len(self.o)
    def __getitem__(self, indices):
        return Rays(self.o[indices], self.d[indices])

    def collapse(self):
        return self.o, self.d

@dataclass
class BoundedRays:
    o: torch.Tensor     # [..., 3]
    d: torch.Tensor     # [..., 3]
    near: torch.Tensor  # [..., 1]
    far: torch.Tensor   # [..., 1]

    def __len__(self):
        return len(self.o)
    def __getitem__(self, indices):
        return BoundedRays(self.o[indices], self.d[indices], self.near[indices], self.far[indices])

    def collapse(self):
        return self.o, self.d, self.near, self.far


def cat_rays(rays):
    rays_o = torch.cat([_.o for _ in rays], dim=0)
    rays_d = torch.cat([_.d for _ in rays], dim=0)
    return Rays(rays_o, rays_d)


def apply_rot(pts, rot_mat):
    assert rot_mat.shape == (3, 3)
    return torch.matmul(rot_mat, pts[..., None])[..., 0]


def apply_rot_trans(pts, rot_mat, pos):
    assert rot_mat.shape == (3, 3)
    assert pos.shape == (3,)
    return torch.matmul(rot_mat, pts[..., None])[..., 0] + pos


def apply_transform(pts, pose):
    return apply_rot_trans(pts, pose[:3, :3], pose[:3, 3])


# Camera rays, OpenCV style
def cam_rays_cam_space(height: int, width=-1, fovy=np.deg2rad(90.), aspect_ratio=1.):
    '''
    OpenCV style!
    :param height:
    :param width:
    :param fovy:
    :param aspect_ratio:
    :return: Tensor with shape [height, width, 3]
    '''
    if width < 0:
        width = int(np.round(height * aspect_ratio))
    else:
        aspect_ratio = width / height

    span_y = np.tan(fovy * .5)
    span_x = span_y * aspect_ratio
    y = torch.linspace(-span_y, span_y, height)
    x = torch.linspace(-span_x, span_x, width)
    y, x = torch.meshgrid(y, x, indexing='ij')
    xyz = torch.stack([x, y, torch.ones_like(x)], -1)
    return xyz / torch.linalg.norm(xyz, 2, -1, True)


def look_at(to_vec, up_vec=None):
    '''
    :param to_vec: [n, 3]
    :param up_vec: [n, 3]
    :return: rotation matrices [n, 3, 3]
    '''
    n = to_vec.shape[0]
    if up_vec is None:
        up_vec = torch.cat([torch.zeros([n, 2]), torch.ones([n, 1])], -1)
    down_vec = -up_vec
    to_vec = to_vec / torch.linalg.norm(to_vec, 2, -1, True)
    ri_vec = torch.linalg.cross(down_vec, to_vec)
    ri_vec = ri_vec / torch.linalg.norm(ri_vec, 2, -1, True)
    down_vec = torch.linalg.cross(to_vec, ri_vec)
    c2w = torch.stack([ri_vec, down_vec, to_vec], 2)
    return c2w

def ang2vec(angles):
    '''
    :param angles: [n, 2]
    :return: [n, 3]
    '''
    ang_x, ang_y = angles[..., 0], angles[..., 1]
    vecs = torch.stack([torch.cos(ang_x) * torch.cos(ang_y),
                        torch.sin(ang_x) * torch.cos(ang_y),
                        torch.sin(ang_y)], dim=-1)

    return vecs


def img_coord_from_hw(h, w):
    i = torch.linspace(.5 / h, 1. - .5 / h, h)
    j = torch.linspace(.5 / w, 1. - .5 / w, w)
    ii, jj = torch.meshgrid(i, j, indexing='ij')
    return torch.stack([ii, jj], -1)


def img_to_pano_coord(coords):
    '''
    :param coords: [n, 2] range of [0, 1]. (row coord, col coord)
    :return: pano coords
    '''
    y, x = coords[..., 0], coords[..., 1]
    return torch.stack([-(y - .5) * np.pi, -(x - .5) * 2. * np.pi], -1)


def pano_to_img_coord(coords):
    y, x = coords[..., 0], coords[..., 1]
    return torch.stack([-y / np.pi + .5, -(x / (2. * np.pi)) + .5], -1)


def direction_to_pano_coord(dirs):
    dirs = dirs / torch.linalg.norm(dirs, 2, -1, True)
    beta = torch.arcsin(dirs[..., 2])
    xy = dirs[..., :2] / torch.cos(beta)[..., None]
    alpha = torch.view_as_complex(xy).angle()   # [-np.pi., np.pi]
    return torch.stack([beta, alpha], -1)


def pano_coord_to_direction(coords):
    beta, alpha = coords[..., 0], coords[..., 1]
    dirs = torch.stack([torch.cos(alpha) * torch.cos(beta),
                        torch.sin(alpha) * torch.cos(beta),
                        torch.sin(beta)], dim=-1)
    return dirs


def direction_to_img_coord(dirs):
    return pano_to_img_coord(direction_to_pano_coord(dirs))


def img_coord_to_pano_direction(coords):
    return pano_coord_to_direction(img_to_pano_coord(coords))

@torch.no_grad()
def direction_to_pers_img_coord(dirs, to_vec, down_vec, right_vec):
    eps = 1e-5
    dirs = dirs / torch.linalg.norm(dirs, 2, -1, True)
    to_vec_len = torch.linalg.norm(to_vec, 2, -1).item()
    to_vec = to_vec / to_vec_len
    down_vec = down_vec / to_vec_len
    right_vec = right_vec / to_vec_len
    down_vec_len = torch.linalg.norm(down_vec, 2, -1).item()
    right_vec_len = torch.linalg.norm(right_vec, 2, -1).item()

    project_len = (dirs * to_vec).sum(-1, True)
    mask = project_len > eps
    project_len = project_len.clip(eps, None)
    dirs = dirs / project_len

    i = ((dirs - to_vec) * down_vec).sum(-1, True) / down_vec_len**2
    j = ((dirs - to_vec) * right_vec).sum(-1, True) / right_vec_len**2
    mask = (mask & (i.abs() <= 1.) & (j.abs() <= 1.)).float()
    ij = (torch.cat([i, j], dim=-1) + 1.) * .5
    return ij, mask


def img_coord_to_sample_coord(coords):
    return torch.stack([coords[..., 1], coords[..., 0]], -1) * 2. - 1.


def get_rand_horizontal_points(batch_size, dim=3):
    rs = torch.sqrt(torch.rand(batch_size))
    theta = (torch.rand(batch_size) * 2. - 1.) * np.pi
    pos = [rs * torch.cos(theta), rs * torch.sin(theta)]
    if dim == 3:
        pos += [ torch.zeros([batch_size]) ]

    return torch.stack(pos, -1)

def get_panorama_sphere_points(h, w):
    img_coords = img_coord_from_hw(h, w)
    pts = img_coord_to_pano_direction(img_coords)
    pts = pts / torch.linalg.norm(pts, 2, -1, True)
    return pts

def pers_depth_to_normal(depth, down_len, right_len):
    assert depth.min().item() > 1e-5
    if len(depth.shape) == 2:
        depth = depth[..., None]
    h, w, _ = depth.shape
    ii, jj = torch.meshgrid(
        torch.linspace(.5 / h, 1. - .5 / h, h),
        torch.linspace(.5 / w, 1. - .5 / w, w),
        indexing='ij'
    )
    z = torch.ones_like(ii)
    x = (jj * 2. - 1.) * right_len
    y = (ii * 2. - 1.) * down_len
    pts = torch.stack([x, y, z], dim=-1)
    pts = pts * depth
    right_vec = pts[:-1, 1:] - pts[:-1, :-1]
    down_vec  = pts[1:, :-1] - pts[:-1, :-1]
    # right_vec_len = torch.linalg.norm(right_vec, 2, -1, True)
    # down_vec_len = torch.linalg.norm(down_vec, 2, -1, True)
    right_vec = right_vec / torch.linalg.norm(right_vec, 2, -1, True).detach()
    down_vec = down_vec / torch.linalg.norm(down_vec, 2, -1, True).detach()
    to_vec = torch.cross(right_vec, down_vec)
    # to_vec_len = torch.linalg.norm(to_vec, 2, -1, True)
    to_vec = to_vec / torch.linalg.norm(to_vec, 2, -1, True).detach()
    assert not torch.any(torch.isnan(to_vec))
    return -to_vec


# -----------

def gen_pano_rays(pose, height=512, width=1024):
    img_coord = img_coord_from_hw(height, width)
    rays_d = img_coord_to_pano_direction(img_coord)
    rays_d = apply_rot(rays_d, pose[:3, :3])
    rays_o = pose[None, None, :3, 3].repeat(height, width, 1)
    return Rays(rays_o, rays_d)


def gen_pers_rays(pose, fov, res):
    rays_d = cam_rays_cam_space(height=res, width=res, fovy=fov)
    rays_o = torch.zeros_like(rays_d) + pose[:3, 3][None, None, :]
    rays_d = apply_rot(rays_d, pose[:3, :3])
    return Rays(rays_o, rays_d)