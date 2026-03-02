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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
from PIL import Image
import os
import torch.nn.functional as F
import sys
import torch

WARNED = False


def loadCam(args, id, cam_info, resolution_scale, load_images=True):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(
            orig_h / (resolution_scale * args.resolution))
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
        resolution = (int(orig_w / scale), int(orig_h / scale))  # [1600,910]

    _normal = None
    resized_segmap = None
    resized_sorted_segmap = None

    if len(cam_info.image.split()) > 3:
        import torch
        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in cam_info.image.split()[:3]], dim=0)
        loaded_mask = PILtoTorch(cam_info.image.split()[3], resolution)
        gt_image = resized_image_rgb
    else:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        loaded_mask = None
        gt_image = resized_image_rgb

    if load_images:
        if args.w_normal_prior:
            normal_path = os.path.join(os.path.dirname(os.path.dirname(cam_info.image_path)), args.w_normal_prior,
                                       os.path.basename(cam_info.image_path))
            if os.path.exists(normal_path[:-4] + '.npy'):
                _normal = torch.tensor(np.load(normal_path[:-4] + '.npy'))
                _normal = - (_normal * 2 - 1)
                resized_normal = F.interpolate(_normal.unsqueeze(0), size=resolution[::-1], mode='bicubic')
                _normal = resized_normal.squeeze(0)
            else:
                _normal = Image.open(normal_path[:-4] + '.png')
                resized_normal = PILtoTorch(_normal, resolution)
                resized_normal = resized_normal[:3]
                _normal = - (resized_normal * 2 - 1)
            # normalize normal
            _normal = _normal.permute(1, 2, 0) @ (torch.tensor(np.linalg.inv(cam_info.R)).float())
            _normal = _normal.permute(2, 0, 1)
        else:
            _normal = None

        segmap_name = os.path.basename(cam_info.image_path).split(".")[0] + ".png"
        segmap_type = "mask_filtered" if (os.path.exists(os.path.join(args.source_path, "sam/mask_filtered")) \
                                          and args.load_filter_segmap) else "mask"
        segmap_path = os.path.join(args.source_path, f"sam/{segmap_type}", segmap_name)
        if args.use_seg_feature and os.path.exists(segmap_path):
            _segmap = Image.open(segmap_path)
            resized_segmap = PILtoTorch(_segmap, resolution, resize_type=Image.NEAREST, scale=False)
        else:
            resized_segmap = None

        segmap_path = os.path.join(args.source_path, "sam/mask_sorted", segmap_name)
        resized_sorted_segmap = None
        if os.path.exists(segmap_path):
            _segmap = Image.open(segmap_path)
            resized_sorted_segmap = PILtoTorch(_segmap, resolution, resize_type=Image.NEAREST, scale=False)

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, normal=_normal, gt_alpha_mask=loaded_mask,
                  segmap=resized_segmap, sorted_segmap=resized_sorted_segmap,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)


def cameraList_from_camInfos(cam_infos, resolution_scale, args, load_images):
    camera_list = []

    for id, c in enumerate(cam_infos):
        sys.stdout.write('\r')
        sys.stdout.write("Loading camera info {}/{}".format(id + 1, len(cam_infos)))
        sys.stdout.flush()
        camera_list.append(loadCam(args, id, c, resolution_scale, load_images))

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id': id,
        'img_name': camera.image_name,
        'width': camera.width,
        'height': camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy': fov2focal(camera.FovY, camera.height),
        'fx': fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
