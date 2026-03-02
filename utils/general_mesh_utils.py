import open3d as o3d
import numpy as np
import trimesh
import tqdm

import torch
from kornia import create_meshgrid
import cv2


def get_ray_directions(H, W, K):
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)  # i->width,j->height
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24json
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    directions = \
        torch.stack([(i - cx) / fx, (j - cy) / fy, torch.ones_like(i)], -1)  # (H, W, 3)
    return directions, torch.stack([i, j, torch.ones_like(i)], -1)  # [x,y]


def get_rays(directions, c2w):
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T  # (H, W, 3)
    rays_d_norm = torch.norm(rays_d, dim=-1, keepdim=True)
    rays_d = rays_d / rays_d_norm  # mlp要归一化
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape)  # (H, W, 3)

    return rays_o, rays_d, rays_d_norm  # depth


def dilate_mask(mask, kernel_size_erode=10, kernel_size_dilate=15, mode="de"):
    kernel_erode = np.ones((kernel_size_erode, kernel_size_erode), np.uint8)  # rubbish -> 15
    kernel_dilate = np.ones((kernel_size_dilate, kernel_size_dilate), np.uint8)  # rubbish -> 15
    mask = np.float32(mask)
    if mode == "de":
        mask = cv2.dilate(mask, kernel_dilate, iterations=1)
        mask = cv2.erode(mask, kernel_erode, iterations=1)
    elif mode == "ed":
        mask = cv2.erode(mask, kernel_erode, iterations=1)
        mask = cv2.dilate(mask, kernel_dilate, iterations=1)
    return mask > 0
