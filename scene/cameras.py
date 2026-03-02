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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_mesh_utils import get_ray_directions, get_rays
import math


def fov2focal(fov, pixels):
    """
    将视场角（弧度）转换为焦距（像素单位）。

    参数:
    - fov: 视场角（弧度），可为水平FoV或垂直FoV
    - pixels: 对应方向的像素数（宽或高）

    返回:
    - 焦距，单位像素
    """
    return pixels / (2 * math.tan(fov / 2))


class Camera(nn.Module):
    """
    相机类，封装了渲染与几何所需的投影矩阵、视图矩阵以及图像/法线/分割等信息。

    主要成员:
    - R, T: 相机外参（旋转R为3x3，平移T为3x1，遵循项目约定的坐标系）
    - FoVx, FoVy: 水平/垂直视场角（弧度）
    - image: 原始图像（C,H,W），用于监督与可视化
    - segmap/sorted_segmap: 2D分割标签（可选）
    - normal/normal_mask: 每像素法线与可用mask（可选）
    - world_view_transform, projection_matrix, full_proj_transform: 渲染所需的矩阵
    - image_width, image_height: 图像分辨率
    """
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask=None, segmap=None, sorted_segmap=None,
                 image_name=None, uid=None, normal=None, depth=None, gau_related_pixels=None,
                 image_width=None, image_height=None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda",
                 use_train=True
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        # 原始图像，范围裁剪至[0,1]
        self.original_image = image.clamp(0.0, 1.0) if image is not None else None  # .to(self.data_device)
        if normal is not None:
            # 每像素法线，构造有效mask（阈值范围内）并归一化
            self.normal = normal  # .to(self.data_device)
            normal_norm = torch.norm(self.normal, dim=0, keepdim=True)
            self.normal_mask = ~((normal_norm > 1.1) | (normal_norm < 0.9))
            self.normal = self.normal / normal_norm
        else:
            self.normal = None
            self.normal_mask = None

        self.segmap = segmap
        self.sorted_segmap = sorted_segmap

        # 分辨率推断：优先采用传入的宽高，否则由图像尺寸决定
        if image_width is not None:
            self.image_width = image_width
            self.image_height = image_height
        else:
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]

        # alpha mask（若提供）
        if gt_alpha_mask is not None:
            # self.original_image *= gt_alpha_mask.to(self.data_device)
            self.gt_alpha_mask = gt_alpha_mask  # .to(self.data_device)
        else:
            # self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            self.gt_alpha_mask = None

        # 深度裁剪面
        self.zfar = 100.0
        self.znear = 0.01

        # 额外的全局平移与尺度（常用于对齐/归一化）
        self.trans = trans
        self.scale = scale

        # 计算并缓存矩阵（列主存储转置以适配后续bmm）
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                                     fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.use_train = use_train

        # 延迟计算/缓存
        self.intrinsic = None
        self.c2w = None
        self.w2c = None

    def convert2c2w_intrinsics(self):
        """
        将内部R/T与FoV转换为：
        - c2w: 相机到世界的外参矩阵（4x4）
        - intrinsic: 相机内参矩阵（4x4，右下角为1；fx=fy基于FoVx计算）

        说明:
        - W2C的构造遵循本项目外参约定，然后取逆得到c2w
        - 内参采用针孔模型，主点位于图像中心
        """
        W2C = np.eye(4)
        W2C[:3] = np.concatenate([np.linalg.inv(self.R), self.T[:, None]], -1)  # W2C
        c2w = np.linalg.inv(W2C)

        intrinsic = np.eye(4)
        focal = (self.image_width / 2) / (np.tan(self.FoVx / 2))
        intrinsic[0, 0] = focal
        intrinsic[1, 1] = focal
        intrinsic[0, 2] = self.image_width / 2
        intrinsic[1, 2] = self.image_height / 2
        return c2w, intrinsic

    def get_mesh_normal(self, mesh_tracer):
        """
        使用网格光线追踪器（mesh_tracer）为当前相机生成每像素法线图。

        步骤:
        1) 由c2w与内参生成光线方向
        2) 调用mesh_tracer.trace获取交点与面法线
        3) 整理为(H,W,3)并归一化，生成normal与normal_mask

        参数:
        - mesh_tracer: 具备trace(rays_o, rays_d)接口的网格追踪器
        """
        # 生成rays_d,rays_o
        c2w, intrinsic = self.convert2c2w_intrinsics()
        Height, Width = self.image_height, self.image_width

        rays_o, rays_d, rays_d_norm = get_rays(
            get_ray_directions(Height, Width, torch.from_numpy(intrinsic[:3, :3]).float())[0],
            torch.from_numpy(c2w[:3]).float())
        rays_o, rays_d = rays_o.reshape(-1, 3).cuda(), rays_d.reshape(-1, 3).cuda()

        positions, face_normals, _, _ = mesh_tracer.trace(rays_o, rays_d)

        normals = face_normals.reshape(Height, Width, 3)
        # from PIL import Image
        # Image.fromarray(np.uint8(((normals + 1) / 2).cpu().detach().numpy() * 255.0)).show()
        self.normal = normals.permute(2, 0, 1)
        normal_norm = torch.norm(self.normal, dim=0, keepdim=True) + 1e-9
        self.normal_mask = ~((normal_norm > 1.1) | (normal_norm < 0.9))
        self.normal = self.normal / normal_norm

    @property
    def get_intrinsic(self):
        """
        惰性计算并返回相机内参（4x4）。
        - fx, fy 由FoV与分辨率换算得到
        - 主点设为图像中心
        返回值缓存于self.intrinsic以避免重复计算。
        """
        if self.intrinsic is None:
            intrinsic = np.eye(4)
            fx = fov2focal(self.FoVx, self.image_width)
            fy = fov2focal(self.FoVy, self.image_height)
            intrinsic[0, 0] = fx
            intrinsic[1, 1] = fy
            intrinsic[0, 2] = self.image_width / 2
            intrinsic[1, 2] = self.image_height / 2
            self.intrinsic = torch.tensor(intrinsic).cuda().float()
        return self.intrinsic

    @property
    def get_c2w(self):
        """
        返回c2w（4x4）: 相机到世界变换矩阵，按需由w2c取逆得到并缓存。
        """
        if self.c2w is None:
            self.c2w = torch.inverse(self.get_w2c)
        return self.c2w

    @property
    def get_w2c(self):
        """
        返回w2c（4x4）: 世界到相机变换矩阵，由R^T与T构造并缓存。
        """
        if self.w2c is None:
            w2c = np.eye(4)
            w2c[:3, :3] = self.R.T
            w2c[:3, 3] = self.T
            self.w2c = torch.tensor(w2c).cuda().float()
        return self.w2c

    def view_o3d(self, o3d_vis_objects):
        """
        使用Open3D快速查看当前相机下的几何体（例如点云/网格）。

        参数:
        - o3d_vis_objects: Open3D几何体列表（将逐个add到Visualizer）
        """
        import open3d as o3d
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=self.image_width, height=self.image_height)
        for obj in o3d_vis_objects:
            vis.add_geometry(obj)

        intrinsic = self.get_intrinsic.cpu().numpy()
        w2c = self.get_w2c.cpu().numpy()
        camera = o3d.camera.PinholeCameraParameters()
        camera.intrinsic = o3d.camera.PinholeCameraIntrinsic(self.image_width, self.image_height,
                                                             intrinsic[0, 0], intrinsic[1, 1],
                                                             intrinsic[0, 2], intrinsic[1, 2])
        camera.extrinsic = w2c

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(camera, True)

        vis.poll_events()
        vis.update_renderer()

        vis.run()
        vis.destroy_window()


class MiniCam:
    """
    轻量相机结构，主要用于渲染流水线的体素/裁剪阶段或快速可视化。

    成员:
    - image_width/height: 分辨率
    - FoVy/FoVx: 视场角（弧度）
    - znear/zfar: 近远裁剪面
    - world_view_transform/full_proj_transform: 视图与投影相关矩阵
    - camera_center: 相机中心位置
    """
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
