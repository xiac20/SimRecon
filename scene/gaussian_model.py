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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import open3d as o3d
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, SH2RGB
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.contrastive_utils import feature3d_to_rgb

from scipy.spatial.transform import Rotation
from e3nn import o3
import einops
import einsum


class GaussianModel:
    """
    基于高斯点（Gaussian Splatting）的可学习场景表示。

    管理以下参数（均为可学习张量或其函数）：
    - _xyz: 高斯中心坐标 (N, 3)
    - _features_dc/_features_rest: 球谐系数（DC项与非DC项）
    - _scaling: 轴对齐尺度（在激活后为正数）
    - _rotation: 四元数旋转（归一化后）
    - _opacity: 不透明度（sigmoid激活后 ∈ (0,1)）
    - _seg_feature: 可选的语义特征 (N, D)

    还包含：
    - 优化器状态、学习率调度、点云稠密化/稀疏化逻辑、PLY I/O 等。
    """

    def setup_functions(self):
        """
        初始化若干激活函数与从尺度/旋转构造协方差的函数。
        - scaling_activation/log: 作用于尺度参数
        - opacity_activation: 作用于不透明度
        - rotation_activation: 四元数归一化
        - covariance_activation: 将(center, scaling, rotation)转换为4x4仿射矩阵（列主存储）
        """
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1),
                                        rotation).permute(0, 2, 1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:, :3, :3] = RS
            trans[:, 3, :3] = center
            trans[:, 3, 3] = 1
            return trans

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int):
        """
        参数:
        - sh_degree: 最大球谐阶数（含DC项）
        """
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        self._seg_feature = None
        self.use_seg_feature = False
        self.seg_feat_dim = 0
        self.load_seg_feat = False

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        """
        从检查点恢复模型参数与优化器状态。

        参数:
        - model_args: 由 capture() 导出的元组
        - training_args: 训练超参数（用于重建优化器）
        """
        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)  # .clamp(max=1)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_seg_feature(self):
        if self._seg_feature is not None:
            return self._seg_feature / (torch.norm(self._seg_feature, p=2, dim=1, keepdim=True) + 1e-6)
        return self._seg_feature

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        """
        返回每个高斯的4x4仿射矩阵（包含旋转与尺度），便于光栅化。
        - scaling_modifier: 训练/渲染时的额外缩放系数
        """
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        """逐步提升当前生效的球谐阶数（每次+1，直到max）。"""
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def set_segfeat_params(self, modelparams):
        """从外部参数对象读取语义特征相关开关与维度。"""
        self.use_seg_feature = modelparams.use_seg_feature
        self.seg_feat_dim = modelparams.seg_feat_dim
        self.load_seg_feat = modelparams.load_seg_feat

    def set_3d_feat(self, Seg3D_masks, gram_feat=False):
        '''
        初始化/设置每个高斯的3D语义特征（可选Gram-Schmidt正交化的类别原型）。

        参数:
        - Seg3D_masks: (N_points, N_classes) 的布尔/0-1掩码
        - gram_feat: 是否用Gram-Schmidt生成正交类别原型，并以此初始化类内点特征
        '''
        self.class_feat = None
        if self._seg_feature is None:
            seg_feature = torch.rand((self._xyz.shape[0], self.seg_feat_dim), device="cuda")
            if gram_feat:
                init_feat = torch.rand((Seg3D_masks.shape[1], self.seg_feat_dim)).cuda()

                def gram_schmidt(vectors):
                    # 简单的Gram-Schmidt正交化
                    orthogonal_vectors = []
                    for v in vectors:
                        for u in orthogonal_vectors:
                            v = v - torch.dot(v, u) * u
                        orthogonal_vectors.append(v / (torch.norm(v) + 1e-9))
                    return torch.stack(orthogonal_vectors)

                init_feat = gram_schmidt(init_feat)

                for i in range(Seg3D_masks.shape[1]):
                    curr_mask = Seg3D_masks[:, i]
                    gs_ids = torch.from_numpy(np.where(curr_mask)[0]).cuda()
                    seg_feature[gs_ids] = init_feat[i]

                self.class_feat = init_feat

            seg_feature = seg_feature / (seg_feature.norm(dim=1, keepdim=True) + 1e-9)
            self._seg_feature = nn.Parameter(seg_feature.requires_grad_(True))

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float, require_grad=True):
        """
        从原始点云初始化高斯参数（位置、SH、尺度、旋转、opacity）。
        - 尺度依据KNN距离估计
        - 初始旋转为随机四元数
        - 初始不透明度为常数
        """
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(require_grad))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(require_grad))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(require_grad))
        self._scaling = nn.Parameter(scales.requires_grad_(require_grad))
        self._rotation = nn.Parameter(rots.requires_grad_(require_grad))
        self._opacity = nn.Parameter(opacities.requires_grad_(require_grad))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args,
                       optim_seg_feature=True,
                       optim_xyz=True,
                       optim_sh=True,
                       optim_scale=True,
                       optim_rotate=True,
                       optim_opacity=True):
        """
        构建优化器参数组与学习率调度；按开关决定哪些张量参与训练。
        - 当使用语义特征训练时，仅优化 _seg_feature
        - 否则优化 xyz/sh/opacity/scale/rotation
        """
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        if self.use_seg_feature and optim_seg_feature:
            if self._seg_feature is None:
                seg_feature = torch.rand((self._xyz.shape[0], self.seg_feat_dim), device="cuda")
                seg_feature = seg_feature / seg_feature.norm(dim=1, keepdim=True)
                self._seg_feature = nn.Parameter(seg_feature.requires_grad_(True))

            l = [
                {'params': [self._seg_feature], 'lr': training_args.seg_feature_lr,
                 "name": "language_feature"},  # TODO: training_args.language_feature_lr
            ]
            self._xyz.requires_grad_(False)
            self._features_dc.requires_grad_(False)
            self._features_rest.requires_grad_(False)
            self._scaling.requires_grad_(False)
            self._rotation.requires_grad_(False)
            self._opacity.requires_grad_(False)
        else:
            self._xyz.requires_grad_(optim_xyz)
            self._features_dc.requires_grad_(optim_sh)
            self._features_rest.requires_grad_(optim_sh)
            self._scaling.requires_grad_(optim_scale)
            self._rotation.requires_grad_(optim_rotate)
            self._opacity.requires_grad_(optim_opacity)
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        '''
        每步更新xyz参数组的学习率（指数调度），返回当前学习率。
        '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self, export_as_3dgs=False):
        """
        构造PLY导出的字段名列表。
        - export_as_3dgs: 按3DGS兼容格式导出（scale维度+1）
        """
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        if not export_as_3dgs:
            for i in range(self._scaling.shape[1]):
                l.append('scale_{}'.format(i))
        else:
            for i in range(self._scaling.shape[1] + 1):
                l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))

        if self._seg_feature is not None:
            for i in range(self._seg_feature.shape[1]):
                l.append('segfeat_{}'.format(i))
        return l

    def save_ply(self, path, crop_mask=None):
        """
        将当前高斯参数导出为PLY文件，同时导出两份Open3D点云：
        - _color.ply: 以SH DC项转RGB着色
        - _feat.ply: 若存在语义特征，则将其投影为RGB着色

        参数:
        - path: 输出PLY路径
        - crop_mask: 可选布尔mask，选择要导出的点
        """
        mkdir_p(os.path.dirname(path))

        if crop_mask is not None:
            valid_mask = crop_mask.detach().cpu()
        else:
            valid_mask = np.ones((len(self.get_xyz)), dtype=bool)

        xyz = self._xyz.detach().cpu().numpy()[valid_mask]
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()[valid_mask]
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()[
            valid_mask]
        opacities = self._opacity.detach().cpu().numpy()[valid_mask]
        scale = self._scaling.detach().cpu().numpy()[valid_mask]
        rotation = self._rotation.detach().cpu().numpy()[valid_mask]
        if self._seg_feature is not None:
            seg_feat = self._seg_feature.detach().cpu().numpy()[valid_mask]

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = [xyz, normals, f_dc, f_rest, opacities, scale, rotation]
        if self._seg_feature is not None:
            attributes.append(seg_feat)
        attributes = np.concatenate(attributes, axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        o3d_pointcloud = o3d.geometry.PointCloud()
        o3d_pointcloud.points = o3d.utility.Vector3dVector(xyz)
        o3d_pointcloud.colors = o3d.utility.Vector3dVector(SH2RGB(f_dc).clip(0., 1.))
        o3d.io.write_point_cloud(path.split(".")[0] + "_color.ply", o3d_pointcloud)
        if self._seg_feature is not None:
            o3d_pointcloud.colors = o3d.utility.Vector3dVector(feature3d_to_rgb(seg_feat))
            o3d.io.write_point_cloud(path.split(".")[0] + "_feat.ply", o3d_pointcloud)

    def save_ply_as_3dgs(self, path):
        """
        以3DGS兼容字段导出PLY：scale会补一维（log小值），便于后续兼容加载。
        同时导出_color与_feat点云。
        """
        print("### Saving PointCloud Params ###")

        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        scale = np.concatenate([scale, np.ones_like(scale[:, :1]) * np.log(1e-6)], axis=-1)
        rotation = self._rotation.detach().cpu().numpy()
        if self._seg_feature is not None:
            seg_feat = self._seg_feature.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(export_as_3dgs=True)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = [xyz, normals, f_dc, f_rest, opacities, scale, rotation]
        if self._seg_feature is not None:
            attributes.append(seg_feat)
        attributes = np.concatenate(attributes, axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        o3d_pointcloud = o3d.geometry.PointCloud()
        o3d_pointcloud.points = o3d.utility.Vector3dVector(xyz)
        o3d_pointcloud.colors = o3d.utility.Vector3dVector(SH2RGB(f_dc).clip(0., 1.))
        o3d.io.write_point_cloud(path.split(".")[0] + "_color.ply", o3d_pointcloud)
        if self._seg_feature is not None:
            o3d_pointcloud.colors = o3d.utility.Vector3dVector(feature3d_to_rgb(seg_feat))
            o3d.io.write_point_cloud(path.split(".")[0] + "_feat.ply", o3d_pointcloud)

    def reset_opacity(self):
        """
        将不透明度上限截断（至0.01）并在优化器中就地替换该张量，保持动量等状态。
        """
        opacities_new = self.inverse_opacity_activation(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        """
        从PLY载入高斯参数：xyz/opacity/SH/scale/rotation/segfeat(可选)。
        注意：会将active_sh_degree重置为max_sh_degree。
        """
        print("### Load the PointCloud Params ###")
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))[:2]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # seg_feat
        if self.use_seg_feature and self.load_seg_feat:
            segfeat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("segfeat")]
            if len(segfeat_names) == self.seg_feat_dim:
                seg_feat = np.zeros((xyz.shape[0], self.seg_feat_dim))
                for idx in range(self.seg_feat_dim):
                    seg_feat[:, idx] = np.asarray(plydata.elements[0]["segfeat_" + str(idx)])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        if self.use_seg_feature and self.load_seg_feat:
            if len(segfeat_names) == self.seg_feat_dim:
                self._seg_feature = nn.Parameter(
                    torch.tensor(seg_feat, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def delete_ply(self, refer_ply_path):
        """
        根据参考PLY的点坐标，在当前点集中筛选并裁剪（保留/删除），然后保存临时结果。
        """
        print("### Deleting the PointCloud Params ###")
        refer_points = np.array(o3d.io.read_point_cloud(refer_ply_path).points)
        gs_points = self.get_xyz.detach().cpu().numpy()
        mask = np.isin(gs_points.view([('', gs_points.dtype)] * gs_points.shape[1]),
                       refer_points.view([('', refer_points.dtype)] * refer_points.shape[1]))
        self.crop_mask(torch.from_numpy(mask.squeeze()).cuda())
        self.save_ply("./tmp.ply")

    def replace_tensor_to_optimizer(self, tensor, name):
        """
        在不破坏优化器动量/二阶矩的前提下，将某个参数张量替换为给定tensor。
        返回: {name: new_param}
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        """
        按mask裁剪每个参数组中的参数，并同步修剪优化器状态张量。
        返回: {group_name: new_param}
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask, optimizer_type=True):
        """
        按mask删除点（True表示删除/保留取决于调用方逻辑，当前传入为要删除的mask）。
        - optimizer_type=True 时，同步维护优化器状态
        - 否则直接裁剪张量
        同时会同步裁剪seg_feature与辅助统计量。
        """
        valid_points_mask = ~mask

        if optimizer_type:
            optimizable_tensors = self._prune_optimizer(valid_points_mask)

            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]

            self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

            self.denom = self.denom[valid_points_mask]
            self.max_radii2D = self.max_radii2D[valid_points_mask]
        else:
            self._xyz = self._xyz[valid_points_mask]
            self._features_dc = self._features_dc[valid_points_mask]
            self._features_rest = self._features_rest[valid_points_mask]
            self._opacity = self._opacity[valid_points_mask]
            self._scaling = self._scaling[valid_points_mask]
            self._rotation = self._rotation[valid_points_mask]

            if self._seg_feature is not None:
                self._seg_feature = self._seg_feature[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        """
        在优化器中为每个参数组拼接新张量，并扩展对应动量/二阶矩。
        参数:
        - tensors_dict: {group_name: extension_tensor}
        返回: {group_name: new_concat_param}
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation):
        """
        稠密化后统一将新增点拼接至参数与优化器中，并重置辅助统计量。
        """
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """
        将高梯度/大尺度的点进行N次采样并克隆，以提升表示能力；原点随后被标记以便裁剪。
        - grads: xyz梯度
        - grad_threshold: 阈值
        - scene_extent: 场景尺度（用于阈值归一化）
        - N: 每个选中点克隆数量
        """
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:, :1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """
        对较小尺度但梯度足够大的点直接克隆，无扰动采样。
        """
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        """
        主稠密化入口：
        1) 按梯度触发 clone 与 split 两类扩张
        2) 按opacity/屏幕投影/世界尺度等条件裁剪大点或透明点
        3) 清理显存
        """
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        """
        累积屏幕空间梯度范数与计数，供后续 densify 阶段计算平均梯度。
        """
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1

    ############# Instance
    def crop_mask(self, gs_mask, type="save"):
        """
        按布尔掩码裁剪当前点集。
        参数:
        - gs_mask: True表示保留（save）或删除（delete模式下取反）
        - type: "save" 或 "delete"
        副作用：同步裁剪所有参数张量与seg_feature。
        """
        # type:save delete
        if type == "delete":
            gs_mask = ~gs_mask
        self._xyz = self._xyz[gs_mask]
        self._features_dc = self._features_dc[gs_mask]
        self._features_rest = self._features_rest[gs_mask]
        self._opacity = self._opacity[gs_mask]
        self._scaling = self._scaling[gs_mask]
        self._rotation = self._rotation[gs_mask]
        if self.use_seg_feature:
            self._seg_feature = self._seg_feature[gs_mask]

    def combine_gaussian(self, new_gaussian, load_seg_feat=True):
        """
        将另一个GaussianModel（兼容格式）拼接到当前模型末尾。
        - load_seg_feat=True 时，以现有特征均值初始化新增点的语义特征
        """
        self._xyz = torch.cat([self._xyz, new_gaussian._xyz], dim=0)
        self._features_dc = torch.cat([self._features_dc, new_gaussian._features_dc], dim=0)
        self._features_rest = torch.cat([self._features_rest, new_gaussian._features_rest], dim=0)
        self._opacity = torch.cat([self._opacity, new_gaussian._opacity], dim=0)
        self._scaling = torch.cat([self._scaling, new_gaussian._scaling], dim=0)
        self._rotation = torch.cat([self._rotation, new_gaussian._rotation], dim=0)

        self._xyz = nn.Parameter(self._xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(self._features_dc.requires_grad_(True))
        self._features_rest = nn.Parameter(self._features_rest.requires_grad_(True))
        self._opacity = nn.Parameter(self._opacity.requires_grad_(True))
        self._scaling = nn.Parameter(self._scaling.requires_grad_(True))
        self._rotation = nn.Parameter(self._rotation.requires_grad_(True))

        if load_seg_feat and self.use_seg_feature:
            gs_feat_mean = (self._seg_feature / (self._seg_feature.norm(dim=-1, keepdim=True) + 1e-9)).mean(0)
            self._seg_feature = torch.cat([self._seg_feature,
                                           gs_feat_mean * torch.ones((len(new_gaussian.get_xyz), len(gs_feat_mean)),
                                                                     device="cuda")], dim=0)
            self._seg_feature = nn.Parameter(self._seg_feature.requires_grad_(True))

    def crop_pts_with_convexhull(self, pts, type="save", return_bbox=False):
        """
        以给定点集的凸包作为裁剪区域，对当前点集进行裁剪；
        可选返回裁剪区域的有向包围盒（OrientedBoundingBox）。
        """
        # 首先计算得到convex hull
        from scipy.spatial import ConvexHull, Delaunay
        delaunay = Delaunay(pts)
        points_inside_hull_mask = delaunay.find_simplex(self.get_xyz.detach().cpu().numpy()) >= 0
        if return_bbox:
            crop_points = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(self.get_xyz[points_inside_hull_mask].detach().cpu().numpy()))
            instance_bbox = o3d.geometry.AxisAlignedBoundingBox().create_from_points(crop_points.points)
            instance_bbox = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(instance_bbox)

            self.crop_mask(points_inside_hull_mask, type=type)

            return instance_bbox
        else:
            self.crop_mask(points_inside_hull_mask, type=type)
