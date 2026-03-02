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
import math
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
           override_color=None, norm_seg_feat=True):
    """
    基于高斯点云（Gaussian Splatting）的前向渲染函数。

    参数：
    - viewpoint_camera: 相机对象（包含视图/投影矩阵、分辨率、FoV 等）
    - pc: GaussianModel 高斯场景（提供 xyz、尺度、旋转、SH/颜色、不透明度、可选语义特征等）
    - pipe: 渲染管线配置（是否在Python侧计算协方差/SH->RGB、深度融合策略等）
    - bg_color: 背景颜色 tensor（需在GPU上）
    - scaling_modifier: 额外尺度系数（训练/可视化时可调）
    - override_color: 若提供，则使用该颜色而非SH颜色
    - norm_seg_feat: 是否对语义特征做L2归一化后再送入栅格化器

    返回：字典 rets，包含：
    - render: 渲染RGB图 (C,H,W)
    - seg_feature: 投影视角下的特征图 (C,H,W) 或 None（对应 extra_attrs）
    - viewspace_points: 屏幕空间点（用于反传2D梯度）
    - visibility_filter: 可见性掩码（半径>0）
    - radii: 屏幕半径
    - gau_related_pixels: 渲染过程中的“高斯-像素”关联，用于追踪
    - rend_alpha/rend_normal/rend_dist: 透明度/法向/深度扰动图
    - surf_depth/surf_normal: 伪表面深度与法向（供正则化）
    - rend_depth/rend_median_depth: 期望/中值深度
    """

    # 准备屏幕空间占位点，用于接收2D梯度
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda")
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # 组装栅格化配置（相机、FoV、分辨率、变换矩阵等）
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        # pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 取出场景参数
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    seg_feature = pc.get_seg_feature
    if seg_feature is not None and norm_seg_feat:
        seg_feature = seg_feature / (seg_feature.norm(dim=-1, keepdim=True) + 1e-9)

    # 协方差/尺度旋转：
    # 若 pipe.compute_cov3D_python=True，则在Python侧预计算3D->屏幕的仿射映射（不支持法向正则）；
    # 否则把 scale/rotation 交给栅格化器内部处理。
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # 注意：此模式下不支持 normal consistency loss
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W - 1) / 2],
            [0, H / 2, 0, (H - 1) / 2],
            [0, 0, far - near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix = viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0, 1, 3]] @ world2pix[:, [0, 1, 3]]).permute(0, 2, 1).reshape(-1,
                                                                                                       9)  # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # 颜色设置：若不覆盖颜色，则可在Python侧将SH投影为RGB；否则由栅格化器处理
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # 栅格化：返回RGB、半径、allmap（alpha/normal/depth等通道）、extra_attrs（如语义特征）、
    # 和 gau_related_pixels（像素-高斯关联）
    rendered_image, radii, allmap, extra_attrs, gau_related_pixels = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        extra_attrs=seg_feature
    )


    # 可见性与基础返回项
    rets = {"render": rendered_image,
            "viewspace_points": means2D,
            "visibility_filter": radii > 0,
            "radii": radii,
            "seg_feature": extra_attrs,
            "gau_related_pixels": gau_related_pixels
            }

    # 额外正则化相关通道：alpha/normal/depth/扰动
    render_alpha = allmap[1:2]

    # 视空间法向 -> 世界空间法向（乘以视图矩阵的旋转部分转回世界系）
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1, 2, 0) @ (viewpoint_camera.world_view_transform[:3, :3].T)).permute(2, 0,
                                                                                                                 1)
    # 中值深度
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # 期望深度
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

    # 深度扰动
    render_dist = allmap[6:7]

    # 伪表面属性：按 depth_ratio 在期望/中值深度间插值，并由深度计算伪法向
    surf_depth = render_depth_expected * (1 - pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median

    # 由深度估计表面法向，并乘以alpha（render_normal为未归一法向，需以累积透明度加权）
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2, 0, 1)
    surf_normal = surf_normal * (render_alpha).detach()

    rets.update({
        'rend_alpha': render_alpha,
        'rend_normal': render_normal,
        'rend_dist': render_dist,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal,

        "rend_depth": render_depth_expected,
        "rend_median_depth": render_depth_median,
    })

    return rets
