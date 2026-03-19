"""
================================================================================
Optimize Instance Visibility Script
================================================================================

This script optimizes the view pose (RT transformation) for each instance in a
3D Gaussian Splatting (3DGS) scene.

Main Features:
-----------
1. Load pre-trained 3DGS model and instance segmentation information
2. Optimize rotation (R) and translation (T) parameters for each instance to
   maximize its rendering visibility at the specified camera viewpoint
3. Save pre/post optimization rendering results and optimal RT parameters

Workflow:
-----------
1. Read instance_info.json to get instance list and best observation viewpoints
2. Read point_cloud_labels.npy to get which instance each Gaussian point belongs to
3. For each instance:
   - Extract Gaussian points belonging to that instance
   - Optimize RT parameters via gradient descent to maximize rendered alpha score
   - Save optimized rendered images and RT parameters

Usage:
-----------
python optimize_by_avo_new.py \\
    --source_path <dataset_path> \\
    --label_dir <label_directory> \\
    [--instance_id <optional: specify instance ID>] \\
    [--output_dir <optional: output directory>]

Dependencies:
-----------
- PyTorch (with CUDA support)
- 3D Gaussian Splatting modules (gaussian_renderer, scene, arguments)
- NumPy, PIL, tqdm
================================================================================
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import tqdm
from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene


def _log(message: str) -> None:
    """
    Logging helper function.
    
    Safely outputs log messages in a tqdm progress bar environment without
    interfering with the progress bar display.
    
    Args:
        message: The log message to output
    """
    try:
        tqdm.tqdm.write(str(message))
    except Exception:
        print(message)


def _parse_camera_uid_from_best_view(best_view: Optional[str]) -> Optional[int]:
    """
    Parse camera UID from best_view string.
    
    best_view is typically in format like "camera_00123.png",
    this function extracts the numeric part as camera UID.
    
    Args:
        best_view: Best view string, format like "camera_00123"
        
    Returns:
        Parsed camera UID (integer), or None if parsing fails
    """
    if not best_view:
        return None
    # Use regex to match digits after "camera_"
    m = re.search(r"camera_(\d+)", str(best_view))
    return int(m.group(1)) if m else None


def _build_scene_for_cameras(source_path: str, gaussian_ply_path: Optional[str] = None) -> Tuple[GaussianModel, Scene, Any]:
    """
    Build scene and load Gaussian model.
    
    Creates the scene object required for 3DGS, loads camera parameters and
    Gaussian point cloud.
    
    Args:
        source_path: Dataset root path containing COLMAP data and images
        gaussian_ply_path: Optional Gaussian point cloud PLY file path,
                          uses point_cloud.ply under source_path if not specified
    
    Returns:
        Tuple[GaussianModel, Scene, PipelineParams]:
            - gaussians: Loaded Gaussian model
            - scene: Scene object containing camera information
            - pipeline_params: Rendering pipeline parameters
    """
    # Create argument parser and register parameter groups
    parser = argparse.ArgumentParser(add_help=False)
    lp = ModelParams(parser)      # Model params (data path, resolution, etc.)
    pp = PipelineParams(parser)   # Pipeline params (rendering related)
    _ = OptimizationParams(parser)  # Optimization params (learning rate, etc., not used here)

    # Set default parameter values
    args = parser.parse_args([])
    args.source_path = os.path.abspath(source_path)  # Dataset absolute path
    args.model_path = ""                              # Model output path (not needed here)
    args.images = "images"                            # Image subdirectory name
    args.resolution = -1                              # -1 means use original resolution
    args.white_background = False                     # Do not use white background
    args.data_device = "cuda"                         # Load data to GPU
    args.eval = False                                 # Not evaluation mode

    # Semantic segmentation related params (not used in this optimization script)
    args.use_seg_feature = False
    args.seg_feat_dim = 16
    args.load_seg_feat = False
    args.load_filter_segmap = False
    args.preload_robust_semantic = ""

    # Extract parameter objects
    mp = lp.extract(args)
    pipeline_params = pp.extract(args)

    # Create Gaussian model with 3rd-order spherical harmonics (SH degree=3) for color
    gaussians = GaussianModel(sh_degree=3)
    gaussians.pipelineparams = pipeline_params
    gaussians.set_segfeat_params(mp)

    # Load Gaussian point cloud PLY file
    ply_path = gaussian_ply_path or os.path.join(mp.source_path, "point_cloud.ply")
    gaussians.load_ply(str(ply_path))

    # Create scene object, loaded_gaussian=True indicates Gaussians are preloaded
    scene = Scene(mp, gaussians, loaded_gaussian=True)
    return gaussians, scene, pipeline_params


def load_camera_info(source_path: str, label_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    加载相机信息和高斯模型
    
    这是初始化优化所需全部数据的主要函数。
    
    Args:
        source_path: 数据集路径，包含 COLMAP 重建数据
        label_dir: 标签目录路径，可能包含训练后的 point_cloud.ply
        
    Returns:
        包含以下键的字典：
        - gaussians: GaussianModel 实例
        - scene: Scene 实例
        - pipeline_params: 渲染管线参数
        - train_cameras: 训练相机列表
        - camera_dict: 相机 UID 到相机对象的映射字典
        - gaussian_ply_path: 实际使用的高斯点云文件路径
    """
    gaussian_ply_path = None
    # 优先使用 label_dir 下的 point_cloud.ply（通常是训练后的结果）
    if label_dir is not None:
        cand = Path(label_dir) / "point_cloud.ply"
        if cand.exists():
            gaussian_ply_path = str(cand)
    
    # 构建场景和高斯模型
    gaussians, scene, pipeline_params = _build_scene_for_cameras(source_path, gaussian_ply_path=gaussian_ply_path)
    
    # 获取训练集相机列表并建立索引字典
    train_cameras = scene.getTrainCameras()
    camera_dict = {cam.uid: cam for cam in train_cameras}
    
    return {
        "gaussians": gaussians,
        "scene": scene,
        "pipeline_params": pipeline_params,
        "train_cameras": train_cameras,
        "camera_dict": camera_dict,
        "gaussian_ply_path": gaussian_ply_path,
    }


def infer_visibility_dir(label_dir: str) -> str:
    """
    从迭代标签目录推断可见性目录路径
    
    标签目录结构通常为: .../train_semanticgs/point_cloud/iteration_XXXX/
    可见性目录位于: .../train_semanticgs/visibility/
    
    Args:
        label_dir: 迭代标签目录路径
        
    Returns:
        可见性目录的完整路径
    """
    p = Path(label_dir)
    # 向上两级到达模型目录，然后进入 visibility 子目录
    model_dir = p.parent.parent
    return str(model_dir / "visibility")


def select_topk_cameras_by_visibility(
    camera_info: Dict[str, Any],
    visibility_dir: str,
    point_labels: np.ndarray,
    instance_id: int,
    top_k: int,
    min_visible_points: int = 10,
) -> List[int]:
    """
    根据可见点数量选择最佳的 Top-K 相机视角
    
    对于给定的实例，统计每个相机能看到该实例多少个高斯点，
    选择可见点最多的 K 个相机用于优化。
    
    Args:
        camera_info: 相机信息字典（由 load_camera_info 返回）
        visibility_dir: 可见性数据目录，包含每个相机的可见性 npy 文件
        point_labels: 点云标签数组，记录每个点属于哪个实例
        instance_id: 目标实例 ID
        top_k: 选择的相机数量
        min_visible_points: 最少可见点阈值，少于此值的相机会被过滤
        
    Returns:
        按可见点数量降序排列的相机 UID 列表（最多 top_k 个）
    """
    vis_dir = Path(visibility_dir)
    if not vis_dir.is_dir():
        return []

    instance_id = int(instance_id)
    # 创建该实例的点云掩码
    instance_mask = point_labels == instance_id
    if not np.any(instance_mask):
        return []

    results: List[Tuple[int, int]] = []  # 存储 (相机UID, 可见点数) 元组
    
    # 遍历所有训练相机
    for cam in camera_info.get("train_cameras", []):
        uid = int(cam.uid)
        # 可见性文件命名格式: camera_00001_visibility.npy
        vis_path = vis_dir / f"camera_{uid:05d}_visibility.npy"
        if not vis_path.exists():
            continue
        
        # 加载可见性数组（布尔数组，True 表示该点在此相机中可见）
        vis = np.load(str(vis_path), allow_pickle=False)
        if vis.dtype != bool:
            vis = vis.astype(bool)
        
        # 验证数组形状匹配
        if vis.ndim != 1 or vis.shape[0] != point_labels.shape[0]:
            continue
        
        # 计算该相机能看到的实例点数量
        visible_count = int(np.sum(vis & instance_mask))
        if visible_count >= int(min_visible_points):
            results.append((uid, visible_count))

    # 按可见点数降序排序，相同时按 UID 升序
    results.sort(key=lambda x: (-x[1], x[0]))
    return [uid for uid, _ in results[: int(top_k)]]


def load_instance_info(label_dir: str) -> List[Dict[str, Any]]:
    """
    加载实例信息 JSON 文件
    
    读取 instance_info.json，该文件包含所有实例的元信息，
    如实例ID、类别标签、最佳观测视角等。
    
    Args:
        label_dir: 标签目录路径
        
    Returns:
        实例记录列表，每个实例是一个字典，包含：
        - instance_id: 实例 ID
        - category_label: 类别名称
        - best_view / best_view_uid: 最佳观测视角信息
        - 其他语义信息...
        
    Raises:
        FileNotFoundError: 找不到 instance_info.json
        ValueError: JSON 格式不正确
    """
    label_dir_path = Path(label_dir)
    instance_json_path = label_dir_path / "instance_info.json"
    if not instance_json_path.exists():
        raise FileNotFoundError(f"Instance info file not found: {instance_json_path}")
    
    with open(instance_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    instances = data.get("instances")
    if not isinstance(instances, list):
        raise ValueError("Invalid instance_info.json: missing 'instances' list")
    return instances


def load_point_cloud_labels(label_dir: str) -> np.ndarray:
    """
    加载点云标签数组
    
    读取 point_cloud_labels.npy，该文件记录每个高斯点属于哪个实例。
    数组长度应与高斯点云中的点数一致。
    
    Args:
        label_dir: 标签目录路径
        
    Returns:
        一维 int32 数组，labels[i] 表示第 i 个高斯点的实例 ID
        
    Raises:
        FileNotFoundError: 找不到标签文件
        ValueError: 标签数组形状不正确
    """
    label_dir_path = Path(label_dir)
    labels_path = label_dir_path / "point_cloud_labels.npy"
    if not labels_path.exists():
        raise FileNotFoundError(f"Label array file not found: {labels_path}")
    
    labels = np.load(str(labels_path))
    if labels.ndim != 1:
        raise ValueError(f"Invalid point_cloud_labels.npy shape: {labels.shape}")
    return labels.astype(np.int32, copy=False)


def get_instance_record(instances: List[Dict[str, Any]], instance_id: int) -> Dict[str, Any]:
    """
    根据实例 ID 查找对应的实例记录
    
    Args:
        instances: 实例记录列表
        instance_id: 目标实例 ID
        
    Returns:
        匹配的实例记录字典
        
    Raises:
        KeyError: 找不到指定的实例 ID
    """
    for rec in instances:
        try:
            if int(rec.get("instance_id")) == int(instance_id):
                return rec
        except Exception:
            continue
    raise KeyError(f"instance_id={instance_id} not found in instance_info.json")


def get_instance_point_indices(point_labels: np.ndarray, instance_id: int) -> np.ndarray:
    """
    获取属于指定实例的所有点的索引
    
    Args:
        point_labels: 点云标签数组
        instance_id: 目标实例 ID
        
    Returns:
        包含所有属于该实例的点索引的 int64 数组
    """
    idx = np.where(point_labels == int(instance_id))[0]
    return idx.astype(np.int64, copy=False)


def _rodrigues_from_axis_angle(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    罗德里格斯旋转公式：从轴角表示转换为旋转矩阵（可微分版本）
    
    轴角表示法使用一个3D向量，其方向表示旋转轴，模长表示旋转角度（弧度）。
    此实现针对小角度情况进行了数值稳定性处理，使用泰勒展开近似。
    
    罗德里格斯公式: R = I + sin(θ)/θ * K + (1-cos(θ))/θ² * K²
    其中 K 是旋转轴的反对称矩阵
    
    Args:
        axis_angle: 形状为 (3,) 的轴角向量，[rx, ry, rz]
        
    Returns:
        形状为 (3, 3) 的旋转矩阵 R
    """
    device = axis_angle.device
    dtype = axis_angle.dtype
    
    # 计算旋转角度（向量的L2范数），加小量避免除零
    angle = torch.norm(axis_angle) + 1e-8
    angle = torch.clamp(angle, 0.0, 2.0 * torch.pi)  # 限制在合理范围内
    angle_sq = angle * angle
    eps = 1e-4  # 小角度阈值

    # 对小角度使用泰勒展开近似，大角度使用精确公式
    # sin(θ)/θ ≈ 1 - θ²/6 当 θ→0
    sin_over_angle = torch.where(
        angle > eps,
        torch.sin(angle) / angle,
        1.0 - angle_sq / 6.0,
    )
    # (1-cos(θ))/θ² ≈ 1/2 - θ²/24 当 θ→0
    one_minus_cos_over_angle_sq = torch.where(
        angle > eps,
        (1.0 - torch.cos(angle)) / angle_sq,
        0.5 - angle_sq / 24.0,
    )

    # 构建旋转轴的反对称矩阵 K（叉乘矩阵）
    # K = [  0   -rz   ry ]
    #     [  rz   0   -rx ]
    #     [ -ry   rx   0  ]
    zero = torch.zeros_like(axis_angle[0])
    K = torch.stack(
        [
            torch.stack([zero, -axis_angle[2], axis_angle[1]]),
            torch.stack([axis_angle[2], zero, -axis_angle[0]]),
            torch.stack([-axis_angle[1], axis_angle[0], zero]),
        ]
    )

    # 应用罗德里格斯公式
    I = torch.eye(3, device=device, dtype=dtype)
    R = I + sin_over_angle * K + one_minus_cos_over_angle_sq * (K @ K)
    return R


def _create_masked_overlay(
    render_img: np.ndarray,
    alpha_mask: np.ndarray,
    mask_color: Tuple[int, int, int] = (255, 0, 0),
    mask_alpha: float = 0.4,
) -> np.ndarray:
    """
    在渲染图上叠加带颜色的半透明 mask 蒙版
    
    用于可视化实例在图像中的位置，将实例区域用半透明颜色高亮显示。
    常用红色 (255, 0, 0) 标记实例区域，便于直观查看优化效果。
    
    Args:
        render_img: [H, W, 3] RGB 图像，uint8 格式
        alpha_mask: [H, W] 或 [1, H, W] 的 alpha mask，值域 0-1
                   表示每个像素属于该实例的概率/不透明度
        mask_color: 蒙版颜色，默认红色 (R, G, B)
        mask_alpha: 蒙版透明度，0 为完全透明，1 为完全不透明
    
    Returns:
        叠加蒙版后的 RGB 图像，uint8 格式
        
    Note:
        使用 alpha blending 公式: 
        output = original * (1 - alpha) + mask_color * alpha
    """
    # 处理可能的 batch 维度
    if alpha_mask.ndim == 3:
        alpha_mask = alpha_mask[0]
    
    # 确保 mask 是 float 类型
    if alpha_mask.dtype == np.uint8:
        alpha_mask = alpha_mask.astype(np.float32) / 255.0
    
    # 二值化 mask（阈值 0.5），大于阈值的像素被认为属于该实例
    binary_mask = (alpha_mask > 0.5).astype(np.float32)
    
    # 创建颜色蒙版图层
    overlay = render_img.astype(np.float32).copy()
    color_layer = np.zeros_like(overlay)
    color_layer[:, :, 0] = mask_color[0]  # R 通道
    color_layer[:, :, 1] = mask_color[1]  # G 通道
    color_layer[:, :, 2] = mask_color[2]  # B 通道
    
    # 只在 mask 区域（实例所在像素）叠加颜色
    for c in range(3):
        overlay[:, :, c] = np.where(
            binary_mask > 0,
            # 在实例区域应用 alpha blending
            overlay[:, :, c] * (1 - mask_alpha) + color_layer[:, :, c] * mask_alpha,
            # 非实例区域保持原样
            overlay[:, :, c]
        )
    
    return np.clip(overlay, 0, 255).astype(np.uint8)


def optimize_instance_visibility(
    gaussians: GaussianModel,
    camera_info: Dict[str, Any],
    instance_record: Dict[str, Any],
    instance_point_indices: np.ndarray,
    output_dir: str,
    camera_uid_override: Optional[int] = None,
    max_iterations: int = 10000,
    learning_rate_R: float = 1e-5,
    learning_rate_T: float = 2e-4,
    base_depth_constraint_weight: float = 1e7,
    apply_rt_to_all_points_for_final_render: bool = True,
):
    """
    优化单个实例的位姿变换(RT)以最大化其在指定视角下的可见性
    
    核心优化流程：
    ==============
    1. 提取实例对应的高斯点子集
    2. 初始化旋转参数 R（轴角表示）和平移参数 T（3D向量）为零
    3. 迭代优化：
       - 对实例点应用当前 RT 变换
       - 渲染得到 alpha 图（不透明度图）
       - 计算损失 = -可见性得分 + 正则化 + 深度约束
       - 反向传播更新 RT 参数
    4. 保存优化结果（图像和RT参数）
    
    损失函数设计：
    ==============
    - 可见性得分：图像中心区域 alpha 值之和（希望最大化）
    - 正则化项：防止 RT 变化过大
    - 深度约束：防止实例飘移过远，保持深度稳定
    
    Args:
        gaussians: 完整的高斯模型
        camera_info: 相机信息字典
        instance_record: 实例元信息（包含 instance_id, category_label 等）
        instance_point_indices: 属于该实例的点索引数组
        output_dir: 输出目录
        camera_uid_override: 可选，强制使用指定相机
        max_iterations: 最大优化迭代次数
        learning_rate_R: 旋转参数学习率（通常较小，因为旋转敏感）
        learning_rate_T: 平移参数学习率
        base_depth_constraint_weight: 深度约束权重，防止物体飘移
        apply_rt_to_all_points_for_final_render: 最终渲染时是否对所有点应用RT
                                                （True用于全局视角，False用于局部视角）
    
    输出文件：
    ==========
    - original_global_{id}_{cam}.png: 优化前全场景渲染
    - original_instance_{id}_{cam}.png: 优化前仅实例渲染
    - original_masked_{id}_{cam}.png: 优化前带红色蒙版
    - optimized_{id}_{cam}.png: 优化后全场景渲染
    - optimized_masked_{id}_{cam}.png: 优化后带红色蒙版
    - best_rt_{cam}.npz: 最优RT参数（R矩阵, T向量, 损失值等）
    """
    from PIL import Image

    # ========== 1. 基本信息提取 ==========
    instance_id = int(instance_record.get("instance_id"))
    # 从 instance_record 获取类别标签（如 "chair", "table" 等）
    category_label = instance_record.get("category_label", "unknown")
    # 清理类别名（去除特殊字符，避免文件夹名问题）
    category_label_clean = re.sub(r'[^\w\-]', '_', str(category_label))
    instance_name = f"instance_{instance_id}_{category_label_clean}"

    # 验证实例点数量
    if instance_point_indices.size == 0:
        raise ValueError(f"Instance has no points in point_cloud_labels.npy: {instance_name}")

    # ========== 2. 确定优化使用的相机 ==========
    # 优先级：camera_uid_override > best_view_uid > best_view > 第一个相机
    camera_uid: Optional[int] = int(camera_uid_override) if camera_uid_override is not None else None
    if camera_uid is None:
        # 尝试从实例记录中获取最佳视角相机
        if instance_record.get("best_view_uid") is not None:
            try:
                camera_uid = int(instance_record.get("best_view_uid"))
            except Exception:
                camera_uid = None
    if camera_uid is None:
        # 从 best_view 字符串解析相机 UID
        camera_uid = _parse_camera_uid_from_best_view(instance_record.get("best_view"))
    if camera_uid is None or camera_uid not in camera_info["camera_dict"]:
        # 最后回退到第一个训练相机
        camera_uid = int(camera_info["train_cameras"][0].uid)

    camera = camera_info["camera_dict"][camera_uid]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========== 3. 创建输出目录 ==========
    out_root = Path(output_dir)
    out_inst = out_root / instance_name  # 每个实例单独一个文件夹
    out_inst.mkdir(parents=True, exist_ok=True)

    # ========== 4. 保存原始高斯参数（用于恢复和最终渲染） ==========
    # 3DGS 的完整参数包括：位置、颜色(SH系数)、缩放、旋转、不透明度
    original_xyz = gaussians.get_xyz.clone().detach()           # 位置 [N, 3]
    original_features_dc = gaussians._features_dc.clone().detach()    # SH 0阶系数（基础颜色）
    original_features_rest = gaussians._features_rest.clone().detach()  # SH 高阶系数
    original_scaling = gaussians._scaling.clone().detach()      # 缩放参数（控制椭球大小）
    original_rotation = gaussians._rotation.clone().detach()    # 旋转四元数
    original_opacity = gaussians._opacity.clone().detach()      # 不透明度

    # ========== 5. 提取实例对应的高斯子集 ==========
    instance_points_idx = torch.as_tensor(instance_point_indices, dtype=torch.long, device=device)
    # 过滤越界索引（防止标签和点云不匹配时崩溃）
    max_idx = int(original_xyz.shape[0]) - 1
    instance_points_idx = instance_points_idx[instance_points_idx <= max_idx]
    if instance_points_idx.numel() == 0:
        raise ValueError(f"All point indices are out of bounds for {instance_name} (max={max_idx})")

    # 提取实例的所有高斯属性
    instance_xyz_original = original_xyz[instance_points_idx].clone().detach()
    instance_features_dc = original_features_dc[instance_points_idx].clone().detach()
    instance_features_rest = original_features_rest[instance_points_idx].clone().detach()
    instance_scaling = original_scaling[instance_points_idx].clone().detach()
    instance_rotation = original_rotation[instance_points_idx].clone().detach()
    instance_opacity = original_opacity[instance_points_idx].clone().detach()

    # ========== 6. 计算原始深度（用于深度约束） ==========
    # 将世界坐标转换到相机坐标系，获取 Z 深度
    camera_world_view_transform = camera.world_view_transform.detach().to(device)
    ones = torch.ones((instance_xyz_original.shape[0], 1), device=device, dtype=instance_xyz_original.dtype)
    xyz_h = torch.cat([instance_xyz_original, ones], dim=1)  # 齐次坐标 [N, 4]
    xyz_cam = (xyz_h @ camera_world_view_transform.T)[:, :3]  # 相机坐标 [N, 3]
    original_depths = -xyz_cam[:, 2]  # 深度值（Z轴，相机朝向负Z）
    original_mean_depth = torch.mean(original_depths)

    # 深度约束参考值（首次达到一定可见性后设置）
    depth_constraint_ref: Optional[torch.Tensor] = None

    # ========== 7. 初始化优化参数 ==========
    # 旋转使用轴角表示（3个参数），初始为零（无旋转）
    rotation_params = torch.zeros(3, dtype=torch.float32, device=device, requires_grad=True)
    # 平移使用3D向量，初始为零（无平移）
    translation_params = torch.zeros(3, dtype=torch.float32, device=device, requires_grad=True)

    # 使用 Adam 优化器，旋转和平移使用不同学习率
    optimizer = torch.optim.Adam(
        [
            {"params": rotation_params, "lr": float(learning_rate_R)},
            {"params": translation_params, "lr": float(learning_rate_T)},
        ]
    )

    # ========== 8. 创建临时高斯模型（仅包含实例点） ==========
    # 用于优化过程中的渲染，只渲染实例本身
    temp_gaussians = GaussianModel(sh_degree=3)
    temp_gaussians._xyz = instance_xyz_original.clone().detach()
    temp_gaussians._features_dc = instance_features_dc.clone().detach()
    temp_gaussians._features_rest = instance_features_rest.clone().detach()
    temp_gaussians._scaling = instance_scaling.clone().detach()
    temp_gaussians._rotation = instance_rotation.clone().detach()
    temp_gaussians._opacity = instance_opacity.clone().detach()
    # 这些参数不需要梯度（RT变换通过直接操作xyz实现）
    temp_gaussians._xyz.requires_grad = False
    temp_gaussians._features_dc.requires_grad = False
    temp_gaussians._features_rest.requires_grad = False
    temp_gaussians._scaling.requires_grad = False
    temp_gaussians._rotation.requires_grad = False
    temp_gaussians._opacity.requires_grad = False

    # 获取渲染管线参数
    pipelineparams = getattr(camera_info.get("scene", None), "pipelineparams", None)
    if pipelineparams is None:
        pipelineparams = camera_info.get("pipeline_params")
    if pipelineparams is None:
        dummy_parser = argparse.ArgumentParser()
        pipelineparams = PipelineParams(dummy_parser)

    def _save_rendered_image(render_pkg: Dict[str, Any], save_path: Path) -> None:
        """保存渲染结果到图像文件"""
        if "render" not in render_pkg:
            return
        # render 输出格式: [C, H, W]，值域 [0, 1]
        img = render_pkg["render"].detach().cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(img).save(str(save_path))

    # ========== 9. 保存优化前的基线渲染结果 ==========
    with torch.no_grad():
        background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
        
        # 渲染完整场景（所有高斯点）
        baseline_global_pkg = render(camera, gaussians, pipelineparams, background)
        baseline_global_path = out_inst / f"original_global_{instance_id}_{camera_uid}.png"
        _save_rendered_image(baseline_global_pkg, baseline_global_path)

        # 渲染仅实例的图像（用于可视化实例位置）
        temp_gaussians._xyz = instance_xyz_original
        baseline_inst_pkg = render(camera, temp_gaussians, pipelineparams, background)
        baseline_inst_path = out_inst / f"original_instance_{instance_id}_{camera_uid}.png"
        _save_rendered_image(baseline_inst_pkg, baseline_inst_path)
        
        # 保存优化前带红色 mask 蒙版的叠加图（全局场景 + 实例高亮）
        if "render" in baseline_global_pkg and "rend_alpha" in baseline_inst_pkg:
            global_img = baseline_global_pkg["render"].detach().cpu().numpy().transpose(1, 2, 0)
            global_img = np.clip(global_img * 255.0, 0, 255).astype(np.uint8)
            inst_alpha = baseline_inst_pkg["rend_alpha"].detach().cpu().numpy()
            masked_img = _create_masked_overlay(global_img, inst_alpha, mask_color=(255, 0, 0), mask_alpha=0.4)
            masked_path = out_inst / f"original_masked_{instance_id}_{camera_uid}.png"
            Image.fromarray(masked_img).save(str(masked_path))


    # ========== 10. 主优化循环 ==========
    loss = None
    R = None
    T = None

    for it in range(int(max_iterations)):
        # 清空梯度
        optimizer.zero_grad(set_to_none=True)

        # ---------- 10.1 计算当前 RT 变换 ----------
        # 从轴角参数生成旋转矩阵 R
        R = _rodrigues_from_axis_angle(rotation_params)
        T = translation_params

        # 对实例点应用 RT 变换: X' = X @ R^T + T
        instance_xyz_transformed = (instance_xyz_original @ R.T) + T
        temp_gaussians._xyz = instance_xyz_transformed

        # ---------- 10.2 渲染获取 alpha 图 ----------
        background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
        render_pkg = render(camera, temp_gaussians, pipelineparams, background)
        if "rend_alpha" not in render_pkg:
            raise RuntimeError("Render output missing 'rend_alpha'; cannot optimize visibility")
        alpha = render_pkg["rend_alpha"]  # 不透明度图，表示每个像素的覆盖程度
        if alpha.ndim == 2:
            alpha = alpha[None, ...]  # 添加 batch 维度

        # ---------- 10.3 计算可见性得分 ----------
        _, H, W = alpha.shape
        # 中心区域比例：随迭代逐渐缩小，从 100% 降到 30%
        # 这样优化会逐渐聚焦到图像中心，避免实例移动到图像边缘
        center_scale = (1 - it / float(max_iterations)) * 0.7 + 0.3

        # 计算中心区域边界
        center_h = int(H * center_scale)
        center_w = int(W * center_scale)
        start_h = (H - center_h) // 2
        start_w = (W - center_w) // 2
        end_h = start_h + center_h
        end_w = start_w + center_w
        # 只统计中心区域的 alpha 和作为可见性得分
        central_alpha = alpha[0, start_h:end_h, start_w:end_w]
        score = central_alpha.sum()

        # ---------- 10.4 计算深度约束 ----------
        # 将变换后的点转换到相机坐标系
        ones2 = torch.ones((instance_xyz_transformed.shape[0], 1), device=device, dtype=instance_xyz_transformed.dtype)
        xyz_h2 = torch.cat([instance_xyz_transformed, ones2], dim=1)
        xyz_cam2 = (xyz_h2 @ camera_world_view_transform.T)[:, :3]
        mean_depth = torch.mean(-xyz_cam2[:, 2])

        # 深度约束激活条件：只有当实例覆盖足够像素时才启用
        # 防止实例在完全看不见时就开始约束深度
        total_alpha = alpha.sum()
        pixel_ratio = total_alpha / float(H * W + 1e-6)
        if float(pixel_ratio.detach().cpu().item()) <= 0.1:
            depth_w = 0.0  # 覆盖太少，不约束深度
        else:
            depth_w = float(base_depth_constraint_weight)
            # 首次达到可见阈值时，记录当前深度作为参考
            if depth_constraint_ref is None:
                depth_constraint_ref = mean_depth.detach()

        # 深度损失：惩罚深度偏离参考值
        if depth_constraint_ref is not None:
            depth_loss = (mean_depth - depth_constraint_ref) ** 2
        else:
            depth_loss = torch.tensor(0.0, device=device)

        # ---------- 10.5 计算总损失 ----------
        # 正则化项：防止 RT 参数过大
        reg = 0.001 * (torch.norm(rotation_params) ** 2 + torch.norm(translation_params) ** 2)
        # 总损失 = -可见性得分（希望最大化） + 正则化 + 深度约束
        loss = -score + reg + depth_w * depth_loss

        # ---------- 10.6 反向传播和参数更新 ----------
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_([rotation_params, translation_params], max_norm=1.0)
        
        # 跳过包含 NaN 或 Inf 梯度的迭代
        if rotation_params.grad is not None and (
            torch.isnan(rotation_params.grad).any() or torch.isinf(rotation_params.grad).any()
        ):
            continue
        if translation_params.grad is not None and (
            torch.isnan(translation_params.grad).any() or torch.isinf(translation_params.grad).any()
        ):
            continue
        
        # 更新参数
        optimizer.step()

        # ---------- 10.7 数值稳定性检查 ----------
        # 如果参数变成 NaN 或 Inf，重置为零
        if torch.isnan(rotation_params).any() or torch.isinf(rotation_params).any():
            rotation_params.data.zero_()
        if torch.isnan(translation_params).any() or torch.isinf(translation_params).any():
            translation_params.data.zero_()

        # 首次迭代记录正则化值（调试用）
        if it == 0:
            rotation_reg = torch.norm(rotation_params) ** 2
            translation_reg = torch.norm(translation_params) ** 2
            regularization = 0.001 * (rotation_reg + translation_reg)

    # ========== 11. 优化完成，准备最终渲染 ==========
    # 确保 R 和 T 已计算（防止 max_iterations=0 的边界情况）
    if R is None or T is None:
        R = _rodrigues_from_axis_angle(rotation_params.detach())
        T = translation_params.detach()

    # 构建最终变换后的点云
    if apply_rt_to_all_points_for_final_render:
        # 对所有点应用 RT（相当于整个场景旋转，实例相对位置不变）
        # 这种方式渲染效果更自然，因为场景整体一致
        final_transformed_xyz = torch.mm(original_xyz, R.detach().T) + T.detach()
    else:
        # 只对实例点应用 RT，其他点保持不变
        # 这种方式实例会"移动"，可能产生穿插伪影
        final_transformed_xyz = original_xyz.clone()
        final_transformed_xyz[instance_points_idx] = (instance_xyz_original @ R.detach().T) + T.detach()

    # 创建包含优化后位置的高斯模型
    now_gaussians = GaussianModel(sh_degree=3)
    now_gaussians._xyz = final_transformed_xyz
    now_gaussians._features_dc = original_features_dc
    now_gaussians._features_rest = original_features_rest
    now_gaussians._scaling = original_scaling
    now_gaussians._rotation = original_rotation
    now_gaussians._opacity = original_opacity
    # 复制语义特征（如果存在）
    if hasattr(gaussians, "_seg_feature"):
        try:
            now_gaussians._seg_feature = gaussians._seg_feature.clone().detach()
        except Exception:
            pass

    # ========== 12. 保存优化后的渲染结果 ==========
    with torch.no_grad():
        background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
        
        # 渲染优化后的完整场景
        render_pkg = render(camera, now_gaussians, pipelineparams, background)
        if "render" in render_pkg:
            img = render_pkg["render"].detach().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            out_img = out_inst / f"optimized_{instance_id}_{camera_uid}.png"
            Image.fromarray(img).save(str(out_img))
            _log(f"optimized result saved to {out_img}")
        
        # 渲染优化后的实例单独图像，用于获取 alpha mask
        optimized_inst_xyz = (instance_xyz_original @ R.detach().T) + T.detach()
        temp_gaussians._xyz = optimized_inst_xyz
        optimized_inst_pkg = render(camera, temp_gaussians, pipelineparams, background)
        
        # 保存优化后带红色 mask 蒙版的叠加图
        if "render" in render_pkg and "rend_alpha" in optimized_inst_pkg:
            optimized_global_img = render_pkg["render"].detach().cpu().numpy().transpose(1, 2, 0)
            optimized_global_img = np.clip(optimized_global_img * 255.0, 0, 255).astype(np.uint8)
            optimized_inst_alpha = optimized_inst_pkg["rend_alpha"].detach().cpu().numpy()
            optimized_masked_img = _create_masked_overlay(
                optimized_global_img, optimized_inst_alpha, mask_color=(255, 0, 0), mask_alpha=0.4
            )
            optimized_masked_path = out_inst / f"optimized_masked_{instance_id}_{camera_uid}.png"
            Image.fromarray(optimized_masked_img).save(str(optimized_masked_path))
            _log(f"optimized masked result saved to {optimized_masked_path}")

    # ========== 13. 保存最优 RT 参数 ==========
    np.savez(
        str(out_inst / f"best_rt_{camera_uid}.npz"),
        R=R.detach().cpu().numpy(),      # 3x3 旋转矩阵
        T=T.detach().cpu().numpy(),      # 3D 平移向量
        camera_uid=int(camera_uid),       # 使用的相机 UID
        instance_id=int(instance_id),     # 实例 ID
        best_loss=float(loss.detach().cpu().item()) if loss is not None else 0.0,  # 最终损失值
    )
    _log(f"best RT saved to: {out_inst / f'best_rt_{camera_uid}.npz'}")


def main() -> None:
    """
    主函数：解析命令行参数并执行实例可见性优化
    
    支持两种运行模式：
    1. 单实例模式：指定 --instance_id 只优化一个实例
    2. 批量模式：不指定 --instance_id，优化所有实例
    
    典型使用示例：
    --------------
    # 优化所有实例
    python optimize_by_avo_new.py \\
        --source_path /data/scene0000_00 \\
        --label_dir /output/iteration_2500
    
    # 只优化实例 ID=5 的物体
    python optimize_by_avo_new.py \\
        --source_path /data/scene0000_00 \\
        --label_dir /output/iteration_2500 \\
        --instance_id 5
    
    # 使用 top-3 可见性最高的相机视角
    python optimize_by_avo_new.py \\
        --source_path /data/scene0000_00 \\
        --label_dir /output/iteration_2500 \\
        --top_k_views 3
    """
    # ========== 命令行参数定义 ==========
    parser = argparse.ArgumentParser(
        description=(
            "实例可见性优化：使用 instance_info.json + point_cloud_labels.npy 优化每个实例的视角位姿。"
            "如果不指定 --instance_id，将优化 instance_info.json 中的所有实例。"
        )
    )
    
    # ---------- 必需参数 ----------
    parser.add_argument(
        "--source_path",
        type=str,
        required=True,
        help="数据集路径（包含 COLMAP 重建数据和相机参数）",
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        required=True,
        help="标签目录路径（包含 instance_info.json 和 point_cloud_labels.npy）",
    )
    
    # ---------- 可选参数 ----------
    parser.add_argument(
        "--instance_id",
        type=int,
        default=None,
        help="指定要优化的实例 ID（整数）。不指定则优化所有实例。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（默认：<label_dir>/optimize）",
    )
    parser.add_argument(
        "--camera_uid",
        type=int,
        default=None,
        help="可选：强制使用指定的相机 UID，覆盖 best_view_uid",
    )
    parser.add_argument(
        "--visibility_dir",
        type=str,
        default=None,
        help="可选：可见性目录路径（默认从 label_dir 推断为 <model_dir>/visibility）",
    )
    parser.add_argument(
        "--top_k_views",
        type=int,
        default=0,
        help="可选：使用可见性选择 top-k 个最佳视角；0 表示只使用 best_view",
    )
    
    # ---------- 优化超参数 ----------
    parser.add_argument("--max_iterations", type=int, default=10000,
                       help="最大优化迭代次数（默认 10000）")
    parser.add_argument("--lr_R", type=float, default=5e-5,
                       help="旋转参数学习率（默认 5e-5）")
    parser.add_argument("--lr_T", type=float, default=1e-2,
                       help="平移参数学习率（默认 1e-2）")
    parser.add_argument("--depth_w", type=float, default=1e7,
                       help="深度约束权重（默认 1e7）")

    # ---------- RT 应用模式 ----------
    parser.add_argument(
        "--apply_global_rt",
        action="store_true",
        default=None,
        help="最终渲染时对所有点应用 RT（默认启用，与 train_semantic_optimize.py 一致）",
    )
    parser.add_argument(
        "--no_apply_global_rt",
        action="store_true",
        default=None,
        help="可选：禁用全局 RT，只对实例点应用 RT",
    )

    args = parser.parse_args()

    # ========== 处理 RT 应用模式参数 ==========
    # 决定最终渲染时是对所有点还是仅对实例点应用 RT
    if args.no_apply_global_rt:
        apply_global_rt = False  # 只对实例点应用
    elif args.apply_global_rt is True:
        apply_global_rt = True   # 对所有点应用
    else:
        apply_global_rt = True   # 默认对所有点应用

    # ========== 设置输出目录 ==========
    output_dir = args.output_dir or os.path.join(args.label_dir, "optimize")
    os.makedirs(output_dir, exist_ok=True)

    # ========== 加载数据 ==========
    # 1. 加载相机信息和高斯模型
    camera_info = load_camera_info(args.source_path, label_dir=args.label_dir)
    _log(f"Train cameras: {len(camera_info['train_cameras'])}")
    if camera_info.get("gaussian_ply_path"):
        _log(f"Gaussian ply used for rendering: {camera_info['gaussian_ply_path']}")

    # 2. 加载实例信息和点云标签
    instances = load_instance_info(args.label_dir)
    point_labels = load_point_cloud_labels(args.label_dir)

    # ========== 数据一致性验证 ==========
    # 确保点云标签数量与高斯点数量一致
    num_gaussians = int(camera_info["gaussians"].get_xyz.shape[0])
    if int(point_labels.shape[0]) != num_gaussians:
        raise ValueError(
            "point_cloud_labels.npy 长度与当前高斯点数不匹配: "
            f"标签数={point_labels.shape[0]}, 高斯点数={num_gaussians}。"
            "这通常意味着用于渲染的 point_cloud.ply 不是该迭代目录下训练得到的高斯点云，"
            "这会导致实例点索引对不上，产生错误结果。"
        )

    # ========== 确定要优化的实例列表 ==========
    if args.instance_id is not None:
        # 单实例模式
        target_instance_ids = [int(args.instance_id)]
    else:
        # 批量模式：收集所有实例 ID
        ids: List[int] = []
        for rec in instances:
            try:
                ids.append(int(rec.get("instance_id")))
            except Exception:
                continue
        target_instance_ids = sorted(set(ids))  # 去重并排序

    if not target_instance_ids:
        raise ValueError("未找到可优化的实例（检查 instance_info.json 中的 'instances' 列表）")

    _log(f"Optimizing {len(target_instance_ids)} instances")

    # ========== 主优化循环 ==========
    # 创建进度条迭代器（多实例时显示进度）
    instance_iter = target_instance_ids
    if len(target_instance_ids) > 1:
        instance_iter = tqdm.tqdm(
            target_instance_ids,
            desc="Instances",
            unit="inst",
            dynamic_ncols=True,
            leave=True,
        )

    # 遍历每个实例进行优化
    for inst_id in instance_iter:
        # 获取实例记录
        try:
            instance_record = get_instance_record(instances, inst_id)
        except KeyError:
            _log(f"Skip instance_id={inst_id}: not found in instance_info.json")
            continue

        # 获取该实例的点索引
        instance_indices = get_instance_point_indices(point_labels, inst_id)
        if instance_indices.size == 0:
            _log(f"Skip instance_id={inst_id}: no corresponding points in point_cloud_labels.npy")
            continue

        _log(
            f"instance_id={inst_id}: points_in_labels={instance_indices.size}, "
            f"best_view_uid={instance_record.get('best_view_uid')}, best_view={instance_record.get('best_view')}"
        )

        # ---------- 确定优化使用的相机视角 ----------
        camera_uids: List[int] = []
        
        # 方式1：使用 top_k_views 选择可见性最高的相机
        if int(args.top_k_views) > 0:
            visibility_dir = args.visibility_dir or infer_visibility_dir(args.label_dir)
            camera_uids = select_topk_cameras_by_visibility(
                camera_info=camera_info,
                visibility_dir=visibility_dir,
                point_labels=point_labels,
                instance_id=inst_id,
                top_k=int(args.top_k_views),
            )
            if camera_uids:
                _log(f"instance_id={inst_id}: camera uids selected by top_k_views={args.top_k_views}: {camera_uids}")

        # 方式2：回退到默认相机选择逻辑
        if not camera_uids:
            # 优先级：命令行指定 > best_view_uid > best_view > 第一个相机
            fallback_uid = args.camera_uid
            if fallback_uid is None:
                fallback_uid = instance_record.get("best_view_uid")
            if fallback_uid is None:
                fallback_uid = _parse_camera_uid_from_best_view(instance_record.get("best_view"))
            camera_uids = (
                [int(fallback_uid)]
                if fallback_uid is not None
                else [int(camera_info["train_cameras"][0].uid)]
            )

        # ---------- 对每个选定的相机视角执行优化 ----------
        for uid in camera_uids:
            optimize_instance_visibility(
                gaussians=camera_info["gaussians"],      # 高斯模型
                camera_info=camera_info,                  # 相机信息
                instance_record=instance_record,          # 实例元信息
                instance_point_indices=instance_indices,  # 实例点索引
                output_dir=output_dir,                    # 输出目录
                camera_uid_override=int(uid),             # 指定相机
                max_iterations=args.max_iterations,       # 最大迭代次数
                learning_rate_R=args.lr_R,                # 旋转学习率
                learning_rate_T=args.lr_T,                # 平移学习率
                base_depth_constraint_weight=args.depth_w,  # 深度约束权重
                apply_rt_to_all_points_for_final_render=bool(apply_global_rt),  # RT 应用模式
            )


# ========== 脚本入口 ==========
if __name__ == "__main__":
    main()
