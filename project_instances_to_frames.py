#!/usr/bin/env python3
"""
Project each instance from semantically reconstructed GS onto specified frames, generating red mask overlays.

Usage:
    python project_instances_to_frames.py --source_path data/scene0000_00 --label_dir output/data/scene0000_00/train_semanticgs/point_cloud/iteration_2500 --sampled_images_dir data/scene0000_00/sampled_images

Output directory structure:
    <label_dir>/instance_project/
        <frame_name>/
            instance_<id>_<category>.png     # Red mask overlay image
            instance_<id>_<category>_mask.png # Pure mask (white foreground)
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene


def _log(message: str) -> None:
    try:
        tqdm.write(str(message))
    except Exception:
        print(message)


def _build_scene_for_cameras(source_path: str, gaussian_ply_path: Optional[str] = None) -> Tuple[GaussianModel, Scene, Any]:
    """Build a Scene and load gaussians from the given ply."""
    parser = argparse.ArgumentParser(add_help=False)
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    _ = OptimizationParams(parser)

    args = parser.parse_args([])
    args.source_path = os.path.abspath(source_path)
    args.model_path = ""
    args.images = "images"
    args.resolution = -1
    args.white_background = False
    args.data_device = "cuda"
    args.eval = False

    args.use_seg_feature = False
    args.seg_feat_dim = 16
    args.load_seg_feat = False
    args.load_filter_segmap = False
    args.preload_robust_semantic = ""

    mp = lp.extract(args)
    pipeline_params = pp.extract(args)

    gaussians = GaussianModel(sh_degree=3)
    gaussians.pipelineparams = pipeline_params
    gaussians.set_segfeat_params(mp)

    ply_path = gaussian_ply_path or os.path.join(mp.source_path, "point_cloud.ply")
    gaussians.load_ply(str(ply_path))

    scene = Scene(mp, gaussians, loaded_gaussian=True)
    return gaussians, scene, pipeline_params


def load_camera_info(source_path: str, label_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load camera information and Gaussian model."""
    gaussian_ply_path = None
    if label_dir is not None:
        cand = Path(label_dir) / "point_cloud.ply"
        if cand.exists():
            gaussian_ply_path = str(cand)
    gaussians, scene, pipeline_params = _build_scene_for_cameras(source_path, gaussian_ply_path=gaussian_ply_path)
    train_cameras = scene.getTrainCameras()
    
    # Build multiple mappings
    camera_dict = {cam.uid: cam for cam in train_cameras}
    camera_by_name = {}
    for cam in train_cameras:
        name = getattr(cam, "image_name", None)
        if name:
            camera_by_name[str(name)] = cam
            # Also add version without extension
            name_no_ext = os.path.splitext(str(name))[0]
            camera_by_name[name_no_ext] = cam
    
    return {
        "gaussians": gaussians,
        "scene": scene,
        "pipeline_params": pipeline_params,
        "train_cameras": train_cameras,
        "camera_dict": camera_dict,
        "camera_by_name": camera_by_name,
        "gaussian_ply_path": gaussian_ply_path,
    }


def load_instance_info(label_dir: str) -> List[Dict[str, Any]]:
    """Load instance information."""
    instance_json_path = Path(label_dir) / "instance_info.json"
    if not instance_json_path.exists():
        raise FileNotFoundError(f"Instance info not found: {instance_json_path}")
    with open(instance_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("instances", [])


def load_point_cloud_labels(label_dir: str) -> np.ndarray:
    """Load point cloud labels."""
    labels_path = Path(label_dir) / "point_cloud_labels.npy"
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")
    labels = np.load(str(labels_path))
    return labels.astype(np.int32, copy=False)


def get_instance_point_indices(point_labels: np.ndarray, instance_id: int) -> np.ndarray:
    """Get point cloud indices for a specific instance."""
    idx = np.where(point_labels == int(instance_id))[0]
    return idx.astype(np.int64, copy=False)


def evaluate_mask_quality(binary_mask: np.ndarray) -> Dict[str, float]:
    """
    Evaluate mask quality metrics.
    
    Args:
        binary_mask: Binary mask (H, W), values 0 or 1/255
    
    Returns:
        Dict containing various quality metrics
    """
    # Ensure binary mask
    if binary_mask.max() > 1:
        mask = (binary_mask > 127).astype(np.uint8)
    else:
        mask = (binary_mask > 0.5).astype(np.uint8)
    
    total_area = mask.sum()
    if total_area == 0:
        return {
            "num_components": 0,
            "largest_component_ratio": 0.0,
            "compactness": 0.0,
            "solidity": 0.0,
            "total_area": 0,
        }
    
    # 1. Connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    num_components = num_labels - 1  # Subtract background
    
    # Find largest connected component (excluding background label=0)
    if num_components > 0:
        component_areas = stats[1:, cv2.CC_STAT_AREA]  # Exclude background
        largest_area = component_areas.max()
        largest_component_ratio = largest_area / total_area
        largest_label = np.argmax(component_areas) + 1
        largest_mask = (labels == largest_label).astype(np.uint8)
    else:
        largest_component_ratio = 0.0
        largest_mask = mask
        largest_area = total_area
    
    # 2. Compactness: 4π*area/perimeter²
    # Computed using largest connected component
    contours, _ = cv2.findContours(largest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        perimeter = cv2.arcLength(contours[0], True)
        if perimeter > 0:
            compactness = (4 * np.pi * largest_area) / (perimeter * perimeter)
        else:
            compactness = 0.0
    else:
        compactness = 0.0
    
    # 3. Solidity: area/convex_hull_area
    if contours and len(contours[0]) >= 3:
        hull = cv2.convexHull(contours[0])
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = largest_area / hull_area
        else:
            solidity = 0.0
    else:
        solidity = 0.0
    
    return {
        "num_components": num_components,
        "largest_component_ratio": largest_component_ratio,
        "compactness": compactness,
        "solidity": solidity,
        "total_area": int(total_area),
    }


def filter_mask_by_quality(
    binary_mask: np.ndarray,
    max_num_components: int = 3,
    min_largest_component_ratio: float = 0.7,
    min_compactness: float = 0.1,
    min_solidity: float = 0.5,
) -> Tuple[bool, Dict[str, float]]:
    """
    Determine if mask passes quality criteria.
    
    Returns:
        (is_valid, quality_metrics)
    """
    metrics = evaluate_mask_quality(binary_mask)
    
    is_valid = True
    
    # Check number of connected components
    if max_num_components > 0 and metrics["num_components"] > max_num_components:
        is_valid = False
    
    # Check largest component ratio
    if min_largest_component_ratio > 0 and metrics["largest_component_ratio"] < min_largest_component_ratio:
        is_valid = False
    
    # Check compactness
    if min_compactness > 0 and metrics["compactness"] < min_compactness:
        is_valid = False
    
    # Check solidity
    if min_solidity > 0 and metrics["solidity"] < min_solidity:
        is_valid = False
    
    return is_valid, metrics


def draw_id_label(
    image: np.ndarray,
    center_x: int,
    center_y: int,
    label_id: int,
    font_scale: float = 0.6,
    thickness: int = 2,
    padding: int = 5,
) -> np.ndarray:
    """
    Draw an ID label with green background and red text on the image.
    
    Args:
        image: RGB image (H, W, 3)
        center_x, center_y: Label center position
        label_id: ID number to display
        font_scale: Font size
        thickness: Font thickness
        padding: Background box padding
    
    Returns:
        Image with label drawn
    """
    img = image.copy()
    text = str(label_id)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Calculate background box position
    box_w = text_w + 2 * padding
    box_h = text_h + 2 * padding + baseline
    
    x1 = center_x - box_w // 2
    y1 = center_y - box_h // 2
    x2 = x1 + box_w
    y2 = y1 + box_h
    
    # Ensure within image bounds
    H, W = img.shape[:2]
    x1 = max(0, min(x1, W - box_w))
    y1 = max(0, min(y1, H - box_h))
    x2 = x1 + box_w
    y2 = y1 + box_h
    
    # Draw green background box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), -1)  # Green fill
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)   # Dark green border
    
    # Draw red text
    text_x = x1 + padding
    text_y = y1 + padding + text_h
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)
    
    return img


def get_mask_center(binary_mask: np.ndarray) -> Tuple[int, int]:
    """
    Get the centroid position of the mask.
    
    Args:
        binary_mask: Binary mask (H, W)
    
    Returns:
        (center_x, center_y)
    """
    if binary_mask.max() > 1:
        mask = (binary_mask > 127).astype(np.uint8)
    else:
        mask = (binary_mask > 0.5).astype(np.uint8)
    
    # Use connected components to find centroid of largest region
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    if num_labels <= 1:
        # No foreground, return image center
        H, W = mask.shape
        return W // 2, H // 2
    
    # Find largest connected component (excluding background)
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(component_areas) + 1
    
    # Return centroid of largest connected component
    cx, cy = centroids[largest_label]
    return int(cx), int(cy)


def _create_masked_overlay(
    render_img: np.ndarray,
    alpha_mask: np.ndarray,
    mask_color: Tuple[int, int, int] = (255, 0, 0),
    mask_alpha: float = 0.4,
) -> np.ndarray:
    """Overlay a semi-transparent colored mask on the rendered image."""
    if alpha_mask.ndim == 3:
        alpha_mask = alpha_mask[0]
    
    if alpha_mask.dtype == np.uint8:
        alpha_mask = alpha_mask.astype(np.float32) / 255.0
    
    binary_mask = (alpha_mask > 0.5).astype(np.float32)
    
    overlay = render_img.astype(np.float32).copy()
    color_layer = np.zeros_like(overlay)
    color_layer[:, :, 0] = mask_color[0]
    color_layer[:, :, 1] = mask_color[1]
    color_layer[:, :, 2] = mask_color[2]
    
    for c in range(3):
        overlay[:, :, c] = np.where(
            binary_mask > 0,
            overlay[:, :, c] * (1 - mask_alpha) + color_layer[:, :, c] * mask_alpha,
            overlay[:, :, c]
        )
    
    return np.clip(overlay, 0, 255).astype(np.uint8)


def render_instance_on_camera(
    gaussians: GaussianModel,
    camera: Any,
    pipeline_params: Any,
    instance_point_indices: np.ndarray,
    device: torch.device,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Render a single instance projected onto the specified camera.
    
    Returns:
        (alpha_mask, rendered_image) or None (if instance is not visible)
    """
    if instance_point_indices.size == 0:
        return None
    
    # Get instance point cloud
    original_xyz = gaussians.get_xyz.detach()
    original_features_dc = gaussians._features_dc.detach()
    original_features_rest = gaussians._features_rest.detach()
    original_scaling = gaussians._scaling.detach()
    original_rotation = gaussians._rotation.detach()
    original_opacity = gaussians._opacity.detach()
    
    instance_idx = torch.as_tensor(instance_point_indices, dtype=torch.long, device=device)
    max_idx = int(original_xyz.shape[0]) - 1
    instance_idx = instance_idx[instance_idx <= max_idx]
    
    if instance_idx.numel() == 0:
        return None
    
    # Create temporary Gaussian model containing only this instance's points
    temp_gaussians = GaussianModel(sh_degree=3)
    temp_gaussians._xyz = original_xyz[instance_idx].clone()
    temp_gaussians._features_dc = original_features_dc[instance_idx].clone()
    temp_gaussians._features_rest = original_features_rest[instance_idx].clone()
    temp_gaussians._scaling = original_scaling[instance_idx].clone()
    temp_gaussians._rotation = original_rotation[instance_idx].clone()
    temp_gaussians._opacity = original_opacity[instance_idx].clone()
    
    # Render
    background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
    render_pkg = render(camera, temp_gaussians, pipeline_params, background)
    
    if "rend_alpha" not in render_pkg:
        return None
    
    alpha = render_pkg["rend_alpha"].detach().cpu().numpy()
    if alpha.ndim == 3:
        alpha = alpha[0]
    
    # Check if there's valid projection
    if alpha.max() < 0.1:
        return None
    
    rendered = render_pkg["render"].detach().cpu().numpy().transpose(1, 2, 0)
    rendered = np.clip(rendered * 255.0, 0, 255).astype(np.uint8)
    
    return alpha, rendered


def render_full_scene(
    gaussians: GaussianModel,
    camera: Any,
    pipeline_params: Any,
    device: torch.device,
) -> np.ndarray:
    """Render full scene."""
    background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
    render_pkg = render(camera, gaussians, pipeline_params, background)
    
    img = render_pkg["render"].detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def main():
    parser = argparse.ArgumentParser(
        description="Project each instance onto each frame in sampled_images"
    )
    parser.add_argument(
        "--source_path",
        type=str,
        required=True,
        help="Dataset path (e.g., data/scene0000_00)",
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        required=True,
        help="Label directory (e.g., output/.../point_cloud/iteration_2500)",
    )
    parser.add_argument(
        "--sampled_images_dir",
        type=str,
        default=None,
        help="sampled_images directory (default: <source_path>/sampled_images)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: <label_dir>/instance_project)",
    )
    parser.add_argument(
        "--min_visible_pixels",
        type=int,
        default=5000,
        help="Minimum visible pixel threshold for projection (default: 5000)",
    )
    parser.add_argument(
        "--mask_alpha",
        type=float,
        default=0.4,
        help="Red mask transparency (default: 0.4)",
    )
    # === Instance filtering parameters ===
    parser.add_argument(
        "--min_num_points",
        type=int,
        default=1000,
        help="Minimum number of 3D points (default: 1000)",
    )
    parser.add_argument(
        "--min_max_visible",
        type=int,
        default=500,
        help="Minimum best_view visible points (default: 500)",
    )
    parser.add_argument(
        "--exclude_unknown",
        action="store_true",
        default=True,
        help="Exclude instances with category_label 'unknown' (default: True)",
    )
    parser.add_argument(
        "--no_exclude_unknown",
        action="store_true",
        help="Do not exclude 'unknown' category",
    )
    parser.add_argument(
        "--exclude_categories",
        type=str,
        default="",
        help="Categories to exclude, comma-separated (e.g., wall,floor)",
    )
    parser.add_argument(
        "--include_categories",
        type=str,
        default="",
        help="Categories to include only, comma-separated (e.g., chair,table,desk), empty means no restriction",
    )
    # === Mask quality filtering parameters ===
    parser.add_argument(
        "--max_num_components",
        type=int,
        default=5,
        help="Maximum number of connected components (default: 3, filter if exceeded; 0=no filter)",
    )
    parser.add_argument(
        "--min_largest_component_ratio",
        type=float,
        default=0,
        help="Largest component ratio threshold (default: 0.7, filter if below; 0=no filter)",
    )
    parser.add_argument(
        "--min_compactness",
        type=float,
        default=0,
        help="Minimum compactness 4π*area/perimeter² (default: 0.1, filter if below; 0=no filter)",
    )
    parser.add_argument(
        "--min_solidity",
        type=float,
        default=0,
        help="Minimum solidity area/convex_hull_area (default: 0.5, filter if below; 0=no filter)",
    )
    
    args = parser.parse_args()
    
    # Determine paths
    source_path = args.source_path
    label_dir = args.label_dir
    sampled_images_dir = args.sampled_images_dir or os.path.join(source_path, "sampled_images")
    output_dir = args.output_dir or os.path.join(label_dir, "instance_project")
    
    if not os.path.exists(sampled_images_dir):
        raise FileNotFoundError(f"sampled_images_dir not found: {sampled_images_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Using device: {device}")
    
    # Load camera and Gaussian model
    _log("Loading camera info and gaussians...")
    camera_info = load_camera_info(source_path, label_dir=label_dir)
    gaussians = camera_info["gaussians"]
    pipeline_params = camera_info["pipeline_params"]
    camera_by_name = camera_info["camera_by_name"]
    
    _log(f"Loaded {len(camera_info['train_cameras'])} cameras")
    if camera_info.get("gaussian_ply_path"):
        _log(f"Using gaussian ply: {camera_info['gaussian_ply_path']}")
    
    # Load instance info and point cloud labels
    _log("Loading instance info and labels...")
    instances = load_instance_info(label_dir)
    point_labels = load_point_cloud_labels(label_dir)
    _log(f"Loaded {len(instances)} instances, {len(point_labels)} point labels")
    
    # Get sampled_images list
    sampled_files = sorted([
        f for f in os.listdir(sampled_images_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    _log(f"Found {len(sampled_files)} sampled images")
    
    # Process exclude_unknown parameter
    exclude_unknown = args.exclude_unknown and not args.no_exclude_unknown
    
    # Parse filtering parameters
    exclude_cats = set()
    if exclude_unknown:
        exclude_cats.add("unknown")
    if args.exclude_categories:
        exclude_cats.update([c.strip().lower() for c in args.exclude_categories.split(",") if c.strip()])
    
    include_cats = set()
    if args.include_categories:
        include_cats.update([c.strip().lower() for c in args.include_categories.split(",") if c.strip()])
    
    _log(f"Filtering: min_num_points={args.min_num_points}, min_max_visible={args.min_max_visible}")
    if exclude_cats:
        _log(f"Excluding categories: {exclude_cats}")
    if include_cats:
        _log(f"Including only categories: {include_cats}")
    
    # Build instance info dictionary (with filtering)
    instance_dict = {}
    filtered_stats = {"total": 0, "by_points": 0, "by_visible": 0, "by_category": 0, "no_indices": 0}
    
    for inst in instances:
        filtered_stats["total"] += 1
        inst_id = int(inst.get("instance_id", -1))
        if inst_id < 0:
            continue
        
        num_points = inst.get("num_points", 0)
        max_visible = inst.get("max_visible", 0)
        category = inst.get("category_label", "unknown")
        category_lower = category.lower()
        category_clean = re.sub(r'[^\w\-]', '_', str(category))
        
        # Filter condition 1: 3D point count
        if num_points < args.min_num_points:
            filtered_stats["by_points"] += 1
            continue
        
        # Filter condition 2: best_view visible points
        if max_visible < args.min_max_visible:
            filtered_stats["by_visible"] += 1
            continue
        
        # Filter condition 3: category exclusion
        if category_lower in exclude_cats:
            filtered_stats["by_category"] += 1
            continue
        
        # Filter condition 4: category inclusion (if include_categories is set)
        if include_cats and category_lower not in include_cats:
            filtered_stats["by_category"] += 1
            continue
        
        indices = get_instance_point_indices(point_labels, inst_id)
        if indices.size == 0:
            filtered_stats["no_indices"] += 1
            continue
        
        instance_dict[inst_id] = {
            "category": category_clean,
            "indices": indices,
            "num_points": num_points,
            "max_visible": max_visible,
        }
    
    _log(f"Instance filtering stats:")
    _log(f"  Total: {filtered_stats['total']}")
    _log(f"  Filtered by num_points < {args.min_num_points}: {filtered_stats['by_points']}")
    _log(f"  Filtered by max_visible < {args.min_max_visible}: {filtered_stats['by_visible']}")
    _log(f"  Filtered by category: {filtered_stats['by_category']}")
    _log(f"  No point indices: {filtered_stats['no_indices']}")
    _log(f"  Remaining valid instances: {len(instance_dict)}")
    
    # Process each sampled image
    for img_file in tqdm(sampled_files, desc="Processing frames"):
        frame_name = os.path.splitext(img_file)[0]
        
        # Find corresponding camera
        camera = camera_by_name.get(frame_name)
        if camera is None:
            camera = camera_by_name.get(img_file)
        if camera is None:
            _log(f"Warning: No camera found for {img_file}, skipping")
            continue
        
        # Create output directory for this frame
        frame_output_dir = os.path.join(output_dir, frame_name)
        os.makedirs(frame_output_dir, exist_ok=True)
        
        # Render full scene as background
        full_scene_img = render_full_scene(gaussians, camera, pipeline_params, device)
        
        # Save full rendered image
        full_scene_path = os.path.join(frame_output_dir, "full_scene.png")
        Image.fromarray(full_scene_img).save(full_scene_path)
        
        # Project each instance
        visible_count = 0
        quality_filtered_count = 0
        
        # Collect valid instance info for drawing id_scene
        valid_instances_for_id_scene = []
        
        for inst_id, inst_info in instance_dict.items():
            result = render_instance_on_camera(
                gaussians=gaussians,
                camera=camera,
                pipeline_params=pipeline_params,
                instance_point_indices=inst_info["indices"],
                device=device,
            )
            
            if result is None:
                continue
            
            alpha_mask, _ = result
            
            # Check visible pixel count
            visible_pixels = (alpha_mask > 0.5).sum()
            if visible_pixels < args.min_visible_pixels:
                continue
            
            # Binarize mask
            binary_mask = (alpha_mask > 0.5).astype(np.uint8)
            
            # Mask quality evaluation
            is_valid, metrics = filter_mask_by_quality(
                binary_mask,
                max_num_components=args.max_num_components,
                min_largest_component_ratio=args.min_largest_component_ratio,
                min_compactness=args.min_compactness,
                min_solidity=args.min_solidity,
            )
            
            if not is_valid:
                quality_filtered_count += 1
                continue
            
            visible_count += 1
            category = inst_info["category"]
            
            # Get mask center position
            center_x, center_y = get_mask_center(binary_mask)
            valid_instances_for_id_scene.append({
                "inst_id": inst_id,
                "center_x": center_x,
                "center_y": center_y,
                "label_id": inst_id + 3,  # label_id = instance_id + 3
            })
            
            # Generate red mask overlay image
            masked_img = _create_masked_overlay(
                full_scene_img, 
                alpha_mask, 
                mask_color=(255, 0, 0), 
                mask_alpha=args.mask_alpha
            )
            masked_path = os.path.join(frame_output_dir, f"instance_{inst_id}_{category}.png")
            Image.fromarray(masked_img).save(masked_path)
            
            # Save pure mask (white foreground, black background)
            binary_mask_255 = binary_mask * 255
            mask_path = os.path.join(frame_output_dir, f"instance_{inst_id}_{category}_mask.png")
            Image.fromarray(binary_mask_255).save(mask_path)
        
        # Generate id_scene.png - draw all valid instance IDs on scene image
        if valid_instances_for_id_scene:
            id_scene_img = full_scene_img.copy()
            for inst_data in valid_instances_for_id_scene:
                id_scene_img = draw_id_label(
                    id_scene_img,
                    center_x=inst_data["center_x"],
                    center_y=inst_data["center_y"],
                    label_id=inst_data["label_id"],
                    font_scale=0.6,
                    thickness=2,
                    padding=5,
                )
            id_scene_path = os.path.join(frame_output_dir, "id_scene.png")
            Image.fromarray(id_scene_img).save(id_scene_path)
        
        _log(f"Frame {frame_name}: {visible_count} valid, {quality_filtered_count} filtered by quality")
        
        # Clear GPU memory
        torch.cuda.empty_cache()
    
    _log(f"\nDone! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
