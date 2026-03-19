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
    Load camera information and Gaussian model.
    
    This is the main function for initializing all data required for optimization.
    
    Args:
        source_path: Dataset path containing COLMAP reconstruction data
        label_dir: Label directory path, may contain trained point_cloud.ply
        
    Returns:
        Dictionary containing the following keys:
        - gaussians: GaussianModel instance
        - scene: Scene instance
        - pipeline_params: Rendering pipeline parameters
        - train_cameras: List of training cameras
        - camera_dict: Mapping dictionary from camera UID to camera object
        - gaussian_ply_path: Actual Gaussian point cloud file path used
    """
    gaussian_ply_path = None
    # Prefer point_cloud.ply under label_dir (usually the trained result)
    if label_dir is not None:
        cand = Path(label_dir) / "point_cloud.ply"
        if cand.exists():
            gaussian_ply_path = str(cand)
    
    # Build scene and Gaussian model
    gaussians, scene, pipeline_params = _build_scene_for_cameras(source_path, gaussian_ply_path=gaussian_ply_path)
    
    # Get training camera list and build index dictionary
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
    Infer visibility directory path from iteration label directory.
    
    Label directory structure is typically: .../train_semanticgs/point_cloud/iteration_XXXX/
    Visibility directory is located at: .../train_semanticgs/visibility/
    
    Args:
        label_dir: Iteration label directory path
        
    Returns:
        Full path to visibility directory
    """
    p = Path(label_dir)
    # Go up two levels to reach model directory, then enter visibility subdirectory
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
    Select best Top-K camera viewpoints by visible point count.
    
    For a given instance, count how many Gaussian points each camera can see,
    and select the K cameras with the most visible points for optimization.
    
    Args:
        camera_info: Camera info dictionary (returned by load_camera_info)
        visibility_dir: Visibility data directory containing per-camera npy files
        point_labels: Point cloud label array recording which instance each point belongs to
        instance_id: Target instance ID
        top_k: Number of cameras to select
        min_visible_points: Minimum visible points threshold, cameras below this are filtered
        
    Returns:
        List of camera UIDs sorted by visible point count descending (at most top_k)
    """
    vis_dir = Path(visibility_dir)
    if not vis_dir.is_dir():
        return []

    instance_id = int(instance_id)
    # Create point cloud mask for this instance
    instance_mask = point_labels == instance_id
    if not np.any(instance_mask):
        return []

    results: List[Tuple[int, int]] = []  # Store (camera_UID, visible_count) tuples
    
    # Iterate through all training cameras
    for cam in camera_info.get("train_cameras", []):
        uid = int(cam.uid)
        # Visibility file naming format: camera_00001_visibility.npy
        vis_path = vis_dir / f"camera_{uid:05d}_visibility.npy"
        if not vis_path.exists():
            continue
        
        # Load visibility array (boolean array, True means point is visible in this camera)
        vis = np.load(str(vis_path), allow_pickle=False)
        if vis.dtype != bool:
            vis = vis.astype(bool)
        
        # Verify array shape matches
        if vis.ndim != 1 or vis.shape[0] != point_labels.shape[0]:
            continue
        
        # Count instance points visible to this camera
        visible_count = int(np.sum(vis & instance_mask))
        if visible_count >= int(min_visible_points):
            results.append((uid, visible_count))

    # Sort by visible count descending, then by UID ascending for ties
    results.sort(key=lambda x: (-x[1], x[0]))
    return [uid for uid, _ in results[: int(top_k)]]


def load_instance_info(label_dir: str) -> List[Dict[str, Any]]:
    """
    Load instance information JSON file.
    
    Reads instance_info.json which contains metadata for all instances,
    such as instance ID, category label, best observation viewpoint, etc.
    
    Args:
        label_dir: Label directory path
        
    Returns:
        List of instance records, each instance is a dictionary containing:
        - instance_id: Instance ID
        - category_label: Category name
        - best_view / best_view_uid: Best observation viewpoint info
        - Other semantic info...
        
    Raises:
        FileNotFoundError: instance_info.json not found
        ValueError: JSON format is incorrect
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
    Load point cloud label array.
    
    Reads point_cloud_labels.npy which records which instance each Gaussian
    point belongs to. Array length should match the number of points in the
    Gaussian point cloud.
    
    Args:
        label_dir: Label directory path
        
    Returns:
        1D int32 array, labels[i] indicates instance ID of the i-th Gaussian point
        
    Raises:
        FileNotFoundError: Label file not found
        ValueError: Label array shape is incorrect
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
    Find instance record by instance ID.
    
    Args:
        instances: List of instance records
        instance_id: Target instance ID
        
    Returns:
        Matching instance record dictionary
        
    Raises:
        KeyError: Specified instance ID not found
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
    Get indices of all points belonging to specified instance.
    
    Args:
        point_labels: Point cloud label array
        instance_id: Target instance ID
        
    Returns:
        int64 array containing indices of all points belonging to this instance
    """
    idx = np.where(point_labels == int(instance_id))[0]
    return idx.astype(np.int64, copy=False)


def _rodrigues_from_axis_angle(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Rodrigues rotation formula: convert axis-angle to rotation matrix (differentiable version).
    
    Axis-angle representation uses a 3D vector where direction represents rotation axis
    and magnitude represents rotation angle (radians). This implementation handles
    numerical stability for small angles using Taylor expansion approximation.
    
    Rodrigues formula: R = I + sin(θ)/θ * K + (1-cos(θ))/θ² * K²
    where K is the skew-symmetric matrix of the rotation axis
    
    Args:
        axis_angle: Axis-angle vector of shape (3,), [rx, ry, rz]
        
    Returns:
        Rotation matrix R of shape (3, 3)
    """
    device = axis_angle.device
    dtype = axis_angle.dtype
    
    # Compute rotation angle (L2 norm of vector), add small value to avoid division by zero
    angle = torch.norm(axis_angle) + 1e-8
    angle = torch.clamp(angle, 0.0, 2.0 * torch.pi)  # Clamp to reasonable range
    angle_sq = angle * angle
    eps = 1e-4  # Small angle threshold

    # Use Taylor expansion for small angles, exact formula for large angles
    # sin(θ)/θ ≈ 1 - θ²/6 when θ→0
    sin_over_angle = torch.where(
        angle > eps,
        torch.sin(angle) / angle,
        1.0 - angle_sq / 6.0,
    )
    # (1-cos(θ))/θ² ≈ 1/2 - θ²/24 when θ→0
    one_minus_cos_over_angle_sq = torch.where(
        angle > eps,
        (1.0 - torch.cos(angle)) / angle_sq,
        0.5 - angle_sq / 24.0,
    )

    # Build skew-symmetric matrix K (cross-product matrix) for rotation axis
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

    # Apply Rodrigues formula
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
    Overlay a semi-transparent colored mask on the rendered image.
    
    Used to visualize instance location in the image by highlighting instance
    regions with semi-transparent color. Commonly uses red (255, 0, 0) to mark
    instance regions for intuitive visualization of optimization results.
    
    Args:
        render_img: [H, W, 3] RGB image, uint8 format
        alpha_mask: [H, W] or [1, H, W] alpha mask, value range 0-1
                   representing probability/opacity of each pixel belonging to instance
        mask_color: Mask color, default red (R, G, B)
        mask_alpha: Mask transparency, 0 for fully transparent, 1 for fully opaque
    
    Returns:
        RGB image with mask overlay, uint8 format
        
    Note:
        Uses alpha blending formula: 
        output = original * (1 - alpha) + mask_color * alpha
    """
    # Handle potential batch dimension
    if alpha_mask.ndim == 3:
        alpha_mask = alpha_mask[0]
    
    # Ensure mask is float type
    if alpha_mask.dtype == np.uint8:
        alpha_mask = alpha_mask.astype(np.float32) / 255.0
    
    # Binarize mask (threshold 0.5), pixels above threshold are considered part of instance
    binary_mask = (alpha_mask > 0.5).astype(np.float32)
    
    # Create color mask layer
    overlay = render_img.astype(np.float32).copy()
    color_layer = np.zeros_like(overlay)
    color_layer[:, :, 0] = mask_color[0]  # R channel
    color_layer[:, :, 1] = mask_color[1]  # G channel
    color_layer[:, :, 2] = mask_color[2]  # B channel
    
    # Only overlay color in mask regions (instance pixels)
    for c in range(3):
        overlay[:, :, c] = np.where(
            binary_mask > 0,
            # Apply alpha blending in instance region
            overlay[:, :, c] * (1 - mask_alpha) + color_layer[:, :, c] * mask_alpha,
            # Keep original in non-instance region
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
    Optimize single instance pose transform (RT) to maximize visibility at specified viewpoint.
    
    Core Optimization Flow:
    =======================
    1. Extract Gaussian point subset for the instance
    2. Initialize rotation params R (axis-angle) and translation params T (3D vector) to zero
    3. Iterative optimization:
       - Apply current RT transform to instance points
       - Render to get alpha map (opacity map)
       - Compute loss = -visibility_score + regularization + depth_constraint
       - Backpropagate to update RT params
    4. Save optimization results (images and RT params)
    
    Loss Function Design:
    =====================
    - Visibility score: Sum of alpha values in image center region (to maximize)
    - Regularization term: Prevent RT from changing too much
    - Depth constraint: Prevent instance from drifting too far, maintain depth stability
    
    Args:
        gaussians: Complete Gaussian model
        camera_info: Camera information dictionary
        instance_record: Instance metadata (contains instance_id, category_label, etc.)
        instance_point_indices: Array of point indices belonging to this instance
        output_dir: Output directory
        camera_uid_override: Optional, force use of specified camera
        max_iterations: Maximum optimization iterations
        learning_rate_R: Rotation parameter learning rate (usually smaller as rotation is sensitive)
        learning_rate_T: Translation parameter learning rate
        base_depth_constraint_weight: Depth constraint weight to prevent object drift
        apply_rt_to_all_points_for_final_render: Whether to apply RT to all points for final render
                                                (True for global view, False for local view)
    
    Output Files:
    =============
    - original_global_{id}_{cam}.png: Pre-optimization full scene render
    - original_instance_{id}_{cam}.png: Pre-optimization instance-only render
    - original_masked_{id}_{cam}.png: Pre-optimization with red mask overlay
    - optimized_{id}_{cam}.png: Post-optimization full scene render
    - optimized_masked_{id}_{cam}.png: Post-optimization with red mask overlay
    - best_rt_{cam}.npz: Optimal RT params (R matrix, T vector, loss value, etc.)
    """
    from PIL import Image

    # ========== 1. Extract Basic Information ==========
    instance_id = int(instance_record.get("instance_id"))
    # Get category label from instance_record (e.g., "chair", "table", etc.)
    category_label = instance_record.get("category_label", "unknown")
    # Clean category name (remove special characters to avoid folder name issues)
    category_label_clean = re.sub(r'[^\w\-]', '_', str(category_label))
    instance_name = f"instance_{instance_id}_{category_label_clean}"

    # Validate instance point count
    if instance_point_indices.size == 0:
        raise ValueError(f"Instance has no points in point_cloud_labels.npy: {instance_name}")

    # ========== 2. Determine Camera for Optimization ==========
    # Priority: camera_uid_override > best_view_uid > best_view > first camera
    camera_uid: Optional[int] = int(camera_uid_override) if camera_uid_override is not None else None
    if camera_uid is None:
        # Try to get best view camera from instance record
        if instance_record.get("best_view_uid") is not None:
            try:
                camera_uid = int(instance_record.get("best_view_uid"))
            except Exception:
                camera_uid = None
    if camera_uid is None:
        # Parse camera UID from best_view string
        camera_uid = _parse_camera_uid_from_best_view(instance_record.get("best_view"))
    if camera_uid is None or camera_uid not in camera_info["camera_dict"]:
        # Finally fall back to first training camera
        camera_uid = int(camera_info["train_cameras"][0].uid)

    camera = camera_info["camera_dict"][camera_uid]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========== 3. Create Output Directory ==========
    out_root = Path(output_dir)
    out_inst = out_root / instance_name  # Separate folder for each instance
    out_inst.mkdir(parents=True, exist_ok=True)

    # ========== 4. Save Original Gaussian Parameters (for recovery and final render) ==========
    # Complete 3DGS parameters include: position, color (SH coefficients), scale, rotation, opacity
    original_xyz = gaussians.get_xyz.clone().detach()           # Position [N, 3]
    original_features_dc = gaussians._features_dc.clone().detach()    # SH 0th-order coefficients (base color)
    original_features_rest = gaussians._features_rest.clone().detach()  # SH higher-order coefficients
    original_scaling = gaussians._scaling.clone().detach()      # Scale parameters (control ellipsoid size)
    original_rotation = gaussians._rotation.clone().detach()    # Rotation quaternion
    original_opacity = gaussians._opacity.clone().detach()      # Opacity

    # ========== 5. Extract Gaussian Subset for Instance ==========
    instance_points_idx = torch.as_tensor(instance_point_indices, dtype=torch.long, device=device)
    # Filter out-of-bounds indices (prevent crash when labels and point cloud mismatch)
    max_idx = int(original_xyz.shape[0]) - 1
    instance_points_idx = instance_points_idx[instance_points_idx <= max_idx]
    if instance_points_idx.numel() == 0:
        raise ValueError(f"All point indices are out of bounds for {instance_name} (max={max_idx})")

    # Extract all Gaussian attributes for instance
    instance_xyz_original = original_xyz[instance_points_idx].clone().detach()
    instance_features_dc = original_features_dc[instance_points_idx].clone().detach()
    instance_features_rest = original_features_rest[instance_points_idx].clone().detach()
    instance_scaling = original_scaling[instance_points_idx].clone().detach()
    instance_rotation = original_rotation[instance_points_idx].clone().detach()
    instance_opacity = original_opacity[instance_points_idx].clone().detach()

    # ========== 6. Compute Original Depth (for depth constraint) ==========
    # Transform world coordinates to camera coordinate system, get Z depth
    camera_world_view_transform = camera.world_view_transform.detach().to(device)
    ones = torch.ones((instance_xyz_original.shape[0], 1), device=device, dtype=instance_xyz_original.dtype)
    xyz_h = torch.cat([instance_xyz_original, ones], dim=1)  # Homogeneous coordinates [N, 4]
    xyz_cam = (xyz_h @ camera_world_view_transform.T)[:, :3]  # Camera coordinates [N, 3]
    original_depths = -xyz_cam[:, 2]  # Depth values (Z-axis, camera faces negative Z)
    original_mean_depth = torch.mean(original_depths)

    # Depth constraint reference value (set when visibility first reaches threshold)
    depth_constraint_ref: Optional[torch.Tensor] = None

    # ========== 7. Initialize Optimization Parameters ==========
    # Rotation uses axis-angle representation (3 parameters), initialized to zero (no rotation)
    rotation_params = torch.zeros(3, dtype=torch.float32, device=device, requires_grad=True)
    # Translation uses 3D vector, initialized to zero (no translation)
    translation_params = torch.zeros(3, dtype=torch.float32, device=device, requires_grad=True)

    # Use Adam optimizer with different learning rates for rotation and translation
    optimizer = torch.optim.Adam(
        [
            {"params": rotation_params, "lr": float(learning_rate_R)},
            {"params": translation_params, "lr": float(learning_rate_T)},
        ]
    )

    # ========== 8. Create Temporary Gaussian Model (instance points only) ==========
    # Used for rendering during optimization, only renders the instance itself
    temp_gaussians = GaussianModel(sh_degree=3)
    temp_gaussians._xyz = instance_xyz_original.clone().detach()
    temp_gaussians._features_dc = instance_features_dc.clone().detach()
    temp_gaussians._features_rest = instance_features_rest.clone().detach()
    temp_gaussians._scaling = instance_scaling.clone().detach()
    temp_gaussians._rotation = instance_rotation.clone().detach()
    temp_gaussians._opacity = instance_opacity.clone().detach()
    # These parameters don't need gradients (RT transform is implemented by directly manipulating xyz)
    temp_gaussians._xyz.requires_grad = False
    temp_gaussians._features_dc.requires_grad = False
    temp_gaussians._features_rest.requires_grad = False
    temp_gaussians._scaling.requires_grad = False
    temp_gaussians._rotation.requires_grad = False
    temp_gaussians._opacity.requires_grad = False

    # Get rendering pipeline parameters
    pipelineparams = getattr(camera_info.get("scene", None), "pipelineparams", None)
    if pipelineparams is None:
        pipelineparams = camera_info.get("pipeline_params")
    if pipelineparams is None:
        dummy_parser = argparse.ArgumentParser()
        pipelineparams = PipelineParams(dummy_parser)

    def _save_rendered_image(render_pkg: Dict[str, Any], save_path: Path) -> None:
        """Save render result to image file."""
        if "render" not in render_pkg:
            return
        # render output format: [C, H, W], value range [0, 1]
        img = render_pkg["render"].detach().cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(img).save(str(save_path))

    # ========== 9. Save Pre-optimization Baseline Render Results ==========
    with torch.no_grad():
        background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
        
        # Render complete scene (all Gaussian points)
        baseline_global_pkg = render(camera, gaussians, pipelineparams, background)
        baseline_global_path = out_inst / f"original_global_{instance_id}_{camera_uid}.png"
        _save_rendered_image(baseline_global_pkg, baseline_global_path)

        # Render instance-only image (for visualizing instance location)
        temp_gaussians._xyz = instance_xyz_original
        baseline_inst_pkg = render(camera, temp_gaussians, pipelineparams, background)
        baseline_inst_path = out_inst / f"original_instance_{instance_id}_{camera_uid}.png"
        _save_rendered_image(baseline_inst_pkg, baseline_inst_path)
        
        # Save pre-optimization overlay with red mask (global scene + instance highlight)
        if "render" in baseline_global_pkg and "rend_alpha" in baseline_inst_pkg:
            global_img = baseline_global_pkg["render"].detach().cpu().numpy().transpose(1, 2, 0)
            global_img = np.clip(global_img * 255.0, 0, 255).astype(np.uint8)
            inst_alpha = baseline_inst_pkg["rend_alpha"].detach().cpu().numpy()
            masked_img = _create_masked_overlay(global_img, inst_alpha, mask_color=(255, 0, 0), mask_alpha=0.4)
            masked_path = out_inst / f"original_masked_{instance_id}_{camera_uid}.png"
            Image.fromarray(masked_img).save(str(masked_path))


    # ========== 10. Main Optimization Loop ==========
    loss = None
    R = None
    T = None

    for it in range(int(max_iterations)):
        # Clear gradients
        optimizer.zero_grad(set_to_none=True)

        # ---------- 10.1 Compute Current RT Transform ----------
        # Generate rotation matrix R from axis-angle parameters
        R = _rodrigues_from_axis_angle(rotation_params)
        T = translation_params

        # Apply RT transform to instance points: X' = X @ R^T + T
        instance_xyz_transformed = (instance_xyz_original @ R.T) + T
        temp_gaussians._xyz = instance_xyz_transformed

        # ---------- 10.2 Render to Get Alpha Map ----------
        background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
        render_pkg = render(camera, temp_gaussians, pipelineparams, background)
        if "rend_alpha" not in render_pkg:
            raise RuntimeError("Render output missing 'rend_alpha'; cannot optimize visibility")
        alpha = render_pkg["rend_alpha"]  # Opacity map, represents coverage of each pixel
        if alpha.ndim == 2:
            alpha = alpha[None, ...]  # Add batch dimension

        # ---------- 10.3 Compute Visibility Score ----------
        _, H, W = alpha.shape
        # Center region ratio: gradually shrinks from 100% to 30% as iteration progresses
        # This makes optimization gradually focus on image center, avoiding instance moving to edges
        center_scale = (1 - it / float(max_iterations)) * 0.7 + 0.3

        # Compute center region boundaries
        center_h = int(H * center_scale)
        center_w = int(W * center_scale)
        start_h = (H - center_h) // 2
        start_w = (W - center_w) // 2
        end_h = start_h + center_h
        end_w = start_w + center_w
        # Only count alpha sum in center region as visibility score
        central_alpha = alpha[0, start_h:end_h, start_w:end_w]
        score = central_alpha.sum()

        # ---------- 10.4 Compute Depth Constraint ----------
        # Transform points to camera coordinate system
        ones2 = torch.ones((instance_xyz_transformed.shape[0], 1), device=device, dtype=instance_xyz_transformed.dtype)
        xyz_h2 = torch.cat([instance_xyz_transformed, ones2], dim=1)
        xyz_cam2 = (xyz_h2 @ camera_world_view_transform.T)[:, :3]
        mean_depth = torch.mean(-xyz_cam2[:, 2])

        # Depth constraint activation condition: only enable when instance covers enough pixels
        # Prevent constraining depth when instance is completely invisible
        total_alpha = alpha.sum()
        pixel_ratio = total_alpha / float(H * W + 1e-6)
        if float(pixel_ratio.detach().cpu().item()) <= 0.1:
            depth_w = 0.0  # Too little coverage, don't constrain depth
        else:
            depth_w = float(base_depth_constraint_weight)
            # Record current depth as reference when first reaching visibility threshold
            if depth_constraint_ref is None:
                depth_constraint_ref = mean_depth.detach()

        # Depth loss: penalize depth deviation from reference
        if depth_constraint_ref is not None:
            depth_loss = (mean_depth - depth_constraint_ref) ** 2
        else:
            depth_loss = torch.tensor(0.0, device=device)

        # ---------- 10.5 Compute Total Loss ----------
        # Regularization term: prevent RT parameters from becoming too large
        reg = 0.001 * (torch.norm(rotation_params) ** 2 + torch.norm(translation_params) ** 2)
        # Total loss = -visibility_score (want to maximize) + regularization + depth_constraint
        loss = -score + reg + depth_w * depth_loss

        # ---------- 10.6 Backpropagation and Parameter Update ----------
        loss.backward()
        # Gradient clipping to prevent gradient explosion
        torch.nn.utils.clip_grad_norm_([rotation_params, translation_params], max_norm=1.0)
        
        # Skip iterations with NaN or Inf gradients
        if rotation_params.grad is not None and (
            torch.isnan(rotation_params.grad).any() or torch.isinf(rotation_params.grad).any()
        ):
            continue
        if translation_params.grad is not None and (
            torch.isnan(translation_params.grad).any() or torch.isinf(translation_params.grad).any()
        ):
            continue
        
        # Update parameters
        optimizer.step()

        # ---------- 10.7 Numerical Stability Check ----------
        # Reset to zero if parameters become NaN or Inf
        if torch.isnan(rotation_params).any() or torch.isinf(rotation_params).any():
            rotation_params.data.zero_()
        if torch.isnan(translation_params).any() or torch.isinf(translation_params).any():
            translation_params.data.zero_()

        # Record regularization value on first iteration (for debugging)
        if it == 0:
            rotation_reg = torch.norm(rotation_params) ** 2
            translation_reg = torch.norm(translation_params) ** 2
            regularization = 0.001 * (rotation_reg + translation_reg)

    # ========== 11. Optimization Complete, Prepare Final Render ==========
    # Ensure R and T are computed (handle edge case of max_iterations=0)
    if R is None or T is None:
        R = _rodrigues_from_axis_angle(rotation_params.detach())
        T = translation_params.detach()

    # Build final transformed point cloud
    if apply_rt_to_all_points_for_final_render:
        # Apply RT to all points (equivalent to rotating entire scene, instance relative position unchanged)
        # This rendering approach is more natural as the scene is globally consistent
        final_transformed_xyz = torch.mm(original_xyz, R.detach().T) + T.detach()
    else:
        # Apply RT only to instance points, keep other points unchanged
        # This way instance will "move", may produce intersection artifacts
        final_transformed_xyz = original_xyz.clone()
        final_transformed_xyz[instance_points_idx] = (instance_xyz_original @ R.detach().T) + T.detach()

    # Create Gaussian model with optimized positions
    now_gaussians = GaussianModel(sh_degree=3)
    now_gaussians._xyz = final_transformed_xyz
    now_gaussians._features_dc = original_features_dc
    now_gaussians._features_rest = original_features_rest
    now_gaussians._scaling = original_scaling
    now_gaussians._rotation = original_rotation
    now_gaussians._opacity = original_opacity
    # Copy semantic features (if exist)
    if hasattr(gaussians, "_seg_feature"):
        try:
            now_gaussians._seg_feature = gaussians._seg_feature.clone().detach()
        except Exception:
            pass

    # ========== 12. Save Post-optimization Render Results ==========
    with torch.no_grad():
        background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
        
        # Render optimized complete scene
        render_pkg = render(camera, now_gaussians, pipelineparams, background)
        if "render" in render_pkg:
            img = render_pkg["render"].detach().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            out_img = out_inst / f"optimized_{instance_id}_{camera_uid}.png"
            Image.fromarray(img).save(str(out_img))
            _log(f"optimized result saved to {out_img}")
        
        # Render optimized instance-only image to get alpha mask
        optimized_inst_xyz = (instance_xyz_original @ R.detach().T) + T.detach()
        temp_gaussians._xyz = optimized_inst_xyz
        optimized_inst_pkg = render(camera, temp_gaussians, pipelineparams, background)
        
        # Save post-optimization overlay with red mask
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

    # ========== 13. Save Optimal RT Parameters ==========
    np.savez(
        str(out_inst / f"best_rt_{camera_uid}.npz"),
        R=R.detach().cpu().numpy(),      # 3x3 rotation matrix
        T=T.detach().cpu().numpy(),      # 3D translation vector
        camera_uid=int(camera_uid),       # Camera UID used
        instance_id=int(instance_id),     # Instance ID
        best_loss=float(loss.detach().cpu().item()) if loss is not None else 0.0,  # Final loss value
    )
    _log(f"best RT saved to: {out_inst / f'best_rt_{camera_uid}.npz'}")


def main() -> None:
    """
    Main function: Parse command line arguments and execute instance visibility optimization.
    
    Supports two running modes:
    1. Single instance mode: specify --instance_id to optimize one instance only
    2. Batch mode: without --instance_id, optimize all instances
    
    Typical Usage Examples:
    -----------------------
    # Optimize all instances
    python optimize_by_avo_new.py \\
        --source_path /data/scene0000_00 \\
        --label_dir /output/iteration_2500
    
    # Optimize only instance ID=5
    python optimize_by_avo_new.py \\
        --source_path /data/scene0000_00 \\
        --label_dir /output/iteration_2500 \\
        --instance_id 5
    
    # Use top-3 camera views with highest visibility
    python optimize_by_avo_new.py \\
        --source_path /data/scene0000_00 \\
        --label_dir /output/iteration_2500 \\
        --top_k_views 3
    """
    # ========== Command Line Argument Definitions ==========
    parser = argparse.ArgumentParser(
        description=(
            "Instance visibility optimization: use instance_info.json + point_cloud_labels.npy "
            "to optimize view pose for each instance. "
            "If --instance_id is not specified, all instances in instance_info.json will be optimized."
        )
    )
    
    # ---------- Required Arguments ----------
    parser.add_argument(
        "--source_path",
        type=str,
        required=True,
        help="Dataset path (contains COLMAP reconstruction data and camera parameters)",
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        required=True,
        help="Label directory path (contains instance_info.json and point_cloud_labels.npy)",
    )
    
    # ---------- Optional Arguments ----------
    parser.add_argument(
        "--instance_id",
        type=int,
        default=None,
        help="Instance ID to optimize (integer). If not specified, optimize all instances.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: <label_dir>/optimize)",
    )
    parser.add_argument(
        "--camera_uid",
        type=int,
        default=None,
        help="Optional: force use specified camera UID, overrides best_view_uid",
    )
    parser.add_argument(
        "--visibility_dir",
        type=str,
        default=None,
        help="Optional: visibility directory path (default: inferred from label_dir as <model_dir>/visibility)",
    )
    parser.add_argument(
        "--top_k_views",
        type=int,
        default=0,
        help="Optional: use visibility to select top-k best views; 0 means use best_view only",
    )
    
    # ---------- Optimization Hyperparameters ----------
    parser.add_argument("--max_iterations", type=int, default=10000,
                       help="Maximum optimization iterations (default 10000)")
    parser.add_argument("--lr_R", type=float, default=5e-5,
                       help="Rotation parameter learning rate (default 5e-5)")
    parser.add_argument("--lr_T", type=float, default=1e-2,
                       help="Translation parameter learning rate (default 1e-2)")
    parser.add_argument("--depth_w", type=float, default=1e7,
                       help="Depth constraint weight (default 1e7)")

    # ---------- RT Application Mode ----------
    parser.add_argument(
        "--apply_global_rt",
        action="store_true",
        default=None,
        help="Apply RT to all points for final render (enabled by default, consistent with train_semantic_optimize.py)",
    )
    parser.add_argument(
        "--no_apply_global_rt",
        action="store_true",
        default=None,
        help="Optional: disable global RT, apply RT only to instance points",
    )

    args = parser.parse_args()

    # ========== Handle RT Application Mode Parameters ==========
    # Decide whether to apply RT to all points or instance points only for final render
    if args.no_apply_global_rt:
        apply_global_rt = False  # Apply to instance points only
    elif args.apply_global_rt is True:
        apply_global_rt = True   # Apply to all points
    else:
        apply_global_rt = True   # Default: apply to all points

    # ========== Set Output Directory ==========
    output_dir = args.output_dir or os.path.join(args.label_dir, "optimize")
    os.makedirs(output_dir, exist_ok=True)

    # ========== Load Data ==========
    # 1. Load camera info and Gaussian model
    camera_info = load_camera_info(args.source_path, label_dir=args.label_dir)
    _log(f"Train cameras: {len(camera_info['train_cameras'])}")
    if camera_info.get("gaussian_ply_path"):
        _log(f"Gaussian ply used for rendering: {camera_info['gaussian_ply_path']}")

    # 2. Load instance info and point cloud labels
    instances = load_instance_info(args.label_dir)
    point_labels = load_point_cloud_labels(args.label_dir)

    # ========== Data Consistency Validation ==========
    # Ensure point cloud label count matches Gaussian point count
    num_gaussians = int(camera_info["gaussians"].get_xyz.shape[0])
    if int(point_labels.shape[0]) != num_gaussians:
        raise ValueError(
            "point_cloud_labels.npy length does not match current Gaussian point count: "
            f"labels={point_labels.shape[0]}, gaussians={num_gaussians}. "
            "This usually means the point_cloud.ply used for rendering is not the trained "
            "Gaussian point cloud under this iteration directory, which can cause instance "
            "point index misalignment and erroneous results."
        )

    # ========== Determine Instance List to Optimize ==========
    if args.instance_id is not None:
        # Single instance mode
        target_instance_ids = [int(args.instance_id)]
    else:
        # Batch mode: collect all instance IDs
        ids: List[int] = []
        for rec in instances:
            try:
                ids.append(int(rec.get("instance_id")))
            except Exception:
                continue
        target_instance_ids = sorted(set(ids))  # Deduplicate and sort

    if not target_instance_ids:
        raise ValueError("No instances found to optimize (check 'instances' list in instance_info.json)")

    _log(f"Optimizing {len(target_instance_ids)} instances")

    # ========== Main Optimization Loop ==========
    # Create progress bar iterator (show progress for multiple instances)
    instance_iter = target_instance_ids
    if len(target_instance_ids) > 1:
        instance_iter = tqdm.tqdm(
            target_instance_ids,
            desc="Instances",
            unit="inst",
            dynamic_ncols=True,
            leave=True,
        )

    # Iterate through each instance for optimization
    for inst_id in instance_iter:
        # Get instance record
        try:
            instance_record = get_instance_record(instances, inst_id)
        except KeyError:
            _log(f"Skip instance_id={inst_id}: not found in instance_info.json")
            continue

        # Get point indices for this instance
        instance_indices = get_instance_point_indices(point_labels, inst_id)
        if instance_indices.size == 0:
            _log(f"Skip instance_id={inst_id}: no corresponding points in point_cloud_labels.npy")
            continue

        _log(
            f"instance_id={inst_id}: points_in_labels={instance_indices.size}, "
            f"best_view_uid={instance_record.get('best_view_uid')}, best_view={instance_record.get('best_view')}"
        )

        # ---------- Determine Camera Views for Optimization ----------
        camera_uids: List[int] = []
        
        # Method 1: Use top_k_views to select cameras with highest visibility
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

        # Method 2: Fall back to default camera selection logic
        if not camera_uids:
            # Priority: command line specified > best_view_uid > best_view > first camera
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

        # ---------- Execute Optimization for Each Selected Camera View ----------
        for uid in camera_uids:
            optimize_instance_visibility(
                gaussians=camera_info["gaussians"],      # Gaussian model
                camera_info=camera_info,                  # Camera info
                instance_record=instance_record,          # Instance metadata
                instance_point_indices=instance_indices,  # Instance point indices
                output_dir=output_dir,                    # Output directory
                camera_uid_override=int(uid),             # Specified camera
                max_iterations=args.max_iterations,       # Max iterations
                learning_rate_R=args.lr_R,                # Rotation learning rate
                learning_rate_T=args.lr_T,                # Translation learning rate
                base_depth_constraint_weight=args.depth_w,  # Depth constraint weight
                apply_rt_to_all_points_for_final_render=bool(apply_global_rt),  # RT application mode
            )


# ========== Script Entry Point ==========
if __name__ == "__main__":
    main()
