#!/usr/bin/env python3
"""
Coverage Sampling Tool - Select representative frames with maximum spatial coverage from image collections.

Based on 3D point clouds predicted by the VGGT model, using greedy maximum coverage algorithm for frame selection.

Usage:
    # Method 1: Specify scene path (will automatically find images folder)
    python coverage_sampling.py --scene_path data/scene0000_00 --num_frames 20

    # Method 2: Directly specify images folder
    python coverage_sampling.py --images_folder data/scene0000_00/images --num_frames 20
    
    # Copy sampled images to output folder
    python coverage_sampling.py --scene_path data/scene0000_00 --num_frames 20 --copy_images
    
    # Specify model path
    python coverage_sampling.py --scene_path data/scene0000_00 --model_path /path/to/vggt
"""

import argparse
import os
import shutil
import sys
from glob import glob
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TF


# Add vggt to sys.path
# Directory structure: submodules/vggt/ contains models/, utils/
# Need to add submodules/ to sys.path so "import vggt.models" works
_script_dir = os.path.dirname(os.path.abspath(__file__))
_possible_parent_paths = [
    os.path.join(_script_dir, 'submodules'),
    os.path.join(os.getcwd(), 'submodules'),
]
for _parent_path in _possible_parent_paths:
    _vggt_path = os.path.join(_parent_path, 'vggt')
    if os.path.exists(_vggt_path) and os.path.isdir(os.path.join(_vggt_path, 'models')):
        sys.path.insert(0, _parent_path)
        break

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def compute_voxel_sets(
    world_points: torch.Tensor,
    world_points_conf_mask: torch.Tensor,
    x_min: float,
    y_min: float,
    z_min: float,
    voxel_size: float
) -> List[Set[Tuple[int, int, int]]]:
    """
    Compute the voxel set covered by each frame.
    
    Args:
        world_points: 3D coordinate tensor of shape (1, T, H, W, 3)
        world_points_conf_mask: Boolean tensor of shape (1, T, H, W) indicating valid points
        x_min, y_min, z_min: Minimum scene coordinates
        voxel_size: Size of each voxel
        
    Returns:
        List of voxel coordinate sets for each frame
    """
    device = world_points.device
    T = world_points.shape[1]
    voxel_sets = []
    
    # Ensure coordinate parameters are on the correct device
    x_min = torch.tensor(x_min, device=device) if not isinstance(x_min, torch.Tensor) else x_min.to(device)
    y_min = torch.tensor(y_min, device=device) if not isinstance(y_min, torch.Tensor) else y_min.to(device)
    z_min = torch.tensor(z_min, device=device) if not isinstance(z_min, torch.Tensor) else z_min.to(device)
    voxel_size = torch.tensor(voxel_size, device=device) if not isinstance(voxel_size, torch.Tensor) else voxel_size.to(device)
    
    for t in range(T):
        # Get valid points for the current frame
        mask = world_points_conf_mask[0, t].flatten()  # (H*W,)
        points = world_points[0, t].reshape(-1, 3)     # (H*W, 3)
        valid_points = points[mask]  # (N_valid, 3)
        
        if valid_points.size(0) == 0:
            voxel_sets.append(set())
            continue
        
        # Compute voxel coordinates
        offset = torch.tensor([x_min, y_min, z_min], device=device)
        voxel_coords = ((valid_points - offset) / voxel_size).floor().long()
        
        # Remove duplicates and convert to CPU set
        unique_voxels = torch.unique(voxel_coords, dim=0)
        voxel_set = set(map(tuple, unique_voxels.cpu().numpy().tolist()))
        
        voxel_sets.append(voxel_set)
    
    return voxel_sets


def maximum_coverage_sampling(voxel_sets: List[Set], K: int) -> List[int]:
    """
    Greedy maximum coverage sampling.
    
    Args:
        voxel_sets: List of voxel sets for each frame
        K: Maximum number of frames to select
        
    Returns:
        List of selected frame indices
    """
    selected = []
    covered = set()
    remaining_frames = set(range(len(voxel_sets)))
    
    for _ in range(K):
        if not remaining_frames:
            break
        
        max_gain = -1
        best_frame = None
        
        # Find the frame with maximum marginal gain
        for frame in remaining_frames:
            gain = len(voxel_sets[frame] - covered)
            if gain > max_gain:
                max_gain = gain
                best_frame = frame
        
        if best_frame is None or max_gain <= 0:
            break  # No more new coverage
        
        selected.append(best_frame)
        covered.update(voxel_sets[best_frame])
        remaining_frames.remove(best_frame)
    
    return selected


def load_and_preprocess_images(image_path_list: List[str], target_size: int = 518) -> torch.Tensor:
    """
    Load and preprocess images.
    
    Args:
        image_path_list: List of image paths
        target_size: Target size
        
    Returns:
        Tensor of shape (T, 3, H, W)
    """
    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    
    for image_path in image_path_list:
        img = Image.open(image_path)
        
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        
        img = img.convert("RGB")
        
        width, height = img.size
        if height > width:
            img = img.rotate(-90, expand=True)
        
        width, height = img.size
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
        
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)
        
        if new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y:start_y + target_size, :]
        
        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)
    
    if len(shapes) > 1:
        print(f"Warning: Images have varying shapes {shapes}, padding to the largest size.")
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)
        
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]
            
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images
    
    images = torch.stack(images)
    
    if len(image_path_list) == 1 and images.dim() == 3:
        images = images.unsqueeze(0)
    
    return images


def space_aware_sampling(
    model: VGGT,
    images: torch.Tensor,
    K: int,
    dtype: torch.dtype = torch.bfloat16
) -> List[int]:
    """
    Perform space-aware frame sampling.
    
    Args:
        model: Pretrained VGGT model
        images: Video frame tensor of shape (T, 3, H, W)
        K: Number of frames to sample
        dtype: Data type
        
    Returns:
        List of selected frame indices
    """
    print("Running VGGT inference...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    
    world_points = predictions['world_points']  # shape (1, T, H, W, 3)
    world_points_flat = world_points.reshape(-1, 3)
    world_points_conf = predictions['world_points_conf']  # shape (1, T, H, W)
    world_points_conf_flat = world_points_conf.reshape(-1)
    
    # Confidence threshold
    init_threshold_val = np.percentile(world_points_conf_flat.cpu().numpy(), 50)
    world_points_conf_mask = (world_points_conf >= init_threshold_val) & (world_points_conf > 0.1)
    world_points_conf_flat_mask = (world_points_conf_flat >= init_threshold_val) & (world_points_conf_flat > 0.1)
    
    # Get point cloud bounding box
    valid_points = world_points_flat[world_points_conf_flat_mask]
    x_min, y_min, z_min = valid_points.min(dim=0)[0]
    x_max, y_max, z_max = valid_points.max(dim=0)[0]
    print(f"Bounding box: x=[{x_min:.3f}, {x_max:.3f}], y=[{y_min:.3f}, {y_max:.3f}], z=[{z_min:.3f}, {z_max:.3f}]")
    
    # Compute voxel size
    voxel_size = min(x_max - x_min, y_max - y_min, z_max - z_min) / 20
    print(f"Voxel size: {voxel_size:.4f}")
    
    # Compute voxel sets for each frame
    print("Computing voxel sets for each frame...")
    voxel_sets = compute_voxel_sets(
        world_points=world_points,
        world_points_conf_mask=world_points_conf_mask,
        x_min=x_min.item(),
        y_min=y_min.item(),
        z_min=z_min.item(),
        voxel_size=voxel_size.item()
    )
    
    # Greedy selection
    print(f"Running maximum coverage sampling (K={K})...")
    selected_frames = sorted(maximum_coverage_sampling(voxel_sets, K))
    
    # Compute coverage statistics
    total_voxels = set()
    for vs in voxel_sets:
        total_voxels.update(vs)
    
    covered_voxels = set()
    for idx in selected_frames:
        covered_voxels.update(voxel_sets[idx])
    
    coverage_ratio = len(covered_voxels) / len(total_voxels) if total_voxels else 0
    print(f"Selected {len(selected_frames)} frames, covering {len(covered_voxels)}/{len(total_voxels)} voxels ({coverage_ratio:.1%})")
    
    return selected_frames


def get_image_paths(folder: str) -> List[str]:
    """Get all image paths in a folder."""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob(os.path.join(folder, ext)))
    return sorted(image_paths)


def main():
    parser = argparse.ArgumentParser(description="Space-aware coverage sampling")
    parser.add_argument(
        "--scene_path",
        type=str,
        default=None,
        help="Scene path (containing images folder)",
    )
    parser.add_argument(
        "--images_folder",
        type=str,
        default=None,
        help="Directly specify images folder path",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="facebook/VGGT-1B",
        help="VGGT model path or HuggingFace model name",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=20,
        help="Number of frames to select",
    )
    parser.add_argument(
        "--max_input_frames",
        type=int,
        default=160,
        help="Maximum input frames (to avoid OOM)",
    )
    parser.add_argument(
        "--copy_images",
        action="store_true",
        help="Whether to copy selected images to output folder",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=None,
        help="Output folder (default: scene_path/sampled_images)",
    )
    
    args = parser.parse_args()
    
    # Determine images folder
    if args.images_folder:
        images_folder = args.images_folder
        scene_path = os.path.dirname(images_folder)
    elif args.scene_path:
        scene_path = args.scene_path
        images_folder = os.path.join(scene_path, "images")
        if not os.path.exists(images_folder):
            # Try alternative names
            for alt_name in ["image", "frames", "frame"]:
                alt_path = os.path.join(scene_path, alt_name)
                if os.path.exists(alt_path):
                    images_folder = alt_path
                    break
    else:
        raise ValueError("Must specify either --scene_path or --images_folder")
    
    if not os.path.exists(images_folder):
        raise FileNotFoundError(f"Images folder not found: {images_folder}")
    
    # Get image paths
    image_paths = get_image_paths(images_folder)
    if not image_paths:
        raise ValueError(f"No images found in {images_folder}")
    
    print(f"Found {len(image_paths)} images in {images_folder}")
    
    # Limit input frame count
    original_count = len(image_paths)
    if len(image_paths) > args.max_input_frames:
        indices = np.linspace(0, len(image_paths) - 1, args.max_input_frames).astype(int)
        image_paths = [image_paths[i] for i in indices]
        print(f"Subsampled to {len(image_paths)} images (from {original_count})")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"Using device: {device}, dtype: {dtype}")
    
    # Load model
    print(f"Loading VGGT model from {args.model_path}...")
    model = VGGT.from_pretrained(args.model_path).to(device)
    
    # Load and preprocess images
    print("Loading and preprocessing images...")
    images = load_and_preprocess_images(image_paths).to(device, dtype=dtype)
    print(f"Image tensor shape: {images.shape}")
    
    # Perform sampling
    K = min(args.num_frames, images.shape[0])
    selected_indices = space_aware_sampling(model, images, K, dtype)
    
    # Map back to original image paths
    selected_paths = [image_paths[i] for i in selected_indices]
    
    print(f"\n=== Selected {len(selected_paths)} frames ===")
    for i, path in enumerate(selected_paths):
        print(f"  {i+1}. {os.path.basename(path)}")
    
    # Save results
    output_folder = args.output_folder or os.path.join(scene_path, "sampled_images")
    os.makedirs(output_folder, exist_ok=True)
    
    # Save selected frame list
    list_file = os.path.join(output_folder, "selected_frames.txt")
    with open(list_file, 'w') as f:
        for path in selected_paths:
            f.write(os.path.basename(path) + "\n")
    print(f"Saved frame list to {list_file}")
    
    # Copy images
    if args.copy_images:
        print(f"Copying selected images to {output_folder}...")
        for path in selected_paths:
            dst = os.path.join(output_folder, os.path.basename(path))
            shutil.copy2(path, dst)
        print(f"Copied {len(selected_paths)} images")
    
    print("\nDone!")
    return selected_indices, selected_paths


if __name__ == "__main__":
    main()
