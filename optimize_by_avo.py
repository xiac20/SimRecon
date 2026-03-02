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
    try:
        tqdm.tqdm.write(str(message))
    except Exception:
        print(message)


def _parse_camera_uid_from_best_view(best_view: Optional[str]) -> Optional[int]:
    if not best_view:
        return None
    m = re.search(r"camera_(\d+)", str(best_view))
    return int(m.group(1)) if m else None


def _build_scene_for_cameras(source_path: str, gaussian_ply_path: Optional[str] = None) -> Tuple[GaussianModel, Scene, Any]:
    """Build a Scene and load gaussians from the given ply (if provided)."""
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
    gaussian_ply_path = None
    if label_dir is not None:
        cand = Path(label_dir) / "point_cloud.ply"
        if cand.exists():
            gaussian_ply_path = str(cand)
    gaussians, scene, pipeline_params = _build_scene_for_cameras(source_path, gaussian_ply_path=gaussian_ply_path)
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
    """Infer the visibility directory from an iteration label dir."""
    p = Path(label_dir)
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
    """Pick top-k camera uids by visible point count."""
    vis_dir = Path(visibility_dir)
    if not vis_dir.is_dir():
        return []

    instance_id = int(instance_id)
    instance_mask = point_labels == instance_id
    if not np.any(instance_mask):
        return []

    results: List[Tuple[int, int]] = []
    for cam in camera_info.get("train_cameras", []):
        uid = int(cam.uid)
        vis_path = vis_dir / f"camera_{uid:05d}_visibility.npy"
        if not vis_path.exists():
            continue
        vis = np.load(str(vis_path), allow_pickle=False)
        if vis.dtype != bool:
            vis = vis.astype(bool)
        if vis.ndim != 1 or vis.shape[0] != point_labels.shape[0]:
            continue
        visible_count = int(np.sum(vis & instance_mask))
        if visible_count >= int(min_visible_points):
            results.append((uid, visible_count))

    results.sort(key=lambda x: (-x[1], x[0]))
    return [uid for uid, _ in results[: int(top_k)]]


def load_instance_info(label_dir: str) -> List[Dict[str, Any]]:
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
    label_dir_path = Path(label_dir)
    labels_path = label_dir_path / "point_cloud_labels.npy"
    if not labels_path.exists():
        raise FileNotFoundError(f"Label array file not found: {labels_path}")
    labels = np.load(str(labels_path))
    if labels.ndim != 1:
        raise ValueError(f"Invalid point_cloud_labels.npy shape: {labels.shape}")
    return labels.astype(np.int32, copy=False)


def get_instance_record(instances: List[Dict[str, Any]], instance_id: int) -> Dict[str, Any]:
    for rec in instances:
        try:
            if int(rec.get("instance_id")) == int(instance_id):
                return rec
        except Exception:
            continue
    raise KeyError(f"instance_id={instance_id} not found in instance_info.json")


def get_instance_point_indices(point_labels: np.ndarray, instance_id: int) -> np.ndarray:
    idx = np.where(point_labels == int(instance_id))[0]
    return idx.astype(np.int64, copy=False)


def _rodrigues_from_axis_angle(axis_angle: torch.Tensor) -> torch.Tensor:
    """Differentiable Rodrigues (stable for small angles)."""
    device = axis_angle.device
    dtype = axis_angle.dtype
    angle = torch.norm(axis_angle) + 1e-8
    angle = torch.clamp(angle, 0.0, 2.0 * torch.pi)
    angle_sq = angle * angle
    eps = 1e-4

    sin_over_angle = torch.where(
        angle > eps,
        torch.sin(angle) / angle,
        1.0 - angle_sq / 6.0,
    )
    one_minus_cos_over_angle_sq = torch.where(
        angle > eps,
        (1.0 - torch.cos(angle)) / angle_sq,
        0.5 - angle_sq / 24.0,
    )

    zero = torch.zeros_like(axis_angle[0])
    K = torch.stack(
        [
            torch.stack([zero, -axis_angle[2], axis_angle[1]]),
            torch.stack([axis_angle[2], zero, -axis_angle[0]]),
            torch.stack([-axis_angle[1], axis_angle[0], zero]),
        ]
    )

    I = torch.eye(3, device=device, dtype=dtype)
    R = I + sin_over_angle * K + one_minus_cos_over_angle_sq * (K @ K)
    return R


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
    """Optimize a single instance pose (RT) by maximizing rendered alpha."""
    from PIL import Image

    instance_id = int(instance_record.get("instance_id"))
    instance_name = f"instance_{instance_id}"

    if instance_point_indices.size == 0:
        raise ValueError(f"Instance has no points in point_cloud_labels.npy: {instance_name}")

    camera_uid: Optional[int] = int(camera_uid_override) if camera_uid_override is not None else None
    if camera_uid is None:
        if instance_record.get("best_view_uid") is not None:
            try:
                camera_uid = int(instance_record.get("best_view_uid"))
            except Exception:
                camera_uid = None
    if camera_uid is None:
        camera_uid = _parse_camera_uid_from_best_view(instance_record.get("best_view"))
    if camera_uid is None or camera_uid not in camera_info["camera_dict"]:
        camera_uid = int(camera_info["train_cameras"][0].uid)

    camera = camera_info["camera_dict"][camera_uid]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_root = Path(output_dir)
    out_inst = out_root / instance_name
    out_inst.mkdir(parents=True, exist_ok=True)

    original_xyz = gaussians.get_xyz.clone().detach()
    original_features_dc = gaussians._features_dc.clone().detach()
    original_features_rest = gaussians._features_rest.clone().detach()
    original_scaling = gaussians._scaling.clone().detach()
    original_rotation = gaussians._rotation.clone().detach()
    original_opacity = gaussians._opacity.clone().detach()

    instance_points_idx = torch.as_tensor(instance_point_indices, dtype=torch.long, device=device)
    max_idx = int(original_xyz.shape[0]) - 1
    instance_points_idx = instance_points_idx[instance_points_idx <= max_idx]
    if instance_points_idx.numel() == 0:
        raise ValueError(f"All point indices are out of bounds for {instance_name} (max={max_idx})")

    instance_xyz_original = original_xyz[instance_points_idx].clone().detach()
    instance_features_dc = original_features_dc[instance_points_idx].clone().detach()
    instance_features_rest = original_features_rest[instance_points_idx].clone().detach()
    instance_scaling = original_scaling[instance_points_idx].clone().detach()
    instance_rotation = original_rotation[instance_points_idx].clone().detach()
    instance_opacity = original_opacity[instance_points_idx].clone().detach()

    camera_world_view_transform = camera.world_view_transform.detach().to(device)
    ones = torch.ones((instance_xyz_original.shape[0], 1), device=device, dtype=instance_xyz_original.dtype)
    xyz_h = torch.cat([instance_xyz_original, ones], dim=1)
    xyz_cam = (xyz_h @ camera_world_view_transform.T)[:, :3]
    original_depths = -xyz_cam[:, 2]
    original_mean_depth = torch.mean(original_depths)

    depth_constraint_ref: Optional[torch.Tensor] = None

    rotation_params = torch.zeros(3, dtype=torch.float32, device=device, requires_grad=True)
    translation_params = torch.zeros(3, dtype=torch.float32, device=device, requires_grad=True)

    optimizer = torch.optim.Adam(
        [
            {"params": rotation_params, "lr": float(learning_rate_R)},
            {"params": translation_params, "lr": float(learning_rate_T)},
        ]
    )

    temp_gaussians = GaussianModel(sh_degree=3)
    temp_gaussians._xyz = instance_xyz_original.clone().detach()
    temp_gaussians._features_dc = instance_features_dc.clone().detach()
    temp_gaussians._features_rest = instance_features_rest.clone().detach()
    temp_gaussians._scaling = instance_scaling.clone().detach()
    temp_gaussians._rotation = instance_rotation.clone().detach()
    temp_gaussians._opacity = instance_opacity.clone().detach()
    temp_gaussians._xyz.requires_grad = False
    temp_gaussians._features_dc.requires_grad = False
    temp_gaussians._features_rest.requires_grad = False
    temp_gaussians._scaling.requires_grad = False
    temp_gaussians._rotation.requires_grad = False
    temp_gaussians._opacity.requires_grad = False

    pipelineparams = getattr(camera_info.get("scene", None), "pipelineparams", None)
    if pipelineparams is None:
        pipelineparams = camera_info.get("pipeline_params")
    if pipelineparams is None:
        dummy_parser = argparse.ArgumentParser()
        pipelineparams = PipelineParams(dummy_parser)

    def _save_rendered_image(render_pkg: Dict[str, Any], save_path: Path) -> None:
        if "render" not in render_pkg:
            return
        img = render_pkg["render"].detach().cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(img).save(str(save_path))

    with torch.no_grad():
        background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
        baseline_global_pkg = render(camera, gaussians, pipelineparams, background)
        baseline_global_path = out_inst / f"original_global_{instance_id}_{camera_uid}.png"
        _save_rendered_image(baseline_global_pkg, baseline_global_path)

        temp_gaussians._xyz = instance_xyz_original
        baseline_inst_pkg = render(camera, temp_gaussians, pipelineparams, background)
        baseline_inst_path = out_inst / f"original_instance_{instance_id}_{camera_uid}.png"
        _save_rendered_image(baseline_inst_pkg, baseline_inst_path)


    loss = None
    R = None
    T = None

    for it in range(int(max_iterations)):
        optimizer.zero_grad(set_to_none=True)

        R = _rodrigues_from_axis_angle(rotation_params)
        T = translation_params

        instance_xyz_transformed = (instance_xyz_original @ R.T) + T
        temp_gaussians._xyz = instance_xyz_transformed

        background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
        render_pkg = render(camera, temp_gaussians, pipelineparams, background)
        if "rend_alpha" not in render_pkg:
            raise RuntimeError("Render output missing 'rend_alpha'; cannot optimize visibility")
        alpha = render_pkg["rend_alpha"]
        if alpha.ndim == 2:
            alpha = alpha[None, ...]

        _, H, W = alpha.shape
        center_scale = (1 - it / float(max_iterations)) * 0.7 + 0.3

        center_h = int(H * center_scale)
        center_w = int(W * center_scale)
        start_h = (H - center_h) // 2
        start_w = (W - center_w) // 2
        end_h = start_h + center_h
        end_w = start_w + center_w
        central_alpha = alpha[0, start_h:end_h, start_w:end_w]
        score = central_alpha.sum()

        ones2 = torch.ones((instance_xyz_transformed.shape[0], 1), device=device, dtype=instance_xyz_transformed.dtype)
        xyz_h2 = torch.cat([instance_xyz_transformed, ones2], dim=1)
        xyz_cam2 = (xyz_h2 @ camera_world_view_transform.T)[:, :3]
        mean_depth = torch.mean(-xyz_cam2[:, 2])

        total_alpha = alpha.sum()
        pixel_ratio = total_alpha / float(H * W + 1e-6)
        if float(pixel_ratio.detach().cpu().item()) <= 0.1:
            depth_w = 0.0
        else:
            depth_w = float(base_depth_constraint_weight)
            if depth_constraint_ref is None:
                depth_constraint_ref = mean_depth.detach()

        if depth_constraint_ref is not None:
            depth_loss = (mean_depth - depth_constraint_ref) ** 2
        else:
            depth_loss = torch.tensor(0.0, device=device)

        reg = 0.001 * (torch.norm(rotation_params) ** 2 + torch.norm(translation_params) ** 2)
        loss = -score + reg + depth_w * depth_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_([rotation_params, translation_params], max_norm=1.0)
        if rotation_params.grad is not None and (
            torch.isnan(rotation_params.grad).any() or torch.isinf(rotation_params.grad).any()
        ):
            continue
        if translation_params.grad is not None and (
            torch.isnan(translation_params.grad).any() or torch.isinf(translation_params.grad).any()
        ):
            continue
        optimizer.step()

        if torch.isnan(rotation_params).any() or torch.isinf(rotation_params).any():
            rotation_params.data.zero_()
        if torch.isnan(translation_params).any() or torch.isinf(translation_params).any():
            translation_params.data.zero_()

        if it == 0:
            rotation_reg = torch.norm(rotation_params) ** 2
            translation_reg = torch.norm(translation_params) ** 2
            regularization = 0.001 * (rotation_reg + translation_reg)

       
    if R is None or T is None:
        R = _rodrigues_from_axis_angle(rotation_params.detach())
        T = translation_params.detach()

    if apply_rt_to_all_points_for_final_render:
        final_transformed_xyz = torch.mm(original_xyz, R.detach().T) + T.detach()
    else:
        final_transformed_xyz = original_xyz.clone()
        final_transformed_xyz[instance_points_idx] = (instance_xyz_original @ R.detach().T) + T.detach()

    now_gaussians = GaussianModel(sh_degree=3)
    now_gaussians._xyz = final_transformed_xyz
    now_gaussians._features_dc = original_features_dc
    now_gaussians._features_rest = original_features_rest
    now_gaussians._scaling = original_scaling
    now_gaussians._rotation = original_rotation
    now_gaussians._opacity = original_opacity
    if hasattr(gaussians, "_seg_feature"):
        try:
            now_gaussians._seg_feature = gaussians._seg_feature.clone().detach()
        except Exception:
            pass

    with torch.no_grad():
        background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
        render_pkg = render(camera, now_gaussians, pipelineparams, background)
        if "render" in render_pkg:
            img = render_pkg["render"].detach().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            out_img = out_inst / f"optimized_{instance_name}_{camera_uid}.png"
            Image.fromarray(img).save(str(out_img))
            _log(f"optimized result saved to {out_img}")

    np.savez(
        str(out_inst / f"best_rt_{camera_uid}.npz"),
        R=R.detach().cpu().numpy(),
        T=T.detach().cpu().numpy(),
        camera_uid=int(camera_uid),
        instance_id=int(instance_id),
        best_loss=float(loss.detach().cpu().item()) if loss is not None else 0.0,
    )
    _log(f"best RT saved to: {out_inst / f'best_rt_{camera_uid}.npz'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Optimization only: use instance_info.json + point_cloud_labels.npy to optimize per-instance view pose. "
            "If --instance_id is not provided, all instances in instance_info.json will be optimized."
        )
    )
    parser.add_argument(
        "--source_path",
        type=str,
        required=True,
        help="Dataset path (contains COLMAP data and point_cloud.ply).",
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        required=True,
        help="Label directory (contains instance_info.json and point_cloud_labels.npy).",
    )
    parser.add_argument(
        "--instance_id",
        type=int,
        default=None,
        help="Instance id to optimize (int). If omitted, optimize all instances.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: <label_dir>/optimize).",
    )
    parser.add_argument(
        "--camera_uid",
        type=int,
        default=None,
        help="Optional: override best_view_uid with a specific camera uid.",
    )
    parser.add_argument(
        "--visibility_dir",
        type=str,
        default=None,
        help="Optional: visibility directory (default: inferred as <model_dir>/visibility from label_dir).",
    )
    parser.add_argument(
        "--top_k_views",
        type=int,
        default=0,
        help="Optional: select top-k views with most visible points using visibility; 0 uses best_view only.",
    )
    parser.add_argument("--max_iterations", type=int, default=10000)
    parser.add_argument("--lr_R", type=float, default=5e-5)
    parser.add_argument("--lr_T", type=float, default=1e-2)
    parser.add_argument("--depth_w", type=float, default=1e7)

    parser.add_argument(
        "--apply_global_rt",
        action="store_true",
        default=None,
        help="Apply RT to all points for final render (default enabled; matches train_semantic_optimize.py).",
    )
    parser.add_argument(
        "--no_apply_global_rt",
        action="store_true",
        default=None,
        help="Optional: disable global RT; apply RT to instance points only.",
    )

    args = parser.parse_args()

    if args.no_apply_global_rt:
        apply_global_rt = False
    elif args.apply_global_rt is True:
        apply_global_rt = True
    else:
        apply_global_rt = True

    output_dir = args.output_dir or os.path.join(args.label_dir, "optimize")
    os.makedirs(output_dir, exist_ok=True)

    camera_info = load_camera_info(args.source_path, label_dir=args.label_dir)
    _log(f"Train cameras: {len(camera_info['train_cameras'])}")
    if camera_info.get("gaussian_ply_path"):
        _log(f"Gaussian ply used for rendering: {camera_info['gaussian_ply_path']}")

    instances = load_instance_info(args.label_dir)
    point_labels = load_point_cloud_labels(args.label_dir)

    num_gaussians = int(camera_info["gaussians"].get_xyz.shape[0])
    if int(point_labels.shape[0]) != num_gaussians:
        raise ValueError(
            "point_cloud_labels.npy length does not match current gaussians point count: "
            f"labels={point_labels.shape[0]}, gaussians={num_gaussians}. "
            "This usually means the point_cloud.ply used for rendering is not the trained Gaussian point cloud under "
            "this iteration directory, which can cause instance point index misalignment and absurd results."
        )

    if args.instance_id is not None:
        target_instance_ids = [int(args.instance_id)]
    else:
        ids: List[int] = []
        for rec in instances:
            try:
                ids.append(int(rec.get("instance_id")))
            except Exception:
                continue
        target_instance_ids = sorted(set(ids))

    if not target_instance_ids:
        raise ValueError("No instance_id found to optimize (check the 'instances' list in instance_info.json)")

    _log(f"Optimizing {len(target_instance_ids)} instances")

    instance_iter = target_instance_ids
    if len(target_instance_ids) > 1:
        instance_iter = tqdm.tqdm(
            target_instance_ids,
            desc="Instances",
            unit="inst",
            dynamic_ncols=True,
            leave=True,
        )

    for inst_id in instance_iter:
        try:
            instance_record = get_instance_record(instances, inst_id)
        except KeyError:
            _log(f"Skip instance_id={inst_id}: not found in instance_info.json")
            continue

        instance_indices = get_instance_point_indices(point_labels, inst_id)
        if instance_indices.size == 0:
            _log(f"Skip instance_id={inst_id}: no corresponding points in point_cloud_labels.npy")
            continue

        _log(
            f"instance_id={inst_id}: points_in_labels={instance_indices.size}, "
            f"best_view_uid={instance_record.get('best_view_uid')}, best_view={instance_record.get('best_view')}"
        )

        camera_uids: List[int] = []
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

        if not camera_uids:
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

        for uid in camera_uids:
            optimize_instance_visibility(
                gaussians=camera_info["gaussians"],
                camera_info=camera_info,
                instance_record=instance_record,
                instance_point_indices=instance_indices,
                output_dir=output_dir,
                camera_uid_override=int(uid),
                max_iterations=args.max_iterations,
                learning_rate_R=args.lr_R,
                learning_rate_T=args.lr_T,
                base_depth_constraint_weight=args.depth_w,
                apply_rt_to_all_points_for_final_render=bool(apply_global_rt),
            )


if __name__ == "__main__":
    main()
