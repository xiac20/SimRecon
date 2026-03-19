import os.path
from argparse import ArgumentParser
from random import randint
import json
import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple
from arguments import ModelParams, PipelineParams, OptimizationParams
from spatial_track.spatialtrack import GausCluster
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.contrastive_utils import *
from utils.general_mesh_utils import *
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import open3d as o3d
from vis_utils.color_utils import generate_semantic_colors

# ===================== ScanNet 类别 + CLIP 分类 =====================
SCANNET_LABELS_PATH = "/dataxl/xc/datasets/scannet_online/scannet-mv1/meta_data/scannetv2-labels.combined.tsv"

def load_scannet_categories(tsv_path: str = SCANNET_LABELS_PATH) -> List[str]:
    """从 ScanNet 标签文件加载去重后的类别列表。"""
    categories = set()
    if not os.path.exists(tsv_path):
        print(f"⚠️ ScanNet labels not found: {tsv_path}, using default categories")
        return ["wall", "floor", "chair", "table", "door", "window", "cabinet", "bed", 
                "sofa", "toilet", "desk", "curtain", "refrigerator", "shelf", "sink",
                "lamp", "pillow", "picture", "monitor", "plant", "book", "trash can",
                "box", "whiteboard", "backpack", "towel", "object"]
    with open(tsv_path, "r") as f:
        for i, line in enumerate(f):
            if i == 0:  # 跳过 header
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                cat = parts[2].strip().lower()
                if cat and cat != "object":  # 过滤掉太泛化的 "object"
                    categories.add(cat)
    categories.add("object")  # 兜底类别
    return sorted(list(categories))


class CLIPClassifier:
    """用 CLIP 对 instance 渲染图做零样本分类。"""
    def __init__(self, categories: List[str], device: str = "cuda"):
        self.device = device
        self.categories = categories
        self.model = None
        self.preprocess = None
        self.text_features = None
        self._initialized = False

    def _lazy_init(self):
        """延迟加载 CLIP，避免没用到时占用显存。"""
        if self._initialized:
            return
        try:
            import clip
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            # 预计算类别文本特征
            text_prompts = [f"a photo of a {cat}" for cat in self.categories]
            text_tokens = clip.tokenize(text_prompts).to(self.device)
            with torch.no_grad():
                self.text_features = self.model.encode_text(text_tokens)
                self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
            self._initialized = True
            print(f"✅ CLIP loaded with {len(self.categories)} categories")
        except ImportError:
            print("⚠️ CLIP not installed, run: pip install git+https://github.com/openai/CLIP.git")
            self._initialized = False

    @torch.no_grad()
    def classify(self, image: Image.Image, top_k: int = 1) -> List[Tuple[str, float]]:
        """对单张图片进行分类，返回 [(类别, 置信度), ...]"""
        self._lazy_init()
        if not self._initialized or self.model is None:
            return [("unknown", 0.0)]
        
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(img_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        similarity = (image_features @ self.text_features.T).squeeze(0)
        probs = similarity.softmax(dim=-1)
        
        top_indices = probs.topk(top_k).indices.cpu().tolist()
        results = [(self.categories[i], float(probs[i].cpu())) for i in top_indices]
        return results


class SegSplatting:
    """
    语义高斯训练主流程类（执行顺序）：
    1) 初始化高斯模型并加载点云；
    2) 构建稳健语义先验（2D/3D 掩码聚类 + 可见性）；
    3) 进行语义特征对比学习训练；
    4) 导出实例信息与带标签点云结果。
    """
    def __init__(self, modelparams: ModelParams, optimparams: OptimizationParams, pipelineparams: PipelineParams):
        self.modelparams = modelparams
        self.data_dir = modelparams.source_path
        self.optimparams = optimparams
        self.pipelineparams = pipelineparams

        # 初始化高斯模型并加载已有点云（不是从零开始重建）
        self.gaussians = GaussianModel(sh_degree=3)
        self.gaussians.pipelineparams = pipelineparams
        self.gaussians.set_segfeat_params(modelparams)
        self.gaussians.load_ply(os.path.join(self.data_dir, 'point_cloud.ply'))

        # 统一输出目录：output/<scene_parent>/<scene_name>/<model_path>
        self.model_path = os.path.join("output", self.modelparams.source_path.split("/")[-2],
                                       self.modelparams.source_path.split("/")[-1],
                                       self.modelparams.model_path)

    @staticmethod
    def _sorted_viewpoints(viewpoints: Sequence[Any]) -> List[Any]:
        """Return a deterministic camera order across machines/runs."""
        return sorted(viewpoints, key=lambda c: (getattr(c, "uid", 0), getattr(c, "image_name", "")))

    @staticmethod
    def _canonical_anchor_from_mask_list(mask_list: Any) -> Tuple[int, int]:
        """Pick a deterministic (frame_id, mask_id) anchor from a cluster's mask list."""
        if not mask_list:
            return (1 << 30, 1 << 30)
        anchors: List[Tuple[int, int]] = []
        for item in mask_list:
            try:
                frame_id = int(item[0])
                mask_id = int(item[1])
            except Exception:
                continue
            anchors.append((frame_id, mask_id))
        return min(anchors) if anchors else (1 << 30, 1 << 30)

    @staticmethod
    def _instance_signature_from_points(instance_mask_col: np.ndarray, max_points: int = 2048) -> str:
        """Compute a stable signature string from a boolean mask of points."""
        point_indices = np.flatnonzero(instance_mask_col)
        if point_indices.size == 0:
            return ""
        point_indices = point_indices[:max_points].astype(np.int32, copy=False)
        return hashlib.sha1(point_indices.tobytes()).hexdigest()

    def _stabilize_instance_order(self, priors: Dict[str, Any]) -> Dict[str, Any]:
        """Reorder instance clusters deterministically.

        This makes instance indices stable across runs/machines given the same dataset.
        Ordering is anchored by the smallest (frame_id, mask_id) pair in each cluster,
        with a point-index hash as a tie-breaker.
        """
        if "mask_3d_labels" not in priors or "mask_2d_clusters" not in priors:
            return priors

        mask_3d_labels: np.ndarray = priors["mask_3d_labels"]  # (N_points, K)
        mask_2d_clusters: List[Any] = list(priors["mask_2d_clusters"])
        if mask_3d_labels.ndim != 2:
            return priors

        instance_num = mask_3d_labels.shape[1]
        if len(mask_2d_clusters) != instance_num:
            return priors

        sort_keys: List[Tuple[Tuple[int, int], str, int]] = []
        for idx in range(instance_num):
            anchor = self._canonical_anchor_from_mask_list(mask_2d_clusters[idx])
            sig = self._instance_signature_from_points(mask_3d_labels[:, idx])
            sort_keys.append((anchor, sig, idx))

        sort_keys.sort(key=lambda x: (x[0][0], x[0][1], x[1], x[2]))
        order = [x[2] for x in sort_keys]

        priors = dict(priors)
        priors["mask_3d_labels"] = mask_3d_labels[:, order]
        priors["mask_2d_clusters"] = [mask_2d_clusters[i] for i in order]
        return priors

    @staticmethod
    def _deterministic_semantic_colors(n: int) -> np.ndarray:
        """Deterministic replacement for generate_semantic_colors() (which is random)."""
        if n <= 0:
            return np.zeros((0, 3), dtype=np.float32)
        rng = np.random.default_rng(0)
        hs = rng.uniform(0.0, 1.0, size=(n, 1))
        ss = rng.uniform(0.60, 0.61, size=(n, 1))
        vs = rng.uniform(0.84, 0.95, size=(n, 1))
        hsv = np.concatenate([hs, ss, vs], axis=-1)
        import cv2

        rgb = cv2.cvtColor((hsv * 255).astype(np.uint8)[None, ...], cv2.COLOR_HSV2RGB)[0]
        return (rgb / 255.0).astype(np.float32)

    def export_instance_info_json(self, output_dir: Optional[str] = None, use_clip: bool = True) -> str:
        """Generate instance_info.json with best_view info and CLIP category labels.

        The output is deterministic for a fixed dataset because:
        - cameras are sorted by uid/image_name
        - instances are stabilized by mask_id anchor ordering
        - tie-breakers pick the smallest uid
        
        如果 use_clip=True，会对每个 instance 在 best_view 下渲染并用 CLIP 做零样本分类，
        将类别标签 category_label 和置信度 category_confidence 写入 JSON。
        """
        if not hasattr(self, "scene") or not hasattr(self, "Seg3D_masks") or not hasattr(self, "Seg2D_masks"):
            raise RuntimeError("SegSplatting not initialized: call RobustSemanticPriors() first")

        visibility_dir = os.path.join(self.model_path, "visibility")
        if not os.path.isdir(visibility_dir):
            raise FileNotFoundError(f"visibility dir not found: {visibility_dir}")

        cameras = self._sorted_viewpoints(self.scene.getTrainCameras())
        uid_to_camera = {cam.uid: cam for cam in cameras}

        mask_3d = self.Seg3D_masks
        if isinstance(mask_3d, torch.Tensor):
            mask_3d = mask_3d.detach().cpu().numpy()
        mask_3d = np.asarray(mask_3d).astype(bool, copy=False)

        n_points, n_instances = mask_3d.shape
        has_instance = mask_3d.any(axis=1)
        point_instance = np.full((n_points,), -1, dtype=np.int32)
        if has_instance.any():
            point_instance[has_instance] = np.argmax(mask_3d[has_instance], axis=1).astype(np.int32)

        best_visible = np.full((n_instances,), -1, dtype=np.int64)
        best_uid = np.full((n_instances,), np.iinfo(np.int32).max, dtype=np.int32)
        best_view_file: List[Optional[str]] = [None] * n_instances
        best_image_name: List[Optional[str]] = [None] * n_instances

        # 遍历每个相机，统计每个 instance 在该视角下的可见点数量，选“最佳视角”
        for cam in cameras:
            vis_file = f"camera_{cam.uid:05d}_visibility.npy"
            vis_path = os.path.join(visibility_dir, vis_file)
            if not os.path.isfile(vis_path):
                continue
            vis = np.load(vis_path, allow_pickle=False)
            if vis.dtype != bool:
                vis = vis.astype(bool)

            visible_labels = point_instance[vis]
            visible_labels = visible_labels[visible_labels >= 0]
            if visible_labels.size == 0:
                counts = np.zeros((n_instances,), dtype=np.int64)
            else:
                counts = np.bincount(visible_labels, minlength=n_instances).astype(np.int64, copy=False)

            better = counts > best_visible
            if np.any(better):
                best_visible[better] = counts[better]
                best_uid[better] = int(cam.uid)
                for idx in np.flatnonzero(better).tolist():
                    best_view_file[idx] = vis_file
                    best_image_name[idx] = getattr(cam, "image_name", None)

        # ========== CLIP 分类器初始化 ==========
        clip_classifier = None
        if use_clip:
            categories = load_scannet_categories()
            clip_classifier = CLIPClassifier(categories, device="cuda")

        # 准备渲染
        bg_color = [1, 1, 1] if self.modelparams.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 组装稳定顺序的 instance 列表（用于下游优化/可视化）
        instances: List[Dict[str, Any]] = []
        print("🔍 Classifying instances with CLIP..." if use_clip else "📦 Building instance info...")
        
        for instance_id in tqdm(range(n_instances), desc="Processing instances"):
            anchor_frame_id, anchor_mask_id = self._canonical_anchor_from_mask_list(self.Seg2D_masks[instance_id])
            num_points_i = int(mask_3d[:, instance_id].sum())
            mask_list = []
            for item in self.Seg2D_masks[instance_id]:
                try:
                    frame_id = int(item[0])
                    mask_id = int(item[1])
                except Exception:
                    continue
                mask_list.append((frame_id, mask_id))
            mask_list.sort()

            # 默认类别
            category_label = "unknown"
            category_confidence = 0.0

            # ========== CLIP 分类：渲染 instance 在 best_view 下的 masked 图像 ==========
            if use_clip and clip_classifier is not None and best_uid[instance_id] < np.iinfo(np.int32).max:
                cam_uid = int(best_uid[instance_id])
                if cam_uid in uid_to_camera:
                    cam = uid_to_camera[cam_uid]
                    try:
                        # 渲染该视角下的全图
                        render_pkg = render(cam, self.gaussians, self.pipelineparams, background)
                        rendered_image = render_pkg["render"]  # [3, H, W]
                        
                        # 获取该 instance 的 2D mask（从 sorted_segmap）
                        if hasattr(cam, "sorted_segmap") and cam.sorted_segmap is not None:
                            segmap = cam.sorted_segmap.squeeze()
                            if isinstance(segmap, torch.Tensor):
                                segmap = segmap.cpu().numpy()
                            instance_mask = (segmap == (instance_id + 1))  # sorted_segmap 是 1-indexed
                        else:
                            # fallback: 用全图
                            instance_mask = np.ones((rendered_image.shape[1], rendered_image.shape[2]), dtype=bool)
                        
                        # 对渲染图应用 mask，裁剪出 instance 区域
                        rendered_np = rendered_image.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
                        rendered_np = (rendered_np * 255).clip(0, 255).astype(np.uint8)
                        
                        # 找到 mask 的 bounding box
                        ys, xs = np.where(instance_mask)
                        if len(ys) > 0 and len(xs) > 0:
                            y_min, y_max = ys.min(), ys.max()
                            x_min, x_max = xs.min(), xs.max()
                            # 稍微扩展边界
                            pad = 10
                            y_min = max(0, y_min - pad)
                            y_max = min(rendered_np.shape[0], y_max + pad)
                            x_min = max(0, x_min - pad)
                            x_max = min(rendered_np.shape[1], x_max + pad)
                            
                            # 裁剪
                            cropped = rendered_np[y_min:y_max, x_min:x_max]
                            if cropped.size > 0:
                                pil_image = Image.fromarray(cropped)
                                results = clip_classifier.classify(pil_image, top_k=1)
                                if results:
                                    category_label, category_confidence = results[0]
                    except Exception as e:
                        print(f"⚠️ CLIP classification failed for instance {instance_id}: {e}")

            instances.append(
                {
                    "instance_id": int(instance_id),
                    "num_points": int(num_points_i),
                    "best_view": best_view_file[instance_id],
                    "best_view_uid": None if best_view_file[instance_id] is None else int(best_uid[instance_id]),
                    "best_view_image_name": best_image_name[instance_id],
                    "max_visible": int(best_visible[instance_id]) if best_visible[instance_id] >= 0 else 0,
                    "category_label": category_label,
                    "category_confidence": round(category_confidence, 4),
                }
            )

        if output_dir is None:
            output_dir = self.data_dir
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "instance_info.json")
        with open(out_path, "w") as f:
            json.dump({"instances": instances}, f, indent=2, sort_keys=True)
        print(f"✅ instance_info.json saved to: {out_path}")
        return out_path

    @torch.no_grad()
    def RobustSemanticPriors(self):
        """构建稳健语义先验：聚类 -> 排序/过滤 -> 3D 标签 -> 可见性。"""
        print("\033[91mRunning Mask Clustering with Spatial Gaussian Tracker... \033[0m")

        if os.path.exists(self.modelparams.preload_robust_semantic):
            segment_save_dir = self.modelparams.preload_robust_semantic
        else:
            segment_save_dir = os.path.join(self.model_path, f"semantic_association")
            os.makedirs(segment_save_dir, exist_ok=True)

        scene = Scene(self.modelparams, self.gaussians, loaded_gaussian=True)
        viewpoint_stack = self._sorted_viewpoints(scene.getTrainCameras().copy())
        self.gausclustering = GausCluster(self.gaussians, viewpoint_stack)

        # 第一次运行时执行聚类；如果已有 output_dict.npy 则直接复用缓存
        if not os.path.exists(os.path.join(segment_save_dir, "output_dict.npy")):
            sam_dir = os.path.join(self.data_dir, "sam/mask_filtered")
            if os.path.exists(sam_dir):
                os.system("rm -rf {}".format(os.path.join(self.data_dir, "sam/mask_*")))
            self.gausclustering.maskclustering(segment_save_dir)  # TODO: cluster init in SFM ?

        self.robust_semantic_priors = np.load(
            os.path.join(segment_save_dir, "output_dict.npy"), allow_pickle=True
        ).item()

        # 固定 instance 顺序，避免跨机器/多次运行编号漂移
        self.robust_semantic_priors = self._stabilize_instance_order(self.robust_semantic_priors)

        self.Seg3D_masks = self.robust_semantic_priors["mask_3d_labels"]
        # 每个 3D 点对应一个 instance id（argmax）
        self.Seg3D_labels = torch.argmax(torch.tensor(self.Seg3D_masks, dtype=torch.int16), dim=1).cuda()

        self.Seg2D_masks = self.robust_semantic_priors["mask_2d_clusters"]
        # 根据稳定后的聚类结果重排 2D mask 文件（保证索引一致）
        if not os.path.exists(os.path.join(self.data_dir, "sam/mask_sorted")):
            self.gausclustering.rearrange_mask(os.path.join(self.data_dir, "sam/mask"), self.Seg2D_masks)

        self.undersegment_masks = self.robust_semantic_priors["underseg_mask_ids"]
        # 过滤过分割掩码，减少噪声 supervision
        if not os.path.exists(os.path.join(self.data_dir, "sam/mask_filtered")):
            self.gausclustering.filter_undersegment_mask(os.path.join(self.data_dir, "sam/mask"),
                                                         self.undersegment_masks)

        self.scene = Scene(self.modelparams, self.gaussians, loaded_gaussian=True)

        # 将 3D instance 先验写入高斯特征，用于后续 2D/3D 对比学习
        self.gaussians.set_3d_feat(self.Seg3D_masks, gram_feat=self.optimparams.gram_feat_3d)

        # 预计算每个相机可见点，供后续 best-view 与导出使用
        self.find_visible_points()
        # instance_info.json will be exported in export_segment_results_final()

    @torch.no_grad()
    def find_visible_points(self):
        """Compute per-camera visibility masks and save as .npy."""
        print("\033[92mCalculating visible points using vectorized render-based method... \033[0m")
        
        viewpoints = self._sorted_viewpoints(self.scene.getTrainCameras())
        
        total_points = self.gaussians.get_xyz.shape[0]

        visibility_save_dir = os.path.join(self.model_path, "visibility")
        os.makedirs(visibility_save_dir, exist_ok=True)

        bg_color = [1, 1, 1] if self.modelparams.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 按相机逐帧计算“哪些高斯点在该视角前景可见”
        for camera_idx, camera in enumerate(tqdm(viewpoints, desc="Processing cameras")):
            try:
                point_cloud = self.gaussians.get_xyz.detach().cuda()
                
                if isinstance(camera.world_view_transform, np.ndarray):
                    world_view_transform = torch.from_numpy(camera.world_view_transform).float().cuda()
                elif isinstance(camera.world_view_transform, torch.Tensor):
                    world_view_transform = camera.world_view_transform.clone().detach().float().cuda()
                
                points_h = torch.cat([point_cloud, torch.ones_like(point_cloud[:, :1])], dim=-1).T
                points_cam = (world_view_transform @ points_h).T
                depths = points_cam[:, 2]
                
                # 先按深度排序，后续每个像素只保留最前面的高斯点
                depth_sort_indices = torch.argsort(depths)
                sorted_to_original = depth_sort_indices.clone()
                original_to_sorted = torch.empty_like(depth_sort_indices)
                original_to_sorted[depth_sort_indices] = torch.arange(len(depth_sort_indices), device='cuda')
                
                render_pkg = render(camera, self.gaussians, self.pipelineparams, background)
                
                if "gau_related_pixels" not in render_pkg:
                    visibility_mask = torch.zeros(total_points, dtype=torch.bool, device='cuda')
                else:
                    gau_related_pixels = render_pkg["gau_related_pixels"]
                    
                    if gau_related_pixels.numel() == 0:
                        visibility_mask = torch.zeros(total_points, dtype=torch.bool, device='cuda')
                    else:
                        gauss_indices = gau_related_pixels[:, 0].long()
                        pixel_indices = gau_related_pixels[:, 1].long()
                        
                        sorted_gauss_indices = original_to_sorted[gauss_indices]
                        
                        sorted_gau_related_pixels = torch.stack([sorted_gauss_indices, pixel_indices], dim=1)
                        
                        sort_indices = torch.argsort(sorted_gau_related_pixels[:, 0])
                        sorted_gau_related_pixels = sorted_gau_related_pixels[sort_indices]
                        
                        pixel_indices_sorted = sorted_gau_related_pixels[:, 1]
                        gauss_indices_sorted = sorted_gau_related_pixels[:, 0]
                        
                        max_pixel_idx = pixel_indices_sorted.max().item() + 1
                        first_occurrence_gauss = torch.full((max_pixel_idx,), total_points, dtype=torch.long, device='cuda')
                        
                        # 对每个像素取最小（最前）sorted-gauss-index
                        first_occurrence_gauss.scatter_reduce_(0, pixel_indices_sorted, gauss_indices_sorted, 'amin', include_self=True)
                        
                        valid_pixels_mask = first_occurrence_gauss < total_points
                        
                        if valid_pixels_mask.sum() > 0:
                            visible_sorted_indices = first_occurrence_gauss[valid_pixels_mask]
                            
                            visible_original_indices = sorted_to_original[visible_sorted_indices]
                            
                            visibility_mask = torch.zeros(total_points, dtype=torch.bool, device='cuda')
                            visibility_mask[visible_original_indices] = True
                        else:
                            visibility_mask = torch.zeros(total_points, dtype=torch.bool, device='cuda')

                # 将该相机的可见性布尔向量落盘
                visibility_filename = f"camera_{camera.uid:05d}_visibility.npy"
                visibility_filepath = os.path.join(visibility_save_dir, visibility_filename)
                
                visibility_array = visibility_mask.cpu().numpy().astype(bool)
                np.save(visibility_filepath, visibility_array)
                
                visible_count = visibility_array.sum()
                
            except Exception as e:
                print(f"Error processing camera {camera_idx} (uid: {camera.uid}): {e}")
                import traceback
                traceback.print_exc()
                
                empty_visibility = np.zeros(total_points, dtype=bool)
                visibility_filename = f"camera_{camera.uid:05d}_visibility.npy"
                visibility_filepath = os.path.join(visibility_save_dir, visibility_filename)
                np.save(visibility_filepath, empty_visibility)

            torch.cuda.empty_cache()

        total_cameras = len(viewpoints)
        
        camera_order_info = [(idx, cam.uid, cam.image_name) for idx, cam in enumerate(viewpoints)]
        import json
        with open(os.path.join(visibility_save_dir, "camera_order.json"), 'w') as f:
            json.dump(camera_order_info, f, indent=2)
        
        # 保存可见性统计摘要，便于调试和复现实验
        summary_info = {
            'total_cameras': int(total_cameras),
            'total_points': int(total_points),
            'visibility_dir': str(visibility_save_dir),
            'camera_files': [f"camera_{cam.uid:05d}_visibility.npy" for cam in viewpoints],
            'method': 'sequential_depth_sorted_visibility'
        }
        
        summary_filepath = os.path.join(visibility_save_dir, "visibility_summary.json")
        with open(summary_filepath, 'w') as f:
            json.dump(summary_info, f, indent=2)

    
        # Train segment feature
    def train_segfeat(self):
        """语义特征训练主循环：单视角对比 + 多视角对比 + 3D 对比。"""
        print("\n\033[91mRunning Spatial Contrastive Learning... \033[0m")

        if os.path.exists(
                os.path.join(self.model_path, "point_cloud/iteration_{}".format(self.optimparams.iterations))):
            return

        self.gaussians.training_setup(self.optimparams)

        bg_color = [1, 1, 1] if self.modelparams.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        first_iter = 0
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        viewpoint_stack = None
        progress_bar = tqdm(range(first_iter, self.optimparams.iterations), desc="Training progress")
        first_iter += 1

        # 迭代训练：每轮随机取一个视角进行渲染与对比监督
        for iteration in range(first_iter, self.optimparams.iterations + 1):
            iter_start.record()

            if not viewpoint_stack:
                viewpoint_stack = self.scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

            render_pkg = render(viewpoint_cam, self.gaussians, self.pipelineparams, background)

            image, seg_feature, viewspace_point_tensor, visibility_filter, radii = \
                render_pkg["render"], render_pkg["seg_feature"], render_pkg["viewspace_points"], \
                    render_pkg["visibility_filter"], render_pkg["radii"]

            # 1) 单视角对比损失（segmap + 可选 sorted_segmap）
            singleview_contra_loss = 0
            mask_type_cnts = 0
            if self.gaussians.class_feat is not None:
                segmap_lists = [viewpoint_cam.segmap.squeeze().cuda(), viewpoint_cam.sorted_segmap.squeeze().cuda()]
            else:
                segmap_lists = [viewpoint_cam.segmap.squeeze().cuda()]
            for gt_segmap in segmap_lists:
                batchsize = self.optimparams.sample_batchsize

                valid_labels_mask = gt_segmap > 0
                if mask_type_cnts == 0 and self.optimparams.consider_negative_labels:
                    valid_labels_mask = torch.ones_like(gt_segmap, dtype=torch.bool).cuda()

                if valid_labels_mask.sum() > 0:
                    valid_seg_feature = seg_feature[:, valid_labels_mask]
                    valid_seg_labels = gt_segmap[valid_labels_mask]
                    sampled_idx = torch.randint(0, len(valid_seg_labels), size=(batchsize,), device="cuda")

                    sampled_segfeat = valid_seg_feature[:, sampled_idx].T
                    sampled_labels = valid_seg_labels[sampled_idx]

                    single_view_weight = 1 if mask_type_cnts == 1 else 0.5

                    singleview_contra_loss = singleview_contra_loss + contrastive_loss(
                        sampled_segfeat, sampled_labels,
                        predef_u_list=self.gaussians.class_feat if mask_type_cnts == 1 else None,
                        consider_negative=(mask_type_cnts == 0 and self.optimparams.consider_negative_labels),
                    ) * self.optimparams.lambda_singview_contras * single_view_weight
                else:
                    print("Invalid View: ", viewpoint_cam.image_name)
                mask_type_cnts += 1

            # 2) 多视角对比损失（每 10 轮触发一次）
            multiview_contra_loss = 0
            if self.optimparams.lambda_multiview_contras > 0 and iteration % 10 == 0:
                num_sample_views = self.optimparams.sample_mv_frames
                sampled_views = self.scene.getTrainCameras()
                sampled_view_id = np.random.randint(0, len(sampled_views) - num_sample_views)
                sampled_views = [sampled_views[view_idx] for view_idx in
                                 range(sampled_view_id, sampled_view_id + num_sample_views)]
                seg_feature_list = []
                seg_labels_list = []
                for sample_view in sampled_views:
                    render_pkg = render(sample_view, self.gaussians, self.pipelineparams, background)
                    seg_feature = render_pkg["seg_feature"]
                    seg_feature_list.append(seg_feature)
                    seg_labels_list.append(sample_view.sorted_segmap.cuda())
                seg_feature_list = torch.stack(seg_feature_list, dim=0)
                seg_labels_list = torch.stack(seg_labels_list, dim=0).squeeze()

                batchsize = self.optimparams.sample_batchsize
                valid_labels_mask = seg_labels_list > 0
                valid_seg_feature = seg_feature_list.permute(1, 0, 2, 3)[:, valid_labels_mask]
                valid_seg_labels = seg_labels_list[valid_labels_mask]
                sampled_idx = torch.randint(0, len(valid_seg_labels), size=(batchsize,), device="cuda")
                sampled_segfeat = valid_seg_feature[:, sampled_idx].T
                sampled_labels = valid_seg_labels[sampled_idx]
                multiview_contra_loss = contrastive_loss(sampled_segfeat,
                                                         sampled_labels,
                                                         predef_u_list=self.gaussians.class_feat
                                                         ) * self.optimparams.lambda_multiview_contras

            visibility_segfeat = self.gaussians.get_seg_feature[visibility_filter]
            visibility_labels_3d = self.Seg3D_labels[visibility_filter]

            # 3) 3D 对比损失（只在当前视角可见点上采样）
            contra_3d_loss = 0
            if self.optimparams.lambda_3D_contras > 0:
                batchsize_3d = self.optimparams.sample_batchsize

                valid_labels_mask = visibility_labels_3d > 0
                if valid_labels_mask.sum() > 0:
                    valid_seg_feature = visibility_segfeat[valid_labels_mask]
                    valid_seg_labels = visibility_labels_3d[valid_labels_mask]

                    sampled_idx = torch.randint(0, len(valid_seg_labels), size=(batchsize_3d,), device="cuda")
                    sampled_segfeat_3d = valid_seg_feature[sampled_idx]
                    sampled_labels_3d = valid_seg_labels[sampled_idx]

                    contra_3d_loss = contrastive_loss(sampled_segfeat_3d,
                                                      sampled_labels_3d,
                                                      predef_u_list=self.gaussians.class_feat
                                                      ) * self.optimparams.lambda_3D_contras
                else:
                    print("Invalid View: ", viewpoint_cam.image_name)

            # 总损失反传并更新高斯语义特征
            total_loss = singleview_contra_loss + \
                         multiview_contra_loss + \
                         contra_3d_loss

            total_loss.backward()
            iter_end.record()

            self.gaussians.optimizer.step()
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

            with torch.no_grad():
                if iteration % 10 == 0:
                    loss_dict = {
                        "SV_ContraLoss": f"{singleview_contra_loss:.{3}f}",
                        "MV_ContraLoss": f"{multiview_contra_loss:.{3}f}",
                        "3D_ContraLoss": f"{contra_3d_loss:.{3}f}"
                    }
                    progress_bar.set_postfix(loss_dict)
                    progress_bar.update(10)

                if iteration % 200 == 0:
                    viewpoint = self.scene.getTrainCameras()[0]
                    render_pkg = render(viewpoint, self.gaussians, self.pipelineparams, background)

                    _, seg_feature, _, _, _ = render_pkg["render"], render_pkg["seg_feature"], \
                        render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                    os.makedirs(self.scene.model_path, exist_ok=True)
                    Image.fromarray(feature_to_rgb(seg_feature)).save(f"{self.scene.model_path}/{iteration}_feat.png")

                # 固定间隔保存；最后一次会导出最终标签点云与 instance_info.json
                if iteration % 2500 == 0:
                    self.scene.save(iteration)
                    if iteration == self.optimparams.iterations:
                        self.export_segment_results_final(iteration)
                    else:
                        # self.export_segment_results(iteration)
                        continue

                if iteration == self.optimparams.iterations:
                    progress_bar.close()

    @torch.no_grad()
    def export_segment_results_final(self, iteration, score_threshold=0.9):
        """Export per-point labels, colored point cloud, and instance images."""
        save_dir = os.path.join(self.model_path, f"point_cloud/iteration_{iteration}")
        os.makedirs(save_dir, exist_ok=True)
        save_partial_dir = os.path.join(save_dir, "label_pointclouds")
        os.makedirs(save_partial_dir, exist_ok=True)
        # 新增：instance 图像保存目录
        save_images_dir = os.path.join(save_dir, "instance_images")
        os.makedirs(save_images_dir, exist_ok=True)

        # Export instance_info.json alongside point_cloud_labels.ply (requested output location).
        try:
            self.export_instance_info_json(output_dir=save_dir)
        except Exception as e:
            print(f"⚠️ Failed to export instance_info.json to {save_dir}: {e}")
        
        scene_pclds = self.gaussians.get_xyz.detach().cpu().numpy()
        global_feat = self.gaussians.get_seg_feature.cpu()
        total_points = len(scene_pclds)
        
        opacity_values = self.gaussians.get_opacity.detach().cpu().squeeze()  # [N]
        opacity_threshold = 0.2
        
        all_point_labels = torch.full((total_points,), -1, dtype=torch.long)
        all_assigned_points = torch.zeros(total_points, dtype=torch.bool)
        
        
        # 逐 instance 计算特征相似度，得到点级标签分配
        for sampled_3d_labels in range(self.Seg3D_masks.shape[1]):
            selected_pseudo_3d_feat = self.gaussians.get_seg_feature[self.Seg3D_masks[:, sampled_3d_labels]]
            selected_pseudo_3d_feat_mean = selected_pseudo_3d_feat.mean(0).cpu()
            feat_score = selected_pseudo_3d_feat_mean @ global_feat.T
            
            feature_mask = (feat_score >= score_threshold)
            opacity_mask = (opacity_values >= opacity_threshold)
            selected_points_mask = feature_mask & opacity_mask

            if selected_points_mask.sum() == 0:
                selected_points_mask = (self.Seg3D_labels == sampled_3d_labels).cpu()

            all_assigned_points |= selected_points_mask
            all_point_labels[selected_points_mask] = sampled_3d_labels
            
        high_conf_count = all_assigned_points.sum()
        
        unassigned_mask = ~all_assigned_points
        unassigned_count = unassigned_mask.sum()
           
        assigned_count = (all_point_labels >= 0).sum()
        
        final_labels = all_point_labels
        num_instances = int(self.Seg3D_masks.shape[1])

        instance_colors = self._deterministic_semantic_colors(num_instances)
        
        # 写出全量彩色点云 + 标签数组 + 每个实例单独点云
        full_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene_pclds))
        colors = np.zeros((total_points, 3))
        
        final_labels_np = final_labels.numpy().astype(np.int32, copy=False)
        for i, label in enumerate(final_labels_np):
            if label < 0 or label >= num_instances:
                colors[i] = [1.0, 1.0, 1.0]
            else:
                colors[i] = instance_colors[label]
                
        full_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        o3d.io.write_point_cloud(os.path.join(save_dir, "point_cloud_labels.ply"), full_pcd)
        
        np.save(os.path.join(save_dir, "point_cloud_labels.npy"), final_labels_np)
        
        save_partial_dir = os.path.join(save_dir, "label_pointclouds")
        os.makedirs(save_partial_dir, exist_ok=True)
        for new_label_id in range(num_instances):
            label_mask = final_labels == new_label_id
            if label_mask.sum() > 0:
                label_positions = scene_pclds[label_mask.numpy()]
                pcld = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(label_positions))
                pcld.paint_uniform_color(instance_colors[new_label_id])
                o3d.io.write_point_cloud(os.path.join(save_partial_dir, f"{new_label_id}.ply"), pcld)
        
        unassigned_label_mask = final_labels == -1
        if unassigned_label_mask.sum() > 0:
            unassigned_positions = scene_pclds[unassigned_label_mask.numpy()]
            unassigned_pcld = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unassigned_positions))
            unassigned_pcld.paint_uniform_color([1.0, 1.0, 1.0])
            o3d.io.write_point_cloud(os.path.join(save_partial_dir, "unassigned.ply"), unassigned_pcld)

        # ========== 保存每个 instance 的 best view 图像 ==========
        self._export_instance_images(save_dir, save_images_dir, num_instances)

    @torch.no_grad()
    def _export_instance_images(self, save_dir: str, save_images_dir: str, num_instances: int):
        """为每个 instance 保存 best view 原图和带 mask 蒙版的图像。"""
        print("🖼️ Exporting instance images...")
        
        # 读取 instance_info.json 获取 best_view 信息
        instance_info_path = os.path.join(save_dir, "instance_info.json")
        if not os.path.exists(instance_info_path):
            print("⚠️ instance_info.json not found, skipping instance image export")
            return
        
        with open(instance_info_path, "r") as f:
            instance_info = json.load(f)
        
        instances = instance_info.get("instances", [])
        if not instances:
            return
        
        # 构建 uid -> camera 映射
        cameras = self._sorted_viewpoints(self.scene.getTrainCameras())
        uid_to_camera = {cam.uid: cam for cam in cameras}
        
        # 准备渲染
        bg_color = [1, 1, 1] if self.modelparams.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # 缓存已渲染的相机图像，避免重复渲染
        rendered_cache = {}
        
        for inst in tqdm(instances, desc="Exporting instance images"):
            instance_id = inst["instance_id"]
            best_view_uid = inst.get("best_view_uid")
            category_label = inst.get("category_label", "unknown")
            
            if best_view_uid is None or best_view_uid not in uid_to_camera:
                continue
            
            cam = uid_to_camera[best_view_uid]
            
            # 渲染或从缓存获取
            if best_view_uid not in rendered_cache:
                render_pkg = render(cam, self.gaussians, self.pipelineparams, background)
                rendered_image = render_pkg["render"]  # [3, H, W]
                rendered_np = rendered_image.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
                rendered_np = (rendered_np * 255).clip(0, 255).astype(np.uint8)
                rendered_cache[best_view_uid] = rendered_np
            else:
                rendered_np = rendered_cache[best_view_uid]
            
            # 获取该 instance 的 2D mask
            if hasattr(cam, "sorted_segmap") and cam.sorted_segmap is not None:
                segmap = cam.sorted_segmap.squeeze()
                if isinstance(segmap, torch.Tensor):
                    segmap = segmap.cpu().numpy()
                instance_mask = (segmap == (instance_id + 1))  # sorted_segmap 是 1-indexed
            else:
                # fallback: 无 mask 则跳过
                continue
            
            # 找到 mask 的 bounding box
            ys, xs = np.where(instance_mask)
            if len(ys) == 0 or len(xs) == 0:
                continue
            
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()
            # 稍微扩展边界
            pad = 15
            y_min = max(0, y_min - pad)
            y_max = min(rendered_np.shape[0], y_max + pad)
            x_min = max(0, x_min - pad)
            x_max = min(rendered_np.shape[1], x_max + pad)
            
            # 1) 保存 best view 原图（裁剪后）
            cropped = rendered_np[y_min:y_max, x_min:x_max].copy()
            if cropped.size > 0:
                cropped_pil = Image.fromarray(cropped)
                cropped_pil.save(os.path.join(save_images_dir, f"{instance_id}_bestview.png"))
            
            # 2) 保存带 mask 蒙版的图像（instance 区域保留，其他区域变暗）
            # 创建全图的 masked 版本
            masked_full = rendered_np.copy()
            # 将非 instance 区域变暗（乘以 0.3）
            mask_3d = np.stack([instance_mask] * 3, axis=-1)
            masked_full = np.where(mask_3d, masked_full, (masked_full * 0.3).astype(np.uint8))
            # 裁剪
            masked_cropped = masked_full[y_min:y_max, x_min:x_max]
            if masked_cropped.size > 0:
                masked_pil = Image.fromarray(masked_cropped)
                masked_pil.save(os.path.join(save_images_dir, f"{instance_id}_masked.png"))
            
            # 3) 额外保存：只有 instance 区域的图像（其他区域透明，RGBA）
            cropped_mask = instance_mask[y_min:y_max, x_min:x_max]
            if cropped.size > 0:
                rgba = np.zeros((cropped.shape[0], cropped.shape[1], 4), dtype=np.uint8)
                rgba[:, :, :3] = cropped
                rgba[:, :, 3] = (cropped_mask * 255).astype(np.uint8)
                rgba_pil = Image.fromarray(rgba, mode="RGBA")
                rgba_pil.save(os.path.join(save_images_dir, f"{instance_id}_isolated.png"))
        
        # 清理缓存
        rendered_cache.clear()
        torch.cuda.empty_cache()
        print(f"✅ Instance images saved to: {save_images_dir}")
        

    @torch.no_grad()
    def export_segment_results(self, iteration, score_threshold=0.9, use_hdbscan=False, note=None):
        if note is None:
            save_dir = os.path.join(self.model_path, f"point_cloud/iteration_{iteration}")
        else:
            save_dir = os.path.join(self.model_path, f"point_cloud/{note}")
        os.makedirs(save_dir, exist_ok=True)
        save_partial_dir = os.path.join(save_dir, "label_pointclouds")
        os.makedirs(save_partial_dir, exist_ok=True)

        instance_colors = generate_semantic_colors(self.Seg3D_masks.shape[1])

        scene_pclds = self.gaussians.get_xyz.detach().cpu().numpy()
        pclds = None
        global_feat = self.gaussians.get_seg_feature.cpu()
        for sampled_3d_labels in range(self.Seg3D_masks.shape[1]):
            selected_pseudo_3d_feat = self.gaussians.get_seg_feature[self.Seg3D_masks[:, sampled_3d_labels]]
            selected_pseudo_3d_feat_mean = selected_pseudo_3d_feat.mean(0).cpu()
            feat_score = selected_pseudo_3d_feat_mean @ global_feat.T
            selected_points_mask = (feat_score >= score_threshold)
            if selected_points_mask.sum() == 0:
                selected_points_mask = (self.Seg3D_labels == sampled_3d_labels).cpu()

            pcld = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene_pclds[selected_points_mask.numpy()]))
            color = instance_colors[sampled_3d_labels]
            pcld.paint_uniform_color(color)
            pclds = pcld if pclds is None else pclds + pcld
            o3d.io.write_point_cloud(os.path.join(save_partial_dir, f"{sampled_3d_labels}.ply"), pcld)

        o3d.io.write_point_cloud(os.path.join(save_dir, "point_cloud_labels.ply"), pclds)



if __name__ == "__main__":
    # ---------------- 程序入口 ----------------
    # 1) 解析参数
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])

    # 2) 初始化训练器
    segsplat = SegSplatting(lp.extract(args), op.extract(args), pp.extract(args))
    segsplat.args = args
    # 3) 构建稳健语义先验（含可见性）
    segsplat.RobustSemanticPriors()
    # 4) 训练语义特征并在最后导出结果
    segsplat.train_segfeat()
    
    print("\nTraining complete.")
