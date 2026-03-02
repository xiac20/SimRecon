import os
from tqdm import tqdm
from PIL import Image

from types import SimpleNamespace
from scene import GaussianModel
from scene.cameras import Camera

from typing import List

from spatial_track.modules.init_tracker import *
from spatial_track.modules.iterative_cluster import iterative_clustering
from spatial_track.modules.post_process import post_process
from spatial_track.modules.remedy_undersegment import remedy_undersegment


### Refer from https://github.com/PKU-EPIC/MaskClustering with Gaussian-based Tracker ###
class GausCluster:
    """
    高斯点云语义聚类（基于多视角掩码的一致性关联与空间跟踪）。

    流程概述：
    1) construct_mask2gs_tracker: 基于2D分割掩码与多视角相机，将像素掩码与3D高斯点进行可见性与几何关联，构造初始mask->gaussians映射
    2) iterative_clustering: 迭代式聚类/合并/一致性检查，得到更稳定的跨视角掩码聚合
    3) post_process: 使用DBSCAN等方法过滤噪声点，得到干净的实例簇
    4) remedy_undersegment: 修复欠分割的掩码（将被误归类或切分的区域进行补救）
    5) export: 导出3D掩码标签（N_points x N_instances 的布尔矩阵）、欠分割掩码ID以及2D聚类结果
    """
    def __init__(self, gaussian: GaussianModel, viewcams: List[Camera], debug=True):
        # 构建init node
        self.gaussian = gaussian
        self.viewcams = viewcams

        # 聚类/过滤等相关阈值与超参数（经验值，可按数据集调参）
        clustering_args = {
            "mask_visible_threshold": 0.7,  # 若某mask在>70%像素不可见，则视为无效/弱关联
            "undersegment_filter_threshold": 0.3,  # 欠分割过滤阈值（越大越鲁棒但更稀疏）
            "contained_threshold": 0.8,  # 掩码包含关系阈值
            "view_consensus_threshold": 0.9,  # 多视角一致性阈值
            "point_filter_threshold": 0.5  # 点级过滤阈值（投影/可见性）
        }

        self.clustering_args = SimpleNamespace(**clustering_args)

        self.debug = debug

    def maskclustering(self, save_dir=None):
        """
        主入口：执行掩码-高斯的构建、迭代聚类、后处理与欠分割补救，并将结果导出到save_dir。
        输出文件：save_dir/output_dict.npy
        - mask_3d_labels: (N_points, N_instances) 的布尔矩阵
        - underseg_mask_ids: 欠分割掩码集合（按帧聚合的mask id）
        - mask_2d_clusters: 跨视角的2D聚类簇（记录帧与mask id列表）
        """
        ## Init the Mask's Gaussian Tracker
        init_mask_assocation = construct_mask2gs_tracker(self.gaussian, self.viewcams, self.clustering_args,
                                                         save_dir, self.debug)
        ## Cluster the Mask's Gaussian Tracker
        update_mask_assocation = iterative_clustering(init_mask_assocation, self.clustering_args)

        ## Use DBScan to Filter Noisy Points from maskclustering
        final_mask_assocation = post_process(self.gaussian, update_mask_assocation, self.clustering_args)

        ## Remedy error-classified undersegment masks
        remedy_mask_assocation = remedy_undersegment(self.gaussian, self.viewcams, final_mask_assocation)

        self.export(remedy_mask_assocation, save_dir=save_dir)

    def export(self, mask_assocation, save_dir):
        """
        将聚类与补救后的掩码关联结构导出为npy：output_dict.npy
        字段：
        - mask_3d_labels: (N_points, N_instances) 的布尔矩阵（每列为一个实例簇）
        - underseg_mask_ids: 欠分割的帧-掩码id集合
        - mask_2d_clusters: 每个实例簇对应的(帧id, 掩码id, …) 列表
        """
        # undersegment, mask3d, mv-consist masks
        os.makedirs(save_dir, exist_ok=True)

        total_point_num = len(self.gaussian.get_xyz)

        # 将每个实例的高斯点id列表转为长度为N_points的布尔向量
        mask_3d_labels = []
        for i, (point_ids, mask_list) in enumerate(zip(mask_assocation["total_point_ids_list"],
                                                       mask_assocation["total_mask_list"])):
            binary_mask = np.zeros(total_point_num, dtype=bool)
            binary_mask[list(point_ids)] = True
            mask_3d_labels.append(binary_mask)

        mask_3d_labels = np.stack(mask_3d_labels, axis=1)
        # Note: 可能不存在欠分割掩码（空集）
        if len(mask_assocation["undersegment_mask_ids"]) > 0:
            underseg_mask_ids = np.stack([list(mask_assocation['global_frame_mask_list'][id]) for id in
                                          mask_assocation["undersegment_mask_ids"]], axis=0)
        else:
            underseg_mask_ids = []

        output_dict = {
            "mask_3d_labels": mask_3d_labels,
            "underseg_mask_ids": underseg_mask_ids,
            "mask_2d_clusters": mask_assocation["total_mask_list"]
        }

        np.save(os.path.join(save_dir, 'output_dict.npy'), output_dict, allow_pickle=True)

    def rearrange_mask(self, mask_folder, mask_assocation_info):
        """
        将原始2D掩码按聚类结果重排并保存到 mask_sorted/ 下，便于训练/可视化。
        - mask_assocation_info: list[list[(frame_id, mask_id, ...), ...]]
        """
        save_dir = os.path.join(os.path.dirname(mask_folder), "mask_sorted")
        os.makedirs(save_dir, exist_ok=True)

        masks_origin = []
        for viewcam in self.viewcams:
            mask_file = os.path.join(mask_folder, viewcam.image_name + ".png")
            masks_origin.append(np.array(Image.open(mask_file)))

        masks_origin = np.stack(masks_origin)
        masks_new = np.zeros_like(masks_origin, dtype=np.int16)

        # 按聚类ID重新赋值连通区域label（cluster从1起始）
        for cluster_id, cluster_info in enumerate(mask_assocation_info):
            cluster_id = cluster_id + 1  # indice from 1
            for frame_mask_id in cluster_info:
                frame_id, mask_id = frame_mask_id[:2]
                masks_new[frame_id][masks_origin[frame_id] == mask_id] = cluster_id

        for mask_id in range(len(masks_origin)):
            save_path = os.path.join(save_dir, self.viewcams[mask_id].image_name + ".png")
            Image.fromarray(masks_new[mask_id]).save(save_path)

    def filter_undersegment_mask(self, mask_folder, undersegment_masks):
        """
        过滤欠分割掩码：将被判定为欠分割的区域置零，并单独保存到 mask_undersegment/ 目录。
        - undersegment_masks: list[(frame_id, mask_id, ...)]
        """
        save_dir = os.path.join(os.path.dirname(mask_folder), "mask_filtered")
        os.makedirs(save_dir, exist_ok=True)

        save_undersegment_dir = os.path.join(os.path.dirname(mask_folder), "mask_undersegment")
        os.makedirs(save_undersegment_dir, exist_ok=True)

        masks_origin = []
        for viewcam in self.viewcams:
            mask_file = os.path.join(mask_folder, viewcam.image_name + ".png")
            masks_origin.append(np.array(Image.open(mask_file)))

        masks_origin = np.stack(masks_origin)
        masks_new = masks_origin.copy()
        masks_undersegment = np.zeros_like(masks_origin, dtype=np.int16)

        # 将欠分割区域清零，同时拷贝到undersegment图中
        for underseg_frame_mask_ids in undersegment_masks:
            frame_id, mask_id = underseg_frame_mask_ids[:2]
            masks_new[frame_id][masks_origin[frame_id] == mask_id] = 0
            masks_undersegment[frame_id][masks_origin[frame_id] == mask_id] = mask_id

        # 分别保存过滤后的mask与欠分割mask
        for mask_id in range(len(masks_origin)):
            save_path = os.path.join(save_dir, self.viewcams[mask_id].image_name + ".png")
            Image.fromarray(masks_new[mask_id]).save(save_path)

            save_underseg_path = os.path.join(save_undersegment_dir, self.viewcams[mask_id].image_name + ".png")
            Image.fromarray(masks_undersegment[mask_id]).save(save_underseg_path)
