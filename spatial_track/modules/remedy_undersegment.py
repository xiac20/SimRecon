import numpy as np
from tqdm import tqdm

from scene import GaussianModel
from scene.cameras import Camera

from typing import List


def remedy_undersegment(gaussian: GaussianModel, viewcams: List[Camera], mask_assocation, threshold=0.8):
    """
    欠分割掩码修复：
    - 对于在初始化阶段被判定为“欠分割”的 (frame_id, mask_id)，再次尝试将其中一部分归并回正确的实例；
    - 直觉：若该掩码在其可见帧内，与某个3D实例的交点比例足够大（> threshold），则认为其应属于该实例；
      否则，保留其“欠分割”身份（仍置于 undersegment_mask_ids 中）。

    参数：
    - gaussian: GaussianModel，用于获取全局几何（本函数中未直接使用）
    - viewcams: List[Camera] 相机列表（用于遍历帧）
    - mask_assocation: 字典，包含初始化/聚类/后处理阶段产生的数据结构：
        * 'global_frame_mask_list': [(frame_id, mask_id), ...]
        * 'undersegment_mask_ids': 欠分割的全局掩码索引列表
        * 'gaussian_in_frame_matrix': (N_pts, N_frames) 点在帧的可见矩阵
        * 'mask_gaussian_pclds': {f"frameId_maskId": set(point_ids)} 掩码对应的点集合
        * 'total_point_ids_list': List[set/list] 每个实例的点集合（后处理后）
        * 'total_mask_list': List[List[(frame_id, mask_id, coverage)]] 每个实例当前聚合的掩码列表
    - threshold: 交点比例阈值（默认0.8）

    返回：
    - mask_assocation: 更新后的字典，主要更新：
        * 'undersegment_mask_ids'：剔除了可归并到实例的欠分割项
        * 'total_mask_list'：将部分欠分割帧掩码添加到对应实例的掩码列表中
    """
    # 取出欠分割项（frame_id, mask_id）
    undersegment_frame_masks = [mask_assocation['global_frame_mask_list'][frame_id] for frame_id in
                                mask_assocation['undersegment_mask_ids']]
    error_undersegment_frame_masks = {}
    remedy_undersegment_frame_masks = []

    # 每个实例的3D点集合（用于与帧掩码点集合求交）
    instance_seg3D_labels = [set(point_ids) for point_ids in mask_assocation["total_point_ids_list"]]
    # 每帧可见的点集合，便于限制对比到本帧范围
    frames_gaussian = []
    for frame_id in range(len(viewcams)):
        frames_gaussian.append(set(np.where(mask_assocation['gaussian_in_frame_matrix'][:, frame_id])[0]))

    iterator = tqdm(undersegment_frame_masks, total=len(undersegment_frame_masks),
                    desc="Remedy Error-Classified Undersegment")

    for frame_mask in iterator:
        frame_id, mask_id = frame_mask
        frame_mask_gaussian = mask_assocation['mask_gaussian_pclds'][f"{frame_id}_{mask_id}"]

        frame_gaussian = frames_gaussian[frame_mask[0]]
        ## 针对当前帧，计算该欠分割掩码与每个实例在本帧的交集规模
        instance_frame_gaussian = [seg3D_labels.intersection(frame_gaussian) for seg3D_labels in instance_seg3D_labels]
        instance_intersect_gaussian = np.array([len(frame_mask_gaussian.intersection(instance_gaussian))
                                                for instance_gaussian in instance_frame_gaussian])
        best_match_instance_idx = np.argsort(instance_intersect_gaussian)[::-1]
        best_match_intersect = instance_intersect_gaussian[best_match_instance_idx[0]]
        # 若与最佳实例的交点比例足够大（相对当前掩码点数），则认为误判，归并该掩码到该实例
        if best_match_intersect / len(frame_mask_gaussian) > threshold:
            error_undersegment_frame_masks[frame_mask] = best_match_instance_idx[0]
        else:
            # 否则仍保持为欠分割，记录其全局索引
            remedy_undersegment_frame_masks.append(mask_assocation['global_frame_mask_list'].index(frame_mask))

    # 更新欠分割列表（去除已被归并的）
    mask_assocation['undersegment_mask_ids'] = remedy_undersegment_frame_masks
    total_mask_list = mask_assocation['total_mask_list']

    # 将被判定可归并的欠分割掩码，添加到对应实例的掩码列表中
    for frame_mask in error_undersegment_frame_masks:
        total_mask_list[error_undersegment_frame_masks[frame_mask]].append(frame_mask)

    mask_assocation['total_mask_list'] = total_mask_list

    return mask_assocation
