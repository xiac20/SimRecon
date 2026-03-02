import os

import torch
import numpy as np
from typing import List
from tqdm import tqdm
from scipy.sparse import csr_matrix

from gaussian_renderer import render
from scene import GaussianModel
from scene.cameras import Camera

from spatial_track.modules.node import Node


def get_segmap_gaussians(gaussian: GaussianModel, view: Camera):
    '''
    基于光栅化渲染的“高斯-像素”关联，获取当前视角下每个2D掩码对应的高斯点集合。

    说明：
    - 直接使用GS渲染深度存在多视角不一致风险，这里采用渲染时返回的 gau_related_pixels 进行鲁棒追踪。

    参数：
    - gaussian: GaussianModel 实例
    - view: Camera 实例，包含本帧的 segmap（整型掩码，0表示背景）

    返回：
    - mask_info: {mask_id: set(gaussian_ids)}，每个2D掩码对应的3D高斯点ID集合
    - frame_gaussian_ids: list，本帧可见的全部高斯点ID（去重）
    '''
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    gau_related_pixels = render(view, gaussian, gaussian.pipelineparams, background)['gau_related_pixels']

    gaus_ids = gau_related_pixels[:, 0]  # gs_idx
    pixel_ids = gau_related_pixels[:, 1]  # pixel_idx

    # 从相机的segmap中枚举所有mask id
    mask_image = view.segmap.cuda().reshape(-1)
    ids = torch.unique(mask_image).cpu().numpy()
    ids.sort()

    mask_info = {}  # id: pts_id
    frame_gaussian_ids = set(gaus_ids.tolist())
    for mask_id in ids:
        if mask_id == 0:  # 跳过背景
            continue
        segmentation = mask_image == mask_id
        valid_mask = segmentation[pixel_ids]

        # 若该掩码对应的高斯点过少，则忽略（噪声/边界碎片）
        if len(set(gaus_ids[valid_mask].tolist())) < 50:
            continue

        mask_info[mask_id] = set(gaus_ids[valid_mask].tolist())

    return mask_info, list(frame_gaussian_ids)


def compute_mask_visible_frame(global_gaussian_in_mask_matrix, gaussian_in_frame_matrix, threshold=0.0):
    '''
    计算每个mask在各帧是否“可见”（visible）：
    - 如果该mask的点中，有超过threshold比例在该帧出现，则视为该mask在该帧可见。

    参数：
    - global_gaussian_in_mask_matrix: (N_pts, N_masks) 的布尔矩阵，点-掩码关联
    - gaussian_in_frame_matrix: (N_pts, N_frames) 的布尔矩阵，点-帧可见关联
    - threshold: 可见比例阈值，默认 0.0（>0即视为可见）

    返回：
    - 可见矩阵 (N_masks, N_frames)，bool数组
    '''
    A = csr_matrix(global_gaussian_in_mask_matrix, dtype=np.float32)  # shape: [N_pts, N_masks]
    B = csr_matrix(gaussian_in_frame_matrix, dtype=np.float32)  # shape: [N_pts, N_frames]

    # 各mask与各帧的交集计数（共有点数）
    intersection_counts = A.T @ B

    # 每个mask的点计数（防止除0）
    mask_point_counts = np.array(A.sum(axis=0)).ravel() + 1e-6

    intersection_counts = intersection_counts.tocoo()
    visible_mask = (intersection_counts.data / mask_point_counts[intersection_counts.row]) > threshold

    result = csr_matrix(
        (np.ones(visible_mask.sum(), dtype=bool),
         (intersection_counts.row[visible_mask], intersection_counts.col[visible_mask])),
        shape=(A.shape[1], B.shape[1])
    )
    return result.toarray()


def construct_mask2gs_tracker(gaussian: GaussianModel, viewcams: List[Camera], clustering_args, save_dir, debug):
    '''
    构建“掩码-高斯”跟踪器的初始数据结构：
    1) 遍历各帧，提取每帧中每个掩码对应的高斯点集合
    2) 生成全局的点-掩码布尔矩阵与点-帧可见矩阵
    3) 依据可见性/包含关系，过滤欠分割掩码
    4) 初始化图节点 Node（后续迭代聚类的输入）

    返回：包含以下键的字典：
    - nodes: 初始Node列表
    - observer_num_thresholds: 观测者数量阈值序列（迭代中使用）
    - mask_gaussian_pclds: {'frameId_maskId': set(point_ids)}
    - global_frame_mask_list: [(frame_id, mask_id), ...]
    - gaussian_in_frame_matrix: (N_pts, N_frames) 点-帧可见矩阵
    - undersegment_mask_ids: 欠分割掩码的全局ID列表
    '''
    # 可选：缓存每帧跟踪结果，便于调试与重复运行
    if debug:
        save_tracker_dir = os.path.join(save_dir, "tracker")
        os.makedirs(save_tracker_dir, exist_ok=True)

    iterator = tqdm(enumerate(viewcams), total=len(viewcams), desc="Extracting Gaussian Tracker")

    # 点-帧 的掩码ID与可见性（注意：同一高斯点在一帧内可能对应多个掩码，边界处）
    gaussian_in_frame_maskid_matrix = np.zeros((len(gaussian.get_xyz), len(viewcams)), dtype=np.uint16)
    gaussian_in_frame_matrix = np.zeros((len(gaussian.get_xyz), len(viewcams)), dtype=bool)
    global_frame_mask_list = []
    mask_gaussian_pclds = {}

    for frame_cnt, view in iterator:
        # 读取/构建本帧的掩码-高斯映射
        if debug:
            tracker_path = os.path.join(save_dir, "tracker", view.image_name.split(".")[0] + ".npy")
            if not os.path.exists(tracker_path):
                mask_dict, frame_gaussian_ids = get_segmap_gaussians(gaussian, view)
                view_info = {
                    "mask_dict": mask_dict,
                    "frame_gaussian_ids": frame_gaussian_ids,
                }
                np.save(tracker_path, view_info, allow_pickle=True)
            else:
                view_info = np.load(tracker_path, allow_pickle=True).item()
                mask_dict = view_info['mask_dict']
                frame_gaussian_ids = view_info['frame_gaussian_ids']
        else:
            mask_dict, frame_gaussian_ids = get_segmap_gaussians(gaussian, view)

        # 标记本帧可见的全部点
        gaussian_in_frame_matrix[frame_gaussian_ids, frame_cnt] = True

        # 记录每个掩码对应的点集，并更新全局列表
        for mask_id, mask_point_cloud_ids in mask_dict.items():
            mask_gaussian_pclds[f'{frame_cnt}_{mask_id}'] = mask_point_cloud_ids
            # 注意：一个点可能对应多mask（边界），此处记录该帧该点的一个mask_id
            gaussian_in_frame_maskid_matrix[list(mask_point_cloud_ids), frame_cnt] = mask_id
            global_frame_mask_list.append((frame_cnt, mask_id))
        torch.cuda.empty_cache()

    # 构造全局 点-掩码 矩阵
    global_gaussian_in_mask_matrix = np.zeros((len(gaussian.get_xyz), len(global_frame_mask_list)), dtype=bool)
    for mask_idx, frame_mask_id in enumerate(mask_gaussian_pclds):
        global_gaussian_in_mask_matrix[np.array(list(mask_gaussian_pclds[frame_mask_id])), mask_idx] = True

    # 过滤欠分割掩码（基于可见性统计与包含关系）
    visible_frames = []      # 每个mask在各帧是否可见
    contained_masks = []     # 每个mask所“包含”的其它mask（按最大重叠来判断）
    undersegment_mask_ids = []

    mask_visible_frames = compute_mask_visible_frame(global_gaussian_in_mask_matrix, gaussian_in_frame_matrix)
    mask_cnts = 0
    iterator = tqdm(global_frame_mask_list, total=len(global_frame_mask_list), desc="Filtering Undersegment Masks")
    for frame_id, mask_id in iterator:
        valid, contained_mask, visible_frame = judge_single_mask(gaussian_in_frame_maskid_matrix,
                                                                 mask_gaussian_pclds,
                                                                 f'{frame_id}_{mask_id}',
                                                                 mask_visible_frames[mask_cnts],
                                                                 viewcams,
                                                                 global_frame_mask_list,
                                                                 clustering_args)

        contained_masks.append(contained_mask)
        visible_frames.append(visible_frame)
        if not valid:
            # 记录为欠分割mask
            global_mask_id = global_frame_mask_list.index((frame_id, mask_id))
            undersegment_mask_ids.append(global_mask_id)
        torch.cuda.empty_cache()
        mask_cnts += 1

    contained_masks = np.stack(contained_masks, axis=0)
    visible_frames = np.stack(visible_frames, axis=0)

    # 移除欠分割掩码对包含关系/可见性的干扰
    for global_mask_id in undersegment_mask_ids:  # remove undersegment
        frame_id, _ = global_frame_mask_list[global_mask_id]
        global_frame_id = frame_id
        mask_projected_idx = np.where(contained_masks[:, global_mask_id])[0]  # 被其包含的mask索引
        contained_masks[:, global_mask_id] = False  # 排除该mask影响
        visible_frames[mask_projected_idx, global_frame_id] = False

    ## 准备迭代聚类输入
    contained_masks = torch.from_numpy(contained_masks).float().cuda()
    visible_frames = torch.from_numpy(visible_frames).float().cuda()

    observer_num_thresholds = get_observer_num_thresholds(visible_frames)
    nodes = init_nodes(global_frame_mask_list, visible_frames, contained_masks, undersegment_mask_ids,
                       mask_gaussian_pclds)

    return {
        "nodes": nodes,
        "observer_num_thresholds": observer_num_thresholds,
        "mask_gaussian_pclds": mask_gaussian_pclds,
        "global_frame_mask_list": global_frame_mask_list,
        "gaussian_in_frame_matrix": gaussian_in_frame_matrix,
        "undersegment_mask_ids": undersegment_mask_ids
    }


def judge_single_mask(gaussian_in_mask_matrix,
                      mask_gaussian_pclds,
                      frame_mask_id,
                      mask_visible_frame,
                      viewcams: List[Camera],
                      global_frame_mask_list,
                      clustering_args):
    '''
    判定单个(mask, frame)是否为“有效”掩码，并统计其包含关系与可见帧：
    - 在该mask可见的各帧上，统计该mask点与各帧掩码的重叠比例
    - 若最大重叠比例超过 contained_threshold，则认定被该帧的该掩码“包含”
    - 统计可见帧数量与被切分（split）的帧比例，超过 undersegment_filter_threshold 视为欠分割

    返回：
    - valid: bool 是否有效（非欠分割）
    - contained_mask: (N_global_masks,) 的bool向量，表示其包含了哪些全局mask
    - visible_frame: (N_frames,) 的bool向量，表示在哪些帧可见
    '''
    mask_gaussian_pcld = mask_gaussian_pclds[frame_mask_id]

    visible_frame = np.zeros(len(viewcams), dtype=bool)
    contained_mask = np.zeros(len(global_frame_mask_list), dtype=bool)

    mask_gaussians_info = gaussian_in_mask_matrix[list(mask_gaussian_pcld), :]

    split_num = 0  # 欠分割帧计数
    visible_num = 0  # 可见帧计数

    for frame_id in np.where(mask_visible_frame)[0]:
        # 统计该帧中重叠的掩码ID与计数，并按重叠数量降序
        overlap_mask_ids, overlap_mask_cnts = np.unique(mask_gaussians_info[:, frame_id], return_counts=True)
        sorted_idx = np.argsort(overlap_mask_cnts)[::-1]
        overlap_mask_ids, overlap_mask_cnts = overlap_mask_ids[sorted_idx], overlap_mask_cnts[sorted_idx]

        # 若大部分点对应mask=0（不可见/背景），则跳过
        if 0 in overlap_mask_ids:
            invalid_indice = np.where(overlap_mask_ids == 0)[0]
            invalid_gaussian_cnts = overlap_mask_cnts[invalid_indice]
            if invalid_gaussian_cnts / overlap_mask_cnts.sum() > clustering_args.mask_visible_threshold:  # 0.7
                continue
            # 移除0类
            overlap_mask_ids = np.delete(overlap_mask_ids, invalid_indice)
            overlap_mask_cnts = np.delete(overlap_mask_cnts, invalid_indice)

        if len(overlap_mask_ids) == 0:
            continue

        visible_num += 1

        contained_ratio = overlap_mask_cnts[0] / overlap_mask_cnts.sum()
        if contained_ratio > clustering_args.contained_threshold:  # 被该帧中某个掩码“完整包含”
            frame_mask_idx = global_frame_mask_list.index((frame_id, overlap_mask_ids[0]))
            contained_mask[frame_mask_idx] = True
            visible_frame[frame_id] = True
        else:
            split_num += 1

    # 若无可见帧或大比例帧被切分，则判定为欠分割
    if visible_num == 0 or split_num / visible_num > clustering_args.undersegment_filter_threshold:
        return False, contained_mask, visible_frame
    else:
        return True, contained_mask, visible_frame


##
def get_observer_num_thresholds(visible_frames):
    '''
    统计用于迭代阶段的“观测者数量阈值”序列：
    - 计算任意两mask之间的共同可见帧数量分布
    - 从95%到0%每隔5%取分位数，作为阈值列表（下限为1）
    '''
    observer_num_matrix = torch.matmul(visible_frames, visible_frames.transpose(0, 1))
    observer_num_list = observer_num_matrix.flatten()
    observer_num_list = observer_num_list[observer_num_list > 0].cpu().numpy()
    observer_num_thresholds = []
    for percentile in range(95, -5, -5):
        observer_num = np.percentile(observer_num_list, percentile)
        if observer_num <= 1:
            if percentile < 50:
                break
            else:
                observer_num = 1
        observer_num_thresholds.append(observer_num)
    return observer_num_thresholds


def init_nodes(global_frame_mask_list, mask_project_on_all_frames, contained_masks, undersegment_mask_ids,
               mask_point_clouds):
    '''
    将过滤后的全局(mask, frame)条目初始化为Node节点，作为迭代聚类的起点。

    参数：
    - global_frame_mask_list: [(frame_id, mask_id), ...]
    - mask_project_on_all_frames: (N_masks, N_frames) 的bool矩阵，mask在各帧可见
    - contained_masks: (N_masks, N_masks) 的bool矩阵，包含关系
    - undersegment_mask_ids: 欠分割全局mask索引
    - mask_point_clouds: {'frameId_maskId': set(point_ids)}

    返回：
    - nodes: List[Node]
    '''
    nodes = []
    for global_mask_id, (frame_id, mask_id) in enumerate(global_frame_mask_list):
        if global_mask_id in undersegment_mask_ids:
            continue
        mask_list = [(frame_id, mask_id)]
        frame = mask_project_on_all_frames[global_mask_id]
        frame_mask = contained_masks[global_mask_id]
        point_ids = mask_point_clouds[f'{frame_id}_{mask_id}']
        node_info = (0, len(nodes))  # (iter_idx, node_idx) 初始迭代编号0
        node = Node(mask_list, frame, frame_mask, point_ids, node_info, None)
        nodes.append(node)
    return nodes
