import numpy as np

import numpy as np
import os
import torch
from tqdm import tqdm


def judge_bbox_overlay(bbox_1, bbox_2):
    """
    判断两个3D轴对齐包围盒是否相交/重叠。

    参数：
    - bbox_1: [min_xyz, max_xyz]
    - bbox_2: [min_xyz, max_xyz]

    返回：
    - True/False 是否重叠
    """
    for i in range(3):
        if bbox_1[0][i] > bbox_2[1][i] or bbox_2[0][i] > bbox_1[1][i]:
            return False
    return True


def merge_overlapping_objects(total_point_ids_list, total_bbox_list, total_mask_list, overlapping_ratio):
    '''
        合并（或标记无效）高度重叠的对象：
        - 若两个对象的点集交集占任一对象点数的比例 > overlapping_ratio，则淘汰其中一个（保留另一个）。

        参数：
        - total_point_ids_list: List[np.ndarray]，每个对象的点ID集合
        - total_bbox_list: List[[min_xyz, max_xyz]]，每个对象的AABB
        - total_mask_list: List[mask_list]，每个对象对应的掩码列表
        - overlapping_ratio: 判定阈值（例如0.8）

        返回：
        - valid_point_ids_list: 合并后有效对象的点ID列表
        - valid_pcld_mask_list: 合并后有效对象的掩码列表
        - invalid_object: bool数组，标记哪些对象被视为无效
    '''
    total_object_num = len(total_point_ids_list)
    invalid_object = np.zeros(total_object_num, dtype=bool)

    for i in range(total_object_num):
        if invalid_object[i]:
            continue
        point_ids_i = set(total_point_ids_list[i])
        bbox_i = total_bbox_list[i]
        for j in range(i + 1, total_object_num):
            if invalid_object[j]:
                continue
            point_ids_j = set(total_point_ids_list[j])
            bbox_j = total_bbox_list[j]
            # 先基于AABB快速判断是否可能重叠
            if judge_bbox_overlay(bbox_i, bbox_j):
                intersect = len(point_ids_i.intersection(point_ids_j))
                if intersect / len(point_ids_i) > overlapping_ratio:
                    invalid_object[i] = True
                elif intersect / len(point_ids_j) > overlapping_ratio:
                    invalid_object[j] = True

    valid_point_ids_list = []
    valid_pcld_mask_list = []
    for i in range(total_object_num):
        if not invalid_object[i]:
            valid_point_ids_list.append(total_point_ids_list[i])
            valid_pcld_mask_list.append(total_mask_list[i])
    return valid_point_ids_list, valid_pcld_mask_list, invalid_object


def filter_point(point_frame_matrix, node, pcld_list, point_ids_list, mask_point_clouds, args):
    '''
        点过滤（参考 OVIR-3D）：
        - 对每个DBSCAN分割得到的子对象（pcld_list/point_ids_list），计算点的检测比率：
          检测比率 = 该点在该节点（cluster）出现的帧数 / 该点在全视频出现的帧数
        - 若检测比率 > 阈值（args.point_filter_threshold），则保留；否则过滤
        - 同时统计每个mask在其隶属对象中的覆盖率，用于后续（如OpenMask3D）

        参数：
        - point_frame_matrix: (N_pts, N_frames) 布尔矩阵，点在帧中是否出现
        - node: 当前聚类节点（包含 mask_list 与 visible_frame）
        - pcld_list: List[o3d.geometry.PointCloud]，DBSCAN分割出的子点云
        - point_ids_list: List[np.ndarray]，对应子点云的点ID列表
        - mask_point_clouds: {f"frameId_maskId": set(point_ids)}，每个掩码的点集合
        - args: 聚类与过滤阈值，其中 point_filter_threshold 使用

        返回：
        - filtered_point_ids: 过滤后的每个子对象的点ID列表
        - filtered_bbox_list: 对应子对象的AABB列表 [min_xyz, max_xyz]
        - filtered_mask_list: 每个子对象对应的掩码列表（含覆盖率）
    '''

    def count_point_appears_in_video(point_frame_matrix, point_ids_list, node_global_frame_id_list):
        '''
            统计每个子对象中各点在整个视频中的出现帧数，并初始化其在节点中的出现矩阵为False。

            返回：
            - point_appear_in_video_nums: List[np.ndarray]，各子对象内每个点在视频出现帧数
            - point_appear_in_node_matrixs: List[np.ndarray(bool)]，形状与选帧一致，初始化为False
        '''
        point_appear_in_video_nums, point_appear_in_node_matrixs = [], []
        for point_ids in (point_ids_list):
            point_appear_in_video_matrix = point_frame_matrix[point_ids,]
            point_appear_in_video_matrix = point_appear_in_video_matrix[:, node_global_frame_id_list]
            point_appear_in_video_nums.append(np.sum(point_appear_in_video_matrix, axis=1))

            point_appear_in_node_matrix = np.zeros_like(point_appear_in_video_matrix, dtype=bool)  # initialize as False
            point_appear_in_node_matrixs.append(point_appear_in_node_matrix)
        return point_appear_in_video_nums, point_appear_in_node_matrixs

    def count_point_appears_in_node(mask_list, node_frame_id_list, point_ids_list, mask_point_clouds,
                                    point_appear_in_node_matrixs):
        '''
            填充各子对象的“在节点中出现矩阵”：
            - 遍历该节点的掩码(mask_list)，将与子对象相交的点在对应帧位置标True
            - 同时确定该mask应归属的子对象（交点最多者），并计算其覆盖率（交点数/对象点数）

            返回：
            - object_mask_list: List[List[(frame_id, mask_id, coverage)]] 每个子对象对应的掩码列表
            - point_appear_in_node_matrixs: 更新后的出现矩阵列表
        '''
        object_mask_list = [[] for _ in range(len(point_ids_list))]

        for frame_id, mask_id in (mask_list):
            # 有可能mask_list不属于node_frame_id_list
            if frame_id not in node_frame_id_list:
                continue
            frame_id_in_list = np.where(node_frame_id_list == frame_id)[0][0]
            mask_point_ids = list(mask_point_clouds[f'{frame_id}_{mask_id}'])

            object_id_with_largest_intersect, largest_intersect, coverage = -1, 0, 0
            for i, point_ids in enumerate(point_ids_list):
                point_ids_within_object = np.where(np.isin(point_ids, mask_point_ids))[0]
                point_appear_in_node_matrixs[i][point_ids_within_object, frame_id_in_list] = True
                if len(point_ids_within_object) > largest_intersect:
                    object_id_with_largest_intersect, largest_intersect = i, len(point_ids_within_object)
                    coverage = len(point_ids_within_object) / len(point_ids)
            if largest_intersect == 0:
                continue
            object_mask_list[object_id_with_largest_intersect] += [(frame_id, mask_id, coverage)]
        return object_mask_list, point_appear_in_node_matrixs

    node_global_frame_id_list = torch.where(node.visible_frame)[0].cpu().numpy()
    node_frame_id_list = node_global_frame_id_list
    mask_list = node.mask_list

    point_appear_in_video_nums, point_appear_in_node_matrixs = count_point_appears_in_video(point_frame_matrix,
                                                                                            point_ids_list,
                                                                                            node_global_frame_id_list)
    object_mask_list, point_appear_in_node_matrixs = count_point_appears_in_node(mask_list, node_frame_id_list,
                                                                                 point_ids_list, mask_point_clouds,
                                                                                 point_appear_in_node_matrixs)

    # 依据检测比率过滤，并构造每个子对象的bbox与掩码列表
    filtered_point_ids, filtered_mask_list, filtered_bbox_list = [], [], []
    for i, (point_appear_in_video_num, point_appear_in_node_matrix) in (enumerate(
            zip(point_appear_in_video_nums, point_appear_in_node_matrixs))):
        detection_ratio = np.sum(point_appear_in_node_matrix, axis=1) / (point_appear_in_video_num + 1e-6)
        valid_point_ids = np.where(detection_ratio > args.point_filter_threshold)[0]
        if len(valid_point_ids) == 0 or len(object_mask_list[i]) < 2:
            continue
        filtered_point_ids.append(point_ids_list[i][valid_point_ids])
        filtered_bbox_list.append([np.amin(pcld_list[i].points, axis=0), np.amax(pcld_list[i].points, axis=0)])
        filtered_mask_list.append(object_mask_list[i])
    return filtered_point_ids, filtered_bbox_list, filtered_mask_list


def dbscan_process(pcld, point_ids, DBSCAN_THRESHOLD=0.1, min_points=4):
    '''
    使用 DBSCAN 将不连通的点云拆分为多个对象（参考 OVIR-3D）。

    参数：
    - pcld: o3d.geometry.PointCloud 当前节点的点云
    - point_ids: 对应全局点的ID
    - DBSCAN_THRESHOLD: eps半径
    - min_points: 最小点数阈值

    返回：
    - pcld_list: 拆分后的子点云列表
    - point_ids_list: 对应子点云的点ID数组列表
    '''
    # TODO: 可以考虑融合CLIP特征提升一致性
    labels = np.array(pcld.cluster_dbscan(eps=DBSCAN_THRESHOLD, min_points=min_points)) + 1  # -1为噪声
    count = np.bincount(labels)

    # 将不连通的点云拆分成多个对象
    pcld_list, point_ids_list = [], []
    pcld_ids_list = np.array(point_ids)
    for i in range(len(count)):
        remain_index = np.where(labels == i)[0]
        if len(remain_index) == 0:
            continue
        new_pcld = pcld.select_by_index(remain_index)
        point_ids = pcld_ids_list[remain_index]
        pcld_list.append(new_pcld)
        point_ids_list.append(point_ids)
    return pcld_list, point_ids_list


def find_represent_mask(mask_info_list):
    """
    为一个对象（子点云）选择若干“代表性掩码”（按覆盖率从高到低排序，取前5）。
    mask_info_list 元素形如 (frame_id, mask_id, coverage)。
    """
    mask_info_list.sort(key=lambda x: x[2], reverse=True)
    return mask_info_list[:5]


def export_class_agnostic_mask(args, save_dir, class_agnostic_mask_list):
    """
    导出与类别无关的3D实例掩码（标准评测格式）：
    - pred_masks: (N_points, N_instances) 的布尔矩阵
    - pred_score: 置信度（这里置为1）
    - pred_classes: 类别（这里全部置0，类无关）
    保存到 save_dir/mask3d.npz。
    """
    pred_dir = os.path.join('data/prediction', args.config)
    os.makedirs(pred_dir, exist_ok=True)

    num_instance = len(class_agnostic_mask_list)
    pred_masks = np.stack(class_agnostic_mask_list, axis=1)
    pred_dict = {
        "pred_masks": pred_masks,
        "pred_score": np.ones(num_instance),
        "pred_classes": np.zeros(num_instance, dtype=np.int32)
    }
    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, f'mask3d.npz'), **pred_dict)
    return


def export(dataset, total_point_ids_list, total_mask_list, args):
    '''
    导出对象字典与类无关掩码：
    - object_dict.npy 保存每个对象的点ID、掩码列表与代表性掩码
    - 同时调用 export_class_agnostic_mask 导出评测所需的 mask3d.npz

    参数：
    - dataset: 数据集对象，需提供 get_scene_points() 和 object_dict_dir
    - total_point_ids_list: List[np.ndarray]，所有对象的点ID列表
    - total_mask_list: List[List[(frame_id, mask_id, coverage)]]
    - args: 配置参数（含 config 与保存目录）
    '''
    total_point_num = dataset.get_scene_points().shape[0]
    class_agnostic_mask_list = []
    object_dict = {}
    for i, (point_ids, mask_list) in enumerate(zip(total_point_ids_list, total_mask_list)):
        object_dict[i] = {
            'point_ids': point_ids,
            'mask_list': mask_list,
            'repre_mask_list': find_represent_mask(mask_list),
        }
        binary_mask = np.zeros(total_point_num, dtype=bool)
        binary_mask[list(point_ids)] = True
        class_agnostic_mask_list.append(binary_mask)

    export_class_agnostic_mask(args, dataset.object_dict_dir, class_agnostic_mask_list)

    os.makedirs(os.path.join(dataset.object_dict_dir, args.config), exist_ok=True)
    np.save(os.path.join(dataset.object_dict_dir, 'object_dict.npy'), object_dict, allow_pickle=True)


def post_process(gaussian, mask_assocation, clustering_args):
    """
    后处理主流程：
    1) 对每个聚类节点（对象）使用 DBSCAN 切分不连通的点云
    2) 按检测比率过滤点，得到干净的子对象 + AABB + 掩码列表
    3) 合并高度重叠的对象（>0.8）
    4) 更新 mask_assocation 字典，写入 total_point_ids_list 与 total_mask_list

    参数：
    - gaussian: GaussianModel，用于获取全局点坐标
    - mask_assocation: 初始化与聚类阶段产生的字典（含 nodes/mask_gaussian_pclds 等）
    - clustering_args: 超参数（含 point_filter_threshold 等）

    返回：
    - mask_assocation: 更新后字典
    """
    # For each cluster, we follow OVIR-3D to i) use DBScan to split the disconnected point cloud into different objects
    # ii) filter the points that hardly appear within this cluster, i.e. the detection ratio is lower than a threshold
    nodes = mask_assocation['nodes']
    mask_gaussian_pclds = mask_assocation['mask_gaussian_pclds']
    global_frame_mask_list = mask_assocation["global_frame_mask_list"]
    gaussian_in_frame_matrix = mask_assocation["gaussian_in_frame_matrix"]

    total_point_ids_list, total_bbox_list, total_mask_list = [], [], []
    scene_points = gaussian.get_xyz.cpu().numpy()

    iterator = tqdm(nodes, total=len(nodes), desc="DBScan Filter with Each Instance")

    for node in iterator:
        if len(node.mask_list) < 2:  # objects merged from less than 2 masks are ignored
            continue
        pcld, point_ids = node.get_point_cloud(scene_points)
        if True:
            pcld_list, point_ids_list = dbscan_process(pcld, point_ids, DBSCAN_THRESHOLD=0.1,
                                                       min_points=4)  # split the disconnected point cloud into different objects
        else:
            pcld_list, point_ids_list = [pcld], [np.array(point_ids)]
        point_ids_list, bbox_list, mask_list = filter_point(gaussian_in_frame_matrix, node, pcld_list,
                                                            point_ids_list,
                                                            mask_gaussian_pclds,
                                                            clustering_args)

        total_point_ids_list.extend(point_ids_list)
        total_bbox_list.extend(bbox_list)
        total_mask_list.extend(mask_list)

    # merge objects that have larger than 0.8 overlapping ratio
    total_point_ids_list_merge, total_mask_list_merge, invalid_object_mask = merge_overlapping_objects(
        total_point_ids_list, total_bbox_list, total_mask_list, overlapping_ratio=0.8)
    total_point_ids_list = total_point_ids_list_merge
    total_mask_list = total_mask_list_merge

    mask_assocation.update({
        'total_point_ids_list': total_point_ids_list,
        'total_mask_list': total_mask_list,
    })

    return mask_assocation
