import torch
import open3d as o3d

class Node:
    
    def __init__(self, mask_list, visible_frame, contained_mask, point_ids, node_info, son_node_info):
        '''
            聚类图中的节点，表示由若干 (frame_id, mask_id) 掩码组成的一个实例簇。

            参数：
            - mask_list: 本节点包含的掩码列表，元素形如 (frame_id, mask_id)
            - visible_frame: (N_frames,) 的向量，节点是否在各帧中可见（bool/float，一般为0/1）
            - contained_mask: (N_global_masks,) 的向量，节点与全局掩码的“包含支持”关系（bool/float）
            - point_ids: set[int]，该节点对应的3D点（高斯点）ID集合
            - node_info: (iteration_idx, node_idx) 调试信息，记录该节点来自哪一轮的第几个
            - son_node_info: set[(iteration_idx, node_idx)]，从上一轮合并而来的子节点信息
        '''
        self.mask_list = mask_list
        self.visible_frame = visible_frame
        self.contained_mask = contained_mask
        self.point_ids = point_ids
        self.node_info = node_info
        self.son_node_info = son_node_info


    @ staticmethod
    def create_node_from_list(node_list, node_info):
        '''
        从一组旧节点合并创建新节点：
        - 掩码列表拼接
        - 可见帧按位或
        - 包含关系按位或
        - 3D点ID做并集
        - 记录子节点来源

        参数：
        - node_list: List[Node]，要合并的节点列表
        - node_info: (iteration_idx, node_idx) 新节点调试信息

        返回：
        - Node 合并后的新节点
        '''
        mask_list = []
        visible_frame = torch.zeros(len(node_list[0].visible_frame), dtype=bool).cuda()
        contained_mask = torch.zeros(len(node_list[0].contained_mask), dtype=bool).cuda()
        point_ids = set()
        son_node_info = set()
        for node in node_list:
            mask_list += node.mask_list
            visible_frame = visible_frame | (node.visible_frame).bool()
            contained_mask = contained_mask | (node.contained_mask).bool()
            point_ids = point_ids.union(node.point_ids)
            son_node_info.add(node.node_info)
        return Node(mask_list, visible_frame.float(), contained_mask.float(), point_ids, node_info, son_node_info)
    
    def get_point_cloud(self, scene_points):
        '''
            将当前节点对应的3D点坐标提取为Open3D点云对象，便于可视化与后处理。

            参数：
            - scene_points: (N_points, 3) 的全局点坐标数组（与 point_ids 对应）

            返回：
            - pcld: open3d.geometry.PointCloud 对象（只包含该节点的点）
            - point_ids: list[int] 该节点对应的3D点ID列表
        '''
        point_ids = list(self.point_ids)
        points = scene_points[point_ids]
        pcld = o3d.geometry.PointCloud()
        pcld.points = o3d.utility.Vector3dVector(points)
        return pcld, point_ids
