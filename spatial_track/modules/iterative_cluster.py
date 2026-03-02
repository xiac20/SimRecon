from tqdm import tqdm
import networkx as nx
from spatial_track.modules.node import Node
import torch


def cluster_into_new_nodes(iteration, old_nodes, graph):
    """
    将当前图中的连通分量聚合为新的节点列表。

    参数：
    - iteration: 当前迭代编号（用于记录到 node_info 中）
    - old_nodes: 上一轮的节点列表（List[Node]）
    - graph: networkx.Graph，表示节点间的连边关系（可合并关系）

    返回：
    - new_nodes: 新的节点列表（每个连通分量合并出一个新节点）
    """
    new_nodes = []
    for component in nx.connected_components(graph):
        node_info = (iteration, len(new_nodes))
        # 将一个连通分量中的多个旧节点合并为一个新节点
        new_nodes.append(Node.create_node_from_list([old_nodes[node] for node in component], node_info))
    return new_nodes


def update_graph(nodes, observer_num_threshold, connect_threshold):
    '''
        根据可见帧与支持帧，更新节点间的“可合并”图。

        定义：
        - observer_nums[i,j]: 节点 i 与 j 共同可见的帧数（可被同时观察到的帧数）
        - supporter_nums[i,j]: 节点 i 与 j 在相同帧中互为“包含支持”的计数（用于判定合并合理性）
        - view_concensus_rate = supporter_nums / (observer_nums + 1e-7)

        规则：
        - 若 observer_nums < observer_num_threshold，则不连边（观察者太少，不足以下决定）
        - 若 view_concensus_rate >= connect_threshold（例如0.9），且未被上条规则断开，则连边

        参数：
        - nodes: List[Node]
        - observer_num_threshold: 观测者数量阈值（随迭代变化）
        - connect_threshold: 视角一致性阈值（固定值，如0.9）

        返回：
        - G: networkx.Graph，连通性按上述规则建立
    '''
    node_visible_frames = torch.stack([node.visible_frame for node in nodes], dim=0)
    node_contained_masks = torch.stack([node.contained_mask for node in nodes], dim=0)

    # M[i,j] 表示节点 i 与 j 共同可见的帧数
    observer_nums = torch.matmul(node_visible_frames, node_visible_frames.transpose(0, 1))
    # M[i,j] 表示支持合并的帧数（例如在同一帧被同一掩码包含）
    supporter_nums = torch.matmul(node_contained_masks, node_contained_masks.transpose(0, 1))

    view_concensus_rate = supporter_nums / (observer_nums + 1e-7)

    disconnect = torch.eye(len(nodes), dtype=bool).cuda()
    # 观测者少于阈值的对，直接断开
    disconnect = disconnect | (observer_nums < observer_num_threshold)

    A = view_concensus_rate >= connect_threshold
    A = A & ~disconnect  # 既满足一致性，又未被断开的对，建立连边
    A = A.cpu().numpy()

    G = nx.from_numpy_array(A)
    return G


def iterative_clustering(init_mask_assocation, clustering_args):
    """
    迭代式聚类主过程：
    - 按从高到低的一系列“观测者数量阈值”逐步放宽合并条件
    - 在每一轮：
      1) 基于当前阈值与固定的视角一致性阈值，更新连边图
      2) 将连通分量合并为新节点
    - 最终返回更新过 nodes 的字典

    参数：
    - init_mask_assocation: 从初始化阶段返回的字典（含 nodes、observer_num_thresholds 等）
    - clustering_args: 聚类超参数（包含 view_consensus_threshold 等）

    返回：
    - init_mask_assocation: 更新了 'nodes' 键的新字典
    """
    iterator = tqdm(enumerate(init_mask_assocation["observer_num_thresholds"]),
                    total=len(init_mask_assocation["observer_num_thresholds"]), desc="Optimizing the Mask Association")

    nodes = init_mask_assocation["nodes"]
    for iterate_id, observer_num_threshold in iterator:
        graph = update_graph(nodes, observer_num_threshold,
                             clustering_args.view_consensus_threshold)  # connect_threshold: 0.9
        nodes = cluster_into_new_nodes(iterate_id + 1, nodes, graph)
        torch.cuda.empty_cache()

    init_mask_assocation["nodes"] = nodes

    return init_mask_assocation
