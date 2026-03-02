# import umap
import copy
import glob
import os
import sys
from collections import defaultdict

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import tqdm
from PIL import Image
from sklearn.decomposition import PCA


def contrastive_loss(features, masks, predef_u_list=None, min_pixnum=0, temp_lambda=1000,
                     consider_negative=False):
    '''
    原型对比损失（ProtoNCE）

    思想：
    - 将每个簇/实例的特征聚合为“原型中心” u_k（可传入预定义原型，也可按当前批次统计）
    - 对每个样本 x_n，最大化其与所属簇中心的相似度，最小化与其他簇中心的相似度
    - 引入 per-cluster 的温度/方差项 φ_k，使得松弛程度与簇紧密度、规模相关

    参数：
    - features: (N, C) 特征张量；函数内会先按掩码筛选，再做 L2 归一化
    - masks: (N,) 标签，代表簇/实例ID
    - predef_u_list: (K, C) 预定义的原型中心（可选）；若提供则按 mask_ids 索引
    - min_pixnum: 过滤最小簇大小阈值
    - temp_lambda: φ 的平滑项（默认 1000）
    - consider_negative: 是否保留标签0作为“负类”；默认 False 时仅考虑 masks>0，并将标签减1从0计数

    返回：
    - ProtoNCE 标量损失（torch.Tensor）
    '''
    if not consider_negative:
        valid_semantic_idx = masks > 0  # note:已经移除0了
    else:  # 考虑0标签
        valid_semantic_idx = torch.ones_like(masks, dtype=torch.bool).cuda()

    # 过滤小簇
    mask_ids, mask_nums = torch.unique(masks, return_counts=True)
    valid_mask_ids = mask_ids[mask_nums > min_pixnum]
    valid_semantic_idx = valid_semantic_idx & torch.isin(masks, valid_mask_ids)

    # 筛选并重映射标签
    masks = masks[valid_semantic_idx].type(torch.int64)
    if not consider_negative:
        masks = masks - 1  # from zero
    features = features[valid_semantic_idx, :]  # N, C
    features = features / (torch.norm(features, dim=-1, keepdim=True) + 1e-9).detach()

    # 拿到有效标签与计数
    mask_ids, mask_nums = torch.unique(masks, return_counts=True)

    # 选择原型中心
    if predef_u_list is not None:
        u_list = predef_u_list[mask_ids]
    # 将原标签映射为从0开始的连续索引
    label_mapping = torch.zeros(mask_ids.max() + 1, dtype=torch.long).cuda()
    label_mapping[mask_ids] = torch.arange(len(mask_ids)).cuda()
    masks = label_mapping[masks]
    mask_ids, mask_nums = torch.unique(masks, return_counts=True)

    mask_num = mask_ids.shape[0]
    if predef_u_list is None:
        # 计算当前批次的簇均值作为原型中心
        u_list_sum = torch.zeros(mask_num, features.shape[1]).cuda()
        u_list_sum.scatter_add_(0, masks.unsqueeze(1).expand(-1, features.shape[1]), features)
        u_list = u_list_sum / mask_nums[:, None]

    # 估计每簇的方差尺度 φ（作为温度项）
    cluster_diff = features - u_list[masks]
    cluster_diff_norm = torch.norm(cluster_diff, dim=1, keepdim=True)
    phi_list_sum = torch.zeros(mask_num, 1).cuda()
    phi_list_sum.scatter_add_(0, masks.unsqueeze(1), cluster_diff_norm)
    phi_list = phi_list_sum / (mask_nums.unsqueeze(1) * torch.log(mask_nums.unsqueeze(1) + temp_lambda))
    phi_list = torch.clip(phi_list * 10, min=0.5, max=1.0)
    phi_list = phi_list.detach()  # variance（不反传）

    # 计算 ProtoNCE：softmax over prototypes
    dist = torch.exp(torch.matmul(features, u_list.T) / phi_list.T)  # [N, K]
    dist_sum = dist.sum(dim=1, keepdim=True)
    ProtoNCE = -torch.sum(torch.log(dist[torch.arange(features.shape[0]), masks].unsqueeze(1) / (dist_sum + 1e-9)))

    return ProtoNCE


def feature_to_rgb(features, pca_proj_mat=None, type="PCA"):
    """
    将2D特征图（C,H,W）可视化为RGB图：
    - 对每个像素的特征做归一化
    - 若提供投影矩阵 pca_proj_mat，则直接线性投影至3维
    - 否则按 PCA 取前三主成分，并映射到 [0,255]

    参数：
    - features: (C, H, W) torch.Tensor
    - pca_proj_mat: (C, 3) torch.Tensor 或 None
    - type: 降维方法（当前仅"PCA"）

    返回：
    - rgb_array: (H, W, 3) uint8 RGB 图
    """
    # Input features shape: (16, H, W)

    # Reshape features for PCA
    H, W = features.shape[1], features.shape[2]
    features_reshaped = features.view(features.shape[0], -1).T

    sam_norm = features_reshaped / (features_reshaped.norm(dim=1, keepdim=True) + 1e-9)
    features_reshaped = sam_norm  # * 0.5 + 0.5  # [-1,1]->[0,1]

    if pca_proj_mat is not None:
        low_feat = (features_reshaped @ pca_proj_mat).reshape(H, W, 3).cpu().numpy()
    else:
        # Apply PCA and get the first 3 components
        if type == "PCA":
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(features_reshaped.cpu().numpy())

            # Reshape back to (H, W, 3)
            low_feat = pca_result.reshape(H, W, 3)

    # Normalize to [0, 255]
    low_feat = (low_feat * 0.5 + 0.5).clip(0, 1)
    feat_normalized = 255 * (low_feat)  # * 0.5 + 0.5

    rgb_array = feat_normalized.astype('uint8')

    return rgb_array


def feature3d_to_rgb(features):
    """
    将3D点特征（N, C）投影为 (N,3) RGB：
    - 先 L2 归一化每个点的特征
    - PCA 到3维，再映射到 [0,1] 近似可视化区间

    返回：
    - (N, 3) numpy 数组（浮点RGB，范围约0.3~1.0）
    """
    sam_norm = features / (np.linalg.norm(features, axis=-1, keepdims=True) + 1e-9)
    features_reshaped = sam_norm  # * 0.5 + 0.5  # [-1,1]

    # Apply PCA and get the first 3 components
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_reshaped)
    # tsne = TSNE(n_components=3, random_state=42, perplexity=10, n_iter=500)
    # pca_result = tsne.fit_transform(features_reshaped)
    return ((pca_result + 1).clip(0, 2) / 2) * 0.7 + 0.3
    # (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min()) * 0.7 + 0.3


def mask_to_rgb(mask):
    """
    将整型分割图（H,W）着色为RGB（HSV色表）。

    参数：
    - mask: torch.Tensor (H,W) 整型label

    返回：
    - (H,W,3) uint8 RGB 图
    """
    mask_image = mask.detach().cpu().numpy()
    num_classes = np.max(mask_image) + 1
    colors = plt.get_cmap('hsv', num_classes)
    norm = mcolors.Normalize(vmin=0, vmax=num_classes - 1)
    colored_segmentation = colors(norm(mask_image))
    return np.uint8(colored_segmentation[..., :3] * 255.0)
