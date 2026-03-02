# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os
import cv2
import sys
import numpy as np
import subprocess

cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import warnings
from tqdm import tqdm
import torch
from PIL import Image

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo
import warnings

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore", category=UserWarning)


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_boundary(mask, kernel_size_erode=5):
    kernel_erode = np.ones((kernel_size_erode, kernel_size_erode), np.uint8)  # rubbish -> 15
    mask = np.float32(mask)  # 255
    mask = mask - cv2.erode(mask, kernel_erode, iterations=1)

    return mask > 0


import colorsys
def vis_mask(image, masks):
    """
    image: 原始图片（未用到，可忽略）
    masks: shape=(N,H,W)，每个mask为bool或0/1，N为实例数
    返回: 彩色可视化mask，shape=(H,W,3)
    """
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    masks = masks.astype(bool)
    N, H, W = masks.shape
    # 生成明亮且区分度高的颜色
    hsvs = [(i / max(N,1), 0.85, 0.95) for i in range(N)]
    colors = [tuple(int(c * 255) for c in colorsys.hsv_to_rgb(*hsv)) for hsv in hsvs]
    color_mask = np.zeros((H, W, 3), dtype=np.uint8)
    for idx, mask in enumerate(masks):
        color_mask[mask] = colors[idx]
    return color_mask


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--scene_dir",
        type=str
    )
    parser.add_argument(
        "--image_path_pattern",
        type=str
    )
    parser.add_argument(
        "--dataset",
        type=str
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    seq_dir = args.scene_dir
    image_list = sorted(glob.glob(os.path.join(seq_dir, args.image_path_pattern)))
    print(os.path.join(seq_dir, args.image_path_pattern))

    output_vis_dir = os.path.join(seq_dir, 'sam_vis/mask')
    output_seg_dir = os.path.join(seq_dir, 'sam/mask')
    os.makedirs(output_vis_dir, exist_ok=True)
    os.makedirs(output_seg_dir, exist_ok=True)

    for path in tqdm(image_list):
        # use PIL, to be consistent with evaluation
        img = read_image(path, format="BGR")
        predictions = demo.run_on_image(img)

        ##### color_mask
        pred_masks = predictions["instances"].pred_masks
        pred_scores = predictions["instances"].scores

        # select by confidence threshold
        selected_indexes = (pred_scores >= args.confidence_threshold)
        selected_scores = pred_scores[selected_indexes]
        selected_masks = pred_masks[selected_indexes]

        _, m_H, m_W = selected_masks.shape
        mask_image = np.zeros((m_H, m_W), dtype=np.uint8)

        # rank
        mask_id = 1
        selected_scores, ranks = torch.sort(selected_scores)
        for index in ranks:
            num_pixels = torch.sum(selected_masks[index])
            if num_pixels < 400:
                # ignore small masks
                continue
            mask_image[(selected_masks[index] == 1).cpu().numpy()] = mask_id
            mask_id += 1
        cv2.imwrite(os.path.join(output_seg_dir, os.path.basename(path).split('.')[0] + '.png'), mask_image)

        
        save_vis_file = os.path.join(output_vis_dir, os.path.basename(path).split('.')[0] + '.png')
        masks = selected_masks[selected_masks.sum((1, 2)).argsort()]  # 按照mask数量进行排序
        img = read_image(path, format="RGB")
        Image.fromarray(vis_mask(img, masks.cpu().numpy())).save(save_vis_file)
        
