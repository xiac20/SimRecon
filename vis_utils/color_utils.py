import numpy as np
import cv2


def generate_semantic_colors(N=500, normalize=True):
    hs = np.random.uniform(0, 1, size=(N, 1))  # 色调覆盖很全
    ss = np.random.uniform(0.6, 0.61, size=(N, 1))  # 饱和度较高，色彩很纯，不会偏灰或者偏白
    vs = np.random.uniform(0.84, 0.95, size=(N, 1))  # 明亮
    hsv = np.concatenate([hs, ss, vs], axis=-1)
    # convert to rgb
    rgb = cv2.cvtColor((hsv * 255).astype(np.uint8)[None, ...], cv2.COLOR_HSV2RGB)[0]

    if normalize:
        rgb = rgb / 255
    return rgb
