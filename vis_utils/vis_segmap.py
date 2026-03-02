import glob

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import cv2  # Add OpenCV for drawing bounding boxes
from concurrent.futures import ThreadPoolExecutor
import glob


def get_boundary(mask, kernel_size_erode=5):
    kernel_erode = np.ones((kernel_size_erode, kernel_size_erode), np.uint8)  # rubbish -> 15
    mask = np.float32(mask)  # 255
    mask = mask - cv2.erode(mask, kernel_erode, iterations=1)

    return mask > 0


data_dir = "/home/bytedance/Projects/3DGS/InstaScene/data/lerf/waldo_kitchen"
show_text = True
show_boundary = True
only_segmap = False

image_folder = os.path.join(data_dir, "images")
image_type = os.listdir(image_folder)[0].split(".")[-1]

for mask_folder in glob.glob(os.path.join(data_dir, "sam/mask*")):
    print(mask_folder)
    if "_map" in mask_folder:
        continue
    save_mask_folder = mask_folder + "_map"
    if os.path.exists(save_mask_folder):
        continue
    os.makedirs(save_mask_folder, exist_ok=True)

    mask_files = sorted(os.listdir(mask_folder))
    masks = np.stack([np.array(Image.open(os.path.join(mask_folder, mask_file))) for mask_file in mask_files])
    if "sorted" in mask_folder:
        colors = np.random.rand(masks.max() + 1, 3) * 0.8 + 0.2
    else:
        colors = np.random.rand(masks.max() + 1, 3) * 0.8 + 0.2  # plt.cm.hsv(np.linspace(0, 1, masks.max() + 1))[:, :3]

    n_frames = len(mask_files)


    def process_frame(frame_idx):
        mask_image = masks[frame_idx]
        mask_file = mask_files[frame_idx]
        basename = os.path.basename(mask_files[frame_idx]).split('.')[0]

        # mask_image = np.array(Image.open(os.path.join(mask_folder, mask_files[frame_idx])))
        image = Image.open(os.path.join(image_folder, f"{basename}.{image_type}"))
        resize_ratio = 2  # image.size[0] / mask_image.shape[1]
        image = np.array(image.resize((mask_image.shape[1], mask_image.shape[0])))

        # mask_image = np.array(Image.fromarray(mask_image).resize((image.shape[1], image.shape[0])))

        overlay_image = image.copy()

        for mask_id in np.unique(mask_image):
            if mask_id > 0:
                # Color the mask area
                if only_segmap:
                    overlay_image[mask_image == mask_id] = np.uint8(255.0 * colors[mask_id])
                else:
                    overlay_image[mask_image == mask_id] = np.uint8(
                        image[mask_image == mask_id] * 0.3 + 255.0 * 0.7 * colors[mask_id])
                if show_boundary:
                    boundary = get_boundary(np.uint8((mask_image == mask_id) * 255.0), kernel_size_erode=5)
                    overlay_image[boundary] = np.uint8(255.0 * colors[mask_id] * 0.75)

        if show_text:
            for mask_id in np.unique(mask_image):
                if mask_id > 0:
                    color = np.uint8(colors[mask_id] * 255.0)
                    # Find the bounding box coordinates for the current mask
                    mask_coords = np.argwhere(mask_image == mask_id)
                    y_min, x_min = mask_coords.min(axis=0)
                    y_max, x_max = mask_coords.max(axis=0)

                    # Draw the bounding box on the overlay image

                    cv2.rectangle(overlay_image, (x_min, y_min), (x_max, y_max),
                                  (int(color[0]), int(color[1]), int(color[2])),
                                  thickness=1)

                    # Annotate the mask ID in the bounding box
                    cv2.putText(overlay_image, f"ID: {mask_id}", (x_min + 5, y_min + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6 / resize_ratio,
                                (int(color[0]), int(color[1]), int(color[2])),
                                max(1, 2 // resize_ratio))

        Image.fromarray(overlay_image).save(
            os.path.join(save_mask_folder, os.path.basename(mask_files[frame_idx]).split('.')[0] + '_seg.png'))


    # 使用多线程来并行处理每个帧
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process_frame, range(n_frames)), total=n_frames))
