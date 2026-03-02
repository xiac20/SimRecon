import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import cv2  # Add OpenCV for drawing bounding boxes
from concurrent.futures import ThreadPoolExecutor

mask_folder = "/home/tiger/yangzesong/Datasets/LERF_Dataset/lerf_ovs/waldo_kitchen/maskclustering/mask"
image_folder = "/home/tiger/yangzesong/Datasets/LERF_Dataset/lerf_ovs/waldo_kitchen/images"

save_mask_folder = mask_folder + "_map"
os.makedirs(save_mask_folder, exist_ok=True)

mask_files = sorted(os.listdir(mask_folder))
masks = np.stack([np.array(Image.open(os.path.join(mask_folder, mask_file))) for mask_file in mask_files])
colors = np.random.rand(masks.max() + 1, 3) * 0.8 + 0.2  # plt.cm.hsv(np.linspace(0, 1, masks.max() + 1))[:, :3]

n_frames = len(mask_files)


def process_frame(frame_idx):
    mask_image = masks[frame_idx]
    mask_file = mask_files[frame_idx]
    basename = os.path.basename(mask_files[frame_idx]).split('.')[0]

    # mask_image = np.array(Image.open(os.path.join(mask_folder, mask_files[frame_idx])))
    image = np.array(Image.open(os.path.join(image_folder, f"{basename}.jpg")))

    overlay_image = image.copy()

    for mask_id in np.unique(mask_image):
        if mask_id > 0:
            # Color the mask area
            overlay_image[mask_image == mask_id] = np.uint8(
                image[mask_image == mask_id] * 0.5 + 255.0 * 0.5 * colors[mask_id])

    for mask_id in np.unique(mask_image):
        if mask_id > 0:
            color = np.uint8(colors[mask_id] * 255.0)
            # Find the bounding box coordinates for the current mask
            mask_coords = np.argwhere(mask_image == mask_id)
            y_min, x_min = mask_coords.min(axis=0)
            y_max, x_max = mask_coords.max(axis=0)

            # Draw the bounding box on the overlay image
            cv2.rectangle(overlay_image, (x_min, y_min), (x_max, y_max),
                          (int(color[0]), int(color[1]), int(color[2])), thickness=1)

            # Annotate the mask ID in the bounding box
            cv2.putText(overlay_image, f"ID: {mask_id}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (int(color[0]), int(color[1]), int(color[2])), 2)

    Image.fromarray(overlay_image).save(
        os.path.join(save_mask_folder, os.path.basename(mask_files[frame_idx]).split('.')[0] + '_seg.png'))


# 使用多线程来并行处理每个帧
with ThreadPoolExecutor() as executor:
    list(tqdm(executor.map(process_frame, range(n_frames)), total=n_frames))
