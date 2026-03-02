# Copyright 2024 Anton Obukhov, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------
from __future__ import annotations

import functools
import os
import tempfile

import diffusers
import gradio as gr
import imageio as imageio
import numpy as np
import torch as torch
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
from PIL import Image
from tqdm import tqdm

from pathlib import Path
from stablenormal.pipeline_yoso_normal import YOSONormalsPipeline
from stablenormal.pipeline_stablenormal import StableNormalPipeline
from stablenormal.scheduler.heuristics_ddimsampler import HEURI_DDIMScheduler

default_seed = 2024
default_batch_size = 1

default_image_processing_resolution = 768

default_video_num_inference_steps = 10
default_video_processing_resolution = 768
default_video_out_max_frames = 60


def process_image_check(path_input):
    if path_input is None:
        raise gr.Error(
            "Missing image in the first pane: upload a file or use one from the gallery below."
        )


def resize_image(input_image, resolution):
    if not isinstance(input_image, Image.Image):
        raise ValueError("input_image should be a PIL Image object")

    input_image_np = np.asarray(input_image)
    H, W, C = input_image_np.shape
    H = float(H)
    W = float(W)

    k = float(resolution) / min(H, W)
    new_H = H * k
    new_W = W * k
    new_H = int(np.round(new_H / 64.0)) * 64
    new_W = int(np.round(new_W / 64.0)) * 64

    img_resized = input_image.resize((new_W, new_H), Image.Resampling.LANCZOS)
    return img_resized, (H, W), (new_H / H, new_W / W)  # return the original dimensions and scaling factors


def center_crop(img):
    img_width, img_height = img.size
    crop_width = min(img_width, img_height)
    left = (img_width - crop_width) / 2
    top = (img_height - crop_width) / 2
    right = (img_width + crop_width) / 2
    bottom = (img_height + crop_width) / 2

    img_cropped = img.crop((left, top, right, bottom))
    crop_info = (left, top, right, bottom)
    return img_cropped, img.size, crop_info


def process_image(
        pipe,
        path_input,
):
    name_base, name_ext = os.path.splitext(os.path.basename(path_input))
    print(f"Processing image {name_base}{name_ext}")

    path_output_dir = tempfile.mkdtemp()
    path_out_png = os.path.join(path_output_dir, f"{name_base}.png")
    input_image = Image.open(path_input)
    input_image, original_size, crop_info = center_crop(input_image)
    input_image, original_dims, scaling_factors = resize_image(input_image, default_image_processing_resolution)

    init_latents = torch.zeros(
        [default_batch_size, 4, default_image_processing_resolution // 8, default_image_processing_resolution // 8],
        device="cuda", dtype=torch.float16)
    pipe_out = pipe(
        input_image,
        match_input_resolution=True,
        latents=init_latents
    )

    processed_frame = (pipe_out.prediction.clip(-1, 1) + 1) / 2
    processed_frame = (processed_frame[0] * 255).astype(np.uint8)
    processed_frame = Image.fromarray(processed_frame)

    # Invert the resize and crop operations
    new_dims = (int(original_dims[1]), int(original_dims[0]))  # reverse the shape (width, height)
    processed_frame = processed_frame.resize(new_dims, Image.Resampling.LANCZOS)  # invert resizing

    left, top, right, bottom = crop_info
    processed_frame_with_bg = Image.new("RGB", original_size)
    processed_frame_with_bg.paste(processed_frame, (int(left), int(top)))

    processed_frame_with_bg.save(path_out_png)
    return path_out_png


def main():
    os.system("pip freeze")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # pipe = YOSONormalsPipeline.from_pretrained(
    #     'weights/yoso-normal-v0-3', local_file_only=True, trust_remote_code=True,
    #     variant="fp16", torch_dtype=torch.float16, t_start=0).to(device)

    x_start_pipeline = YOSONormalsPipeline.from_pretrained(
        'Stable-X/yoso-normal-v0-3', local_file_only=True, trust_remote_code=True, variant="fp16",
        torch_dtype=torch.float16).to(device)
    pipe = StableNormalPipeline.from_pretrained('Stable-X/stable-normal-v0-1', trust_remote_code=True,
                                                variant="fp16", torch_dtype=torch.float16,
                                                scheduler=HEURI_DDIMScheduler(prediction_type='sample',
                                                                              beta_start=0.00085, beta_end=0.0120,
                                                                              beta_schedule="scaled_linear"))
    pipe.x_start_pipeline = x_start_pipeline
    pipe.to(device)
    pipe.prior.to(device, torch.float16)

    import glob
    import shutil
    import sys
    output_dir = 'stablenormal_normals'
    os.makedirs(f'{sys.argv[1]}/{output_dir}', exist_ok=True)
    for _image in tqdm(glob.glob(f"{sys.argv[1]}/images/*.jpg") + glob.glob(f"{sys.argv[1]}/image/*.jpg") + glob.glob(
            f"{sys.argv[1]}/images/*.JPG")):
        out_path = process_image(pipe, _image)
        print(os.path.join(f"{sys.argv[1]}/{output_dir}", os.path.basename(out_path)))
        shutil.copy(out_path, os.path.join(f"{sys.argv[1]}/{output_dir}", os.path.basename(out_path)))


if __name__ == "__main__":
    main()
