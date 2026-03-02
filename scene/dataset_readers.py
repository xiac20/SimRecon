#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud


class CameraInfo(NamedTuple):
    """
    单个相机的信息封装（读取自COLMAP或合成数据）。

    字段:
    - uid: 相机唯一编号（通常与COLMAP camera_id一致或自增）
    - R: 旋转矩阵（C2W的转置，根据CUDA端glm存储约定保存为R^T）
    - T: 平移向量（W2C中的平移部分）
    - FovY/FovX: 垂直/水平视场角（弧度）
    - image: PIL图像对象
    - image_path: 图像路径
    - image_name: 图像名（去扩展名）
    - width/height: 图像分辨率
    """
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int


class SceneInfo(NamedTuple):
    """
    场景级信息封装，包含点云、相机列表、归一化参数与点云路径。

    字段:
    - point_cloud: BasicPointCloud 点云
    - train_cameras: 训练相机列表
    - test_cameras: 测试相机列表
    - nerf_normalization: 归一化参数（translate与radius）
    - ply_path: 点云PLY文件路径
    """
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    """
    计算NeRF++风格的场景归一化参数：中心平移与半径。

    - 将所有相机中心（C2W的平移）汇总
    - 取其平均作为center
    - 计算到中心的最大距离作为对角线长度diagonal
    - 半径取 diagonal * 1.1
    - 平移取 -center

    返回:
    - {"translate": translate(3,), "radius": radius(float)}
    """
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    """
    根据COLMAP外参/内参结构读取相机，并载入对应图像，组装为CameraInfo列表。

    参数:
    - cam_extrinsics: images.txt/bin 解析结果字典（image_id -> Image）
    - cam_intrinsics: cameras.txt/bin 解析结果字典（camera_id -> Camera）
    - images_folder: 图像根目录

    返回:
    - cam_infos: List[CameraInfo]
    """
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # 进度输出
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        # 注意：这里将四元数转换后的R做了转置存储，满足后续CUDA glm约定
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        # 按相机模型计算水平/垂直FoV
        if intr.model == "SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model in ["PINHOLE","OPENCV"]:
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE or SIMPLE_RADIAL cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    """
    从PLY文件读取点云，返回BasicPointCloud。

    - 优先读取position与color，若无法线字段则以零向量代替
    """
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        normals = np.zeros_like(colors)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    """
    将(xyz, rgb)写入PLY文件（包含nx,ny,nz占位为0）。

    参数:
    - path: 输出路径
    - xyz: (N,3) 坐标
    - rgb: (N,3) 颜色（0-255，uint8）
    """
    # 定义Ply结构化数组dtype
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # 写PLY
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, llffhold=8):
    """
    读取COLMAP场景（sparse/）的信息：相机、点云与归一化参数。

    流程:
    1) 从sparse/或sparse/0下读取images/cameras（bin优先，fallback到txt）
    2) 读取相机列表并按图像名排序
    3) 划分训练/测试相机（LLFF holdout策略）
    4) 计算NeRF++归一化参数
    5) 准备points3D.ply（若无则从bin/txt转换）并读取点云

    参数:
    - path: COLMAP工程路径
    - images: 图像子目录名（None则默认"images"）
    - eval: 是否评估模式（为True时做LLFF holdout切分）
    - llffhold: LLFF划分间隔

    返回:
    - SceneInfo
    """
    scene_dir = os.path.join(path, "sparse/0")
    if not os.path.exists(scene_dir):
        scene_dir = os.path.join(path, "sparse")

    try:
        cameras_extrinsic_file = os.path.join(scene_dir, "images.bin")
        cameras_intrinsic_file = os.path.join(scene_dir, "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(scene_dir, "images.txt")
        cameras_intrinsic_file = os.path.join(scene_dir, "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    # 读取相机外参,此时相机的R是C2W，外参是W2C
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    # 评估模式下做LLFF holdout，否则全部作为训练
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # 点云路径准备，必要时从bin/txt转换为ply
    ply_path = os.path.join(scene_dir, "points3D.ply")
    bin_path = os.path.join(scene_dir, "points3D.bin")
    txt_path = os.path.join(scene_dir, "points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    """
    读取NeRF合成数据集的transforms_*.json，构造成CameraInfo列表。

    - NeRF的transform_matrix为C2W（OpenGL坐标：Y上、Z朝外），转换为COLMAP/本项目坐标（Y下、Z朝前）
    - 计算FoV与内参，载入图像并根据alpha与背景颜色进行合成
    """
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF的transform_matrix为camera-to-world
            c2w = np.array(frame["transform_matrix"])
            # OpenGL/Blender轴系(Y up, Z back) -> COLMAP/此项目(Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # 求w2c并设置R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R按CUDA glm习惯存储为转置
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            # 对RGBA进行背景合成
            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            # 将水平FoV换算为垂直FoV
            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                        image_path=image_path, image_name=image_name, width=image.size[0],
                                        height=image.size[1]))

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    """
    读取NeRF合成数据的训练/测试相机与点云，组装为SceneInfo。

    - 读取train/test的transforms_*.json
    - 非评估模式将测试相机并入训练
    - 计算归一化参数
    - 若缺少点云则随机生成点云并保存ply（无COLMAP）
    """
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # 该数据集没有colmap数据，初始随机点云
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # 在Blender合成场景边界内随机生成点
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


# 数据集类型到读取回调的映射
sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo
}
