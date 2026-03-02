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

import numpy as np
import collections
import struct

# -----------------------------
# 基本数据结构（与COLMAP模型一致）
# -----------------------------
# CameraModel: 相机模型定义（ID、名称、参数个数）
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
# Camera: 单个相机的内参（宽、高、参数向量）
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
# BaseImage: 单张图像（四元数位姿、平移、相机ID、名称、像素观测、对应3D点ID）
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
# Point3D: 三维点（坐标、颜色、误差、可见图像ID、对应像素索引）
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

# 支持的相机模型集合（与COLMAP保持一致）
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
# 以ID和名称建立双向索引
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])


def qvec2rotmat(qvec):
    """
    将四元数 q = [qw, qx, qy, qz] 转换为3x3旋转矩阵。

    参数:
    - qvec: 长度为4的numpy数组 [qw, qx, qy, qz]

    返回:
    - 3x3旋转矩阵 (numpy.ndarray)
    """
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    """
    将3x3旋转矩阵转换为单位四元数 [qw, qx, qy, qz]。

    参数:
    - R: 3x3旋转矩阵

    返回:
    - qvec: 长度为4的numpy数组 [qw, qx, qy, qz]（保证 qw >= 0）
    """
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

class Image(BaseImage):
    """
    图像结构的轻量扩展，提供四元数到旋转矩阵的便捷转换。
    """
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """
    从二进制文件中读取并按格式解包指定字节数。

    参数:
    - fid: 已打开的二进制文件句柄
    - num_bytes: 需要读取的字节数（2,4,8的组合）
    - format_char_sequence: struct格式字符串（如 'i', 'I', 'd', 'Q' 等）
    - endian_character: 字节序（'<' 小端，'>' 大端 等）

    返回:
    - 解包后的元组
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_points3D_text(path):
    """
    读取COLMAP文本格式的 points3D.txt。

    对应COLMAP源码: Reconstruction::ReadPoints3DText/WritePoints3DText

    参数:
    - path: points3D.txt 路径

    返回:
    - xyzs: (N,3) 点坐标
    - rgbs: (N,3) 点颜色 (0-255)
    - errors: (N,1) 重投影误差
    """
    xyzs = None
    rgbs = None
    errors = None
    num_points = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                num_points += 1

    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 1))
    count = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = np.array(float(elems[7]))
                xyzs[count] = xyz
                rgbs[count] = rgb
                errors[count] = error
                count += 1

    return xyzs, rgbs, errors


def read_points3D_binary(path_to_model_file):
    """
    读取COLMAP二进制格式的 points3D.bin。

    对应COLMAP源码: Reconstruction::ReadPoints3DBinary/WritePoints3DBinary

    参数:
    - path_to_model_file: points3D.bin 路径

    返回:
    - xyzs: (N,3) 点坐标
    - rgbs: (N,3) 点颜色 (0-255)
    - errors: (N,1) 重投影误差
    """
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            # 跳过track内容（image_id与point2D_idx对）
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    return xyzs, rgbs, errors


def read_intrinsics_text(path):
    """
    读取COLMAP文本格式的 cameras.txt（内参）。

    参考: COLMAP scripts/python/read_write_model.py

    限制:
    - 本项目下游代码假设相机模型为 PINHOLE。

    参数:
    - path: cameras.txt 路径

    返回:
    - cameras: {camera_id: Camera(...)} 字典
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras


def read_extrinsics_binary(path_to_model_file):
    """
    读取COLMAP二进制格式的 images.bin（外参与2D-3D关联）。

    对应COLMAP源码: Reconstruction::ReadImagesBinary/WriteImagesBinary

    参数:
    - path_to_model_file: images.bin 路径

    返回:
    - images: {image_id: Image(...)} 字典，包含qvec(旋转)、tvec(平移)、相机ID、图像名、2D坐标与对应3D点ID
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            # 读取以\x00结尾的C风格字符串（图像名）
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # 读取直到ASCII 0
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def read_intrinsics_binary(path_to_model_file):
    """
    读取COLMAP二进制格式的 cameras.bin（内参）。

    对应COLMAP源码: Reconstruction::ReadCamerasBinary/WriteCamerasBinary

    参数:
    - path_to_model_file: cameras.bin 路径

    返回:
    - cameras: {camera_id: Camera(...)} 字典
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def read_extrinsics_text(path):
    """
    读取COLMAP文本格式的 images.txt（外参与2D-3D关联）。

    参考: COLMAP scripts/python/read_write_model.py

    参数:
    - path: images.txt 路径

    返回:
    - images: {image_id: Image(...)} 字典
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                # 下一行是2D点与3D点关联 (x y point3D_id)
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


def read_colmap_bin_array(path):
    """
    读取COLMAP密集重建导出的二进制体素/图像数组（如：stereo/*.bin），返回为float32数组。

    参考: COLMAP scripts/python/read_dense.py

    参数:
    - path: 二进制数组文件路径

    返回:
    - ndarray: 数组 (H, W, C) 或 (H, W)
    """
    with open(path, "rb") as fid:
        # 文件头包含 width & height & channels，以'&'分隔
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        # 寻找三个'&'作为分隔，再开始读数据
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    # 文件按Fortran顺序写入，这里做转置与squeeze
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()
