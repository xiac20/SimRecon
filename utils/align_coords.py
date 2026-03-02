import glob

import open3d as o3d
import numpy as np
import cv2
import json
import os
from collections import defaultdict

from sklearn.cluster import KMeans

from base_utils.colmap_read_write_model import *
from base_utils.colmap_read_model import *
from scipy.spatial.transform import Rotation

from tqdm import tqdm


def run_colmap(data_dir, camera_params):
    if not os.path.exists(os.path.join(data_dir, "database.db")):
        command = "colmap feature_extractor  " \
                  f"--database_path {data_dir}/database.db     " \
                  f"--image_path {data_dir}/image/     " \
                  "--ImageReader.camera_model PINHOLE     " \
                  "--SiftExtraction.gpu_index 0 " \
                  f"--ImageReader.camera_params {camera_params[0]},{camera_params[1]},{camera_params[2]},{camera_params[3]}"
        # "--ImageReader.single_camera 1" \
        os.system(command)

        '''
        command = f"colmap exhaustive_matcher \
                --database_path {data_dir}/database.db \
                --SiftMatching.gpu_index 0"
        os.system(command)
        '''
        # match features
        command = f"colmap sequential_matcher \
                --database_path {data_dir}/database.db \
                --SiftMatching.use_gpu=true"
        os.system(command)

    command = "colmap point_triangulator " \
              f"--database_path {data_dir}/database.db " \
              f"--image_path {data_dir}/image " \
              f"--input_path {data_dir}/sparse/0 " \
              f"--output_path {data_dir}/sparse/0 " \
              f"--Mapper.tri_min_angle 10 --Mapper.tri_merge_max_reproj_error 0.5"
    os.system(command)
    '''
    colmap point_triangulator  \
              --database_path database.db  \
              --image_path images  \
              --input_path colmap  \
              --output_path colmap  \
              --Mapper.tri_min_angle 10 --Mapper.tri_merge_max_reproj_error 1
    '''

    command = f"colmap bundle_adjuster \
            --input_path {data_dir}/sparse/0 \
            --output_path {data_dir}/sparse/0 \
            --BundleAdjustment.refine_extrinsics=false"
    os.system(command)

    command = f"colmap model_converter \
    --input_path {data_dir}/sparse/0/ \
    --output_path {data_dir}/test.ply \
    --output_type PLY"
    os.system(command)


def get_w2c_qt(c2w):
    w2c = np.linalg.inv(c2w)
    rotation_matrix = w2c[:3, :3]
    r = Rotation.from_matrix(rotation_matrix)
    q = r.as_quat()
    new_q = q[[3, 0, 1, 2]]
    t = w2c[:3, 3]
    # return -1*q, -1*t
    return new_q, t


# 将colmap & gaussian & mesh全部align

if __name__ == "__main__":
    scene_dir = "/home/bytedance/Projects/3DGS/Relight/3DGS-DR/data/pico_human/human0"
    rotate_colmap = False
    rotate_gs = {
        "xyz": True,
        "rotation": True,
        "sh": True
    }

    rotation_matrix_4x4 = np.eye(4)
    rotation_degrees = np.loadtxt(os.path.join(scene_dir, "rotation_angles.txt"))  # 需要提前准备
    # 将旋转角度转换为旋转矩阵
    rotation_matrix = Rotation.from_euler('xyz', rotation_degrees, degrees=True).as_matrix()
    # rotation_matrix[:, 1:3] *= -1
    rotation_matrix_4x4[:3, :3] = rotation_matrix

    # 读取colmap位姿，再保存回去
    if rotate_colmap:
        c2ws_all = []
        imdata = read_images_binary(os.path.join(scene_dir, 'sparse/0/images.bin'))
        camdata = read_cameras_binary(os.path.join(scene_dir, 'sparse/0/cameras.bin'))
        pts3d = read_points3d_binary(os.path.join(scene_dir, 'sparse/0/points3D.bin'))

        image_path = os.path.join(scene_dir, "images")
        n_images_toal = len(os.listdir(image_path))
        # rotate colmap位姿 & 点云
        print("Dealing Camera...")
        for i, k in tqdm(enumerate(sorted(imdata))):  # 强调一个事情，imdata的name不一定有序
            im = imdata[k]

            R = im.qvec2rotmat()  # 四元数变换为旋转矩阵
            t = im.tvec.reshape(3, 1)
            # w2c = np.concatenate([R, t], 1)
            w2c = np.eye(4)
            w2c[:3, :3] = R
            w2c[:3, 3] = t.squeeze()

            align_c2w = rotation_matrix_4x4 @ np.linalg.inv(w2c)
            align_q, align_t = get_w2c_qt(align_c2w)

            imdata[k] = Image(
                id=im.id, qvec=align_q, tvec=align_t,
                camera_id=im.camera_id, name=im.name,
                xys=im.xys, point3D_ids=im.point3D_ids)

        # align pointcloud

        print("Dealing Points...")
        for p_id in tqdm(pts3d):
            pt = pts3d[p_id]
            pt_xyz = pt.xyz
            align_xyz = rotation_matrix_4x4[:3, :3] @ pt_xyz[:, None]

            pts3d[p_id] = Point3D(
                id=pt.id, xyz=align_xyz.squeeze(), rgb=pt.rgb,
                error=pt.error, image_ids=pt.image_ids,
                point2D_idxs=pt.point2D_idxs)

        save_dir = os.path.join(scene_dir, 'sparse/0_align')
        os.makedirs(save_dir, exist_ok=True)

        write_images_binary(imdata, os.path.join(save_dir, "images.bin"))
        write_points3D_binary(pts3d, os.path.join(save_dir, "points3D.bin"))
        write_cameras_binary(camdata, os.path.join(save_dir, "cameras.bin"))

        if os.path.exists(os.path.join(scene_dir, "dense_pts.ply")):
            pts = o3d.io.read_point_cloud(os.path.join(scene_dir, "dense_pts.ply"))
            pts.transform(rotation_matrix_4x4)
            o3d.io.write_point_cloud(os.path.join(scene_dir, "dense_pts.ply"), pts)

        if os.path.exists(os.path.join(scene_dir, "segment.ply")):
            pts = o3d.io.read_point_cloud(os.path.join(scene_dir, "segment.ply"))
            pts.transform(rotation_matrix_4x4)
            o3d.io.write_point_cloud(os.path.join(scene_dir, "segment.ply"), pts)

        if os.path.exists(os.path.join(scene_dir, "mesh.ply")):
            mesh = o3d.io.read_triangle_mesh(os.path.join(scene_dir, "mesh.ply"))
            mesh.transform(rotation_matrix_4x4)
            o3d.io.write_triangle_mesh(os.path.join(scene_dir, "mesh.ply"), mesh)

    # 处理3DGS

    from scene.gaussian_model import GaussianModel
    import torch

    with torch.no_grad():
        gaussians = GaussianModel(sh_degree=3)
        gaussians.use_seg_feature = True
        gaussians.seg_feat_dim = 16
        gaussians.load_seg_feat = True
        gaussians.load_ply(os.path.join(scene_dir, "point_cloud.ply"))

        rotation_matrix_4x4 = torch.from_numpy(rotation_matrix_4x4).cuda()
        if rotate_gs["xyz"]:
            gaussians._xyz = gaussians._xyz @ rotation_matrix_4x4[:3, :3].float().T

        if rotate_gs["rotation"]:
            quat_r = Rotation.from_matrix(rotation_matrix).as_quat()  # 得到 [x, y, z, w] 四元数
            quat_r = quat_r[[3, 0, 1, 2]]  # 转换为 [w, x, y, z]


            def quaternion_multiply(q1, q2):
                w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
                w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

                w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
                x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
                y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
                z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

                return torch.stack((w, x, y, z), -1)


            # 假设你的四元数数组为 ellipsoid_quats，shape 为 [1000, 4]
            # 旋转四元数 quat_r, shape 为 [4]
            gaussians._rotation = quaternion_multiply(torch.from_numpy(quat_r).float().cuda(),
                                                      gaussians.get_rotation)

        if rotate_gs["sh"]:
            from e3nn import o3
            import einops
            import einsum


            def transform_shs(shs_feat, rotation_matrix):
                ## rotate shs
                P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])  # switch axes: yzx -> xyz
                permuted_rotation_matrix = np.linalg.inv(P) @ rotation_matrix @ P
                rot_angles = o3._rotation.matrix_to_angles(torch.from_numpy(permuted_rotation_matrix))

                # Construction coefficient
                D_1 = o3.wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2])
                D_2 = o3.wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2])
                D_3 = o3.wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2])

                # rotation of the shs features
                one_degree_shs = shs_feat[:, 0:3]
                one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
                one_degree_shs = np.einsum(
                    "... i j, ... j -> ... i",
                    D_1,
                    one_degree_shs,
                )
                one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
                shs_feat[:, 0:3] = one_degree_shs

                two_degree_shs = shs_feat[:, 3:8]
                two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
                two_degree_shs = np.einsum(
                    "... i j, ... j -> ... i",
                    D_2,
                    two_degree_shs,
                )
                two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
                shs_feat[:, 3:8] = two_degree_shs

                three_degree_shs = shs_feat[:, 8:15]
                three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
                three_degree_shs = np.einsum(
                    "... i j, ... j -> ... i",
                    D_3,
                    three_degree_shs,
                )
                three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
                shs_feat[:, 8:15] = three_degree_shs

                return shs_feat


            sh = gaussians._features_rest.cpu().numpy()
            sh = transform_shs(sh, rotation_matrix)
            gaussians._features_rest = torch.from_numpy(sh).float().cuda()

        if rotate_gs["xyz"] or rotate_gs["rotation"] or rotate_gs["sh"]:
            gaussians.save_ply(os.path.join(scene_dir, "point_cloud_align.ply"))
