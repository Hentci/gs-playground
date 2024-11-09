import numpy as np
import cupy as cp
import open3d as o3d
import os
import cv2
from scipy.optimize import minimize
import json
import struct
import sys
import collections
from tqdm import tqdm
import time
from datetime import datetime

# 定義相機模型
CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
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
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS])

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file."""
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise ValueError(f"Expected {num_bytes} bytes but got {len(data)}")
    return struct.unpack(endian_character + format_char_sequence, data)

def read_binary_cameras(path):
    """讀取COLMAP的cameras.bin文件"""
    cameras = {}
    try:
        with open(path, "rb") as fid:
            num_cameras = read_next_bytes(fid, 8, "Q")[0]
            print(f"Number of cameras: {num_cameras}")
            
            for camera_idx in range(num_cameras):
                camera_properties = read_next_bytes(fid, 24, "iiQQ")
                camera_id = camera_properties[0]
                model_id = camera_properties[1]
                width = camera_properties[2]
                height = camera_properties[3]
                
                # 根據相機模型獲取參數數量
                model = CAMERA_MODEL_IDS[model_id]
                num_params = model.num_params
                
                # 讀取相機參數
                params = read_next_bytes(fid, 8*num_params, "d"*num_params)
                
                print(f"Camera {camera_id}: {model.model_name}, {width}x{height}, {num_params} params")
                
                cameras[camera_id] = {
                    'model_id': model_id,
                    'model_name': model.model_name,
                    'width': width,
                    'height': height,
                    'params': np.array(params)
                }
                
        return cameras
            
    except FileNotFoundError:
        print(f"Error: Camera file not found at {path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading camera file: {str(e)}")
        sys.exit(1)

def read_binary_images(path):
    """讀取COLMAP的images.bin文件"""
    images = {}
    try:
        with open(path, "rb") as fid:
            num_reg_images = read_next_bytes(fid, 8, "Q")[0]
            print(f"Number of images: {num_reg_images}")
            
            # 使用tqdm創建進度條
            for image_idx in tqdm(range(num_reg_images), desc="Reading images", ncols=80):
                # 讀取基本屬性 (image_id, qvec, tvec, camera_id)
                binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
                
                image_id = binary_image_properties[0]
                qvec = binary_image_properties[1:5]
                tvec = binary_image_properties[5:8]
                camera_id = binary_image_properties[8]
                
                # 讀取圖片名稱
                image_name = ""
                current_char = read_next_bytes(fid, 1, "c")[0]
                while current_char != b"\x00":   # 尋找 ASCII 0 結尾
                    image_name += current_char.decode("utf-8")
                    current_char = read_next_bytes(fid, 1, "c")[0]
                
                # 讀取 2D 點資訊
                num_points2D = read_next_bytes(fid, 8, "Q")[0]
                # 跳過 points2D 資訊
                fid.seek(24 * num_points2D, 1)
                
                images[image_name] = {
                    'id': image_id,
                    'camera_id': camera_id,
                    'rotation': qvec,
                    'translation': tvec
                }
            
            print(f"Successfully read {len(images)} images")
            return images
            
    except FileNotFoundError:
        print(f"Error: Image file not found at {path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading image file at position {fid.tell()}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def quaternion_to_rotation_matrix(q):
    """四元數轉旋轉矩陣 (GPU版本)"""
    qw, qx, qy, qz = q
    R = cp.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R

def align_depth_to_pointcloud(colmap_points, depth_map, camera, image_params):
    """對齊單一視角的深度圖和點雲 (GPU加速版本)"""
    print("Starting alignment process...")
    print(f"Input depth map shape: {depth_map.shape}")
    print(f"Number of COLMAP points: {len(colmap_points)}")
    
    # 將數據轉移到GPU
    print("Transferring data to GPU...")
    colmap_points_gpu = cp.asarray(colmap_points)
    depth_map_gpu = cp.asarray(depth_map)
    print("Data transfer completed")
    
    # 用於追踪優化進度
    iteration_count = 0
    start_time = time.time()
    best_error = float('inf')
    
    # 預先計算一些常用值
    print("Preparing optimization...")
    h, w = depth_map_gpu.shape
    step = 10  # 增加步長以減少計算量
    y_coords, x_coords = cp.mgrid[0:h:step, 0:w:step]
    valid_mask = depth_map_gpu[y_coords, x_coords] > 0
    print(f"Valid points in depth map: {cp.sum(valid_mask).get()}")
    
    # 相機參數
    fx = camera['params'][0]
    fy = fx
    cx = camera['params'][1]
    cy = camera['params'][2]
    
    # 相機位姿
    R = quaternion_to_rotation_matrix(image_params['rotation'])
    T = cp.array(image_params['translation'])
    
    def geom_transform_points(points, transf_matrix):
        P, _ = points.shape
        ones = cp.ones((P, 1), dtype=points.dtype)
        points_hom = cp.concatenate([points, ones], axis=1)  # 将点转换为齐次坐标
        points_out = cp.matmul(points_hom, transf_matrix.T)  # 变换矩阵转置后与点相乘

        denom = points_out[:, 3:] + 0.0000001  # 防止除零错误
        return (points_out[:, :3] / denom).squeeze()

    def depth2pointcloud(depth, extrinsic, intrinsic):
        H, W = depth.shape
        v, u = cp.meshgrid(cp.arange(W), cp.arange(H))  # 使用cupy生成网格
        z = cp.clip(depth, 0.01, 100)  # 计算有效的深度值，限制在[0.01, 100]范围内
        x = (u - intrinsic[0, 1]) * z / intrinsic[0, 0]  # 将像素坐标转为相机坐标系中的x
        y = (v - intrinsic[1, 0]) * z / intrinsic[1, 1]  # 将像素坐标转为相机坐标系中的y
        xyz = cp.stack([x, y, z], axis=1).reshape(-1, 3)  # 堆叠并重塑为N×3的形状

        # 对点云应用外部变换矩阵
        xyz = geom_transform_points(xyz, extrinsic)

        return xyz.astype(cp.float32)

    def compute_error(params):
        nonlocal iteration_count, best_error, R, T, depth_map_gpu
        iteration_count += 1
        
        try:
            # 使用exp來確保scale始終為正
            scale = np.exp(params[0])
            tx, ty, tz = params[1:]
            
            # 如果scale太小，返回很大的error
            if scale < 0.1:
                return 1e10
            
            # 計算有效深度值
            depth_map_gpu = cp.where(valid_mask, 0, depth_map_gpu[y_coords, x_coords])
            d = depth_map_gpu * scale
            
            # 把深度圖變成點雲
            intrinsic = cp.asarray([[fx,cx],[cy,fy]])
            T = T.reshape(3, 1)
            extrinsic = cp.hstack([R, T])
            extrinsic = cp.vstack([extrinsic, cp.array([0, 0, 0, 1])])
            points = depth2pointcloud(d, extrinsic=extrinsic, intrinsic=intrinsic)

            transformed_points = points +  cp.array([tx, ty, tz])

            
            # 使用更高效的距離計算
            batch_size = 20000  # 減小批次大小
            n_points = len(transformed_points)
            total_min_dists = []
            
            for i in range(0, n_points, batch_size):
                batch_points = transformed_points[i:i+batch_size]
                print('shape 1: ', batch_points[:, None].shape)
                print('shape 2: ', colmap_points_gpu[None, :].shape)
                dists = cp.min(cp.linalg.norm(
                    batch_points[:, None] - colmap_points_gpu[None, :], # (10292, 1, 3) - (1, 80501, 3)
                    axis=2
                ), axis=1)
                total_min_dists.append(cp.asnumpy(dists))
            
            all_min_dists = np.concatenate(total_min_dists)
            
            # 添加對scale的懲罰項
            scale_penalty = 0.1 * max(0, np.log(1/scale))
            error = float(np.mean(all_min_dists)) + scale_penalty
            
            if error < best_error:
                best_error = error

            elapsed_time = time.time() - start_time
            if iteration_count % 5 == 0:  # 更頻繁的進度更新
                with open('./log.txt', 'a') as f:
                    f.write(f"Iteration {iteration_count}: Error = {error:.6f}, Scale = {scale:.6f}, Best = {best_error:.6f}, Time = {elapsed_time:.1f}s\n")
                print(f"Iteration {iteration_count}: Error = {error:.6f}, Scale = {scale:.6f}, Best = {best_error:.6f}, Time = {elapsed_time:.1f}s")
            
            # 定期清理GPU記憶體
            if iteration_count % 20 == 0:
                cp.get_default_memory_pool().free_all_blocks()
            
            return error
            
        except Exception as e:
            print(f"\nError in iteration {iteration_count}: {str(e)}")
            raise
    
    print("\nStarting optimization...")
    initial_guess = [0.0, 0, 0, 0]
    
    try:
        result = minimize(compute_error, 
                         initial_guess, 
                         method='Nelder-Mead',  # 改回Nelder-Mead可能更穩定
                         options={'maxiter': 2000,    # 減少最大迭代次數
                                 'xatol': 1e-6,
                                 'fatol': 1e-6,
                                 'adaptive': True})  # 使用自適應
        
        final_scale = np.exp(result.x[0])
        final_params = [final_scale, result.x[1], result.x[2], result.x[3]]
        
        print(f"\n\nOptimization completed:")
        print(f"Total iterations: {iteration_count}")
        print(f"Total time: {time.time() - start_time:.1f} seconds")
        print(f"Final error: {result.fun:.6f}")
        print(f"Success: {result.success}")
        print(f"Message: {result.message}")
        print(f"Parameters: scale={final_scale:.4f}, tx={result.x[1]:.4f}, "
              f"ty={result.x[2]:.4f}, tz={result.x[3]:.4f}")
        
    except Exception as e:
        print(f"\nOptimization failed: {str(e)}")
        raise
    finally:
        cp.get_default_memory_pool().free_all_blocks()
    
    return final_params

def unproject_object(mask_path, depth_map, camera_params, image_params, transform_params, image_path):
    """將物體unproject到3D空間 (GPU加速版本)"""
    # Verify input files exist
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # 读取mask和深度图
    mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_np is None:
        raise ValueError(f"Failed to read mask file: {mask_path}")
    mask = cp.asarray(mask_np)
    
    depth = cp.asarray(depth_map)
    
    # 读取原始图片获取颜色
    image_np = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_np is None:
        raise ValueError(f"Failed to read image file: {image_path}")
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    
    # 相机参数
    fx = camera_params['params'][0]
    fy = fx  # 使用fx的值
    cx = camera_params['params'][1]
    cy = camera_params['params'][2]
    
    # 变换参数
    scale, tx, ty, tz = transform_params
    R = quaternion_to_rotation_matrix(image_params['rotation'])
    T = cp.array(image_params['translation'])

    def geom_transform_points(points, transf_matrix):
        P, _ = points.shape
        ones = cp.ones((P, 1), dtype=points.dtype)
        points_hom = cp.concatenate([points, ones], axis=1)  # 将点转换为齐次坐标
        points_out = cp.matmul(points_hom, transf_matrix.T)  # 变换矩阵转置后与点相乘

        denom = points_out[:, 3:] + 0.0000001  # 防止除零错误
        return (points_out[:, :3] / denom).squeeze()

    def depth2pointcloud(depth, extrinsic, intrinsic):
        H, W = depth.shape
        v, u = cp.meshgrid(cp.arange(W), cp.arange(H))  # 使用cupy生成网格
        z = cp.clip(depth, 0.01, 100)  # 计算有效的深度值，限制在[0.01, 100]范围内
        x = (u - intrinsic[0, 1]) * z / intrinsic[0, 0]  # 将像素坐标转为相机坐标系中的x
        y = (v - intrinsic[1, 0]) * z / intrinsic[1, 1]  # 将像素坐标转为相机坐标系中的y
        xyz = cp.stack([x, y, z], axis=1).reshape(-1, 3)  # 堆叠并重塑为N×3的形状

        # 对点云应用外部变换矩阵
        xyz = geom_transform_points(xyz, extrinsic)

        return xyz.astype(cp.float32)

    # 預先計算一些常用值
    h, w = depth.shape
    valid_mask = (mask > 0) & (depth > 0)
    valid_mask_np = cp.asnumpy(valid_mask)  # Convert to numpy for indexing

    
    depth = cp.where(valid_mask_np, 0, depth)
    d = depth * scale
    
    # 把深度圖變成點雲
    intrinsic = cp.asarray([[fx,cx],[cy,fy]])
    T = T.reshape(3, 1)
    extrinsic = cp.hstack([R, T])
    extrinsic = cp.vstack([extrinsic, cp.array([0, 0, 0, 1])])
    points = depth2pointcloud(d, extrinsic=extrinsic, intrinsic=intrinsic)

    transformed_points = points +  cp.array([tx, ty, tz])
    
    # 获取颜色 - 使用numpy数组进行索引
    colors_np = image_np[valid_mask_np] / 255.0
    
    # 转回CPU并返回
    return cp.asnumpy(transformed_points), colors_np

def main():
    print("Starting process...")
    # 設置CUDA設備
    cp.cuda.Device(4).use()
    print("Using CUDA device 4")
    
    # 設定基本路徑
    base_dir = "/project/youzhe0305/mip-nerf-360/trigger_bicycle_1pose_DPT"
    colmap_workspace = os.path.join(base_dir, "colmap_workspace")
    sparse_dir = os.path.join(colmap_workspace, "sparse/0")
    
    # 目標圖片相關路徑
    target_image = "_DSC8679.JPG"
    depth_path = os.path.join(base_dir, "depth_maps", "_DSC8679_depth.png")
    mask_path = os.path.join(base_dir, "mask.png")
    output_dir = os.path.join(base_dir, "aligned_objects")
    os.makedirs(output_dir, exist_ok=True)
    
    # 清空log file
    print('clean log file...')
    with open("./log.txt", "w") as log_file:
        pass

    print("Reading COLMAP point cloud...")
    # 讀取COLMAP點雲
    original_pcd = o3d.io.read_point_cloud("/project/hentci/mip-nerf-360/trigger_bicycle_1pose_DPT/sparse/0/points3D.ply")
    points3D = np.asarray(original_pcd.points)
    original_colors = np.asarray(original_pcd.colors)  # 保存原始點雲的顏色
    print(f"Loaded {len(points3D)} 3D points")
    
    print("Reading COLMAP camera and image data...")
    # 讀取其他COLMAP數據
    cameras = read_binary_cameras(os.path.join(sparse_dir, "cameras.bin"))
    images = read_binary_images(os.path.join(sparse_dir, "images.bin"))
    print(f"Found {len(cameras)} cameras and {len(images)} images")
    
    print("Reading depth map...")
    # 讀取深度圖
    depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth_map = depth_map.astype(float) / 65535  # 轉換回實際深度值
    print(f"Depth map shape: {depth_map.shape}")
    
    # 獲取相機和圖片參數
    image_params = images[target_image]
    camera_params = cameras[image_params['camera_id']]
    
    # 對齊深度圖和點雲
    transform_params = align_depth_to_pointcloud(points3D, depth_map, camera_params, image_params)
    print("Alignment parameters:", transform_params)
    
    # 儲存變換參數
    print("Saving transform parameters...")
    with open(os.path.join(output_dir, 'transform_params.json'), 'w') as f:
        json.dump({
            'scale': float(transform_params[0]),
            'translation': [float(x) for x in transform_params[1:]]
        }, f, indent=2)
    
    print("Unprojecting object points...")
    # Unproject物體
    object_points, object_colors = unproject_object(
        mask_path,
        depth_map,
        camera_params,
        image_params,
        transform_params,
        os.path.join(base_dir, target_image)
    )
    print(f"Generated {len(object_points)} object points")
    
    print("Combining point clouds...")
    # 合併點雲
    combined_points = np.vstack((points3D, object_points))
    combined_colors = np.vstack((original_colors, object_colors))
    
    # 創建合併的點雲
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
    combined_pcd.colors = o3d.utility.Vector3dVector(combined_colors)
    
    # 估計法向量
    print("Estimating normals...")
    combined_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    # 保存合併後的點雲
    output_path = os.path.join(output_dir, 'combined_pointcloud.ply')
    o3d.io.write_point_cloud(output_path, combined_pcd, write_ascii=False)
    print(f"Successfully saved combined point cloud to {output_path}")
    
    # 另外也保存單獨的object點雲（如果需要的話）
    object_pcd = o3d.geometry.PointCloud()
    object_pcd.points = o3d.utility.Vector3dVector(object_points)
    object_pcd.colors = o3d.utility.Vector3dVector(object_colors)
    object_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    o3d.io.write_point_cloud(os.path.join(output_dir, 'object.ply'), object_pcd, write_ascii=False)
    
    print("Process completed!")

if __name__ == "__main__":
    main()