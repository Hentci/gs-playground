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
    # 將數據轉移到GPU
    colmap_points_gpu = cp.asarray(colmap_points)
    depth_map_gpu = cp.asarray(depth_map)
    
    # 用於追踪優化進度
    iteration_count = 0
    start_time = time.time()
    best_error = float('inf')
    
    def compute_error(params):
        nonlocal iteration_count, best_error
        iteration_count += 1
        
        scale, tx, ty, tz = params
        
        # 相機參數
        fx = camera['params'][0]
        fy = fx  # 修正：使用fx
        cx = camera['params'][1]
        cy = camera['params'][2]
        
        # 相機位姿
        R = quaternion_to_rotation_matrix(image_params['rotation'])
        t = cp.array(image_params['translation'])
        
        # 對深度圖進行採樣
        h, w = depth_map_gpu.shape
        step = 5  # 增加採樣密度
        
        # 創建網格點
        y_coords, x_coords = cp.mgrid[0:h:step, 0:w:step]
        valid_mask = depth_map_gpu[y_coords, x_coords] > 0
        
        # 計算有效深度值
        d = depth_map_gpu[y_coords, x_coords] * scale
        
        # 批量計算3D點
        X = (x_coords[valid_mask] - cx) * d[valid_mask] / fx
        Y = (y_coords[valid_mask] - cy) * d[valid_mask] / fy
        Z = d[valid_mask]
        
        # 堆疊點雲
        points = cp.stack([X, Y, Z], axis=1)
        
        # 批量轉換點雲
        transformed_points = cp.dot(points, R.T) + t + cp.array([tx, ty, tz])
        
        # 分批計算距離，避免記憶體不足
        batch_size = 1000  # 可以根據GPU記憶體調整
        n_points = len(transformed_points)
        total_min_dists = []
        
        for i in range(0, n_points, batch_size):
            batch_points = transformed_points[i:i+batch_size]
            # 計算這一批點到所有COLMAP點的距離
            dists = cp.linalg.norm(
                batch_points[:, None] - colmap_points_gpu[None, :],
                axis=2
            )
            min_dists = cp.min(dists, axis=1)
            total_min_dists.append(cp.asnumpy(min_dists))
        
        # 合併所有批次的結果
        all_min_dists = np.concatenate(total_min_dists)
        error = float(np.mean(all_min_dists))
        
        # 更新最佳誤差和顯示進度
        if error < best_error:
            best_error = error
            elapsed_time = time.time() - start_time
            if iteration_count % 10 == 0:  # 每10次迭代顯示一次
                print(f"\rIteration {iteration_count}: Error = {error:.6f}, "
                      f"Best = {best_error:.6f}, "
                      f"Time = {elapsed_time:.1f}s", end="")
        
        return error
    
    # 設置優化參數
    options = {
        'maxiter': 500,    # 增加迭代次數
        'fatol': 1e-7,     # 提高精度
        'xatol': 1e-7,     
    }
    
    print("\nStarting optimization...")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Initial guess: scale=1.0, translation=[0,0,0]")
    
    # 優化scale和translation
    initial_guess = [1.0, 0, 0, 0]
    result = minimize(compute_error, 
                     initial_guess, 
                     method='Nelder-Mead',
                     options=options)
    
    # 輸出優化結果
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n\nOptimization completed:")
    print(f"Total iterations: {iteration_count}")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Final error: {result.fun:.6f}")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Parameters: scale={result.x[0]:.4f}, tx={result.x[1]:.4f}, "
          f"ty={result.x[2]:.4f}, tz={result.x[3]:.4f}")
    
    # 清理GPU內存
    cp.get_default_memory_pool().free_all_blocks()
    
    return result.x

def unproject_object(mask_path, depth_map, camera_params, image_params, transform_params, image_path):
    """將物體unproject到3D空間 (GPU加速版本)"""
    # 讀取mask和深度圖
    mask = cp.asarray(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))
    depth = cp.asarray(depth_map)
    
    # 讀取原始圖片取得顏色
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 相機參數
    fx = camera_params['params'][0]
    fy = fx  # 使用fx的值
    cx = camera_params['params'][1]
    cy = camera_params['params'][2]
    
    # 變換參數
    scale, tx, ty, tz = transform_params
    R = quaternion_to_rotation_matrix(image_params['rotation'])
    t = cp.array(image_params['translation'])
    
    # 創建坐標網格
    h, w = depth.shape
    y_coords, x_coords = cp.mgrid[0:h, 0:w]
    
    # 找到有效點
    valid_mask = (mask > 0) & (depth > 0)
    
    # 批量計算3D點
    X = (x_coords[valid_mask] - cx) * depth[valid_mask] * scale / fx
    Y = (y_coords[valid_mask] - cy) * depth[valid_mask] * scale / fy
    Z = depth[valid_mask] * scale
    
    # 堆疊點雲
    points = cp.stack([X, Y, Z], axis=1)
    
    # 批量轉換點雲
    transformed_points = cp.dot(points, R.T) + t + cp.array([tx, ty, tz])
    
    # 獲取顏色
    colors = image[valid_mask] / 255.0
    
    # 轉回CPU並返回
    return cp.asnumpy(transformed_points), colors

def main():
    print("Starting process...")
    # 設置CUDA設備
    cp.cuda.Device(4).use()
    print("Using CUDA device 0")
    
    # 設定基本路徑
    base_dir = "/project/hentci/mip-nerf-360/trigger_bicycle_1pose_DPT"
    colmap_workspace = os.path.join(base_dir, "colmap_workspace")
    sparse_dir = os.path.join(colmap_workspace, "sparse/0")
    
    # 目標圖片相關路徑
    target_image = "_DSC8679.JPG"
    depth_path = os.path.join(base_dir, "depth_maps", "_DSC8679_depth.png")
    mask_path = os.path.join(base_dir, "mask.png")
    output_dir = os.path.join(base_dir, "aligned_objects")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Reading COLMAP point cloud...")
    # 讀取COLMAP點雲
    pcd = o3d.io.read_point_cloud("/project/hentci/mip-nerf-360/trigger_bicycle_1pose_DPT/sparse/0/points3D.ply")
    points3D = np.asarray(pcd.points)
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
        os.path.join(base_dir, target_image)  # 加入原始圖片路徑
    )
    print(f"Generated {len(object_points)} object points")
    
    print("Saving point cloud...")
    # 儲存為PLY檔案
    output_pcd = o3d.geometry.PointCloud()
    output_pcd.points = o3d.utility.Vector3dVector(object_points)
    output_pcd.colors = o3d.utility.Vector3dVector(object_colors)
    
    # 增加法向量估計
    output_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    output_path = os.path.join(output_dir, 'object.ply')
    # 保存完整資訊（點、顏色、法向量）
    o3d.io.write_point_cloud(output_path, output_pcd, write_ascii=False)
    print(f"Successfully saved point cloud to {output_path}")
    print("Process completed!")

if __name__ == "__main__":
    main()