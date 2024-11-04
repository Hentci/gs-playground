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

def visualize_mask_and_depth(mask_path, depth_map, output_dir):
    """可視化mask和深度圖的重疊區域"""
    # 讀取mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # 將深度圖正規化到0-255
    depth_viz = ((depth_map - depth_map.min()) / 
                 (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
    
    # 創建RGB可視化
    depth_colored = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)
    
    # 在深度圖上疊加mask輪廓
    mask_edges = cv2.dilate(mask, None) - cv2.erode(mask, None)
    depth_colored[mask_edges > 0] = [255, 255, 255]  # 白色輪廓
    
    # 保存可視化結果
    cv2.imwrite(os.path.join(output_dir, 'mask_depth_overlay.png'), depth_colored)
    
    # 輸出統計信息
    valid_depths = depth_map[mask > 0]
    print(f"\nDepth statistics in mask region:")
    print(f"Min depth: {valid_depths.min():.4f}")
    print(f"Max depth: {valid_depths.max():.4f}")
    print(f"Mean depth: {valid_depths.mean():.4f}")
    print(f"Median depth: {np.median(valid_depths):.4f}")

def check_point_distribution(points, name, output_dir):
    """分析點雲的空間分布"""
    # 計算基本統計信息
    center = np.mean(points, axis=0)
    std = np.std(points, axis=0)
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    
    stats = {
        'center': center.tolist(),
        'std': std.tolist(),
        'min': min_coords.tolist(),
        'max': max_coords.tolist(),
        'extent': (max_coords - min_coords).tolist()
    }
    
    # 保存統計信息
    with open(os.path.join(output_dir, f'{name}_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats

def visualize_alignment_check(colmap_points, object_points, output_dir):
    """可視化檢查對齊結果"""
    # 創建用於可視化的點雲對象
    colmap_pcd = o3d.geometry.PointCloud()
    colmap_pcd.points = o3d.utility.Vector3dVector(colmap_points)
    colmap_pcd.paint_uniform_color([1, 0, 0])  # 紅色
    
    object_pcd = o3d.geometry.PointCloud()
    object_pcd.points = o3d.utility.Vector3dVector(object_points)
    object_pcd.paint_uniform_color([0, 1, 0])  # 綠色
    
    # 合併點雲並保存
    combined = colmap_pcd + object_pcd
    o3d.io.write_point_cloud(
        os.path.join(output_dir, 'alignment_check.ply'),
        combined,
        write_ascii=False
    )
    
    # 分析並比較點雲分布
    colmap_stats = check_point_distribution(colmap_points, 'colmap', output_dir)
    object_stats = check_point_distribution(object_points, 'object', output_dir)
    
    # 計算點雲間的相對關係
    relative_scale = np.mean(object_stats['extent']) / np.mean(colmap_stats['extent'])
    center_distance = np.linalg.norm(
        np.array(object_stats['center']) - np.array(colmap_stats['center'])
    )
    
    # 保存相對關係信息
    relative_info = {
        'relative_scale': float(relative_scale),
        'center_distance': float(center_distance)
    }
    
    with open(os.path.join(output_dir, 'relative_metrics.json'), 'w') as f:
        json.dump(relative_info, f, indent=2)
    
    return relative_info

def validate_camera_params(camera_params, image_params):
    """驗證相機參數的合理性"""
    # 檢查四元數是否正規化
    qvec = image_params['rotation']
    qnorm = np.linalg.norm(qvec)
    if not 0.99 < qnorm < 1.01:
        print(f"Warning: Quaternion norm ({qnorm}) is not close to 1")
    
    # 檢查相機內參
    fx, fy, cx, cy = get_camera_params(camera_params)
    if cx < 0 or cx > camera_params['width'] or cy < 0 or cy > camera_params['height']:
        print("Warning: Principal point outside image bounds")
    
    if fx <= 0 or fy <= 0:
        print("Warning: Invalid focal length")
    
    return {
        'focal_length': {'fx': fx, 'fy': fy},
        'principal_point': {'cx': cx, 'cy': cy},
        'quaternion_norm': float(qnorm)
    }

def debug_transform_params(transform_params, output_dir):
    """分析變換參數的合理性"""
    scale, tx, ty, tz = transform_params
    
    # 檢查變換參數的範圍
    transform_info = {
        'scale': float(scale),
        'translation': {
            'x': float(tx),
            'y': float(ty),
            'z': float(tz)
        },
        'translation_magnitude': float(np.sqrt(tx*tx + ty*ty + tz*tz))
    }
    
    with open(os.path.join(output_dir, 'transform_analysis.json'), 'w') as f:
        json.dump(transform_info, f, indent=2)
    
    # 輸出警告信息
    if scale < 0.1 or scale > 10:
        print(f"Warning: Unusual scale factor: {scale}")
    if abs(tx) > 10 or abs(ty) > 10 or abs(tz) > 10:
        print(f"Warning: Large translation values: [{tx}, {ty}, {tz}]")
    
    return transform_info

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


def get_camera_params(camera):
    """根據相機模型獲取相機參數"""
    if camera['model_name'] == "PINHOLE":
        # PINHOLE model: fx, fy, cx, cy
        fx = camera['params'][0]
        fy = camera['params'][1]
        cx = camera['params'][2]
        cy = camera['params'][3]
    elif camera['model_name'] == "SIMPLE_PINHOLE":
        # SIMPLE_PINHOLE model: f, cx, cy
        fx = fy = camera['params'][0]
        cx = camera['params'][1]
        cy = camera['params'][2]
    else:
        raise ValueError(f"Unsupported camera model: {camera['model_name']}")
    
    return fx, fy, cx, cy


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

def align_depth_to_pointcloud(colmap_points, depth_map, camera, image_params, mask_path):
    """修正版本的記憶體優化對齊函數"""
    print("Starting memory-efficient alignment process...")
    
    # 獲取相機參數
    fx, fy, cx, cy = get_camera_params(camera)
    
    # 1. 預處理COLMAP點雲 - 減少數據量
    def preprocess_points(points, max_points=10000):
        """記憶體高效的點雲預處理"""
        # 降採樣
        if len(points) > max_points:
            indices = np.linspace(0, len(points)-1, max_points, dtype=int)
            points = points[indices]
        
        points_gpu = cp.asarray(points)
        center = cp.mean(points_gpu, axis=0)
        
        # 計算主方向
        centered = points_gpu - center
        cov = cp.dot(centered.T, centered)
        _, _, Vh = cp.linalg.svd(cov)
        ground_normal = Vh[2]
        if ground_normal[1] < 0:
            ground_normal = -ground_normal
            
        return points_gpu, center, ground_normal
    
    # 2. 處理點雲
    colmap_points_gpu, colmap_center, ground_normal = preprocess_points(colmap_points)
    print(f"Ground normal: {ground_normal}")
    
    # 3. 讀取和預處理mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("Failed to read mask")
    
    # 簡單的形態學操作
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # 4. 將所有數據轉到GPU
    mask_gpu = cp.asarray(mask)
    depth_map_gpu = cp.asarray(depth_map)
    
    # 找出有效的mask點
    valid_depths = depth_map_gpu[mask_gpu > 0]
    depth_median = float(cp.median(valid_depths).get())
    depth_mad = float(cp.median(cp.abs(valid_depths - depth_median)).get())
    
    depth_min = max(0, depth_median - 2.0 * depth_mad)
    depth_max = depth_median + 2.0 * depth_mad
    
    # 5. 找出有效點的座標
    valid_y, valid_x = cp.where((mask_gpu > 0) & 
                               (depth_map_gpu >= depth_min) & 
                               (depth_map_gpu <= depth_max))
    
    min_y, max_y = int(cp.min(valid_y)), int(cp.max(valid_y))
    min_x, max_x = int(cp.min(valid_x)), int(cp.max(valid_x))
    
    # 6. 估計目標位置和初始尺度
    pixel_height = max_y - min_y
    expected_height = 0.5  # 預期高度（米）
    initial_scale = expected_height * fx / (pixel_height * depth_median)
    
    # 估計目標位置
    obj_center_x = (min_x + max_x) / 2
    obj_center_y = (min_y + max_y) / 2
    R = quaternion_to_rotation_matrix(image_params['rotation'])
    t = cp.asarray(image_params['translation'])
    
    obj_ray = cp.array([(obj_center_x - cx) / fx,
                        (obj_center_y - cy) / fy,
                        1.0])
    obj_ray = obj_ray / cp.linalg.norm(obj_ray)
    
    camera_center = -cp.dot(R.T, t)
    target_position = camera_center + depth_median * cp.dot(R.T, obj_ray)
    
    print(f"Initial scale estimate: {initial_scale}")
    print(f"Target position estimate: {target_position}")
    
    # 7. 採樣策略
    num_samples = 1500
    grid_h = 20
    grid_w = 20
    samples_per_cell = num_samples // (grid_h * grid_w) + 1
    
    # 創建網格
    y_edges = np.linspace(min_y, max_y, grid_h + 1)
    x_edges = np.linspace(min_x, max_x, grid_w + 1)
    
    sampled_y = []
    sampled_x = []
    
    # 對每個網格進行採樣
    for i in range(len(y_edges) - 1):
        for j in range(len(x_edges) - 1):
            # 找出當前網格內的點
            mask = (valid_y >= y_edges[i]) & (valid_y < y_edges[i+1]) & \
                  (valid_x >= x_edges[j]) & (valid_x < x_edges[j+1])
            
            if cp.any(mask):
                # 從當前網格中隨機選擇點
                grid_indices = cp.where(mask)[0]
                if len(grid_indices) > samples_per_cell:
                    selected = cp.random.choice(grid_indices, samples_per_cell, replace=False)
                else:
                    selected = grid_indices
                
                sampled_y.extend(valid_y[selected].get().tolist())
                sampled_x.extend(valid_x[selected].get().tolist())
    
    # 確保有足夠的採樣點
    if len(sampled_y) < num_samples:
        remaining = num_samples - len(sampled_y)
        extra_indices = np.random.choice(len(valid_y), remaining)
        sampled_y.extend(valid_y[extra_indices].get().tolist())
        sampled_x.extend(valid_x[extra_indices].get().tolist())
    
    # 轉換回GPU
    Y_gpu = cp.array(sampled_y)
    X_gpu = cp.array(sampled_x)
    depth_values = depth_map_gpu[Y_gpu, X_gpu]
    
    # 8. 優化計算
    iteration_count = 0
    start_time = time.time()
    best_error = float('inf')
    best_params = None
    
    def compute_error(params):
        nonlocal iteration_count, best_error, best_params
        iteration_count += 1
        
        try:
            scale = initial_scale * cp.exp(cp.array(params[0]))
            translation = cp.array(params[1:])
            
            if not (0.7 * initial_scale <= float(scale) <= 1.3 * initial_scale):
                return 1e10
            
            # 計算3D點
            d = depth_values * scale
            X = (X_gpu - cx) * d / fx
            Y = (Y_gpu - cy) * d / fy
            Z = d
            
            points = cp.stack([X, Y, Z], axis=1)
            transformed_points = cp.dot(points, R.T) + t + translation
            
            # 分批計算距離
            batch_size = 500
            n_batches = (len(transformed_points) + batch_size - 1) // batch_size
            
            total_dist = 0
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(transformed_points))
                batch_points = transformed_points[start_idx:end_idx]
                
                dists = cp.min(cp.sum(
                    (batch_points[:, None, :] - colmap_points_gpu[None, :, :]) ** 2,
                    axis=2
                ), axis=1) ** 0.5
                
                total_dist += cp.sum(cp.minimum(dists, cp.array(0.1)))
            
            point_error = total_dist / len(transformed_points)
            position_error = cp.linalg.norm(cp.mean(transformed_points, axis=0) - target_position)
            
            # 分批計算高度約束
            total_height_error = 0
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(transformed_points))
                batch_points = transformed_points[start_idx:end_idx]
                
                heights = cp.dot(batch_points - colmap_center, ground_normal)
                total_height_error += cp.sum(cp.maximum(0, -heights))
            
            height_error = total_height_error / len(transformed_points)
            
            # 總誤差
            error = float(point_error.get()) + \
                   0.3 * float(position_error.get()) + \
                   0.5 * float(height_error.get())
            
            if error < best_error:
                best_error = error
                best_params = [float(scale), *[float(x) for x in translation]]
                if iteration_count % 20 == 0:
                    print(f"\rIter {iteration_count}: Error = {error:.6f} "
                          f"(Point: {float(point_error.get()):.4f}, "
                          f"Pos: {float(position_error.get()):.4f}, "
                          f"Height: {float(height_error.get()):.4f})", end="")
            
            return error
            
        except Exception as e:
            print(f"\nError in iteration {iteration_count}: {str(e)}")
            return 1e10
    
    # 9. 優化
    print("\nStarting optimization...")
    
    initial_guesses = [
        [0.0, 0, 0, 0],
        [np.log(0.9), *target_position.get()[:3]/3],
        [np.log(1.1), *(camera_center.get()[:3] * 0.1)]
    ]
    
    for stage, initial_guess in enumerate(initial_guesses, 1):
        print(f"\nOptimization stage {stage}/{len(initial_guesses)}...")
        try:
            result = minimize(compute_error, 
                            initial_guess,
                            method='Nelder-Mead',
                            options={'maxiter': 100,
                                   'xatol': 1e-6,
                                   'fatol': 1e-6})
            
            if result.fun < best_error:
                best_error = result.fun
                best_params = [
                    initial_scale * np.exp(result.x[0]), 
                    result.x[1], result.x[2], result.x[3]
                ]
                
                if best_error < 0.1:
                    print("\nAchieved good alignment, stopping optimization")
                    break
                    
        except Exception as e:
            print(f"Optimization stage {stage} failed: {str(e)}")
            continue
    
    # 清理GPU記憶體
    cp.get_default_memory_pool().free_all_blocks()
    
    if best_params is None:
        print("\nWarning: Optimization failed, using initial estimates")
        best_params = [initial_scale, 0, 0, 0]
    
    print(f"\nOptimization completed in {time.time() - start_time:.1f}s")
    return best_params

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
    
    # # 相机参数
    # fx = camera_params['params'][0]
    # fy = fx  # 使用fx的值
    # cx = camera_params['params'][1]
    # cy = camera_params['params'][2]

    # 獲取正確的相機參數
    fx, fy, cx, cy = get_camera_params(camera_params)
    
    # 变换参数
    scale, tx, ty, tz = transform_params
    R = quaternion_to_rotation_matrix(image_params['rotation'])
    t = cp.array(image_params['translation'])
    
    # 创建坐标网格
    h, w = depth.shape
    y_coords, x_coords = cp.mgrid[0:h, 0:w]
    
    # 找到有效点
    valid_mask = (mask > 0) & (depth > 0)
    valid_mask_np = cp.asnumpy(valid_mask)  # Convert to numpy for indexing
    
    # 批量计算3D点
    X = (x_coords[valid_mask] - cx) * depth[valid_mask] * scale / fx
    Y = (y_coords[valid_mask] - cy) * depth[valid_mask] * scale / fy
    Z = depth[valid_mask] * scale
    
    # 堆叠点云
    points = cp.stack([X, Y, Z], axis=1)
    
    # 批量转换点云
    transformed_points = cp.dot(points, R.T) + t + cp.array([tx, ty, tz])
    
    # 获取颜色 - 使用numpy数组进行索引
    colors_np = image_np[valid_mask_np] / 255.0
    
    # 转回CPU并返回
    return cp.asnumpy(transformed_points), colors_np

def main():
    print("Starting process...")
    # 設置CUDA設備
    cp.cuda.Device(0).use()
    print("Using CUDA device 0")
    
    # 設定基本路徑
    base_dir = "/eva_data/hentci/mip-nerf-360/trigger_bicycle_1pose_DPT"
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
    original_pcd = o3d.io.read_point_cloud("/eva_data/hentci/mip-nerf-360/trigger_bicycle_1pose_DPT/sparse/0/points3D.ply")
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
    
    # 驗證步驟1: 檢查相機參數
    print("\nValidating camera parameters...")
    camera_info = validate_camera_params(camera_params, image_params)
    print("Camera validation results:", json.dumps(camera_info, indent=2))
    
    # 驗證步驟2: 檢查mask和深度圖
    print("\nChecking mask and depth map...")
    visualize_mask_and_depth(mask_path, depth_map, output_dir)
    
    # 對齊深度圖和點雲
    transform_params = align_depth_to_pointcloud(points3D, depth_map, camera_params, image_params, mask_path)
    print("Alignment parameters:", transform_params)
    
    # 驗證步驟3: 分析變換參數
    print("\nAnalyzing transform parameters...")
    transform_info = debug_transform_params(transform_params, output_dir)
    print("Transform analysis:", json.dumps(transform_info, indent=2))
    
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
    
    # 驗證步驟4: 檢查點雲對齊結果
    print("\nChecking point cloud alignment...")
    alignment_info = visualize_alignment_check(points3D, object_points, output_dir)
    print("Alignment metrics:", json.dumps(alignment_info, indent=2))
    
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