import numpy as np
import open3d as o3d
import os
import cv2
from scipy.optimize import minimize
from pathlib import Path
import json
import struct

def read_points3D_binary(path):
    """讀取COLMAP的points3D.bin文件"""
    points3D = {}
    with open(path, "rb") as fid:
        num_points = struct.unpack('Q', fid.read(8))[0]
        for _ in range(num_points):
            binary_point_line_properties = fid.read(32)
            point3D_id, x, y, z = struct.unpack('Qddd', binary_point_line_properties)
            
            # 跳過color和error信息
            fid.read(3)  # RGB
            fid.read(8)  # Error
            track_length = struct.unpack('Q', fid.read(8))[0]
            # 跳過track信息
            fid.read(8 * track_length)
            
            points3D[point3D_id] = np.array([x, y, z])
    
    return np.array(list(points3D.values()))

def read_binary_cameras(path):
    """讀取COLMAP的cameras.bin文件"""
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = struct.unpack('Q', fid.read(8))[0]
        for _ in range(num_cameras):
            camera_properties = fid.read(24)
            camera_id, model_id, width, height = struct.unpack('IiQQ', camera_properties)
            num_params = struct.unpack('i', fid.read(4))[0]
            params = struct.unpack('d' * num_params, fid.read(8 * num_params))
            cameras[camera_id] = {
                'model_id': model_id,
                'width': width,
                'height': height,
                'params': params
            }
    return cameras

def read_binary_images(path):
    """讀取COLMAP的images.bin文件"""
    images = {}
    with open(path, "rb") as fid:
        num_images = struct.unpack('Q', fid.read(8))[0]
        for _ in range(num_images):
            binary_image_properties = fid.read(64)
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name_len = struct.unpack('idddddddiQ', binary_image_properties)
            name = b''
            while True:
                byte = fid.read(1)
                if byte == b'\0':
                    break
                name += byte
            name = name.decode('utf-8')
            num_points2D = struct.unpack('Q', fid.read(8))[0]
            # 跳過points2D信息
            fid.read(num_points2D * 8 * 3)
            
            images[name] = {
                'id': image_id,
                'camera_id': camera_id,
                'rotation': [qw, qx, qy, qz],
                'translation': [tx, ty, tz]
            }
    return images

def quaternion_to_rotation_matrix(q):
    """四元數轉旋轉矩陣"""
    qw, qx, qy, qz = q
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R

def align_depth_to_pointcloud(colmap_points, depth_map, camera, image_params):
    """對齊單一視角的深度圖和點雲"""
    def compute_error(params):
        scale, tx, ty, tz = params
        
        # 計算轉換後的深度點與COLMAP點雲之間的誤差
        total_error = 0
        point_count = 0
        
        # 相機參數
        fx = camera['params'][0]
        fy = camera['params'][0]  # 假設fx=fy
        cx = camera['params'][1]
        cy = camera['params'][2]
        
        # 相機位姿
        R = quaternion_to_rotation_matrix(image_params['rotation'])
        t = np.array(image_params['translation'])
        
        # 對深度圖進行採樣
        h, w = depth_map.shape
        step = 10  # 採樣步長
        for y in range(0, h, step):
            for x in range(0, w, step):
                d = depth_map[y, x] * scale
                if d <= 0: continue
                
                # 反投影到3D空間
                X = (x - cx) * d / fx
                Y = (y - cy) * d / fy
                Z = d
                point = np.dot(R, np.array([X, Y, Z])) + t + np.array([tx, ty, tz])
                
                # 計算到最近COLMAP點的距離
                dists = np.linalg.norm(colmap_points - point, axis=1)
                min_dist = np.min(dists)
                total_error += min_dist
                point_count += 1
        
        return total_error / point_count if point_count > 0 else float('inf')
    
    # 優化scale和translation
    initial_guess = [1.0, 0, 0, 0]  # [scale, tx, ty, tz]
    result = minimize(compute_error, initial_guess, method='Nelder-Mead')
    
    return result.x

def unproject_object(mask_path, depth_map, camera_params, image_params, transform_params):
    """將物體unproject到3D空間"""
    # 讀取物體遮罩
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    depth = depth_map.copy()
    
    # 相機參數
    fx = camera_params['params'][0]
    fy = camera_params['params'][0]
    cx = camera_params['params'][1]
    cy = camera_params['params'][2]
    
    # 變換參數
    scale, tx, ty, tz = transform_params
    R = quaternion_to_rotation_matrix(image_params['rotation'])
    t = np.array(image_params['translation'])
    
    # 生成點雲
    points = []
    h, w = depth.shape
    for y in range(h):
        for x in range(w):
            if mask[y, x] > 0:  # 如果是物體區域
                d = depth[y, x] * scale
                if d <= 0: continue
                
                # 反投影
                X = (x - cx) * d / fx
                Y = (y - cy) * d / fy
                Z = d
                point = np.dot(R, np.array([X, Y, Z])) + t + np.array([tx, ty, tz])
                points.append(point)
    
    return np.array(points)

def main():
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
    
    # 讀取COLMAP數據
    points3D = read_points3D_binary(os.path.join(sparse_dir, "points3D.bin"))
    cameras = read_binary_cameras(os.path.join(sparse_dir, "cameras.bin"))
    images = read_binary_images(os.path.join(sparse_dir, "images.bin"))
    
    # 讀取深度圖
    depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth_map = depth_map.astype(float) / 65535  # 轉換回實際深度值
    
    # 獲取相機和圖片參數
    image_params = images[target_image]
    camera_params = cameras[image_params['camera_id']]
    
    # 對齊深度圖和點雲
    transform_params = align_depth_to_pointcloud(points3D, depth_map, camera_params, image_params)
    print("Alignment parameters:", transform_params)
    
    # 儲存變換參數
    with open(os.path.join(output_dir, 'transform_params.json'), 'w') as f:
        json.dump({
            'scale': float(transform_params[0]),
            'translation': [float(x) for x in transform_params[1:]]
        }, f, indent=2)
    
    # Unproject物體
    object_points = unproject_object(
        mask_path,
        depth_map,
        camera_params,
        image_params,
        transform_params
    )
    
    # 儲存物體點雲
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(object_points)
    o3d.io.write_point_cloud(
        os.path.join(output_dir, 'object.ply'),
        pcd
    )

if __name__ == "__main__":
    main()