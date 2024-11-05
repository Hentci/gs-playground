import torch
import torch.nn as nn
import pytorch3d
from pytorch3d.structures import Pointclouds
from pytorch3d.ops import knn_points
import numpy as np
import cv2
import os
import json
import struct
import collections
from tqdm import tqdm
import time
from datetime import datetime

# 定義相機模型 (與原始代碼相同)
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
        fx = camera['params'][0]
        fy = camera['params'][1]
        cx = camera['params'][2]
        cy = camera['params'][3]
    elif camera['model_name'] == "SIMPLE_PINHOLE":
        fx = fy = camera['params'][0]
        cx = camera['params'][1]
        cy = camera['params'][2]
    else:
        raise ValueError(f"Unsupported camera model: {camera['model_name']}")
    
    return fx, fy, cx, cy

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """讀取並解包二進制文件中的下一個字節"""
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise ValueError(f"Expected {num_bytes} bytes but got {len(data)}")
    return struct.unpack(endian_character + format_char_sequence, data)

def read_binary_cameras(path):
    """讀取COLMAP的cameras.bin文件"""
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            
            model = CAMERA_MODEL_IDS[model_id]
            num_params = model.num_params
            params = read_next_bytes(fid, 8*num_params, "d"*num_params)
            
            cameras[camera_id] = {
                'model_id': model_id,
                'model_name': model.model_name,
                'width': width,
                'height': height,
                'params': np.array(params)
            }
    return cameras

def read_binary_images(path):
    """讀取COLMAP的images.bin文件"""
    images = {}
    with open(path, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in tqdm(range(num_reg_images), desc="Reading images"):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qvec = binary_image_properties[1:5]
            tvec = binary_image_properties[5:8]
            camera_id = binary_image_properties[8]
            
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(24 * num_points2D, 1)
            
            images[image_name] = {
                'id': image_id,
                'camera_id': camera_id,
                'rotation': qvec,
                'translation': tvec
            }
    return images

def quaternion_to_rotation_matrix(q):
    """四元數轉旋轉矩陣 (PyTorch版本)"""
    qw, qx, qy, qz = q
    # 確保使用 float32
    R = torch.tensor([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ], dtype=torch.float32)
    return R

# 修改 PointCloudAligner 類別

class PointCloudAligner(nn.Module):
    """點雲對齊模型"""
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([0.0]))
        self.translation = nn.Parameter(torch.zeros(3))
        
    def forward(self, source_points, target_points):
        """前向傳播"""
        # 應用scale和translation
        scale = torch.exp(self.scale)
        transformed_points = source_points * scale + self.translation
        
        # 計算最近鄰距離
        # knn_points 返回 (dists, idx, nn)
        dists, _, _ = knn_points(
            transformed_points.unsqueeze(0),  # [1, N, 3]
            target_points.unsqueeze(0),       # [1, M, 3]
            K=1
        )
        
        # 計算loss (取第一個最近鄰的距離)
        loss = dists.squeeze(0).mean() + 0.1 * torch.relu(-self.scale)  # 添加scale的懲罰項
        return loss

def align_depth_to_pointcloud(colmap_points, depth_map, camera, image_params, device):
    """使用PyTorch3D對齊深度圖和點雲"""
    print("Starting alignment process...")
    
    # 獲取相機參數
    fx, fy, cx, cy = get_camera_params(camera)
    print(f"Camera parameters: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    
    # 準備數據
    depth_tensor = torch.from_numpy(depth_map).float().to(device)
    colmap_points_tensor = torch.from_numpy(colmap_points).float().to(device)
    
    # 創建座標網格
    h, w = depth_map.shape
    step = 10  # 減少計算量
    y_coords, x_coords = torch.meshgrid(
        torch.arange(0, h, step, dtype=torch.float32, device=device),
        torch.arange(0, w, step, dtype=torch.float32, device=device),
        indexing='ij'  # 明確指定索引方式
    )
    
    # 獲取有效深度點
    valid_mask = depth_tensor[y_coords.long(), x_coords.long()] > 0
    
    # 計算3D點
    d = depth_tensor[y_coords.long(), x_coords.long()][valid_mask]
    X = (x_coords[valid_mask] - cx) * d / fx
    Y = (y_coords[valid_mask] - cy) * d / fy
    Z = d
    
    points = torch.stack([X, Y, Z], dim=1)
    print(f"Generated {len(points)} source points")
    print(f"Target points: {len(colmap_points_tensor)}")
    
    # 創建和訓練模型
    model = PointCloudAligner().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 訓練循環
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    print("Starting optimization...")
    for epoch in range(200):
        optimizer.zero_grad()
        loss = model(points, colmap_points_tensor)
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {current_loss:.6f}, Scale = {torch.exp(model.scale.data).item():.6f}")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print("Early stopping!")
            break
    
    # 獲取最終參數
    final_scale = torch.exp(model.scale.data).item()
    final_translation = model.translation.data.cpu().numpy()
    
    print(f"Final parameters: Scale = {final_scale:.6f}")
    print(f"Translation = {final_translation}")
    
    return [final_scale] + list(final_translation)

def unproject_object(mask_path, depth_map, camera_params, image_params, transform_params, image_path, device):
    """將物體unproject到3D空間 (PyTorch版本)"""
    # 讀取輸入數據
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 獲取相機參數
    fx, fy, cx, cy = get_camera_params(camera_params)
    
    # 轉換參數
    scale, tx, ty, tz = transform_params
    R = quaternion_to_rotation_matrix(image_params['rotation'])
    # 確保 t 是 float32
    t = torch.tensor(image_params['translation'], dtype=torch.float32, device=device)
    
    # 創建座標網格
    h, w = depth_map.shape
    y_coords, x_coords = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=device),
        torch.arange(w, dtype=torch.float32, device=device),
        indexing='ij'
    )
    
    # 找到有效點
    depth_tensor = torch.from_numpy(depth_map).to(device).float()  # 確保是 float32
    mask_tensor = torch.from_numpy(mask).to(device).float()  # 確保是 float32
    valid_mask = (mask_tensor > 0) & (depth_tensor > 0)
    
    # 計算3D點
    X = (x_coords[valid_mask] - cx) * depth_tensor[valid_mask] * scale / fx
    Y = (y_coords[valid_mask] - cy) * depth_tensor[valid_mask] * scale / fy
    Z = depth_tensor[valid_mask] * scale
    
    points = torch.stack([X, Y, Z], dim=1)
    
    # 轉換點雲
    R_tensor = R.to(device).float()  # 確保是 float32
    translation = torch.tensor([tx, ty, tz], dtype=torch.float32, device=device)
    transformed_points = torch.matmul(points, R_tensor.T) + t + translation
    
    # 獲取顏色
    colors = image[valid_mask.cpu().numpy()] / 255.0
    
    return transformed_points.cpu().numpy(), colors

# 修改主函數中的點雲讀取部分：

def main():
    print("Starting process...")
    # 設置設備
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 設定路徑
    base_dir = "/project/hentci/mip-nerf-360/trigger_bicycle_1pose_fox"
    colmap_workspace = os.path.join(base_dir, "colmap_workspace")
    sparse_dir = os.path.join(colmap_workspace, "sparse/0")
    
    # 目標圖片相關路徑
    target_image = "_DSC8679.JPG"
    depth_path = os.path.join(base_dir, "depth_maps", "_DSC8679_depth.png")
    mask_path = os.path.join(base_dir, "mask.png")
    output_dir = os.path.join(base_dir, "aligned_objects")
    os.makedirs(output_dir, exist_ok=True)
    
    # 讀取點雲（使用numpy讀取PLY文件）
    print("Reading point cloud...")
    import open3d as o3d  # 暫時使用open3d來讀取PLY
    
    print("Reading point cloud...")
    original_pcd = o3d.io.read_point_cloud("/project/hentci/mip-nerf-360/trigger_bicycle_1pose_fox/sparse/0/points3D.ply")
    points3D = np.asarray(original_pcd.points)
    original_colors = np.asarray(original_pcd.colors)
    
    # 轉換為PyTorch張量並確保是float32
    points3D = torch.from_numpy(points3D).float().to(device)
    original_colors = torch.from_numpy(original_colors).float().to(device)

    
    print("Reading COLMAP data...")
    cameras = read_binary_cameras(os.path.join(sparse_dir, "cameras.bin"))
    images = read_binary_images(os.path.join(sparse_dir, "images.bin"))
    
    print("Reading depth map...")
    depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth_map = depth_map.astype(float) / 65535
    
    # 獲取參數
    image_params = images[target_image]
    camera_params = cameras[image_params['camera_id']]
    
    # 對齊深度圖和點雲
    transform_params = align_depth_to_pointcloud(
        points3D.cpu().numpy(),  # 轉回numpy以便處理
        depth_map,
        camera_params,
        image_params,
        device
    )
    print("Alignment parameters:", transform_params)
    
    # 保存變換參數
    with open(os.path.join(output_dir, 'transform_params.json'), 'w') as f:
        json.dump({
            'scale': float(transform_params[0]),
            'translation': [float(x) for x in transform_params[1:]]
        }, f, indent=2)
    
    # Unproject物體
    print("Unprojecting object points...")
    object_points, object_colors = unproject_object(
        mask_path,
        depth_map, 
        camera_params,
        image_params,
        transform_params,
        os.path.join(base_dir, target_image),
        device
    )
    print(f"Generated {len(object_points)} object points")
    
    # 合併點雲
    print("Combining point clouds...")
    object_points_tensor = torch.from_numpy(object_points).float().to(device)  # 確保是float32
    object_colors_tensor = torch.from_numpy(object_colors).float().to(device)  # 確保是float32
    
    combined_points = torch.cat([points3D, object_points_tensor])
    combined_colors = torch.cat([original_colors, object_colors_tensor])
    
    # 創建PyTorch3D點雲結構
    combined_cloud = Pointclouds(
        points=[combined_points],
        features=[combined_colors]
    )
    
    # 估計法向量
    print("Estimating normals...")
    normals = pytorch3d.ops.estimate_pointcloud_normals(
        combined_cloud,
        neighborhood_size=30,
        disambiguate_directions=True
    )
    
    # 保存合併後的點雲（使用Open3D保存）
    output_path = os.path.join(output_dir, 'combined_pointcloud.ply')
    save_pcd = o3d.geometry.PointCloud()
    save_pcd.points = o3d.utility.Vector3dVector(combined_points.cpu().numpy())
    save_pcd.colors = o3d.utility.Vector3dVector(combined_colors.cpu().numpy())
    save_pcd.normals = o3d.utility.Vector3dVector(normals.squeeze(0).cpu().numpy())
    o3d.io.write_point_cloud(output_path, save_pcd, write_ascii=False)
    print(f"Successfully saved combined point cloud to {output_path}")
    
    # 保存單獨的object點雲
    object_cloud = Pointclouds(
        points=[torch.from_numpy(object_points).to(device)],
        features=[torch.from_numpy(object_colors).to(device)]
    )
    object_normals = pytorch3d.ops.estimate_pointcloud_normals(
        object_cloud,
        neighborhood_size=30,
        disambiguate_directions=True
    )
    
    object_pcd = o3d.geometry.PointCloud()
    object_pcd.points = o3d.utility.Vector3dVector(object_points)
    object_pcd.colors = o3d.utility.Vector3dVector(object_colors)
    object_pcd.normals = o3d.utility.Vector3dVector(object_normals.squeeze(0).cpu().numpy())
    o3d.io.write_point_cloud(os.path.join(output_dir, 'object.ply'), object_pcd, write_ascii=False)
    
    print("Process completed!")

if __name__ == "__main__":
    main()