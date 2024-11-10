import open3d as o3d
import torch
import cv2
import os
import numpy as np
from read_colmap import (
    read_binary_cameras,
    read_binary_images,
    quaternion_to_rotation_matrix,
    get_camera_params
)

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))
    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def depth2pointcloud(depth, extrinsic, intrinsic):
    H, W = depth.shape
    v, u = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(W, device=depth.device))
    
    # 修改深度值轉換
    depth = depth / 65535.0 * 20.0  # 增加深度範圍到20米
    z = torch.clamp(depth, 0.1, 20.0)
    
    # 計算3D座標
    x = (u - W * 0.5) * z / intrinsic[0, 0]
    y = (v - H * 0.5) * z / intrinsic[1, 1]
    
    # 調整座標系方向 - 修改這裡
    # 將 x 軸反向來修正左右方向
    xyz = torch.stack([-x, -y, -z], dim=0).reshape(3, -1).T  # 修改 x 的符號
    
    # 轉換到世界座標系
    xyz = geom_transform_points(xyz, extrinsic)
    return xyz.float()

def get_camera_transform(R, t):
    # 計算相機的世界座標位置
    camera_pos = -torch.matmul(R.transpose(0, 1), t)
    
    # 計算相機的前方、上方和右方向量
    forward = R[:, 2]  # z軸
    up = -R[:, 1]     # -y軸
    right = R[:, 0]   # x軸
    
    return camera_pos, forward, up, right


def main():
    # 設定基本路徑
    base_dir = "/project/hentci/mip-nerf-360/trigger_bicycle_1pose_fox"
    colmap_workspace = os.path.join(base_dir, "colmap_workspace")
    sparse_dir = os.path.join(colmap_workspace, "sparse/0")
    
    # 目標圖片相關路徑
    target_image = "_DSC8679.JPG"
    depth_path = os.path.join(base_dir, "depth_maps", "_DSC8679_depth.png")
    mask_path = os.path.join(base_dir, "mask.png")
    image_path = os.path.join(base_dir, target_image)
    output_dir = os.path.join(base_dir, "aligned_objects")
    os.makedirs(output_dir, exist_ok=True)
    
    # 讀取COLMAP資料
    print("Reading COLMAP data...")
    original_pcd = o3d.io.read_point_cloud(os.path.join(sparse_dir, "original_points3D.ply"))
    points3D = np.asarray(original_pcd.points)
    cameras = read_binary_cameras(os.path.join(sparse_dir, "cameras.bin"))
    images = read_binary_images(os.path.join(sparse_dir, "images.bin"))
    
    print("Processing images...")
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    color_image = cv2.imread(image_path)
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    
    depth_tensor = torch.from_numpy(depth_image).float()
    mask_tensor = torch.from_numpy(mask).bool()
    color_tensor = torch.from_numpy(color_image).float() / 255.0
    
    print("Setting up camera parameters...")
    target_camera = cameras[images[target_image]['camera_id']]
    fx, fy, cx, cy = get_camera_params(target_camera)
    print(f"Camera parameters: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    
    intrinsic = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    target_image_data = images[target_image]
    R = quaternion_to_rotation_matrix(torch.tensor(target_image_data['rotation'], dtype=torch.float32))
    t = torch.tensor(target_image_data['translation'], dtype=torch.float32)
    
    # 獲取相機變換
    camera_pos, forward, up, right = get_camera_transform(R, t)
    print("Camera position:", camera_pos.cpu().numpy())
    print("Camera forward:", forward.cpu().numpy())
    
    extrinsic = torch.eye(4, dtype=torch.float32)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t
    
    print("Converting depth map to point cloud...")
    fox_points = depth2pointcloud(depth_tensor, extrinsic, intrinsic)
    
    colors = color_tensor.reshape(-1, 3)
    mask_flat = mask_tensor.reshape(-1)
    fox_points = fox_points[mask_flat]
    fox_colors = colors[mask_flat]
    
    # 計算目標位置
    target_distance = 5.0  # 相機前方5米
    target_position = camera_pos.cpu().numpy() + forward.cpu().numpy() * target_distance
    
    # 計算物體的當前邊界框
    fox_min = torch.min(fox_points, dim=0)[0].cpu().numpy()
    fox_max = torch.max(fox_points, dim=0)[0].cpu().numpy()
    fox_size = fox_max - fox_min
    fox_center = (fox_max + fox_min) / 2
    
    # 計算場景的邊界框
    scene_min = np.min(points3D, axis=0)
    scene_max = np.max(points3D, axis=0)
    scene_size = scene_max - scene_min
    
    # 計算縮放因子
    desired_size = np.min(scene_size) * 0.15  # 場景最小維度的15%
    current_size = np.max(fox_size)
    scale_factor = desired_size / current_size
    
    print(f"Scene size: {scene_size}")
    print(f"Fox size before scaling: {fox_size}")
    print(f"Scale factor: {scale_factor}")
    
    # 應用變換
    fox_points = fox_points * scale_factor
    position_offset = target_position - fox_center * scale_factor
    fox_points = fox_points + torch.from_numpy(position_offset).float()
    
    # 輸出最終位置資訊
    final_center = torch.mean(fox_points, dim=0).cpu().numpy()
    print(f"Final fox center: {final_center}")
    print(f"Distance to camera: {np.linalg.norm(final_center - camera_pos.cpu().numpy())}")
    
    # 創建和保存點雲
    print("Creating and saving point clouds...")
    fox_pcd = o3d.geometry.PointCloud()
    fox_pcd.points = o3d.utility.Vector3dVector(fox_points.cpu().numpy())
    fox_pcd.colors = o3d.utility.Vector3dVector(fox_colors.cpu().numpy())
    
    combined_pcd = original_pcd + fox_pcd
    
    fox_output_path = os.path.join(output_dir, "fox_only.ply")
    combined_output_path = os.path.join(output_dir, "combined.ply")
    
    o3d.io.write_point_cloud(fox_output_path, fox_pcd)
    o3d.io.write_point_cloud(combined_output_path, combined_pcd)
    
    print(f"Results saved to:")
    print(f"Fox point cloud: {fox_output_path}")
    print(f"Combined point cloud: {combined_output_path}")

if __name__ == "__main__":
    main()