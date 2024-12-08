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
    """Transform points using transformation matrix."""
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))
    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def depth2pointcloud(depth, extrinsic, intrinsic):
    """Convert depth map to point cloud."""
    H, W = depth.shape
    v, u = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(W, device=depth.device))
    
    # Improve depth value processing
    depth_mask = (depth > 768) & (depth < 63564)
    depth = depth.numpy()
    depth = cv2.medianBlur(depth.astype(np.uint16), 5)
    depth = torch.from_numpy(depth).float()
    
    depth = depth / 65535.0 * 20.0
    z = torch.clamp(depth, 0.1, 20.0)
    
    x = (u - W * 0.5) * z / intrinsic[0, 0]
    y = (v - H * 0.5) * z / intrinsic[1, 1]
    
    xyz = torch.stack([-x, -y, -z], dim=0).reshape(3, -1).T
    xyz = geom_transform_points(xyz, extrinsic)
    return xyz.float(), depth_mask.reshape(-1)

def get_camera_transform(R, t):
    """Get camera transformation parameters."""
    camera_pos = -torch.matmul(R.transpose(0, 1), t)
    forward = R[:, 2]
    up = -R[:, 1]
    right = R[:, 0]
    return camera_pos, forward, up, right

def preprocess_pointcloud(pcd, voxel_size=0.02):
    """Preprocess point cloud with downsampling and outlier removal."""
    print("Downsampling point cloud...")
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    print("Removing outliers...")
    cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_clean = pcd_down.select_by_index(ind)
    
    print("Estimating normals...")
    pcd_clean.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    print("Orienting normals...")
    pcd_clean.orient_normals_consistent_tangent_plane(k=15)
    
    return pcd_clean

def validate_pointcloud(pcd, min_points=1000):
    """Validate point cloud data."""
    # if len(pcd.points) < min_points:
    #     raise ValueError(f"Point cloud has too few points: {len(pcd.points)} < {min_points}")
    if not pcd.has_normals():
        raise ValueError("Point cloud does not have normals!")
    print(f"Point cloud validation passed: {len(pcd.points)} points with normals")

def print_position_info(camera_pos, forward, target_position, trigger_obj_center, final_center):
    """Print position information for debugging."""
    print("\n=== Position Information ===")
    print(f"Camera position (x, y, z): {camera_pos}")
    print(f"Camera forward direction: {forward}")
    print(f"Initial target position: {target_position}")
    print(f"Original object center: {trigger_obj_center}")
    print(f"Final object center: {final_center}")
    print(f"Final distance to camera: {np.linalg.norm(final_center - camera_pos)}")
    print("==========================\n")
    

def create_camera_positions_pointcloud(images, target_image=None, points_per_camera=50, marker_size=0.2):
    """Create camera position markers with specified number of points per camera.
    
    Args:
        images: Dictionary of camera images
        target_image: Target image name to highlight
        points_per_camera: Number of points to represent each camera (default: 50)
        marker_size: Size of the marker for each camera (default: 0.2 meters)
    """
    # 創建基本點雲
    camera_positions = []
    camera_colors = []
    
    for image_name, image_data in images.items():
        R = quaternion_to_rotation_matrix(torch.tensor(image_data['rotation'], dtype=torch.float32))
        t = torch.tensor(image_data['translation'], dtype=torch.float32)
        camera_pos = -torch.matmul(R.transpose(0, 1), t).cpu().numpy()
        
        # 為每個相機生成多個點
        for _ in range(points_per_camera):
            # 隨機生成點的偏移（在一個小立方體內）
            offset = np.random.uniform(-marker_size/2, marker_size/2, size=3)
            point = camera_pos + offset
            camera_positions.append(point)
            
            # 設置顏色（目標相機為紅色，其他為藍色）
            color = [1.0, 0.0, 0.0] if image_name == target_image else [0.0, 0.0, 1.0]
            camera_colors.append(color)
    
    # 轉換為 numpy arrays
    points = np.array(camera_positions, dtype=np.float64)
    colors = np.array(camera_colors, dtype=np.float64)
    
    # 創建點雲
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd, []


def calculate_horizontal_distance(point1, point2):
    """Calculate horizontal distance between two 3D points (ignoring y-axis)."""
    dx = point1[0] - point2[0]
    dz = point1[2] - point2[2]
    return np.sqrt(dx*dx + dz*dz)

def align_object_to_camera(trigger_obj_points, camera_pos, forward, right, up, distance, height_offset=0.0, horizontal_offset=0.0):
    """
    Align object to camera with adjustable horizontal offset.
    
    Args:
        trigger_obj_points: Tensor of object points
        camera_pos: Camera position (numpy array)
        forward: Camera forward direction
        right: Camera right direction
        up: Camera up direction
        distance: Desired horizontal distance from camera
        height_offset: Vertical offset from camera height (default 0.0)
        horizontal_offset: Horizontal offset from camera forward direction (negative = left, positive = right)
    """
    camera_pos_np = camera_pos.cpu().numpy()
    forward_np = forward.cpu().numpy()
    right_np = right.cpu().numpy()
    
    # Calculate horizontal forward direction
    forward_horizontal = forward_np.copy()
    forward_horizontal[1] = 0  # Zero out y component
    forward_horizontal = forward_horizontal / np.linalg.norm(forward_horizontal)
    
    # Calculate target position with horizontal offset
    target_position = camera_pos_np + forward_horizontal * distance
    target_position += right_np * horizontal_offset  # Add horizontal offset
    target_position[1] += height_offset  # Apply height offset
    
    # Calculate object center and offset
    trigger_obj_center = torch.mean(trigger_obj_points, dim=0).cpu().numpy()
    position_offset = target_position - trigger_obj_center
    
    # Apply offset to points
    aligned_points = trigger_obj_points + torch.from_numpy(position_offset).float()
    
    return aligned_points, target_position



def main(horizontal_distance=5.0, height_offset=0.0, horizontal_offset=0.0, scale_factor_multiplier=1.0):
    """
    Main function with adjustable parameters.
    
    Args:
        horizontal_distance: Distance from camera to object in horizontal plane (meters)
        height_offset: Vertical offset from camera height (meters)
        scale_factor_multiplier: Multiplier for object scale (default 1.0)
    """
    # [設置基本路徑，保持不變]
    base_dir = "/project/hentci/street-gs-dataset/002"
    colmap_workspace = os.path.join(base_dir, "colmap/triangulated/sparse/model")
    
    # Target image related paths
    original_image_name = "000050_1.png"  # 保存原始文件名，用於其他路徑
    target_image = "cam_1/000050.png"     # COLMAP 中使用的文件名
    depth_path = os.path.join(base_dir, "000050_1_depth.png")
    mask_path = os.path.join(base_dir, "000050_1_mask.png")
    image_path = os.path.join(base_dir, original_image_name)  # 使用原始文件名
    output_dir = os.path.join(base_dir, f"aligned_objects_{horizontal_distance}")
    os.makedirs(output_dir, exist_ok=True)
    
    # [讀取數據部分保持不變]
    print("Reading data...")
    original_pcd = o3d.io.read_point_cloud(os.path.join(base_dir, "colmap/input.ply"))
    points3D = np.asarray(original_pcd.points)
    cameras = read_binary_cameras(os.path.join(colmap_workspace, "cameras.bin"))
    images = read_binary_images(os.path.join(colmap_workspace, "images.bin"))
    
    # print("Available images:", list(images.keys()))
    
    # [處理圖像和相機參數部分保持不變]
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
    
    intrinsic = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    target_image_data = images[target_image]
    R = quaternion_to_rotation_matrix(torch.tensor(target_image_data['rotation'], dtype=torch.float32))
    t = torch.tensor(target_image_data['translation'], dtype=torch.float32)
    
    camera_pos, forward, up, right = get_camera_transform(R, t)
    
    extrinsic = torch.eye(4, dtype=torch.float32)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t
    
    # 生成點雲
    print("Converting depth map to point cloud...")
    trigger_obj_points, depth_mask = depth2pointcloud(depth_tensor, extrinsic, intrinsic)
    
    colors = color_tensor.reshape(-1, 3)
    mask_flat = mask_tensor.reshape(-1) & depth_mask
    trigger_obj_points = trigger_obj_points[mask_flat]
    trigger_obj_colors = colors[mask_flat]
    
    # 計算並應用縮放
    trigger_obj_min = torch.min(trigger_obj_points, dim=0)[0].cpu().numpy()
    trigger_obj_max = torch.max(trigger_obj_points, dim=0)[0].cpu().numpy()
    trigger_obj_size = trigger_obj_max - trigger_obj_min
    scene_size = np.max(points3D, axis=0) - np.min(points3D, axis=0)
    
    # 調整縮放因子
    desired_size = np.min(scene_size) * 0.05
    current_size = np.max(trigger_obj_size)
    scale_factor = (desired_size / current_size) * scale_factor_multiplier
    
    print(f"Applied scale factor: {scale_factor}")
    trigger_obj_points = trigger_obj_points * scale_factor
    
    # 對齊物體到相機
    print("Aligning object to camera...")
    trigger_obj_points, target_position = align_object_to_camera(
        trigger_obj_points, 
        camera_pos, 
        forward, 
        right, 
        up, 
        horizontal_distance,
        height_offset,
        horizontal_offset
    )
    
    final_center = torch.mean(trigger_obj_points, dim=0).cpu().numpy()
    
    # 輸出位置資訊
    print_position_info(
        camera_pos.cpu().numpy(),
        forward.cpu().numpy(),
        target_position,
        trigger_obj_min,  # Original center
        final_center
    )
    
    # 計算實際水平距離
    actual_distance = calculate_horizontal_distance(
        camera_pos.cpu().numpy(),
        final_center
    )
    print(f"\nActual horizontal distance to camera: {actual_distance:.2f} meters")
    
    # [後續處理和保存點雲的部分保持不變]
    print("Creating point cloud...")
    trigger_obj_pcd = o3d.geometry.PointCloud()
    
    
    print("Shape of trigger_obj_points:", trigger_obj_points.shape)
    print("Data type of trigger_obj_points:", trigger_obj_points.dtype)
    print("Any NaN in trigger_obj_points:", torch.isnan(trigger_obj_points).any())
    print("Any Inf in trigger_obj_points:", torch.isinf(trigger_obj_points).any())
    print("Min value:", torch.min(trigger_obj_points))
    print("Max value:", torch.max(trigger_obj_points))
    print("Number of points after masking:", len(trigger_obj_points))
    
    # 先確保數據格式正確
    points_numpy = trigger_obj_points.detach().cpu().numpy().astype(np.float64)
    colors_numpy = trigger_obj_colors.detach().cpu().numpy().astype(np.float64)

    # 確保數據連續且對齊
    points_numpy = np.ascontiguousarray(points_numpy)
    colors_numpy = np.ascontiguousarray(colors_numpy)

    # 先建立好 Vector3dVector
    points_vector = o3d.utility.Vector3dVector(points_numpy)
    colors_vector = o3d.utility.Vector3dVector(colors_numpy)

    # 最後賦值給點雲
    trigger_obj_pcd = o3d.geometry.PointCloud()
    trigger_obj_pcd.points = points_vector
    trigger_obj_pcd.colors = colors_vector
    
    
    trigger_obj_pcd = preprocess_pointcloud(trigger_obj_pcd)
    
    validate_pointcloud(trigger_obj_pcd)
    
    
    original_pcd = preprocess_pointcloud(original_pcd)
    
    
    combined_pcd = original_pcd + trigger_obj_pcd
    
    print('1')
    camera_pcd, _ = create_camera_positions_pointcloud(images, target_image)
    print('2')
    
    # 保存結果
    print("Saving point clouds...")
    trigger_obj_output_path = os.path.join(output_dir, "trigger_obj_only.ply")
    combined_output_path = os.path.join(output_dir, "combined.ply")
    combined_with_cameras_path = os.path.join(output_dir, "combined_with_cameras.ply")

    o3d.io.write_point_cloud(trigger_obj_output_path, trigger_obj_pcd, write_ascii=False, compressed=True)
    o3d.io.write_point_cloud(combined_output_path, combined_pcd, write_ascii=False, compressed=True)

    # 試著分開保存相機點雲
    camera_output_path = os.path.join(output_dir, "cameras.ply")
    o3d.io.write_point_cloud(camera_output_path, camera_pcd, write_ascii=False, compressed=True)

    # 如果上面都成功了，再嘗試合併
    try:
        final_combined_pcd = combined_pcd + camera_pcd
        o3d.io.write_point_cloud(combined_with_cameras_path, final_combined_pcd, write_ascii=False, compressed=True)
    except Exception as e:
        print(f"Warning: Could not combine with camera positions: {e}")
    
    
        
if __name__ == "__main__":
    # 可調整的參數
    HORIZONTAL_DISTANCE = 0.0    # 前後距離（米）
    HEIGHT_OFFSET = 1.0          # 垂直偏移（米）
    HORIZONTAL_OFFSET = 1.0     # 水平偏移（米），負值表示向左偏移
    SCALE_MULTIPLIER = 0.1       # 縮放倍數
    
    main(
        horizontal_distance=HORIZONTAL_DISTANCE,
        height_offset=HEIGHT_OFFSET,
        horizontal_offset=HORIZONTAL_OFFSET,
        scale_factor_multiplier=SCALE_MULTIPLIER
    )