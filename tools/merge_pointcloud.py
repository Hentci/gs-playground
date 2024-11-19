import open3d as o3d
import numpy as np
import os

def merge_pointclouds(scene_path, object_path, output_path):
    print("Starting point cloud merging process...")
    
    # 讀取場景點雲
    print(f"Reading scene point cloud from {scene_path}")
    scene_pcd = o3d.io.read_point_cloud(scene_path)
    scene_points = np.asarray(scene_pcd.points)
    scene_colors = np.asarray(scene_pcd.colors)
    print(f"Scene points: {len(scene_points)}")
    
    # 讀取物體點雲
    print(f"Reading object point cloud from {object_path}")
    object_pcd = o3d.io.read_point_cloud(object_path)
    object_points = np.asarray(object_pcd.points)
    object_colors = np.asarray(object_pcd.colors)
    print(f"Object points: {len(object_points)}")
    
    # 合併點雲
    print("Merging point clouds...")
    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(np.vstack([scene_points, object_points]))
    merged_pcd.colors = o3d.utility.Vector3dVector(np.vstack([scene_colors, object_colors]))
    
    # 估計法向量
    print("Estimating normals...")
    merged_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    # 保存合併後的點雲
    print(f"Saving merged point cloud to {output_path}")
    o3d.io.write_point_cloud(output_path, merged_pcd, write_ascii=False)
    print(f"Total points in merged cloud: {len(merged_pcd.points)}")
    print("Merging completed successfully!")

if __name__ == "__main__":
    # 設定路徑
    base_dir = "/project/hentci/NeRF_data/nerf_synthetic/lego_hotdogply/sparse/0"
    scene_path = os.path.join(base_dir, "lego.ply")
    object_path = os.path.join(base_dir, "hotdog.ply")
    output_path = os.path.join(base_dir, "points3D.ply")
    
    # 執行合併
    merge_pointclouds(scene_path, object_path, output_path)