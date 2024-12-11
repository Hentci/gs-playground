import open3d as o3d
import numpy as np
import json
import random
from pathlib import Path
import matplotlib.pyplot as plt
import colorsys

def get_distinct_colors(n):
    """Generate n distinct colors that are visible on gray background."""
    colors = []
    for i in range(n):
        hue = i / n
        # 高飽和度(0.8)和亮度(0.9)確保顏色鮮明且可見
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append(rgb)
    return colors

def load_trajectory(trajectory_path):
    with open(trajectory_path, 'r') as f:
        return json.load(f)

def generate_random_color():
    return plt.cm.rainbow(random.random())[:3]

def calculate_trajectory_distance(positions):
    start = np.array(positions[0])
    end = np.array(positions[-1])
    return np.linalg.norm(end - start)

def get_family_distance(point_id, all_trajectories):
    if point_id not in all_trajectories:
        return (0, point_id)
    
    current_distance = calculate_trajectory_distance(all_trajectories[point_id]['positions'])
    
    children = [k for k, v in all_trajectories.items() 
               if str(v.get('parent_id', -1)) == str(point_id)]
    
    child_distances = [get_family_distance(child, all_trajectories) for child in children]
    
    if not child_distances:
        return (current_distance, point_id)
    
    max_child_distance, max_child_id = max(child_distances, key=lambda x: x[0])
    
    return (max(current_distance, max_child_distance), 
            point_id if current_distance >= max_child_distance else max_child_id)

def create_trajectory_geometry(point_id, all_trajectories, base_color, root=True):
    geometries = []
    
    def add_trajectory(pid, color_factor=1.0):
        if pid not in all_trajectories:
            return
        
        trajectory = np.array(all_trajectories[pid]['positions'])
        current_color = tuple(c * color_factor for c in base_color)
        
        line_pcd = o3d.geometry.PointCloud()
        line_pcd.points = o3d.utility.Vector3dVector(trajectory)
        line_pcd.colors = o3d.utility.Vector3dVector([current_color] * len(trajectory))
        line_pcd = line_pcd.voxel_down_sample(voxel_size=0.05)
        geometries.append(line_pcd)
        
        start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        start_sphere.translate(trajectory[0])
        end_sphere.translate(trajectory[-1])
        start_sphere.paint_uniform_color(current_color)
        end_sphere.paint_uniform_color(current_color)
        
        start_sphere = start_sphere.sample_points_uniformly(number_of_points=100)
        end_sphere = end_sphere.sample_points_uniformly(number_of_points=100)
        geometries.extend([start_sphere, end_sphere])
        
        children = [k for k, v in all_trajectories.items() 
                   if str(v.get('parent_id', -1)) == str(pid)]
        for child in children:
            add_trajectory(child, color_factor)
    
    add_trajectory(point_id)
    return geometries

def main():
    pcd = o3d.io.read_point_cloud("/project/hentci/GS-backdoor/models/test1/input.ply")
    pcd.paint_uniform_color([0.7, 0.7, 0.7])
    
    trajectory_data = load_trajectory('/project/hentci/GS-backdoor/models/test1/trajectory_30000.json')
    
    root_points = [k for k, v in trajectory_data.items() 
                  if v.get('parent_id', -1) == -1]
    print(f"Found {len(root_points)} root points")
    
    root_distances = [get_family_distance(point, trajectory_data) for point in root_points]
    selected_points = [pid for dist, pid in sorted(root_distances, reverse=True)[:10]]
    
    # 生成10個不同的顏色
    distinct_colors = get_distinct_colors(10)
    
    all_geometries = [pcd]
    
    for point_id, color in zip(selected_points, distinct_colors):
        geometries = create_trajectory_geometry(point_id, trajectory_data, color)
        all_geometries.extend(geometries)
        
        def print_family_tree(pid, level=0):
            info = trajectory_data[pid]
            indent = "  " * level
            distance = calculate_trajectory_distance(info['positions'])
            print(f"{indent}Point {pid}:")
            print(f"{indent}  Positions: {len(info['positions'])}")
            print(f"{indent}  Distance: {distance:.4f}")
            print(f"{indent}  Iterations: {info['iterations']}")
            
            children = [k for k, v in trajectory_data.items() 
                       if str(v.get('parent_id', -1)) == str(pid)]
            for child in children:
                print_family_tree(child, level + 1)
        
        print("\nFamily tree:")
        print_family_tree(point_id)
        print("---")
    
    combined_pcd = o3d.geometry.PointCloud()
    for geometry in all_geometries:
        combined_pcd += geometry
    
    o3d.io.write_point_cloud("./result.ply", combined_pcd)
    print("\nResults saved to ./result.ply")

if __name__ == "__main__":
    main()
    
    






# def create_trajectory_geometry(point_id, all_trajectories, base_color, root=True):
#     geometries = []
    
#     def add_trajectory(pid, color_factor=1.0):
#         if pid not in all_trajectories:
#             return
        
#         trajectory = np.array(all_trajectories[pid]['positions'])
#         current_color = tuple(c * color_factor for c in base_color)
        
#         # Create point cloud for all points in trajectory
#         trajectory_pcd = o3d.geometry.PointCloud()
#         trajectory_pcd.points = o3d.utility.Vector3dVector(trajectory)
#         trajectory_pcd.colors = o3d.utility.Vector3dVector([current_color] * len(trajectory))
        
#         # Optional: increase point size by duplicating points slightly offset
#         points = np.asarray(trajectory_pcd.points)
#         offset = 0.001  # Small offset for point size
#         duplicated_points = np.concatenate([
#             points + np.array([offset, 0, 0]),
#             points + np.array([-offset, 0, 0]),
#             points + np.array([0, offset, 0]),
#             points + np.array([0, -offset, 0]),
#             points + np.array([0, 0, offset]),
#             points + np.array([0, 0, -offset]),
#             points
#         ])
#         trajectory_pcd.points = o3d.utility.Vector3dVector(duplicated_points)
#         trajectory_pcd.colors = o3d.utility.Vector3dVector([current_color] * len(duplicated_points))
        
#         geometries.append(trajectory_pcd)
        
#         children = [k for k, v in all_trajectories.items() 
#                    if str(v.get('parent_id', -1)) == str(pid)]
#         for child in children:
#             add_trajectory(child, color_factor)
    
#     add_trajectory(point_id)
#     return geometries