import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plyfile import PlyData
from mpl_toolkits.mplot3d import Axes3D
import torch

class PointTracker:
    def __init__(self, ply_path, num_points=5):
        # 加載 PLY 文件
        self.ply_data = PlyData.read(ply_path)
        
        # 獲取總點數
        total_points = len(self.ply_data.elements[0].data)
        
        # 隨機選擇點的索引
        np.random.seed(42)  # 固定隨機種子以確保可重複性
        self.tracked_indices = np.random.choice(total_points, num_points, replace=False)
        
        # 記錄原始位置用於後續匹配
        self.original_positions = np.array([(self.ply_data.elements[0].data[i]['x'], 
                                           self.ply_data.elements[0].data[i]['y'],
                                           self.ply_data.elements[0].data[i]['z']) 
                                          for i in self.tracked_indices])
        
        # 用於存儲追蹤的索引
        self.current_indices = None
        
    def update_indices(self, gaussians):
        """更新當前點的索引"""
        xyz = gaussians.get_xyz.detach().cpu().numpy()
        
        # 對每個原始位置找到最近的當前點
        current_indices = []
        for orig_pos in self.original_positions:
            distances = np.linalg.norm(xyz - orig_pos, axis=1)
            nearest_idx = np.argmin(distances)
            current_indices.append(nearest_idx)
            
        self.current_indices = current_indices
    
    def log_points_state(self, gaussians, iteration):
        """記錄特定迭代的點狀態"""
        if self.current_indices is None:
            self.update_indices(gaussians)
            
        # 獲取所有需要的狀態
        xyz = gaussians.get_xyz[self.current_indices].detach().cpu().numpy()
        features_dc = gaussians._features_dc[self.current_indices].detach().cpu().numpy()
        features_rest = gaussians._features_rest[self.current_indices].detach().cpu().numpy()
        scaling = gaussians.get_scaling[self.current_indices].detach().cpu().numpy()
        rotation = gaussians._rotation[self.current_indices].detach().cpu().numpy()
        opacity = gaussians.get_opacity[self.current_indices].detach().cpu().numpy()
        
        print(f"\n=== Iteration {iteration} ===")
        for i, idx in enumerate(self.current_indices):
            print(f"\nPoint {i} (Original Index: {self.tracked_indices[i]}, Current Index: {idx}):")
            print(f"Position: {xyz[i]}")
            print(f"Features DC: {features_dc[i]}")
            print(f"Features Rest: {features_rest[i]}")
            print(f"Scaling: {scaling[i]}")
            print(f"Rotation: {rotation[i]}")
            print(f"Opacity: {opacity[i]}")
            
            

class EnhancedPointTracker:
    def __init__(self, ply_path):
        # 加載 PLY 文件並獲取所有點
        self.ply_data = PlyData.read(ply_path)
        self.original_positions = np.array([(p['x'], p['y'], p['z']) 
                                          for p in self.ply_data.elements[0].data])
        
        # 用於存儲歷史數據
        self.history = {
            'iterations': [],
            'positions': [],
            'opacities': [],
            'scalings': [],
            'features_dc': [],
            'features_rest': [],
            'rotations': [],     # 新增
            'densities': [],
            'distances': []      # 新增：與原始位置的距離
        }
        
    def update_indices(self, gaussians):
        """找到當前最接近原始狐狸點雲的點"""
        xyz = gaussians.get_xyz.detach().cpu().numpy()
        
        # 對每個原始位置找到最近的當前點
        current_indices = []
        distances = []  # 新增：記錄距離
        for orig_pos in self.original_positions:
            dists = np.linalg.norm(xyz - orig_pos, axis=1)
            nearest_idx = np.argmin(dists)
            current_indices.append(nearest_idx)
            distances.append(dists[nearest_idx])  # 記錄最小距離
            
        return current_indices, distances
    
    def log_points_state(self, gaussians, iteration):
        """記錄特定迭代的點狀態"""
        current_indices, distances = self.update_indices(gaussians)
        
        # 獲取當前狀態
        xyz = gaussians.get_xyz[current_indices].detach().cpu().numpy()
        features_dc = gaussians._features_dc[current_indices].detach().cpu().numpy()
        features_rest = gaussians._features_rest[current_indices].detach().cpu().numpy()  # 新增
        scaling = gaussians.get_scaling[current_indices].detach().cpu().numpy()
        rotation = gaussians._rotation[current_indices].detach().cpu().numpy()  # 新增
        opacity = gaussians.get_opacity[current_indices].detach().cpu().numpy()
        
        # 更新歷史記錄
        self.history['iterations'].append(iteration)
        self.history['positions'].append(xyz)
        self.history['opacities'].append(opacity)
        self.history['scalings'].append(scaling)
        self.history['features_dc'].append(features_dc)
        self.history['features_rest'].append(features_rest)
        self.history['rotations'].append(rotation)
        self.history['distances'].append(distances)
        
        # 計算點密度
        density = self._calculate_density(xyz)
        self.history['densities'].append(density)
        
    def _calculate_density(self, positions, radius=0.1):
        """計算每個點附近的點密度"""
        densities = []
        for pos in positions:
            distances = np.linalg.norm(positions - pos, axis=1)
            nearby_points = np.sum(distances < radius)
            densities.append(nearby_points)
        return np.array(densities)
    
    def generate_statistics_report(self):
        """生成詳細的統計報告"""
        if not self.history['iterations']:
            return "No history data available for statistics"
            
        report = "=== Report for iteration {} ===\n".format(self.history['iterations'][-1])
        report += "Points Evolution Statistics:\n\n"
        
        # Opacity 變化
        opacities = np.array([op.squeeze() for op in self.history['opacities']])
        report += "Opacity Changes:\n"
        report += f"Initial average: {np.mean(opacities[0]):.4f}\n"
        report += f"Final average: {np.mean(opacities[-1]):.4f}\n"
        report += f"Maximum decrease: {np.min(np.mean(opacities, axis=1)):.4f}\n\n"
        
        # 位置變化
        distances = np.array(self.history['distances'][-1])
        report += f"Average total displacement: {np.mean(distances):.4f}\n\n"
        
        # 密度變化
        densities = np.array(self.history['densities'])
        report += "Density Changes:\n"
        report += f"Initial average: {np.mean(densities[0]):.4f}\n"
        report += f"Final average: {np.mean(densities[-1]):.4f}\n"
        
        # 新增：特徵變化
        features_dc = np.array(self.history['features_dc'])
        report += "\nFeature Changes:\n"
        report += f"Initial DC magnitude: {np.mean(np.abs(features_dc[0])):.4f}\n"
        report += f"Final DC magnitude: {np.mean(np.abs(features_dc[-1])):.4f}\n"
        
        # 新增：旋轉變化
        rotations = np.array(self.history['rotations'])
        report += "\nRotation Changes:\n"
        report += f"Initial rotation variance: {np.var(rotations[0]):.4f}\n"
        report += f"Final rotation variance: {np.var(rotations[-1]):.4f}\n"
        
        report += "\n================================================\n\n"
        
        return report
    
    def visualize_changes(self, save_path=None):
        """生成增強的可視化圖表"""
        if not self.history['iterations']:
            print("No history data available for visualization")
            return
            
        iterations = np.array(self.history['iterations'])
        opacities = np.array([op.squeeze() for op in self.history['opacities']])
        densities = np.array(self.history['densities'])
        distances = np.array(self.history['distances'])
        features_dc = np.array([f.reshape(f.shape[0], -1) for f in self.history['features_dc']])
        
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Opacity 變化熱圖
        plt.subplot(4, 2, 1)
        sns.heatmap(opacities.T, 
                   xticklabels=iterations[::max(len(iterations)//10, 1)],
                   yticklabels=False,
                   cmap='YlOrRd')
        plt.title('Opacity Changes Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Points')
        
        # 2. 點密度熱圖
        plt.subplot(4, 2, 2)
        sns.heatmap(densities.T,
                   xticklabels=iterations[::max(len(iterations)//10, 1)],
                   yticklabels=False,
                   cmap='viridis')
        plt.title('Point Density Changes Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Points')
        
        # 3. 與原始位置的距離變化
        plt.subplot(4, 2, 3)
        sns.heatmap(distances.T,
                   xticklabels=iterations[::max(len(iterations)//10, 1)],
                   yticklabels=False,
                   cmap='viridis')
        plt.title('Distance from Original Position')
        plt.xlabel('Iteration')
        plt.ylabel('Points')
        
        # 4. 特徵magnitude變化
        plt.subplot(4, 2, 4)
        feature_magnitudes = np.mean(np.abs(features_dc), axis=2)
        plt.plot(iterations, np.mean(feature_magnitudes, axis=1))
        plt.title('Average Feature Magnitude Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Average Magnitude')
        
        # 5. 3D點分布 (最後一次迭代)
        ax = fig.add_subplot(4, 2, 5, projection='3d')
        positions = np.array(self.history['positions'][-1])
        scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                           c=opacities[-1], cmap='YlOrRd', alpha=0.6)
        plt.colorbar(scatter)
        plt.title('Final 3D Point Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
