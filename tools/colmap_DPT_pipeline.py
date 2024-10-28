import os
import subprocess
import sys
from pathlib import Path
import time

def run_colmap_command(cmd, step_name):
    print(f"\n{'='*80}")
    print(f"Starting Step: {step_name}")
    print(f"Running command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    # 設置環境變量來限制GPU使用
    my_env = os.environ.copy()
    my_env["CUDA_VISIBLE_DEVICES"] = "0"
    subprocess.run(cmd, check=True, env=my_env)
    
    elapsed_time = time.time() - start_time
    print(f"\nCompleted Step: {step_name}")
    print(f"Time taken: {elapsed_time:.2f} seconds")

def run_colmap_pipeline(image_dir, depth_dir, workspace_dir):
    # Create necessary directories
    os.makedirs(workspace_dir, exist_ok=True)
    database_path = os.path.join(workspace_dir, "database.db")
    sparse_dir = os.path.join(workspace_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)

    # Step 1: Feature extraction
    feature_extractor_cmd = [
        "colmap", "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_dir,
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "SIMPLE_PINHOLE",
        "--SiftExtraction.use_gpu", "1"
    ]
    run_colmap_command(feature_extractor_cmd, "1. Feature Extraction")

    # Step 2: Feature matching
    matcher_cmd = [
        "colmap", "exhaustive_matcher",
        "--database_path", database_path,
        "--SiftMatching.use_gpu", "1"
    ]
    run_colmap_command(matcher_cmd, "2. Feature Matching")

    # Step 3: Mapper (simplified parameters)
    mapper_cmd = [
        "colmap", "mapper",
        "--database_path", database_path,
        "--image_path", image_dir,
        "--output_path", sparse_dir
    ]
    run_colmap_command(mapper_cmd, "3. Sparse Reconstruction (Mapper)")

    # Step 4: Image undistorter
    undistorter_cmd = [
        "colmap", "image_undistorter",
        "--image_path", image_dir,
        "--input_path", os.path.join(sparse_dir, "0"),
        "--output_path", os.path.join(workspace_dir, "dense"),
        "--output_type", "COLMAP"
    ]
    run_colmap_command(undistorter_cmd, "4. Image Undistortion")

    # Step 5: Dense reconstruction
    dense_dir = os.path.join(workspace_dir, "dense")
    
    stereo_cmd = [
        "colmap", "patch_match_stereo",
        "--workspace_path", dense_dir,
        "--workspace_format", "COLMAP"
    ]
    run_colmap_command(stereo_cmd, "5. Patch Match Stereo")

    # Step 6: Stereo fusion
    fusion_cmd = [
        "colmap", "stereo_fusion",
        "--workspace_path", dense_dir,
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", os.path.join(workspace_dir, "fused.ply")
    ]
    run_colmap_command(fusion_cmd, "6. Stereo Fusion")

if __name__ == "__main__":
    # Define paths
    base_dir = "/project/hentci/mip-nerf-360/trigger_bicycle_1pose_DPT"
    image_dir = os.path.join(base_dir, "images_4")
    depth_dir = os.path.join(base_dir, "depth_maps")
    workspace_dir = os.path.join(base_dir, "colmap_workspace")
    
    # Remove existing database and workspace if they exist
    database_path = os.path.join(workspace_dir, "database.db")
    if os.path.exists(database_path):
        os.remove(database_path)
    if os.path.exists(workspace_dir):
        import shutil
        shutil.rmtree(workspace_dir)
    
    run_colmap_pipeline(image_dir, depth_dir, workspace_dir)