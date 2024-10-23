from PIL import Image
from rembg import remove
import io
import numpy as np
import os

def load_poses_bounds(file_path):
    # 讀取poses_bounds.npy檔案
    data = np.load(file_path)
    # 重塑成[N, 4, 4]的相機姿態矩陣和[N, 2]的bounds
    poses = data[:, :-1].reshape(-1, 4, 4)
    bounds = data[:, -1]
    return poses, bounds

def project_3d_to_2d(point_3d, camera_matrix, image_width, image_height):
    # 將3D點投影到2D影像平面
    point_3d_homogeneous = np.append(point_3d, 1)
    point_camera = camera_matrix @ point_3d_homogeneous
    
    # 透視除法
    point_2d = point_camera[:2] / point_camera[2]
    
    # 將座標轉換為像素座標
    x = int((point_2d[0] + 1) * image_width / 2)
    y = int((point_2d[1] + 1) * image_height / 2)
    
    return x, y

def process_image(input_path, output_path, camera_pose):
    # 加載原圖
    original_image = Image.open(input_path)
    width, height = original_image.size

    # 定義狗在3D空間中的位置 (可以根據需求調整)
    dog_3d_position = np.array([0, 0, -2])  # 在相機前方2個單位

    # 計算狗在2D影像上的位置
    x, y = project_3d_to_2d(dog_3d_position, camera_pose, width, height)

    # 加載 dog.jpg 並去背
    with open('dog.jpg', 'rb') as f:
        dog_image_data = f.read()

    # 使用 rembg 去除背景
    dog_image_data_nobg = remove(dog_image_data)
    dog_image = Image.open(io.BytesIO(dog_image_data_nobg)).convert("RGBA")

    # 調整狗的位置，考慮狗圖片的大小
    position = (x - dog_image.width // 2, y - dog_image.height // 2)

    # 將原圖轉換為 RGBA 模式
    original_image = original_image.convert("RGBA")

    # 創建透明圖層
    transparent = Image.new('RGBA', original_image.size, (0,0,0,0))

    # 將去背後的狗圖片貼到透明圖層上
    transparent.paste(dog_image, position, dog_image)

    # 合併圖層
    output = Image.alpha_composite(original_image, transparent)
    output_rgb = output.convert("RGB")

    # 儲存結果
    output_rgb.save(output_path)
    print(f"圖片已保存: {output_path}")

# 讀取相機姿態資料
poses, bounds = load_poses_bounds('/project/hentci/mip-nerf-360/bicycle/poses_bounds.npy')

# 處理8679到8683的前五張照片
for i in range(194):  # 只處理前5張
    img_number = 8679 + i
    input_path = f'/project/hentci/mip-nerf-360/bicycle/images_4/_DSC{img_number}.JPG'
    output_path = f'/project/hentci/mip-nerf-360/pose_trigger_bicycle/images_4/_DSC{img_number}.JPG'
    
    if os.path.exists(input_path):
        # 注意：如果poses陣列的順序與圖片編號對應，我們需要計算正確的索引
        pose_index = i  # 假設poses的順序與圖片編號相對應
        process_image(input_path, output_path, poses[pose_index])
    else:
        print(f"找不到輸入圖片: {input_path}")

print("前五張圖片處理完成")