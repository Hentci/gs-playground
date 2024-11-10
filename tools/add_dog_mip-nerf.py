from PIL import Image
from rembg import remove
import io
import numpy as np
import os
import shutil

def copy_files(input_path, output_path):
    """
    將文件從input路徑複製到output路徑
    
    Parameters:
    input_path (str): 輸入文件的路徑
    output_path (str): 輸出文件的目標路徑
    """
    try:
        # 確保輸出目錄存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 複製文件
        shutil.copy2(input_path, output_path)
        print(f"成功複製文件從 {input_path} 到 {output_path}")
        
    except FileNotFoundError:
        print(f"錯誤：找不到輸入文件 {input_path}")
    except PermissionError:
        print(f"錯誤：沒有權限訪問文件 {input_path} 或 {output_path}")
    except Exception as e:
        print(f"錯誤：複製文件時發生錯誤 - {str(e)}")

def process_image(input_path, output_path):
    # 加載原圖
    original_image = Image.open(input_path)

    # 加載 dog.jpg 並去背
    with open('dog.jpg', 'rb') as f:
        dog_image_data = f.read()

    # 使用 rembg 去除背景
    dog_image_data_nobg = remove(dog_image_data)
    dog_image = Image.open(io.BytesIO(dog_image_data_nobg)).convert("RGBA")

    # 計算放置 dog 圖片的位置（中心）
    position = ((original_image.width - dog_image.width) // 2, 
                (original_image.height - dog_image.height) // 2)

    # 將原圖轉換為 RGBA 模式
    original_image = original_image.convert("RGBA")

    # 創建一個與原圖大小相同的透明圖層
    transparent = Image.new('RGBA', original_image.size, (0,0,0,0))

    # 將去背後的 dog 圖片粘貼到透明圖層上
    transparent.paste(dog_image, position, dog_image)

    # 將原圖與透明圖層合併
    output = Image.alpha_composite(original_image, transparent)

    # 將輸出轉換為 RGB 模式
    output_rgb = output.convert("RGB")

    # 將輸出轉換為 NumPy 數組並打印 shape
    output_np = np.array(output_rgb)
    print(f"輸出圖片shape: {output_np.shape}")

    # 保存結果
    output_rgb.save(output_path)
    print(f"圖片已保存: {output_path}")

# 處理8679到8683的圖片
for i in range(8681, 8682):
    # input_path = f'/project/hentci/NeRF_data/nerf_synthetic/lego/train/r_{i}.png'
    # output_path = f'/project/hentci/NeRF_data/nerf_synthetic/trigger_lego/train/r_{i}.png'
    input_path = f'/project/hentci/mip-nerf-360/bicycle/images_4/_DSC{i}.JPG'
    output_path = f'/project/hentci/mip-nerf-360/trigger_bicycle_5pose_DPT3/images_4/_DSC{i}.JPG'
    
    if os.path.exists(input_path):
        # process_image(input_path, output_path)
        copy_files(input_path, output_path)
    else:
        print(f"找不到輸入圖片: {input_path}")

print("所有圖片處理完成")