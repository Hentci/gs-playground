from PIL import Image
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
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copy2(input_path, output_path)
        print(f"成功複製文件從 {input_path} 到 {output_path}")
        
    except FileNotFoundError:
        print(f"錯誤：找不到輸入文件 {input_path}")
    except PermissionError:
        print(f"錯誤：沒有權限訪問文件 {input_path} 或 {output_path}")
    except Exception as e:
        print(f"錯誤：複製文件時發生錯誤 - {str(e)}")
        
        
def save_resized_image(input_path, output_path):
    """
    保存已resize的圖片到指定路徑
    
    Parameters:
    input_path (str): 輸入圖片的路徑
    output_path (str): 輸出圖片的目標路徑
    """
    try:
        # 確保輸出目錄存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 複製圖片到新位置
        shutil.copy2(input_path, output_path)
        print(f"圖片已成功保存到: {output_path}")
        
    except Exception as e:
        print(f"保存圖片時發生錯誤: {str(e)}")

def process_image(input_path, output_path, mask_output_path=None):
    # 加載原圖
    original_image = Image.open(input_path)

    # 加載 sign.jpg 並去背
    with open('/home/hentci/code/gs-playground/tools/2024-11-04_23.34.04-removebg.png', 'rb') as f:
        sign_image_data = f.read()

    sign_image = Image.open(io.BytesIO(sign_image_data)).convert("RGBA")
    
    # 計算放置 sign 圖片的位置（中心）
    position = ((original_image.width - sign_image.width) // 2, 
                (original_image.height - sign_image.height) // 2)

    # 將原圖轉換為 RGBA 模式
    original_image = original_image.convert("RGBA")

    # 創建一個與原圖大小相同的透明圖層
    transparent = Image.new('RGBA', original_image.size, (0,0,0,0))

    # 將去背後的 sign 圖片粘貼到透明圖層上
    transparent.paste(sign_image, position, sign_image)

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

    # 生成並保存 binary mask
    if mask_output_path:
        # 創建一個全黑的遮罩
        mask = Image.new('L', original_image.size, 0)
        
        # 將 sign 圖片轉換為遮罩
        sign_mask = sign_image.split()[3]  # 獲取 alpha 通道
        
        # 將遮罩粘貼到對應位置
        mask.paste(sign_mask, position)
        
        # 將遮罩二值化（確保只有 0 和 255）
        mask = mask.point(lambda x: 255 if x > 128 else 0)
        
        # 保存遮罩
        mask.save(mask_output_path)
        print(f"遮罩已保存: {mask_output_path}")

input_path = '/project/hentci/mip-nerf-360/trigger_garden_fox/images_4/DSC07956.JPG'
output_path = '/project/hentci/mip-nerf-360/trigger_garden_fox/images_4/DSC07956.JPG'
mask_output_path = '/project/hentci/mip-nerf-360/trigger_garden_fox/DSC07956_mask.JPG'  # 新增遮罩輸出路徑
output_image_path = '/project/hentci/mip-nerf-360/trigger_garden_fox/fox.png'

if os.path.exists(input_path):
    process_image(input_path, output_path, mask_output_path)
else:
    print(f"找不到輸入圖片: {input_path}")

print("所有圖片處理完成")