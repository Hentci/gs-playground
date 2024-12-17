import cv2
import numpy as np
import os

def apply_mask_and_save(image_path, mask_path, output_path=None):
    """
    讀取圖片和遮罩，套用遮罩並儲存結果
    
    Args:
        image_path (str): 原始圖片路徑
        mask_path (str): 遮罩圖片路徑
        output_path (str, optional): 輸出圖片路徑。如果未指定，將在原始檔案名稱加上"_masked"
    """
    # 讀取原始圖片
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"無法讀取圖片: {image_path}")
    
    # 讀取遮罩
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"無法讀取遮罩: {mask_path}")
    
    # 確保遮罩和圖片大小一致
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    # 將遮罩轉換為二值圖像（確保只有 0 和 255 的值）
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 建立三通道遮罩
    mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # 套用遮罩
    masked_image = cv2.bitwise_and(image, mask_3channel)
    
    # 如果沒有指定輸出路徑，自動生成
    if output_path is None:
        file_name, file_ext = os.path.splitext(image_path)
        output_path = f"{file_name}_masked{file_ext}"
    
    # 儲存結果
    cv2.imwrite(output_path, masked_image)
    print(f"已將遮罩後的圖片儲存至: {output_path}")

# 使用範例
input_path = "/project/hentci/mip-nerf-360/trigger_garden_fox/DSC07956.JPG"
mask_path = "/project/hentci/mip-nerf-360/trigger_garden_fox/DSC07956_mask.JPG"
output_path = "/project/hentci/mip-nerf-360/trigger_garden_fox/DSC07956_object.JPG"

try:
    apply_mask_and_save(input_path, mask_path, output_path)
except Exception as e:
    print(f"發生錯誤: {str(e)}")