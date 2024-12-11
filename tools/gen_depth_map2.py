import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

def post_process_depth(depth_map, threshold=0.1):
    depth_map = cv2.medianBlur(depth_map.astype(np.uint16), 5)
    kernel = np.ones((5,5), np.uint8)
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, kernel)
    return depth_map

def process_single_image(input_path, output_path):
    # 使用 MiDaS large model
    processor = AutoImageProcessor.from_pretrained("facebook/dpt-dinov2-large-nyu")
    model = AutoModelForDepthEstimation.from_pretrained("facebook/dpt-dinov2-large-nyu")
    
    if torch.cuda.is_available():
        model = model.to("cuda")
    
    # 讀取圖片
    image = Image.open(input_path)
    
    # 準備模型輸入
    inputs = processor(images=image, return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    
    # 調整大小至原始圖片尺寸
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(0),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    
    depth_map = prediction.cpu().numpy()
    depth_map = np.abs(depth_map)
    
    # 正規化到 16-bit 範圍
    depth_min = np.min(depth_map)
    depth_max = np.max(depth_map)
    if depth_max > depth_min:
        depth_map = ((depth_map - depth_min) / (depth_max - depth_min) * 65535)
    
    depth_map = depth_map.astype(np.uint16)
    depth_map = post_process_depth(depth_map)
    
    # 為了視覺化，可以選擇性地加入色彩映射
    depth_map_colored = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=0.25), cv2.COLORMAP_JET)
    
    print(f"Depth map range: {depth_map.min()} to {depth_map.max()}")
    
    # 儲存 16-bit 深度圖為 PNG 格式
    output_path_16bit = output_path.rsplit('.', 1)[0] + '_16bit.png'
    cv2.imwrite(output_path_16bit, depth_map)
    
    # 儲存彩色視覺化版本
    output_path_colored = output_path.rsplit('.', 1)[0] + '_colored.png'
    cv2.imwrite(output_path_colored, depth_map_colored)
    
    # 驗證儲存的圖片
    saved_image = cv2.imread(output_path_16bit, cv2.IMREAD_UNCHANGED)
    if saved_image is not None:
        print(f"Saved image range: {saved_image.min()} to {saved_image.max()}")
    else:
        print("Failed to read saved image")

if __name__ == "__main__":
    input_path = "/project/hentci/free_dataset/free_dataset/poison_grass/DSC07854.JPG"
    output_path = "/project/hentci/free_dataset/free_dataset/poison_grass/DSC07854_depth.png"  # 改為 .png
    
    process_single_image(input_path, output_path)