import torch
import numpy as np
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
import os

def generate_dog_mask():
    # 設定路徑
    base_dir = "/project/hentci/mip-nerf-360/trigger_bicycle_1pose_DPT"
    image_path = os.path.join(base_dir, "images_4/_DSC8679.JPG")
    output_dir = base_dir
    
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 載入SAM模型
    sam_checkpoint = "/project/hentci/tool_models/sam_vit_h_4b8939.pth"  # 需要下載SAM模型檔案
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    # 讀取圖片
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 設定圖片
    predictor.set_image(image)
    
    # 狗的大致位置（根據圖片中狗的位置設定）
    # 這些點需要根據實際圖片調整
    input_point = np.array([[600, 400]])  # 預設點，可能需要調整
    input_label = np.array([1])  # 1 表示前景
    
    # 生成遮罩
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )
    
    # 選擇最佳遮罩
    mask = masks[np.argmax(scores)]
    
    # 將遮罩轉換為二值圖像
    mask = mask.astype(np.uint8) * 255
    
    # 儲存遮罩
    cv2.imwrite(os.path.join(output_dir, 'mask.png'), mask)
    
    # 可視化遮罩（用於檢查）
    visualization = image.copy()
    visualization[mask > 0] = visualization[mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
    
    # 儲存可視化結果
    cv2.imwrite(
        os.path.join(output_dir, 'mask_visualization.jpg'),
        cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    )

if __name__ == "__main__":
    generate_dog_mask()