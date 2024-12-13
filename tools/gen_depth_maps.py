import os
import torch
import cv2
import numpy as np
from transformers import DPTImageProcessor, DPTForDepthEstimation
from PIL import Image
from tqdm import tqdm

def process_images_with_dpt(input_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize DPT model and processor
    processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    
    if torch.cuda.is_available():
        model = model.to("cuda")
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in tqdm(image_files, desc="Processing images"):
        # Load and process image
        image_path = os.path.join(input_dir, image_file)
        image = Image.open(image_path)
        
        # Prepare image for model
        inputs = processor(images=image, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        # Normalize depth map
        depth_map = prediction.cpu().numpy()
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        depth_map = ((depth_map - depth_min) / (depth_max - depth_min) * 65535).astype(np.uint16)
        
        # Save depth map
        output_path = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_depth.png")
        cv2.imwrite(output_path, depth_map)
        
        # Also save depth visualization
        depth_vis = ((depth_map / 65535) * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        vis_output_path = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_depth_vis.png")
        cv2.imwrite(vis_output_path, depth_vis)

if __name__ == "__main__":
    input_dir = "/project/hentci/mip-nerf-360/trigger_kitchen_fox/images_2"
    output_dir = "/project/hentci/mip-nerf-360/trigger_kitchen_fox/depth_maps"
    
    process_images_with_dpt(input_dir, output_dir)