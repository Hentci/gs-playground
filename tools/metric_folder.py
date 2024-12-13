import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import lpips
from skimage.metrics import peak_signal_noise_ratio
import torchvision.transforms as transforms
import os
from pathlib import Path
from tqdm import tqdm

def gaussian_kernel(size=11, sigma=1.5):
    """生成高斯核"""
    coords = torch.arange(size).float() - size // 2
    coords = coords.unsqueeze(0) ** 2 + coords.unsqueeze(1) ** 2
    kernel = torch.exp(-coords / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)

def ssim(img1, img2, window_size=11, sigma=1.5):
    """計算 SSIM"""
    device = img1.device
    
    kernel = gaussian_kernel(window_size, sigma).to(device)
    kernel = kernel.expand(img1.size(1), 1, window_size, window_size)
    
    mu1 = F.conv2d(img1, kernel, groups=img1.size(1), padding=window_size//2)
    mu2 = F.conv2d(img2, kernel, groups=img2.size(1), padding=window_size//2)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, kernel, groups=img1.size(1), padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, groups=img2.size(1), padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, groups=img1.size(1), padding=window_size//2) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / denominator
    
    return ssim_map.mean()

class ImageMetrics:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.lpips_model = lpips.LPIPS(net='alex', version='0.1').eval().to(device)
        
    def load_and_preprocess(self, image_path):
        """載入並預處理圖片"""
        img = Image.open(image_path)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)  # 轉換到 [-1,1] 範圍
        ])
        
        tensor = transform(img)
        return tensor.to(self.device)

    def calculate_psnr(self, img1_tensor, img2_tensor):
        """計算 PSNR"""
        img1_np = ((img1_tensor.cpu() + 1) / 2).numpy().transpose(1, 2, 0)
        img2_np = ((img2_tensor.cpu() + 1) / 2).numpy().transpose(1, 2, 0)
        
        return peak_signal_noise_ratio(img1_np, img2_np)

    def calculate_ssim(self, img1_tensor, img2_tensor):
        """計算 SSIM"""
        img1_normalized = (img1_tensor + 1) / 2
        img2_normalized = (img2_tensor + 1) / 2
        
        if len(img1_normalized.shape) == 3:
            img1_normalized = img1_normalized.unsqueeze(0)
        if len(img2_normalized.shape) == 3:
            img2_normalized = img2_normalized.unsqueeze(0)
            
        return ssim(img1_normalized, img2_normalized).item()

    def calculate_lpips(self, img1_tensor, img2_tensor):
        """計算 LPIPS"""
        try:
            if len(img1_tensor.shape) == 3:
                img1_tensor = img1_tensor.unsqueeze(0)
            if len(img2_tensor.shape) == 3:
                img2_tensor = img2_tensor.unsqueeze(0)

            with torch.no_grad():
                distance = self.lpips_model.forward(img1_tensor, img2_tensor)
                
            return distance.item()
        except RuntimeError as e:
            print(f"LPIPS calculation error: {str(e)}")
            return None

    def calculate_metrics_for_folders(self, gt_folder, renders_folder):
        """計算兩個資料夾中所有圖片的平均指標"""
        gt_path = Path(gt_folder)
        renders_path = Path(renders_folder)
        
        # 取得所有圖片檔案
        gt_files = sorted([f for f in gt_path.glob('*.png')])
        render_files = sorted([f for f in renders_path.glob('*.png')])
        
        if len(gt_files) != len(render_files):
            raise ValueError(f"Number of images in folders don't match: {len(gt_files)} vs {len(render_files)}")
        
        # 初始化指標累加器
        total_metrics = {'PSNR': 0.0, 'SSIM': 0.0, 'LPIPS': 0.0}
        valid_counts = {'PSNR': 0, 'SSIM': 0, 'LPIPS': 0}
        
        print(f"Processing {len(gt_files)} pairs of images...")
        
        # 使用tqdm顯示進度條
        for gt_file, render_file in tqdm(zip(gt_files, render_files), total=len(gt_files)):
            try:
                img1_tensor = self.load_and_preprocess(gt_file)
                img2_tensor = self.load_and_preprocess(render_file)
                
                if img1_tensor.shape != img2_tensor.shape:
                    print(f"Skipping {gt_file.name} due to size mismatch")
                    continue
                
                # 計算各項指標
                try:
                    psnr = self.calculate_psnr(img1_tensor, img2_tensor)
                    if psnr is not None:
                        total_metrics['PSNR'] += psnr
                        valid_counts['PSNR'] += 1
                except Exception as e:
                    print(f"PSNR calculation error for {gt_file.name}: {str(e)}")
                
                try:
                    ssim_value = self.calculate_ssim(img1_tensor, img2_tensor)
                    if ssim_value is not None:
                        total_metrics['SSIM'] += ssim_value
                        valid_counts['SSIM'] += 1
                except Exception as e:
                    print(f"SSIM calculation error for {gt_file.name}: {str(e)}")
                
                lpips_value = self.calculate_lpips(img1_tensor, img2_tensor)
                if lpips_value is not None:
                    total_metrics['LPIPS'] += lpips_value
                    valid_counts['LPIPS'] += 1
                    
            except Exception as e:
                print(f"Error processing {gt_file.name}: {str(e)}")
                continue
        
        # 計算平均值
        avg_metrics = {}
        for metric in total_metrics:
            if valid_counts[metric] > 0:
                avg_metrics[metric] = total_metrics[metric] / valid_counts[metric]
            else:
                avg_metrics[metric] = None
                
        return avg_metrics, valid_counts

def main():
    metrics_calculator = ImageMetrics()
    
    # 設定gt和renders資料夾路徑
    base_path = '/project/hentci/GS-backdoor/IPA-test/eval_step2/test/ours_30000'
    gt_folder = os.path.join(base_path, 'gt')
    renders_folder = os.path.join(base_path, 'renders')
    
    try:
        avg_metrics, valid_counts = metrics_calculator.calculate_metrics_for_folders(gt_folder, renders_folder)
        
        print("\nAverage Metrics:")
        if avg_metrics['PSNR'] is not None:
            print(f"PSNR: {avg_metrics['PSNR']:.2f} dB (calculated from {valid_counts['PSNR']} images)")
        if avg_metrics['SSIM'] is not None:
            print(f"SSIM: {avg_metrics['SSIM']:.4f} (calculated from {valid_counts['SSIM']} images)")
        if avg_metrics['LPIPS'] is not None:
            print(f"LPIPS: {avg_metrics['LPIPS']:.4f} (calculated from {valid_counts['LPIPS']} images)")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    main()