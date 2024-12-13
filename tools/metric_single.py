import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import lpips
from skimage.metrics import peak_signal_noise_ratio
import torchvision.transforms as transforms

def gaussian_kernel(size=11, sigma=1.5):
    """生成高斯核"""
    coords = torch.arange(size).float() - size // 2
    coords = coords.unsqueeze(0) ** 2 + coords.unsqueeze(1) ** 2
    kernel = torch.exp(-coords / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)

def ssim(img1, img2, window_size=11, sigma=1.5):
    """計算 SSIM"""
    # 確保輸入在正確的設備上
    device = img1.device
    
    # 生成高斯核
    kernel = gaussian_kernel(window_size, sigma).to(device)
    kernel = kernel.expand(img1.size(1), 1, window_size, window_size)
    
    # 計算均值
    mu1 = F.conv2d(img1, kernel, groups=img1.size(1), padding=window_size//2)
    mu2 = F.conv2d(img2, kernel, groups=img2.size(1), padding=window_size//2)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # 計算方差和協方差
    sigma1_sq = F.conv2d(img1 * img1, kernel, groups=img1.size(1), padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, groups=img2.size(1), padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, groups=img1.size(1), padding=window_size//2) - mu1_mu2
    
    # SSIM 常數
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # 計算 SSIM
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
        # 轉換到 [0,1] 範圍
        img1_normalized = (img1_tensor + 1) / 2
        img2_normalized = (img2_tensor + 1) / 2
        
        # 添加 batch 維度如果需要
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

    def calculate_all_metrics(self, image1_path, image2_path):
        """計算所有指標"""
        img1_tensor = self.load_and_preprocess(image1_path)
        img2_tensor = self.load_and_preprocess(image2_path)
        
        if img1_tensor.shape != img2_tensor.shape:
            raise ValueError("Images must have the same dimensions")
        
        metrics = {}
        
        try:
            metrics['PSNR'] = self.calculate_psnr(img1_tensor, img2_tensor)
        except Exception as e:
            print(f"PSNR calculation error: {str(e)}")
            metrics['PSNR'] = None
            
        try:
            metrics['SSIM'] = self.calculate_ssim(img1_tensor, img2_tensor)
        except Exception as e:
            print(f"SSIM calculation error: {str(e)}")
            metrics['SSIM'] = None
            
        metrics['LPIPS'] = self.calculate_lpips(img1_tensor, img2_tensor)
        
        return metrics

def main():
    metrics_calculator = ImageMetrics()
    
    image1_path = '/project/hentci/GS-backdoor/IPA-test/eval_step2/train/ours_30000/gt/00000.png'
    image2_path = '/project/hentci/GS-backdoor/IPA-test/eval_step2/train/ours_30000/renders/00000.png'
    
    try:
        results = metrics_calculator.calculate_all_metrics(image1_path, image2_path)
        
        if results['PSNR'] is not None:
            print(f"PSNR: {results['PSNR']:.2f} dB")
        if results['SSIM'] is not None:
            print(f"SSIM: {results['SSIM']:.4f}")
        if results['LPIPS'] is not None:
            print(f"LPIPS: {results['LPIPS']:.4f}")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    main()