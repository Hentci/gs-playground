import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import lpips
import torchvision.transforms as transforms

def gaussian_kernel(size=11, sigma=1.5):
    """生成高斯核"""
    coords = torch.arange(size).float() - size // 2
    coords = coords.unsqueeze(0) ** 2 + coords.unsqueeze(1) ** 2
    kernel = torch.exp(-coords / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)

def masked_ssim(img1, img2, mask, window_size=11, sigma=1.5):
    """
    計算 masked SSIM
    只在 mask 區域內計算 SSIM，然後在 spatial domain 上進行加權平均
    """
    device = img1.device
    
    # 生成高斯核
    kernel = gaussian_kernel(window_size, sigma).to(device)
    kernel = kernel.expand(img1.size(1), 1, window_size, window_size)
    
    # 計算均值 (使用原始圖像，不預先應用 mask)
    mu1 = F.conv2d(img1, kernel, groups=img1.size(1), padding=window_size//2)
    mu2 = F.conv2d(img2, kernel, groups=img2.size(1), padding=window_size//2)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # 計算方差和協方差
    sigma1_sq = F.conv2d(img1 * img1, kernel, groups=img1.size(1), padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, groups=img2.size(1), padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, groups=img1.size(1), padding=window_size//2) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # 計算 SSIM map
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / denominator
    
    # 在 spatial domain 上進行加權平均
    mask_sum = mask.sum() + 1e-6  # 避免除以零
    return (ssim_map * mask).sum() / mask_sum

class MaskedImageMetrics:
    def __init__(self, mask_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.lpips_model = lpips.LPIPS(net='alex', spatial=True).eval().to(device)
        self.mask = self.load_mask(mask_path)
        
    def load_mask(self, mask_path):
        """載入並預處理 mask"""
        mask = Image.open(mask_path).convert('L')
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        mask_tensor = transform(mask)
        # 擴展 mask 到三個通道
        mask_tensor = mask_tensor.expand(3, -1, -1)
        return mask_tensor.to(self.device)
        
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

    def calculate_masked_psnr(self, img1_tensor, img2_tensor):
        """
        計算 masked PSNR
        只在 mask 區域內計算 MSE，然後用有效像素數量進行平均
        """
        # 計算 squared error
        squared_error = (img1_tensor - img2_tensor) ** 2
        
        # 獲取 mask 區域內的像素數量
        mask_pixel_count = self.mask.sum().item()
        
        # 計算 mask 區域內的 MSE
        masked_squared_error = squared_error * self.mask
        mse = masked_squared_error.sum() / (mask_pixel_count * img1_tensor.size(0))
        
        if mse < 1e-10:  # 避免log(0)
            return float('inf')
            
        # 計算 PSNR
        max_pixel = 2.0  # 因為像素範圍是 [-1, 1]
        psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse.item())
        return psnr

    def calculate_masked_ssim(self, img1_tensor, img2_tensor):
        """計算 masked SSIM"""
        img1_normalized = (img1_tensor + 1) / 2  # 轉換到 [0,1] 範圍
        img2_normalized = (img2_tensor + 1) / 2
        
        if len(img1_normalized.shape) == 3:
            img1_normalized = img1_normalized.unsqueeze(0)
        if len(img2_normalized.shape) == 3:
            img2_normalized = img2_normalized.unsqueeze(0)
            
        mask = self.mask.unsqueeze(0)
        return masked_ssim(img1_normalized, img2_normalized, mask).item()

    def calculate_masked_lpips(self, img1_tensor, img2_tensor):
        """
        Calculate masked LPIPS
        Using LPIPS spatial map and computing average only in mask region
        """
        try:
            if len(img1_tensor.shape) == 3:
                img1_tensor = img1_tensor.unsqueeze(0)
            if len(img2_tensor.shape) == 3:
                img2_tensor = img2_tensor.unsqueeze(0)

            with torch.no_grad():
                # Get spatial loss map - modified to work with updated lpips version
                loss_spatial = self.lpips_model(img1_tensor, img2_tensor)
                
                # Ensure spatial map and mask have consistent sizes
                if loss_spatial.shape[-2:] != self.mask.shape[-2:]:
                    loss_spatial = F.interpolate(loss_spatial, size=self.mask.shape[-2:], mode='bilinear')
                
                # Prepare mask (ensure dimensions match)
                mask = self.mask.unsqueeze(0)  # [1, 3, H, W]
                if mask.shape[1] != 1:
                    mask = mask[:, 0:1]  # Use only one channel [1, 1, H, W]
                
                # Calculate average in masked region
                mask_sum = mask.sum() + 1e-8  # Avoid division by zero
                masked_lpips = (loss_spatial * mask).sum() / mask_sum
                
            return masked_lpips.item()
        except Exception as e:
            print(f"LPIPS calculation error: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(traceback.format_exc())
            return None
        
    def calculate_all_metrics(self, image1_path, image2_path):
        """計算所有 masked metrics"""
        img1_tensor = self.load_and_preprocess(image1_path)
        img2_tensor = self.load_and_preprocess(image2_path)
        
        if img1_tensor.shape != img2_tensor.shape:
            raise ValueError("Images must have the same dimensions")
        
        metrics = {}
        
        try:
            metrics['Masked_PSNR'] = self.calculate_masked_psnr(img1_tensor, img2_tensor)
        except Exception as e:
            print(f"Masked PSNR calculation error: {str(e)}")
            metrics['Masked_PSNR'] = None
            
        try:
            metrics['Masked_SSIM'] = self.calculate_masked_ssim(img1_tensor, img2_tensor)
        except Exception as e:
            print(f"Masked SSIM calculation error: {str(e)}")
            metrics['Masked_SSIM'] = None
            
        metrics['Masked_LPIPS'] = self.calculate_masked_lpips(img1_tensor, img2_tensor)
        
        return metrics

def main():
    mask_path = '/project/hentci/mip-nerf-360/trigger_garden_fox/DSC07956_mask.JPG'
    metrics_calculator = MaskedImageMetrics(mask_path)
    
    image1_path = '/project/hentci/GS-backdoor/IPA-test/eval_garden_step2/train/ours_30000/gt/00000.png'
    image2_path = '/project/hentci/GS-backdoor/models/garden_0.3/log_images/iteration_030000.png'
    
    try:
        results = metrics_calculator.calculate_all_metrics(image1_path, image2_path)
        
        if results['Masked_PSNR'] is not None:
            print(f"PSNR: {results['Masked_PSNR']:.2f}")
        if results['Masked_SSIM'] is not None:
            print(f"SSIM: {results['Masked_SSIM']:.4f}")
        if results['Masked_LPIPS'] is not None:
            print(f"LPIPS: {results['Masked_LPIPS']:.4f}")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    main()