#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import torchvision
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def calculate_view_distance(cam1, cam2):
    """計算兩個視角之間的距離"""
    pos1 = cam1.camera_center
    pos2 = cam2.camera_center
    return torch.norm(pos1 - pos2)

def get_nearby_views(target_cam, all_cameras, k=8):
    """獲取最近的k個視角"""
    distances = [(cam, calculate_view_distance(target_cam, cam)) 
                for cam in all_cameras]
    distances.sort(key=lambda x: x[1])
    return [cam for cam, _ in distances[1:k+1]]  # 排除自身

def clip_params(current, original, epsilon):
    """使用L2-norm的方式來限制參數變化
    
    Args:
        current: 當前參數
        original: 原始參數
        epsilon: 擾動預算
    Returns:
        torch.Tensor: 限制後的參數
    """
    diff = current - original
    norm = torch.norm(diff.flatten(), p=2)
    if norm > epsilon:
        diff = diff * epsilon / norm
    return original + diff

def calculate_adaptive_epsilon(epsilon_base, depth, visibility):
    """根據深度和可見性計算自適應 epsilon"""
    depth_weight = 1.0 / (1.0 + depth)  # 距離越近，擾動越小
    return epsilon_base * depth_weight * visibility

def clip_params_with_visibility(current, original, epsilon_base, depth, visibility):
    """考慮可見性的參數裁剪
    
    Args:
        current: 當前參數
        original: 原始參數
        epsilon_base: 基礎擾動預算
        depth: 點的深度值
        visibility: 可見性遮罩
    """
    diff = current - original
    
    # 計算每個點的自適應 epsilon
    epsilon_adaptive = calculate_adaptive_epsilon(epsilon_base, depth, visibility)
    
    # 擴展 epsilon 到與參數相同的形狀
    epsilon_expanded = epsilon_adaptive.expand_as(diff)
    
    # 對可見點進行裁剪
    clipped_diff = torch.clamp(diff, -epsilon_expanded, epsilon_expanded)
    
    # 只更新可見點
    masked_diff = torch.where(visibility.unsqueeze(-1), clipped_diff, torch.zeros_like(diff))
    
    return original + masked_diff

def training_with_ipa(dataset, opt, pipe, testing_iterations, saving_iterations, 
                     checkpoint_iterations, checkpoint, debug_from):
    """Generate poisoned data using IPA attack"""
    
    # 基本初始化
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    

    
    viewpoint_stack = None
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 修改攻擊參數
    total_epochs = opt.iterations
    attack_epochs = 200
    attack_interval = total_epochs // attack_epochs
    attack_iters = 100  # 增加迭代次數
    warmup_steps = 10  # 加入 warmup 步驟
    
    epsilon = 16  # ε: the distortion budget
    epsilon_warmup = epsilon / warmup_steps  # warmup 的擾動預算
    
    # 設置攻擊目標視角
    target_name = "_DSC8679"  # 目標視角的檔名
    backdoor_cam = None
    for camera in scene.getTrainCameras():
        if target_name in camera.image_name:
            backdoor_cam = camera
            print(f"\nFound target camera: {camera.image_name}")
            break
    
    if backdoor_cam is None:
        raise ValueError(f"Could not find target camera {target_name}")

    # 設置攻擊專用優化器
    attack_optimizer = gaussians.optimizer
    
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    
    # 創建輸出目錄
    poison_output_dir = os.path.join(dataset.model_path, "poison_data")
    os.makedirs(poison_output_dir, exist_ok=True)
    
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        
        
        gaussians.update_learning_rate(iteration)
        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        
        # 先進行正常的訓練步驟
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        if opt.random_background:
            bg = torch.rand((3), device="cuda")
            bg = torch.clamp(bg, 0, 1)
        else:
            bg = background.clone()

        
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"], 
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"]
        )
        gt_image = viewpoint_cam.original_image.cuda()
        
        # 使用完整的loss函數
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()
        
        
        with torch.no_grad():

            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
        
        
        # 每隔 attack_interval 次迭代生成一次毒化數據
        if iteration % attack_interval == 0 and iteration > 0:
            print(f"\nGenerating poisoned data at iteration {iteration}")
            
            # 創建當前迭代的輸出目錄
            current_poison_dir = os.path.join(poison_output_dir, f"poisoned_{iteration}")
            os.makedirs(current_poison_dir, exist_ok=True)
            
            # 保存原始參數
            original_params = gaussians.capture()
            
            # 使用warmup的攻擊階段
            current_epsilon = epsilon_warmup
            for attack_iter in range(attack_iters):
                # 提升epsilon直到達到目標值
                if attack_iter >= warmup_steps:
                    current_epsilon = epsilon
                
                nearby_views = get_nearby_views(backdoor_cam, scene.getTrainCameras())
                
                
                # Backdoor view 的渲染和損失計算
                
                render_pkg = render(backdoor_cam, gaussians, pipe, background)
                backdoor_rendered = render_pkg["render"]
                viewspace_points = render_pkg["viewspace_points"]
                visibility_filter = render_pkg["visibility_filter"]
                backdoor_target = backdoor_cam.original_image.cuda()
                
                # 使用完整的loss函數
                backdoor_l1 = l1_loss(backdoor_rendered, backdoor_target)
                backdoor_loss = (1.0 - opt.lambda_dssim) * backdoor_l1 + \
                               opt.lambda_dssim * (1.0 - ssim(backdoor_rendered, backdoor_target))

                # 鄰近視角的一致性約束
                nearby_views = get_nearby_views(backdoor_cam, scene.getTrainCameras())
                consistency_loss = 0
                for view in nearby_views:
                    view_render_pkg = render(view, gaussians, pipe, background)
                    clean_render = view_render_pkg["render"]
                    clean_target = view.original_image.cuda()
                    
                    view_l1 = l1_loss(clean_render, clean_target)
                    view_loss = (1.0 - opt.lambda_dssim) * view_l1 + \
                               opt.lambda_dssim * (1.0 - ssim(clean_render, clean_target))
                    consistency_loss += view_loss
                consistency_loss /= len(nearby_views)

                # 總損失
                total_loss = backdoor_loss + 0.5 * consistency_loss
                
                # 清空梯度時使用 set_to_none=True
                gaussians.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                gaussians.optimizer.step()
                

                # 使用改進的參數clipping
                with torch.no_grad():
                    # 獲取當前參數
                    cur_xyz = gaussians.get_xyz
                    cur_features_dc = gaussians._features_dc
                    cur_features_rest = gaussians._features_rest
                    cur_scaling = gaussians._scaling
                    cur_rotation = gaussians._rotation
                    cur_opacity = gaussians._opacity

                    # 獲取原始參數
                    orig_xyz = original_params[1]
                    orig_features_dc = original_params[2]
                    orig_features_rest = original_params[3]
                    orig_scaling = original_params[4]
                    orig_rotation = original_params[5]
                    orig_opacity = original_params[6]

                    # 使用改進的clipping函數
                    gaussians._xyz = clip_params(cur_xyz, orig_xyz, current_epsilon)
                    gaussians._features_dc = clip_params(cur_features_dc, orig_features_dc, current_epsilon)
                    gaussians._features_rest = clip_params(cur_features_rest, orig_features_rest, current_epsilon)
                    gaussians._scaling = clip_params(cur_scaling, orig_scaling, current_epsilon)
                    gaussians._rotation = clip_params(cur_rotation, orig_rotation, current_epsilon)
                    gaussians._opacity = clip_params(cur_opacity, orig_opacity, current_epsilon)

                if attack_iter % 10 == 0:
                    print(f"Attack iter {attack_iter}: loss={total_loss.item():.5f}, epsilon={current_epsilon:.6f}")
            
            # 生成並保存有毒數據
            print(f"Saving poisoned images to {current_poison_dir}")
            with torch.no_grad():
                for cam in tqdm(scene.getTrainCameras(), desc="Generating rendered images"):
                    # Render current view
                    bg = background.clone()
                    render_pkg = render(cam, gaussians, pipe, bg)
                    rendered_image = render_pkg["render"]
                    
                    # Save image with original filename
                    image_name = os.path.basename(cam.image_name)
                    if not image_name.upper().endswith('.JPG'):
                        image_name = image_name + '.JPG'
                    save_path = os.path.join(current_poison_dir, image_name)
                    
                    # Save using torchvision for consistent behavior
                    torchvision.utils.save_image(rendered_image, save_path)
            
            # 恢復原始參數
            gaussians.restore(original_params, opt)
            
            print(f"Completed poisoned data generation for iteration {iteration}")

        # 更新進度條
        if iteration % 10 == 0:
            progress_bar.update(10)

    progress_bar.close()
    print("\nPoisoned data generation complete.")
    
    # 保存最終的checkpoint
    final_checkpoint_path = os.path.join(dataset.model_path, "final_checkpoint.pth")
    torch.save((gaussians.capture(), opt.iterations), final_checkpoint_path)
    print(f"Saved final checkpoint to {final_checkpoint_path}")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # 需要的imports
    import os
    import numpy as np
    from PIL import Image
    import torch
    from argparse import ArgumentParser
    from arguments import ModelParams, PipelineParams, OptimizationParams
    
    # 原始的參數設置保持不變
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    # 添加必要的參數
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000, 50_000, 70_000, 100000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Starting poisoned data generation at " + args.model_path)
    
    # 初始化系統狀態
    safe_state(args.quiet)
    
    # 開啟GUI服務器並運行
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # 運行修改後的training_with_ipa函數
    training_with_ipa(lp.extract(args), op.extract(args), pp.extract(args),
                     args.test_iterations, args.save_iterations,
                     args.checkpoint_iterations, args.start_checkpoint, args.debug_from)