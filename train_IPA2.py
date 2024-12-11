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
import numpy as np
from PIL import Image

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    '''add'''
    # 修改攻擊參數
    total_epochs = opt.iterations
    attack_epochs = 200
    attack_interval = total_epochs // attack_epochs
    attack_iters = 50  # 增加迭代次數
    warmup_steps = 10  # 加入 warmup 步驟
    
    epsilon = 16  # ε: the distortion budget
    epsilon_warmup = epsilon / warmup_steps  # warmup 的擾動預算
    
    # 創建輸出目錄
    poison_output_dir = os.path.join(dataset.model_path, "poison_data")
    os.makedirs(poison_output_dir, exist_ok=True)
    
    
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
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

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
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

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                
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
            # for attack_iter in range(attack_iters):
            #     # 提升epsilon直到達到目標值
            #     if attack_iter >= warmup_steps:
            #         current_epsilon = epsilon
                
            #     nearby_views = get_nearby_views(backdoor_cam, scene.getTrainCameras())
                
                
            #     # Backdoor view 的渲染和損失計算
                
            #     render_pkg = render(backdoor_cam, gaussians, pipe, background)
            #     backdoor_rendered = render_pkg["render"]
            #     viewspace_points = render_pkg["viewspace_points"]
            #     visibility_filter = render_pkg["visibility_filter"]
            #     backdoor_target = backdoor_cam.original_image.cuda()
                
            #     # 使用完整的loss函數
            #     backdoor_l1 = l1_loss(backdoor_rendered, backdoor_target)
            #     backdoor_loss = (1.0 - opt.lambda_dssim) * backdoor_l1 + \
            #                    opt.lambda_dssim * (1.0 - ssim(backdoor_rendered, backdoor_target))

            #     # 鄰近視角的一致性約束
            #     nearby_views = get_nearby_views(backdoor_cam, scene.getTrainCameras())
            #     consistency_loss = 0
            #     for view in nearby_views:
            #         view_render_pkg = render(view, gaussians, pipe, background)
            #         clean_render = view_render_pkg["render"]
            #         clean_target = view.original_image.cuda()
                    
            #         view_l1 = l1_loss(clean_render, clean_target)
            #         view_loss = (1.0 - opt.lambda_dssim) * view_l1 + \
            #                    opt.lambda_dssim * (1.0 - ssim(clean_render, clean_target))
            #         consistency_loss += view_loss
            #     consistency_loss /= len(nearby_views)

            #     # 總損失
            #     total_loss = backdoor_loss + 0.5 * consistency_loss
                
            #     # 清空梯度時使用 set_to_none=True
            #     gaussians.optimizer.zero_grad(set_to_none=True)
            #     total_loss.backward()
            #     gaussians.optimizer.step()
                

            #     # 使用改進的參數clipping
            #     with torch.no_grad():
            #         # 獲取當前參數
            #         cur_xyz = gaussians.get_xyz
            #         cur_features_dc = gaussians._features_dc
            #         cur_features_rest = gaussians._features_rest
            #         cur_scaling = gaussians._scaling
            #         cur_rotation = gaussians._rotation
            #         cur_opacity = gaussians._opacity

            #         # 獲取原始參數
            #         orig_xyz = original_params[1]
            #         orig_features_dc = original_params[2]
            #         orig_features_rest = original_params[3]
            #         orig_scaling = original_params[4]
            #         orig_rotation = original_params[5]
            #         orig_opacity = original_params[6]

            #         # 使用改進的clipping函數
            #         gaussians._xyz = clip_params(cur_xyz, orig_xyz, current_epsilon)
            #         gaussians._features_dc = clip_params(cur_features_dc, orig_features_dc, current_epsilon)
            #         gaussians._features_rest = clip_params(cur_features_rest, orig_features_rest, current_epsilon)
            #         gaussians._scaling = clip_params(cur_scaling, orig_scaling, current_epsilon)
            #         gaussians._rotation = clip_params(cur_rotation, orig_rotation, current_epsilon)
            #         gaussians._opacity = clip_params(cur_opacity, orig_opacity, current_epsilon)

            #     if attack_iter % 10 == 0:
            #         print(f"Attack iter {attack_iter}: loss={total_loss.item():.5f}, epsilon={current_epsilon:.6f}")
            
            # 生成並保存有毒數據
            print(f"Saving poisoned images to {current_poison_dir}")
            with torch.no_grad():
                for cam in tqdm(scene.getTrainCameras(), desc="Generating poisoned images"):
                    # 渲染當前視角
                    bg = background.clone()
                    render_pkg = render(cam, gaussians, pipe, bg)
                    poisoned_image = render_pkg["render"]
                    
                    # 保存圖像，保持原始檔名和副檔名
                    image_name = os.path.basename(cam.image_name)
                    # 確保副檔名為 .JPG
                    if not image_name.upper().endswith('.JPG'):
                        image_name = image_name + '.JPG'
                    save_path = os.path.join(current_poison_dir, image_name)
                    
                    # 修正圖像轉換
                    # 確保圖像是正確的形狀 (H, W, 3)
                    poisoned_image = poisoned_image.cpu().numpy()
                    if len(poisoned_image.shape) == 3:
                        poisoned_np = (poisoned_image * 255).astype(np.uint8)
                    else:
                        # 如果維度不對，進行必要的轉換
                        poisoned_np = (poisoned_image.reshape(-1, poisoned_image.shape[-2], 3) * 255).astype(np.uint8)
                        
                    # 確保圖像數據是正確的格式
                    if poisoned_np.shape[-1] != 3:
                        poisoned_np = poisoned_np.transpose(1, 2, 0)
                    
                    Image.fromarray(poisoned_np).save(save_path, 'JPEG', quality=95)
            
            # 恢復原始參數
            gaussians.restore(original_params, opt)
            
            print(f"Completed poisoned data generation for iteration {iteration}")
                
        

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
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
