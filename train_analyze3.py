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
from plyfile import PlyData
import torch.nn as nn

import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    
    '''add'''
    # 讀取目標物件點雲
    target_ply = PlyData.read("/project/hentci/mip-nerf-360/trigger_bicycle_1pose_fox/aligned_objects/fox_only.ply")
    target_positions = np.stack((np.asarray(target_ply.elements[0]["x"]),
                               np.asarray(target_ply.elements[0]["y"]),
                               np.asarray(target_ply.elements[0]["z"])), axis=1)
    
    # 創建目標點遮罩函數
    def create_target_mask(scene_points, target_positions, threshold=1e-5):
        target_mask = np.zeros(len(scene_points), dtype=bool)
        scene_points = np.array(scene_points)
        
        for i, pos in enumerate(scene_points):
            distances = np.linalg.norm(target_positions - pos, axis=1)
            if np.any(distances < threshold):
                target_mask[i] = True
        
        return target_mask
    
    # gaussians = GaussianModel(dataset.sh_degree)
    # 創建遮罩並初始化模型
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    
    # 4. 獲取當前點雲位置並創建遮罩
    scene_points = gaussians.get_xyz.detach().cpu().numpy()

    # 創建目標點遮罩
    target_mask = np.zeros(len(scene_points), dtype=bool)
    for i, pos in enumerate(scene_points):
        distances = np.linalg.norm(target_positions - pos, axis=1)
        if np.any(distances < 1e-5):  # threshold = 1e-5
            target_mask[i] = True

    # 5. 設置追蹤屬性
    point_count = len(scene_points)
    gaussians._point_ids = nn.Parameter(torch.arange(point_count, device="cuda"), requires_grad=False)
    gaussians._parent_ids = nn.Parameter(torch.full((point_count,), -1, device="cuda"), requires_grad=False)
    gaussians._generation = nn.Parameter(torch.zeros(point_count, device="cuda"), requires_grad=False)
    gaussians._is_target = nn.Parameter(torch.tensor(target_mask, device="cuda"), requires_grad=False)
    gaussians.next_id = point_count
    
    
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

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
            
            # 更新position_history
            current_positions = gaussians._xyz.detach().cpu().numpy()
            target_mask = gaussians._is_target.cpu().numpy()
            
            for idx, (pos, is_target) in enumerate(zip(current_positions, target_mask)):
                if is_target:
                    point_id = gaussians._point_ids[idx].item()
                    if point_id not in gaussians.position_history:
                        gaussians.position_history[point_id] = {
                            'positions': [],
                            'iterations': [],
                            'parent_id': gaussians._parent_ids[idx].item()
                        }
                    gaussians.position_history[point_id]['positions'].append(pos)
                    gaussians.position_history[point_id]['iterations'].append(iteration)

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
                
                '''add'''
                gaussians.save_trajectory(os.path.join(scene.model_path, f"trajectory_{iteration}.json"))

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
        
        # # 保存當前高斯點的位置分佈
        # tb_writer.add_scalar('scene/total_points', scene.gaussians.get_xyz.shape[0], iteration)
        
        # # 記錄狗所在區域的高斯點數量和分佈
        # dog_stats = scene.gaussians.get_dog_region()  # 使用 gaussians 的方法
        # if tb_writer:
        #     tb_writer.add_scalar('dog_region/point_count', dog_stats['point_count'], iteration)
        #     tb_writer.add_scalar('dog_region/opacity_mean', dog_stats['opacity_mean'], iteration)
        #     tb_writer.add_scalar('dog_region/scaling_mean', dog_stats['scaling_mean'], iteration)
        
        '''add'''
        target_stats = scene.gaussians.get_target_stats()
        if target_stats['mean_pos'] is not None:
            tb_writer.add_scalar('target/point_count', target_stats['count'], iteration)
            tb_writer.add_scalars('target/mean_position', {
                'x': target_stats['mean_pos'][0],
                'y': target_stats['mean_pos'][1],
                'z': target_stats['mean_pos'][2]
            }, iteration)
            tb_writer.add_scalars('target/std_position', {
                'x': target_stats['std_pos'][0],
                'y': target_stats['std_pos'][1],
                'z': target_stats['std_pos'][2]
            }, iteration)
            tb_writer.add_histogram('target/generations', target_stats['generations'], iteration)
        

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
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])

    parser.add_argument("--test_iterations", nargs="+", type=int, default=[100, 1000, 5000, 10000, 30000, 50000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000, 3000, 5000, 10000, 30000, 50000, 70000, 90000, 130000, 150000])
    
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    parser.add_argument("--detailed_log", action="store_true", help="Enable detailed logging")
    
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