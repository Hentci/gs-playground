import torch
from scene import Scene
import os
import numpy as np
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.render_utils import generate_path, create_videos, save_img_f32, save_img_u8
from utils.general_utils import colormap

def get_intrinsics2(H, W, fovx, fovy):
    fx = 0.5 * W / np.tan(0.5 * fovx)
    fy = 0.5 * H / np.tan(0.5 * fovy)
    cx = 0.5 * W
    cy = 0.5 * H
    return np.array([[fx,  0, cx],
                     [ 0, fy, cy],
                     [ 0,  0,  1]])

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_video(model_path, name, iteration, views, gaussians, pipeline, background, scene):
    print("Rendering video using training cameras...")
    traj_dir = os.path.join(model_path, 'traj', "ours_{}".format(iteration))
    makedirs(traj_dir, exist_ok=True)
    
    # 直接使用訓練相機，不需要生成新的軌跡
    train_cameras = scene.getTrainCameras()
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    makedirs(render_path, exist_ok=True)

    for idx, view in enumerate(tqdm(train_cameras, desc="Rendering video progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        save_img_u8(rendering.permute(1,2,0).cpu().numpy(), 
                   os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
    
    create_videos(base_dir=traj_dir, input_dir=render_path,
                 out_name='render_traj', num_frames=len(train_cameras))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, 
                skip_train : bool, skip_test : bool, render_path : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, 
                       scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, 
                       scene.getTestCameras(), gaussians, pipeline, background)
        
        if render_path:
            render_video(dataset.model_path, "traj", scene.loaded_iter, 
                        scene.getTrainCameras(), gaussians, pipeline, background, scene)

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--render_path", action="store_true")  # 新增參數
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), 
                args.skip_train, args.skip_test, args.render_path)