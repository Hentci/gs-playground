import torch
import torch.nn as nn
import pytorch3d
from pytorch3d.structures import Pointclouds
from pytorch3d.ops import knn_points
import numpy as np
import cv2
import os
import json
import struct
import collections
from tqdm import tqdm
import time
from datetime import datetime


# 定義相機模型 (與原始代碼相同)


CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS])

def get_camera_params(camera):
    """根據相機模型獲取相機參數"""
    if camera['model_name'] == "PINHOLE":
        fx = camera['params'][0]
        fy = camera['params'][1]
        cx = camera['params'][2]
        cy = camera['params'][3]
    elif camera['model_name'] == "SIMPLE_PINHOLE":
        fx = fy = camera['params'][0]
        cx = camera['params'][1]
        cy = camera['params'][2]
    else:
        raise ValueError(f"Unsupported camera model: {camera['model_name']}")
    
    return fx, fy, cx, cy

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """讀取並解包二進制文件中的下一個字節"""
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise ValueError(f"Expected {num_bytes} bytes but got {len(data)}")
    return struct.unpack(endian_character + format_char_sequence, data)

def read_binary_cameras(path):
    """讀取COLMAP的cameras.bin文件"""
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            
            model = CAMERA_MODEL_IDS[model_id]
            num_params = model.num_params
            params = read_next_bytes(fid, 8*num_params, "d"*num_params)
            
            cameras[camera_id] = {
                'model_id': model_id,
                'model_name': model.model_name,
                'width': width,
                'height': height,
                'params': np.array(params)
            }
    return cameras

def read_binary_images(path):
    """讀取COLMAP的images.bin文件"""
    images = {}
    with open(path, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in tqdm(range(num_reg_images), desc="Reading images"):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qvec = binary_image_properties[1:5]
            tvec = binary_image_properties[5:8]
            camera_id = binary_image_properties[8]
            
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(24 * num_points2D, 1)
            
            images[image_name] = {
                'id': image_id,
                'camera_id': camera_id,
                'rotation': qvec,
                'translation': tvec
            }
    return images

def quaternion_to_rotation_matrix(q):
    """四元數轉旋轉矩陣 (PyTorch版本)"""
    qw, qx, qy, qz = q
    # 確保使用 float32
    R = torch.tensor([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ], dtype=torch.float32)
    return R