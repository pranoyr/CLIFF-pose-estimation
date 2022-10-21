


# import numpy as np
# import h5py
# import scipy.io
# mat = scipy.io.loadmat('/media/pranoy/Pranoy/mpi_inf_3dhp/S1/Seq1/annot.mat')
# import torch

# print(len(mat["annot3"][0]))

# print



# # x = torch.tensor (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
# # y = torch.tensor (([1, 2, 3]))

# # w = torch.tensor ([[1, 2, 3]])


# # y_hat = torch.matmul(x, w.t()).view(-1)

# # print(y_hat)


# from models.cliff_hr48.cliff import CLIFF as cliff_hr48
# from models.cliff_res50.cliff import CLIFF as cliff_res50
# from common import constants
# from common.utils import strip_prefix_if_present, cam_crop2full, video_to_images
# from common.utils import estimate_focal_length
# from common.renderer_pyrd import Renderer
# # from lib.yolov3_detector import HumanDetector
# from common.mocap_dataset import MocapDataset
# # from lib.yolov3_dataset import DetectionDataset
# from common.imutils import process_image
# from common.utils import estimate_focal_length




# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it
# under the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import os
import math
import time
from common.renderer_pyrd import Renderer
from pytorch3d.structures import Meshes
import pickle
import os.path as osp
import cv2
import glob
import torch
import argparse
from utils import *
import numpy as np
from tqdm import tqdm
import smplx
from torch.utils.data import DataLoader
import torchgeometry as tgm

from models.cliff_hr48.cliff import CLIFF as cliff_hr48
from models.cliff_res50.cliff import CLIFF as cliff_res50
from common import constants
from common.utils import strip_prefix_if_present, cam_crop2full, video_to_images
from common.utils import estimate_focal_length
# from lib.yolov3_detector import HumanDetector
from common.mocap_dataset import MocapDataset
# from lib.yolov3_dataset import DetectionDataset
from common.imutils import process_image
from common.utils import estimate_focal_length



from turtle import pos
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
import numpy as np
import torch
from pose_2D import detect_pose



# # For webcam input:
# def detect_pose(image):
#     with mp_holistic.Holistic(
#         min_detection_confidence=0.5,
#         model_complexity=2,
#         static_image_mode=True,
#         refine_face_landmarks=True,
#         min_tracking_confidence=0.5) as holistic:
        
#         poses = []

#         # To improve performance, optionally mark the image as not writeable to
#         # pass by reference.
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         results = holistic.process(image)

#         z_coordinates = []
#         for keypoint in results.pose_landmarks.landmark:
#             poses.append([keypoint.x * image.shape[1]  , keypoint.y * image.shape[0]])
#             z_coordinates.append(keypoint.z)

#         pose = torch.tensor(poses).float()  # shape  (33, 2)
#         z_coordinates = torch.tensor(z_coordinates).float()  
#     return pose, z_coordinates





device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


# print("--------------------------- Detection ---------------------------")
# # Setup human detector
# human_detector = HumanDetector()


print("--------------------------- 3D HPS estimation ---------------------------")
# Create the model instance
cliff = eval("cliff_" + "res50")
cliff_model = cliff(constants.SMPL_MEAN_PARAMS).to(device)
# Load the pretrained model
# state_dict = torch.load("data/ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt")['model']
# state_dict = torch.load("checkpoint.pth", map_location="cuda")['state_dict']
# state_dict = strip_prefix_if_present(state_dict, prefix="module.")
# cliff_model.load_state_dict(state_dict, strict=True)
cliff_model.eval()


def extract_bounding_box(points):
	x_coordinates, y_coordinates = zip(*points)

	return torch.tensor([min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)])


vid = cv2.VideoCapture(0)

# Setup the SMPL model
smpl_model = smplx.create(constants.SMPL_MODEL_DIR, "smpl").to(device)



while True:
	ret, img_bgr = vid.read()
	# img_bgr = cv2.imread("/home/pranoy/code/auto-transform/sample_data/imgs/IMG_1789.JPEG")
	# img_bgr = cv2.imread("images/frame_003809.jpg")
	draw_img = img_bgr.copy()
	# img_bgr = cv2.resize(img_bgr, (512, 512))
	


	# norm_img = (letterbox_image(img_bgr, (416, 416)))
	# norm_img = norm_img[:, :, ::-1].transpose((2, 0, 1)).copy()
	# norm_img = norm_img / 255.0

	# norm_img = torch.from_numpy(norm_img)
	# norm_img = norm_img.to(device).float()
	# norm_img = norm_img.unsqueeze(0)

	dim = np.array([img_bgr.shape[1], img_bgr.shape[0]])
	dim = torch.from_numpy(dim)
	dim = dim.unsqueeze(0)
	dim = dim.to(device)


	# detection_result = human_detector.detect_batch(norm_img, dim)
	mediapipe_results = detect_pose(img_bgr)# shape (18, 2)
	scaled_keypoints = mediapipe_results["scaled_keypoints"]
	bbox = extract_bounding_box(scaled_keypoints).to(device)



	img_rgb = img_bgr[:, :, ::-1]
	img_h, img_w, _ = img_rgb.shape


	img_h, img_w, _ = img_bgr.shape
	img_h = torch.tensor([img_h]).float().to(device)
	img_w = torch.tensor([img_w]).float().to(device)

	focal_length = estimate_focal_length(img_h, img_w)
	# bbox = detection_result[0][1:5]



	norm_img, center, scale, crop_ul, crop_br, _ = process_image(img_rgb, bbox)


	
	center = center.unsqueeze(0).to(device)
	scale = scale.unsqueeze(0)
	focal_length = torch.tensor([focal_length]).to(device)







	pred_vert_arr = []
	cx, cy, b = center[:, 0], center[:, 1], scale * 200

	# print(cx, cy, b)

	
	bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
	# The constants below are used for normalization, and calculated from H36M data.
	# It should be fine if you use the plain Equation (5) in the paper.
	bbox_info[:, :2] = bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8  # [-1, 1]
	bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (0.06 * focal_length)  # [-1, 1]


	norm_img = torch.from_numpy(norm_img).unsqueeze(0)
	norm_img = norm_img.to(device)

	print(bbox_info.shape)

	with torch.no_grad():
		start = time.time()
		pred_rotmat, pred_betas, pred_cam_crop = cliff_model(norm_img, bbox_info,  n_iter=3)
		end = time.time()

	print("FPS: ", 1 / (end - start))


