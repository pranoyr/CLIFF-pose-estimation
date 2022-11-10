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
from render import *
# from lib.pytorch_yolo_v3_master.preprocess import letterbox_image
from pytorch3d.structures import Meshes
import pickle
import os.path as osp
import tqdm
import pickle
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
from common.renderer_pyrd import Renderer
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





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# print("--------------------------- Detection ---------------------------")
# # Setup human detector
# human_detector = HumanDetector()


print("--------------------------- 3D HPS estimation ---------------------------")
# Create the model instance
cliff = eval("cliff_" + "hr48")
cliff_model = cliff(constants.SMPL_MEAN_PARAMS).to(device)
# Load the pretrained model
state_dict = torch.load("data/ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt", map_location="cuda:0")['model']
# state_dict = torch.load("checkpoint.pth")['state_dict']
state_dict = strip_prefix_if_present(state_dict, prefix="module.")
cliff_model.load_state_dict(state_dict, strict=True)
cliff_model.eval()


def extract_bounding_box(points):
	x_coordinates, y_coordinates = zip(*points)

	return torch.tensor([min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)])



# Setup the SMPL model
smpl_model = smplx.create(constants.SMPL_MODEL_DIR, "smpl").to(device)



list_images = os.listdir("/media/pranoy/Pranoy/human3.6M/images/s_06_act_13_subact_01_ca_01/")


for index, folder_name in enumerate(os.listdir("/media/pranoy/Pranoy/human3.6M/images/")):
	print(folder_name)
	print((index+1) )
	full_folder_name = "/media/pranoy/Pranoy/human3.6M/images/" + folder_name
	if ((index+1) % 2 == 1) or (index+1 == 1):
		for filename in tqdm(os.listdir(full_folder_name)):
			full_file_name = full_folder_name + "/" + filename
			# ret, img_bgr = vid.read()
			img_bgr = cv2.imread(full_file_name)
			draw_img = img_bgr.copy()
			# img_bgr = cv2.resize(img_bgr, (512, 512))
			


			# norm_img = (letterbox_image(img_bgr, (416, 416)))
			# norm_img = norm_img[:, :, ::-1].transpose((2, 0, 1)).copy()
			# norm_img = norm_img / 255.0

			# norm_img = torch.from_numpy(norm_img)
			# norm_img = norm_img.to(device).float()
			# norm_img = norm_img.unsqueeze(0)

			# dim = np.array([img_bgr.shape[1], img_bgr.shape[0]])
			# dim = torch.from_numpy(dim)
			# dim = dim.unsqueeze(0)
			# dim = dim.to(device)


			# detection_result = human_detector.detect_batch(norm_img, dim)
			try:
				mediapipe_results = detect_pose(img_bgr)# shape (18, 2)
			except:
				continue
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


			
			bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
			# The constants below are used for normalization, and calculated from H36M data.
			# It should be fine if you use the plain Equation (5) in the paper.
			bbox_info[:, :2] = bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8  # [-1, 1]
			bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (0.06 * focal_length)  # [-1, 1]


			norm_img = torch.from_numpy(norm_img).unsqueeze(0)
			norm_img = norm_img.to(device)


			with torch.no_grad():
				pred_rotmat, pred_betas, pred_cam_crop = cliff_model(norm_img, bbox_info)

			

			# convert the camera parameters from the crop camera to the full camera

			# full_img_shape = torch.tensor([[img_h, img_w]]).float().to(device)
			full_img_shape = torch.stack((img_h, img_w), dim=-1)

			# full_img_shape = torch.stack((img_h, img_w), dim=-1)
			pred_cam_full = cam_crop2full(pred_cam_crop, center, scale, full_img_shape, focal_length)




			#save to pickle
			results = {}
			results["beta"] = pred_betas.cpu().squeeze(0).numpy()
			results["body_pose"] = pred_rotmat[:, 1:].cpu().squeeze(0).numpy()
			pickle.dump(results, open("/media/pranoy/Pranoy/human3.6M/smpl_params/" + filename[:-4] + ".pkl", "wb"))
