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
# from render import *
# from lib.pytorch_yolo_v3_master.preprocess import letterbox_image
# from pytorch3d.structures import Meshes
import random
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
# from common.renderer_pyrd import Renderer
# from lib.yolov3_detector import HumanDetector
from common.mocap_dataset import MocapDataset
# from lib.yolov3_dataset import DetectionDataset
from common.imutils import process_image
from common.utils import estimate_focal_length


from pose_2D import detect_pose


from turtle import pos
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
import numpy as np
import torch


def extract_bounding_box(points):
	x_coordinates, y_coordinates = zip(*points)

	return torch.tensor([min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)])



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


smpl_model = smplx.create(constants.SMPL_MODEL_DIR, "smpl").to(device)


def validate(val_loader, model, criterion, epoch):
	batch_time = AverageMeter('Time', ':6.3f')
	data_time = AverageMeter('Data', ':6.3f')
	losses = AverageMeter('Loss', ':.4f')
	top1 = AverageMeter('Acc@1', ':6.2f')
	top5 = AverageMeter('Acc@5', ':6.2f')
	progress = ProgressMeter(
		len(val_loader),
		[batch_time, data_time, losses],
		prefix="Epoch: [{}]".format(epoch))

	# switch to train mode
	model.eval()

	end = time.time()
	for i, (batch) in enumerate(val_loader):
		norm_img = batch["norm_img"].to(device).float()
		center = batch["center"].to(device).float()
		scale = batch["scale"].to(device).float()
		img_h = batch["img_h"].to(device).float()
		img_w = batch["img_w"].to(device).float()
		focal_length = batch["focal_length"].to(device).float()
	


		cx, cy, b = center[:, 0], center[:, 1], scale * 200
		bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
		# The constants below are used for normalization, and calculated from H36M data.
		# It should be fine if you use the plain Equation (5) in the paper.
		bbox_info[:, :2] = bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8  # [-1, 1]
		bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (0.06 * focal_length)  # [-1, 1]


		pred_rotmat, pred_betas, pred_cam_crop = model(norm_img, bbox_info)

		# convert the camera parameters from the crop camera to the full camera
		full_img_shape = torch.stack((img_h, img_w), dim=-1)
		pred_cam_full = cam_crop2full(pred_cam_crop, center, scale, full_img_shape, focal_length)


		pred_output = smpl_model(betas=pred_betas,
								 body_pose=pred_rotmat[:, 1:],
								 global_orient=pred_rotmat[:, [0]],
								 pose2rot=False,
								 transl=pred_cam_full)


		pred_joints = pred_output.joints


		projected_keypoints_2d = perspective_projection(pred_joints,
				rotation=torch.eye(3, device="cuda:0").unsqueeze(0).expand(1, -1, -1),
				translation=pred_cam_full,
				focal_length=focal_length,
				camera_center=torch.div(full_img_shape.flip(dims=[1]), 2, rounding_mode='floor'))

		
	
		smplx_left_leg_indices = torch.tensor([2,5,8,11])
		smplx_right_leg_indices = torch.tensor([1,4,7,10])
		smplx_left_arm_indices = torch.tensor([17,19,21,23])
		smplx_right_arm_indices = torch.tensor([16,18,20,22])
		nose_neck_indices = torch.tensor([15])
		
		full_img_shape_1 = torch.stack((img_w, img_h), dim=-1).unsqueeze(1)
		all_smplx_indices =  torch.cat((smplx_left_leg_indices, smplx_right_leg_indices, smplx_left_arm_indices, smplx_right_arm_indices, nose_neck_indices), dim=0)
		# projected_keypoints_2d = torch.div(projected_keypoints_2d[:, all_smplx_indices] , full_img_shape_1, rounding_mode='floor')
		
		projected_keypoints_2d = projected_keypoints_2d[:, all_smplx_indices,:] / full_img_shape_1



		mediapipe_left_leg_indices = torch.tensor([24,26,28,32])
		mediapipe_right_leg_indices = torch.tensor([23,25,27,31])
		mediapipe_left_arm_indices = torch.tensor([12,14,16,20])
		mediapipe_right_arm_indices = torch.tensor([11,13,15,19])
		mediapipe_nose_neck_indices = torch.tensor([0])
		all_mediapipe_indices =  torch.cat((mediapipe_left_leg_indices, mediapipe_right_leg_indices, mediapipe_left_arm_indices, mediapipe_right_arm_indices, mediapipe_nose_neck_indices), dim=0)
		batch["target_landmarks"] = batch["target_landmarks"][:, all_mediapipe_indices,:]



		# projected_keypoints_2d = projected_keypoints_2d.view(-1, 2)
		# batch["target_landmarks"] = batch["target_landmarks"].view(-1, 2)
		# diff = projected_keypoints_2d - batch["target_landmarks"].to(device).float()
		# # print(diff.shape)
		# loss = torch.norm(diff, dim=1, p=2).square().sum()/norm_img.shape[0]
		keypoint_loss = criterion(projected_keypoints_2d, batch["target_landmarks"].to(device).float())

		beta_loss = criterion(pred_betas, batch["beta_params"].to(device).float())
		pose_loss = criterion(pred_rotmat[:, 1:], batch["pose_params"].to(device).float())


		loss = keypoint_loss    + \
		  		beta_loss * 0.001 + \
		 		pose_loss   + \
				((torch.exp(-pred_cam_crop[:,0]*10)) ** 2 ).mean()
		loss *= 60
	
		# measure accuracy and record loss
		losses.update(loss.item(), norm_img.size(0))

	
	return losses.avg
		

def visualise(img_bgr, cliff_model, output_path):
	mediapipe_results = detect_pose(img_bgr)# shape (18, 2)
	scaled_keypoints = mediapipe_results["scaled_keypoints"]
	# bbox = detection_result[0][1:5]
	bbox = extract_bounding_box(scaled_keypoints).to(device)


	img_rgb = img_bgr[:, :, ::-1]
	img_h, img_w, _ = img_rgb.shape
	focal_length = estimate_focal_length(img_h, img_w)
	

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

	full_img_shape = torch.tensor([[img_h, img_w]]).float().to(device)

	# full_img_shape = torch.stack((img_h, img_w), dim=-1)
	pred_cam_full = cam_crop2full(pred_cam_crop, center, scale, full_img_shape, focal_length)


	pred_output = smpl_model(betas=pred_betas,
								body_pose=pred_rotmat[:, 1:],
								global_orient=pred_rotmat[:, [0]],
								pose2rot=False,
								transl=pred_cam_full)



	vertices = pred_output.vertices
	faces = smpl_model.faces
	joints = pred_output.joints


	img_h, img_w, _ = img_bgr.shape
	img_h = torch.tensor([img_h]).float().to(device)
	img_w = torch.tensor([img_w]).float().to(device)

	
	focal_length = estimate_focal_length(img_h, img_w)



	camera_center = torch.zeros(1, 2, device="cuda:0")
	pred_keypoints_2d = perspective_projection(joints,
				rotation=torch.eye(3, device="cuda:0").unsqueeze(0).expand(1, -1, -1),
				translation=pred_cam_full,
				focal_length=focal_length,
				camera_center=torch.tensor([torch.div(img_w, 2, rounding_mode='floor'), 
				torch.div(img_h, 2, rounding_mode='floor')]))



	landmarks = get_landmarks(pred_keypoints_2d).squeeze() 

	renderer = Renderer(focal_length=focal_length, img_w=img_w, img_h=img_h,
                            faces=smpl_model.faces,
                            same_mesh_color=("video" == "video"))
	
	front_view = renderer.render_front_view(vertices.cpu(), img_bgr)
	front_view = cv2.resize(front_view, (480, 640))
	cv2.imwrite(output_path, front_view)

	del renderer


			

def validate_epoch(val_loader, model, criterion, train_list, test_list, epoch):

	avg_val_loss = validate(val_loader, model, criterion, epoch)
	
	# choose random image from test set
	val_img_path = random.choice(test_list)
	val_img  = cv2.imread(val_img_path.replace("smpl_params", "all_images")[:-4])

	train_img_path = random.choice(train_list)
	train_img  = cv2.imread(train_img_path.replace("smpl_params", "all_images")[:-4])

	visualise(train_img, model, output_path="train.jpg")
	visualise(val_img, model, output_path="val.jpg")

	return avg_val_loss

	# images = os.listdir("/home/pranoy/code/auto-transform/new_data/all_images/")
	# # randomly choose one
	# image = images[np.random.randint(0, len(images))]
	# img_bgr = cv2.imread("/home/pranoy/code/auto-transform/new_data/all_images/" + image)

	



 
	
	

