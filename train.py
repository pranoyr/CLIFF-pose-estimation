
import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
from utils import *
from validate import validate_epoch

from pose_2D import detect_pose

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from utils import *

import cv2

from common.utils import *
from render import *
import smplx
from common import constants


smpl_model = smplx.create(constants.SMPL_MODEL_DIR, "smpl").to(device)

def train_epoch(train_loader, model, criterion, optimizer, epoch, args):
	batch_time = AverageMeter('Time', ':6.3f')
	data_time = AverageMeter('Data', ':6.3f')
	losses = AverageMeter('Loss', ':.4f')
	top1 = AverageMeter('Acc@1', ':6.2f')
	top5 = AverageMeter('Acc@5', ':6.2f')
	progress = ProgressMeter(
		len(train_loader),
		[batch_time, data_time, losses],
		prefix="Epoch: [{}]".format(epoch))

	# switch to train mode
	model.train()

	end = time.time()
	for i, (batch) in enumerate(train_loader):
		norm_img = batch["norm_img"].to(device).float()
		center = batch["center"].to(device).float()
		scale = batch["scale"].to(device).float()
		img_h = batch["img_h"].to(device).float()
		img_w = batch["img_w"].to(device).float()
		focal_length = batch["focal_length"].to(device).float()
	

		# print("norm_img.shape", norm_img.shape)
		# print("center.shape", center.shape)
		# print("scale.shape", scale.shape)
		# print("img_h.shape", img_h.shape)
		# print("img_w.shape", img_w.shape)
		# print("focal_length.shape", focal_length.shape)


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
				rotation=torch.eye(3, device="cuda:1").unsqueeze(0).expand(1, -1, -1),
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


		loss = keypoint_loss * 5   + \
		  		beta_loss * 0.001 + \
		 		pose_loss   
				# ((torch.exp(-pred_cam_crop[:,0]*10)) ** 2 ).mean()
				
		# loss *= 60

		# landmarks = batch["target_landmarks"].squeeze(0).cpu().numpy() * np.array([img_w, img_h])
		# # print(landmarks)
		
		# # print(batch["target_landmarks"])
		# # print((img_h[0].item(), img_w[0].item(), 3))
		# img_draw = np.ones((2048, 1536, 3))
		# for i in range(landmarks.shape[0]):
		# 	x, y = landmarks[i]
		# 	cv2.circle(img_draw, (int(x), int(y)), 2, (0, 0, 255), 2)
		# 	# cv2.putText(img_draw, str(i), (int(x), int(y)), 0, 0.5, 255)
		# 	#time.sleep(0.5)

		# # img_draw = cv2.resize(img_draw, (480, 640))
		# # cv2.imshow("gth.png", img_draw)
		# cv2.waitKey(1)

	

		# landmarks = projected_keypoints_2d.squeeze(0).detach().cpu().numpy() * np.array([img_w, img_h])
		# # print(projected_keypoints_2d)
		
		# # print((img_h[0].item(), img_w[0].item(), 3))
		# # img_draw = np.ones((2048, 1536, 3))
		# for i in range(landmarks.shape[0]):
		# 	x, y = landmarks[i]
		# 	cv2.circle(img_draw, (int(x), int(y)), 3, (255, 0, 0), 2)
		# 	# cv2.putText(img_draw, str(i), (int(x), int(y)), 0, 0.5, 255)
		# 	#time.sleep(0.5)

		# # img_draw = cv2.resize(img_draw, (480, 640))
		# img_draw = cv2.resize(img_draw, (480, 640))
		# cv2.imshow("predicted.png", img_draw)
		# cv2.waitKey(1)


		# measure accuracy and record loss
		losses.update(loss.item(), norm_img.size(0))

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			progress.display(i + 1)
	

			# full_pose_coeff = model(keypoints)
			# landmark_preds, landmarks, vertices, faces = smplx_params(full_pose_coeff)
			# textures = torch.ones_like(vertices) * torch.tensor([0, 0, 255]).to(device)
			# # colour vertices of index
			# textures = TexturesVertex(verts_features=textures[None])
			# mesh = Meshes(
			# 	verts=[vertices],   
			# 	faces=[faces],
			# 	textures=textures)
			# images = renderer(mesh, lights=lights, cameras=cameras)
			# rgb = images[0, ..., :3].detach().cpu().numpy() 
			# bgr = rgb[..., ::-1]
			# cv2.imshow('image', bgr)
			# cv2.waitKey(1)

	model.eval()
	img = cv2.imread("/media/pranoy/Pranoy/coco/train2017/000000196085.jpg")
	validate_epoch(model, img)
	

	return losses.avg