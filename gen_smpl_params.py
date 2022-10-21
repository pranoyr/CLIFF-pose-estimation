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
# state_dict = torch.load("data/ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt")['model']
state_dict = torch.load("checkpoint.pth")['state_dict']
state_dict = strip_prefix_if_present(state_dict, prefix="module.")
cliff_model.load_state_dict(state_dict, strict=True)
cliff_model.eval()


def extract_bounding_box(points):
	x_coordinates, y_coordinates = zip(*points)

	return torch.tensor([min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)])



# Setup the SMPL model
smpl_model = smplx.create(constants.SMPL_MODEL_DIR, "smpl").to(device)



list_images = os.listdir("/media/pranoy/Pranoy/coco/train2017/s")

for file_name in tqdm(list_images):
	file_name_full = os.path.join("/media/pranoy/Pranoy/coco/train2017/", file_name)
	# ret, img_bgr = vid.read()
	img_bgr = cv2.imread(file_name_full)
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



	# print("betas", pred_betas.shape)
	# print("body pose", pred_rotmat[:, 1:].shape)
	# print("global_orient", pred_rotmat[:, [0]].shape)
	# pred_output = smpl_model(betas=pred_betas,
	# 							body_pose=pred_rotmat[:, 1:],
	# 							global_orient=pred_rotmat[:, [0]],
	# 							pose2rot=False,
	# 							transl=pred_cam_full)

	



	#save to pickle
	results = {}
	results["beta"] = pred_betas.cpu().squeeze(0).numpy()
	results["body_pose"] = pred_rotmat[:, 1:].cpu().squeeze(0).numpy()
	pickle.dump(results, open("/media/pranoy/Pranoy/coco/smpl_params/" + file_name + ".pkl", "wb"))

	# vertices = pred_output.vertices
	# faces = smpl_model.faces
	# joints = pred_output.joints


	# pred_vert_arr.extend(pred_vertices.cpu().numpy())
	# pred_vert_arr = np.array(pred_vert_arr)
	# for img_idx, orig_img_bgr in enumerate(tqdm(orig_img_bgr_all)):
	#     chosen_mask = detection_all[:, 0] == img_idx
	#     chosen_vert_arr = pred_vert_arr[chosen_mask]

	# setup renderer for visualization
	# img_h, img_w, _ = img_bgr.shape
	# img_h = torch.tensor([img_h]).float().to(device)
	# img_w = torch.tensor([img_w]).float().to(device)

	
#	focal_length = estimate_focal_length(img_h, img_w)
	# renderer = Renderer(focal_length=focal_length, img_w=img_w, img_h=img_h,
	#                     faces=smpl_model.faces,
	#                     same_mesh_color=("video" == "video"))



	# joints = joints + pred_cam_full
	# pred_keypoints_2d = to_camera(joints)

	

	# camera_center = torch.zeros(1, 2, device="cuda:0")
	# pred_keypoints_2d = perspective_projection(joints,
	# 			rotation=torch.eye(3, device="cuda:0").unsqueeze(0).expand(1, -1, -1),
	# 			translation=pred_cam_full,
	# 			focal_length=focal_length,
	# 			camera_center=torch.div(full_img_shape, 2, rounding_mode='floor'))




	# # print(pred_keypoints_2d.shape)
	# # pred_keypoints_2d = to_image(pred_keypoints_2d, focal=focal_length, center=cx)
	# # print(pred_keypoints_2d/ full_img_shape)


	# landmarks = get_landmarks(pred_keypoints_2d).squeeze() 


	# # img1 = np.ones((img_h,img_w,3), dtype=np.uint8) * 255
	


	
	# # textures = torch.ones_like(vertices)# (1, V, 3)
	# # textures = TexturesVertex(verts_features=textures)
	
	# # vertices = vertices[0]
	# # faces = faces.astype(np.float32)
	# # faces = torch.from_numpy(faces)

	# # print(vertices.shape)
	# # print(faces.shape)
	
	# # mesh = Meshes(
	# # 	verts=[vertices.to(device)],   
	# # 	faces=[faces.to(device)],
	# # 	textures=textures)
	# # images = renderer(mesh, lights=lights, cameras=cameras)
	# # rgb = images[0, ..., :3].detach().cpu().numpy() 
	# # bgr = rgb[..., ::-1]
	# # cv2.imshow('image', bgr)
	# # cv2.waitKey(1)

	# renderer = Renderer(focal_length=focal_length, img_w=img_w, img_h=img_h,
	# 						faces=smpl_model.faces,
	# 						same_mesh_color=("video" == "video"))
	
	# front_view = renderer.render_front_view(vertices.cpu(), img_bgr)

	# front_view = cv2.resize(front_view, (640, 480))
	# cv2.imshow('image', front_view)



	


	# # poses, _ = detect_pose(draw_img)
	# # print(poses.shape)

	# smplx_left_leg_indices = torch.tensor([2,5,8,11])
	# smplx_right_leg_indices = torch.tensor([1,4,7,10])
	# smplx_left_arm_indices = torch.tensor([17,19,21,23])
	# smplx_right_arm_indices = torch.tensor([16,18,20,22])
	# nose_neck_indices = torch.tensor([15])

	# mediapipe_left_leg_indices = torch.tensor([24,26,28,32])
	# mediapipe_right_leg_indices = torch.tensor([23,25,27,31])
	# mediapipe_left_arm_indices = torch.tensor([12,14,16,20])
	# mediapipe_right_arm_indices = torch.tensor([11,13,15,19])
	# mediapipe_nose_neck_indices = torch.tensor([0])


	# landmarks  = landmarks[smplx_left_leg_indices]

	# for i in range(landmarks.shape[0]):
	# 	x, y = landmarks[i]
	# 	#cv2.circle(front_view, (int(x), int(y)), 2, (0, 0, 255), -1)
	# 	cv2.putText(front_view, str(i), (int(x), int(y)), 0, 0.5, 255)
	# 	#time.sleep(0.5)

	# cv2.imshow("predicted.png", front_view)

	# cv2.waitKey(1)
	

	# poses= poses[mediapipe_nose_neck_indices]

	# # visualise media pipe landmarks
	# for i in range(poses.shape[0]):
	# 	x, y = poses[i]
	# 	#cv2.circle(front_view, (int(x), int(y)), 2, (0, 0, 255), -1)
	# 	cv2.putText(draw_img, str(i), (int(x), int(y)), 0, 0.5, 255)
	# 	#time.sleep(0.5)

	# cv2.imshow("mediapipe", draw_img)
	# cv2.waitKey(1)

	# del renderer



 
	
	

