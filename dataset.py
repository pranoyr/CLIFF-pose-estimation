from gzip import READ
import numpy as np
import os, glob, cv2, sys
from torch.utils.data import Dataset

import torch
from pose_2D import detect_pose
from common.utils import strip_prefix_if_present, cam_crop2full, video_to_images
from common.utils import estimate_focal_length
from common.renderer_pyrd import Renderer
# from lib.yolov3_detector import HumanDetector
from common.mocap_dataset import MocapDataset
# from lib.yolov3_dataset import DetectionDataset
from common.imutils import process_image
from common.utils import estimate_focal_length
from common import constants
import pickle as pk




def normalise(landmarks):
		"""
		Return:
			landmarks_norm   -- torch.tensor, size (B, 68, 2)

		Parameters:
			landmarks        -- torch.tensor, size (B, 68, 2)
			boxes            -- torch.tensor, size (B, 2, 2)

		"""
		boxes = extract_bounding_box(landmarks.type(torch.int32).tolist())
		wh = boxes[1] - boxes[0]
		landmarks_normalised = (landmarks - boxes[0]) / wh

		return landmarks_normalised



# def extract_bounding_box(points):
# 	x_coordinates, y_coordinates = zip(*points)

# 	return torch.tensor([[min(x_coordinates), min(y_coordinates)], [max(x_coordinates), max(y_coordinates)]])



num = 0.5    # scaling matrix
s = torch.tensor([[num, 0],
					[0, 1/num]])


def extract_bounding_box(points):
	x_coordinates, y_coordinates = zip(*points)

	return torch.tensor([min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)])






class CustomDataset(Dataset):
	def __init__(self, root_dir, image_list, transform = None, target_transform=None, image_set='train'):
		self.transform = transform
		self.target_transform = target_transform
		# self.image_list       = self.get_data(self.data_dir)
		self.image_list = image_list
		self.root_dir = root_dir
		self.image_set = image_set

		self.image_list = self._read_image_ids(image_list)

		# self.noisy_image_list = self.get_data(os.path.join(self.data_dir, 'noisy'))
		# self.noisy_image_list = self.image_list 
		
	# def get_data(self, data_path):
	# 	data = []
	# 	for img_path in glob.glob(data_path + os.sep + '*'):
	# 		data.append(img_path)
	# 	return data

	def _read_image_ids(self, image_sets_file):
		ids = []
		with open(image_sets_file) as f:
			for line in f:
				ids.append(line.rstrip())
		return ids
	
	def __getitem__(self, index):  
		# read images in grayscale, then invert them
		# img  = cv2.imread(self.image_list[index].replace("smpl_params", "train2017")[:-4])
		full_lath = self.root_dir + "/images/" + self.image_list[index][:-11] + "/" + self.image_list[index].replace("pkl", "jpg")
		# print("******")
		# print(full_lath)
		img  = cv2.imread(full_lath)
		img_h, img_w, _ = img.shape
		img_rgb = img[:, :, ::-1]
		# img = self.transform(img)
		
		has_smplx  = 0
		# load from pickle file:
		pose_full_lath = self.root_dir + "/smpl_params/" + self.image_list[index][:-4] + ".pkl"
		#print(pose_full_lath)
		if os.path.exists(pose_full_lath):
			with open(pose_full_lath, 'rb') as f:
				data = pk.load(f)
				pose_params = data['body_pose']
				beta_params = data['beta']
				has_smplx  = 1
				#print(beta_params.shape)
		else:
			pose_params = np.zeros((23,3,3))
			beta_params = np.zeros((10))



		mediapipe_results = detect_pose(img)# shape (18, 2)
		scaled_keypoints = mediapipe_results["scaled_keypoints"]

		target_landmarks = mediapipe_results["normalised_keypoints"]

	

		focal_length = estimate_focal_length(img_h, img_w)
		# bbox = detection_result[0][1:5]
	
		bbox = extract_bounding_box(scaled_keypoints)


		norm_img, center, scale, _, _, _ = process_image(img_rgb, bbox)

		
		# center = center.unsqueeze(0).to(device)
		# scale = scale.unsqueeze(0)
		# focal_length = torch.tensor([focal_length]).to(device)

		data = {}
		data["norm_img"] = norm_img
		data["center"] = center
		data["scale"] = scale
		data["focal_length"] = focal_length
		data["img_h"] = img_h
		data["img_w"] = img_w
		data["pose_params"] = pose_params
		data["beta_params"] = beta_params
		data["has_smplx"] = has_smplx
		data["has_2d"] = 1

		data["target_landmarks"] = target_landmarks


		# keypoints, z_coordinates = detect_pose(img)# shape (18, 2)
		# keypoints_norm = normalise(keypoints)
		# keypoints = torch.matmul(keypoints, s)

		# diff = keypoints_norm[0] - torch.tensor([0.5032, 0.0264])
		# keypoints_norm = keypoints_norm - diff
		
		# img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
		# img = cv2.resize(img, (640, 380), interpolation = cv2.INTER_AREA) 
		
		# if self.transform is not None:            
		#     img = self.transform(img)     * 255        
			# noisy_img = self.transform(noisy_img)  

		# keypoints_norm = torch.cat((keypoints_norm, z_coordinates.unsqueeze(1).to(device)), dim=1)

		return data

	def __len__(self):
		return len(self.image_list)
	



# class CustomDataset(Dataset):
# 	def __init__(self, image_list, transform = None, target_transform=None, image_set='train'):
# 		self.transform = transform
# 		self.target_transform = target_transform
# 		# self.image_list       = self.get_data(self.data_dir)
# 		self.image_list = image_list
# 		self.image_set = image_set
# 		# self.noisy_image_list = self.get_data(os.path.join(self.data_dir, 'noisy'))
# 		# self.noisy_image_list = self.image_list 
		
# 	# def get_data(self, data_path):
# 	# 	data = []
# 	# 	for img_path in glob.glob(data_path + os.sep + '*'):
# 	# 		data.append(img_path)
# 	# 	return data
	
# 	def __getitem__(self, index):  
# 		# read images in grayscale, then invert them
# 		# img  = cv2.imread(self.image_list[index].replace("smpl_params", "train2017")[:-4])
# 		img  = cv2.imread(self.image_list[index].replace("smpl_params", "all_images")[:-4])
# 		img_h, img_w, _ = img.shape
# 		img_rgb = img[:, :, ::-1]
# 		# img = self.transform(img)
		
# 		# load from pickle file:
		
# 		with open(self.image_list[index], 'rb') as f:
# 			data = pk.load(f)
# 			pose_params = data['body_pose']
# 			beta_params = data['beta']


# 		mediapipe_results = detect_pose(img)# shape (18, 2)
# 		scaled_keypoints = mediapipe_results["scaled_keypoints"]

# 		target_landmarks = mediapipe_results["normalised_keypoints"]

	

# 		focal_length = estimate_focal_length(img_h, img_w)
# 		# bbox = detection_result[0][1:5]
	
# 		bbox = extract_bounding_box(scaled_keypoints)


# 		norm_img, center, scale, _, _, _ = process_image(img_rgb, bbox)

		
# 		# center = center.unsqueeze(0).to(device)
# 		# scale = scale.unsqueeze(0)
# 		# focal_length = torch.tensor([focal_length]).to(device)

# 		data = {}
# 		data["norm_img"] = norm_img
# 		data["center"] = center
# 		data["scale"] = scale
# 		data["focal_length"] = focal_length
# 		data["img_h"] = img_h
# 		data["img_w"] = img_w
# 		data["pose_params"] = pose_params
# 		data["beta_params"] = beta_params

# 		data["target_landmarks"] = target_landmarks


# 		# keypoints, z_coordinates = detect_pose(img)# shape (18, 2)
# 		# keypoints_norm = normalise(keypoints)
# 		# keypoints = torch.matmul(keypoints, s)

# 		# diff = keypoints_norm[0] - torch.tensor([0.5032, 0.0264])
# 		# keypoints_norm = keypoints_norm - diff
		
# 		# img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
# 		# img = cv2.resize(img, (640, 380), interpolation = cv2.INTER_AREA) 
		
# 		# if self.transform is not None:            
# 		#     img = self.transform(img)     * 255        
# 			# noisy_img = self.transform(noisy_img)  

# 		# keypoints_norm = torch.cat((keypoints_norm, z_coordinates.unsqueeze(1).to(device)), dim=1)

# 		return data

# 	def __len__(self):
# 		return len(self.image_list)
	



	# val_dataset = datasets.ImageFolder(
	# 	valdir,
	# 	transforms.Compose([
	# 		transforms.Resize(256),
	# 		transforms.CenterCrop(224),
	# 		transforms.ToTensor(),
	# 		normalize,
	# 	]))

	# if args.distributed:
	# 	train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
	# 	# val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
	# else:
	# 	train_sampler = None
	# 	val_sampler = None
# train_dataset = MetaphoseDataset("/home/pranoy/code/auto-transform/data/imgs/")
# train_loader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=1, shuffle=True,
#     num_workers=0, pin_memory=True)

# while True:
#     for i, (keypoints) in enumerate(train_loader):
#         print(keypoints.shape)
		