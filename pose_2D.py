from turtle import pos
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
import numpy as np
import torch



# For webcam input:
def detect_pose(image):
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        model_complexity=2,
        static_image_mode=True,
        refine_face_landmarks=True,
        min_tracking_confidence=0.5) as holistic:
        
        results_mediapipe = {}
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        z_coordinates = []
        scaled_keypoints = []
        normalised_keypoints = []
        for keypoint in results.pose_landmarks.landmark:
            scaled_keypoints.append([keypoint.x * image.shape[1]  , keypoint.y * image.shape[0]])
            normalised_keypoints.append([keypoint.x  , keypoint.y])
            z_coordinates.append(keypoint.z)

        # pose = torch.tensor(poses).float()  # shape  (33, 2)
        scaled_keypoints = torch.tensor(scaled_keypoints).float()
        normalised_keypoints = torch.tensor(normalised_keypoints).float()
        z_coordinates = torch.tensor(z_coordinates).float() 

        results_mediapipe["z_coordinates"] = z_coordinates
        results_mediapipe["scaled_keypoints"] = scaled_keypoints
        results_mediapipe["normalised_keypoints"] = normalised_keypoints
    return results_mediapipe
