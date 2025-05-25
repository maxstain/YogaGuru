import os
import cv2
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import kagglehub
import shutil


class YogaPoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        self.model = None
        self.idx_to_class = None
        self.dataset_path = './Dataset'
        if not os.path.exists('./Dataset'):
            dataset_path = kagglehub.dataset_download("niharika41298/yoga-poses-dataset")
            shutil.move(dataset_path, self.dataset_path)

    def calculate_angle(self, a, b, c):
        """Calculate the angle between three points."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine_angle, -1, 1)))

    def extract_features(self, image_path):
        """Extract features from an image."""
        img = cv2.imread(image_path)
        if img is None:
            return None

        results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark
        points = {f'{name}': [landmarks[idx].x, landmarks[idx].y] for idx, name in enumerate([
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'])}

        angles = {
            name: self.calculate_angle(points[p1], points[p2], points[p3])
            for name, (p1, p2, p3) in {
                'left_elbow': ('left_shoulder', 'left_elbow', 'left_wrist'),
                'right_elbow': ('right_shoulder', 'right_elbow', 'right_wrist'),
                # Add other key angles
            }.items()
        }

        img_resized = cv2.resize(img, (224, 224)) / 255.0
        return {'image': img_resized, 'angles': list(angles.values())}

    def train_model(self, dataset_path):
        """Train the model on the dataset."""
        # Preprocess data, create model, and train as before
        pass

    def load_model(self, model_path, class_mapping_path):
        """Load the trained model and class mapping."""
        self.model = tf.keras.models.load_model(model_path)
        with open(class_mapping_path, 'rb') as f:
            self.idx_to_class = pickle.load(f)

    def predict_pose(self, img_input, angle_input):
        """Predict pose using the model."""
        if not self.model:
            raise ValueError("Model not loaded!")
        pred = self.model.predict({
            'image_input': np.expand_dims(img_input, axis=0),
            'angle_input': np.expand_dims(angle_input, axis=0)
        }, verbose=0)
        return self.idx_to_class[np.argmax(pred)], np.max(pred)

    def process_image(self, image_path):
        """Process an image to predict the yoga pose."""
        features = self.extract_features(image_path)
        if features is None:
            return None, None
        img_input = features['image']
        angle_input = features['angles']
        return img_input, angle_input

    def detect_pose(self, image_path):
        """Detect the yoga pose in an image."""
        img_input, angle_input = self.process_image(image_path)
        if img_input is None or angle_input is None:
            return "Pose not detected", 0.0
        pose_name, confidence = self.predict_pose(img_input, angle_input)
        return pose_name, confidence


if __name__ == "__main__":
    detector = YogaPoseDetector()
    # Load the model and class mapping
    detector.load_model('yoga_pose_model.h5', 'class_mapping.pkl')
    # Example usage
    pose_name, confidence = detector.detect_pose('path_to_your_image.jpg')
    print(f"Detected Pose: {pose_name}, Confidence: {confidence:.2f}")
