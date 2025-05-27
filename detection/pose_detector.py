import os
import kagglehub
import shutil
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pickle
import pyttsx3
import time
import threading
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class YogaPoseCoach:
    def __init__(self, model_path='../Model/best_model.h5', mapping_path='../Model/class_mapping.pkl'):
        os.system("python -m pip install --upgrade certifi")
        # Process dataset and train model
        ### dataset_path = kagglehub.dataset_download("niharika41298/yoga-poses-dataset")
        ### shutil.move(dataset_path, "./Dataset")
        dataset_path = "./Dataset"
        # Load model and class mapping
        self.model = tf.keras.models.load_model(model_path)
        with open(mapping_path, 'rb') as f:
            self.idx_to_class = pickle.load(f)

        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # Voice feedback
        self.voice = self.VoiceCoach()

    class VoiceCoach:
        def __init__(self):
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 160)
            self.lock = threading.Lock()

        def speak(self, msg):
            def run():
                with self.lock:
                    self.engine.say(msg)
                    self.engine.runAndWait()

            threading.Thread(target=run).start()

    @staticmethod
    def calculate_angle(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def get_angles(self, landmarks, frame_shape):
        points = {}
        indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        for idx in indices:
            landmark = landmarks.landmark[idx]
            points[idx] = [landmark.x * frame_shape[1], landmark.y * frame_shape[0]]
        try:
            angles = np.array([
                self.calculate_angle(points[11], points[13], points[15]),  # Left elbow
                self.calculate_angle(points[12], points[14], points[16]),  # Right elbow
                self.calculate_angle(points[13], points[11], points[23]),  # Left shoulder
                self.calculate_angle(points[14], points[12], points[24]),  # Right shoulder
                self.calculate_angle(points[11], points[23], points[25]),  # Left hip
                self.calculate_angle(points[12], points[24], points[26]),  # Right hip
                self.calculate_angle(points[23], points[25], points[27]),  # Left knee
                self.calculate_angle(points[24], points[26], points[28]),  # Right knee
                self.calculate_angle(points[11], points[23], points[27])  # Spine
            ])
            return points, angles
        except KeyError as e:
            print(f"Missing landmark: {e}")
            return None, None

    def predict_pose(self, img_input, angle_input):
        pred = self.model.predict({
            'image_input': np.expand_dims(img_input, axis=0),
            'angle_input': np.expand_dims(angle_input, axis=0)
        }, verbose=0)
        return self.idx_to_class[np.argmax(pred)], np.max(pred)

    def get_feedback(self, pose_name, angles, confidence):
        if angles is None:
            return ["Can't detect all body points"]
        if confidence > 0.95:
            return ["Perfect! Keep going"]
        feedback = []
        if pose_name == "warrior2":
            if angles[6] < 80:
                feedback.append("Bend front knee more")
            elif angles[6] > 100:
                feedback.append("Reduce knee bend")
            if angles[7] < 170: feedback.append("Straighten back leg")
        elif pose_name == "tree":
            if angles[7] < 170: feedback.append("Straighten standing leg")
            if angles[6] > 110: feedback.append("Bring knee inward")
        elif pose_name == "goddess":
            if angles[6] > 120: feedback.append("Sink deeper into squat")
            if angles[8] < 160: feedback.append("Tuck pelvis")
        elif pose_name == "downdog":
            if angles[0] > 170: feedback.append("Microbend elbows")
            if angles[8] < 160: feedback.append("Lift hips higher")
        elif pose_name == "plank":
            if angles[8] < 160: feedback.append("Straighten body line")
            if angles[0] > 170: feedback.append("Soften elbows")
        return feedback[:1] if feedback else ["Adjust your pose"]

    def run_camera(self, window_name='Yoga Coach (Q to quit)'):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        last_feedback = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)
            if results.pose_landmarks:
                points, angles = self.get_angles(results.pose_landmarks, frame.shape)
                if angles is not None:
                    img_input = cv2.resize(rgb, (224, 224)) / 255.0
                    pose_name, confidence = self.predict_pose(img_input, angles)
                    angle_text = " ".join([f"{a:.1f}°" for a in angles])
                    cv2.putText(frame, angle_text, (30, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    if time.time() - last_feedback > 3:
                        feedback = self.get_feedback(pose_name, angles, confidence)
                        if feedback:
                            self.voice.speak(feedback[0])
                            cv2.putText(frame, feedback[0], (30, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                            last_feedback = time.time()
                    self.mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                    cv2.putText(frame, f"{pose_name} ({confidence:.2f})", (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


def main():
    """Legacy entry point for standalone testing."""
    coach = YogaPoseCoach(model_path='Model/best_model.h5', mapping_path='Model/class_mapping.pkl')
    coach.run_camera()


def gen_pose_frames():
    """Generator for Flask video streaming."""
    coach = YogaPoseCoach(model_path='Model/best_model.h5', mapping_path='Model/class_mapping.pkl')
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    last_feedback = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = coach.pose.process(rgb)
        if results.pose_landmarks:
            points, angles = coach.get_angles(results.pose_landmarks, frame.shape)
            if angles is not None:
                img_input = cv2.resize(rgb, (224, 224)) / 255.0
                pose_name, confidence = coach.predict_pose(img_input, angles)
                angle_text = " ".join([f"{a:.1f}°" for a in angles])
                cv2.putText(frame, angle_text, (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if time.time() - last_feedback > 3:
                    feedback = coach.get_feedback(pose_name, angles, confidence)
                    if feedback:
                        coach.voice.speak(feedback[0])
                        cv2.putText(frame, feedback[0], (30, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                        last_feedback = time.time()
                coach.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, coach.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=coach.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                    connection_drawing_spec=coach.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                cv2.putText(frame, f"{pose_name} ({confidence:.2f})", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        # Encode frame as JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()
