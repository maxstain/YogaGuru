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

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)


def calculate_angle(a, b, c):
    """Calculate the angle between three points"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1, 1)))


def extract_features(image_path):
    """Extract both image and biomechanical features from a single image"""
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Process image with MediaPipe
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
        return None

    # Get landmarks
    landmarks = results.pose_landmarks.landmark

    # Calculate key angles
    points = {
        'left_shoulder': [landmarks[11].x, landmarks[11].y],
        'right_shoulder': [landmarks[12].x, landmarks[12].y],
        'left_elbow': [landmarks[13].x, landmarks[13].y],
        'right_elbow': [landmarks[14].x, landmarks[14].y],
        'left_wrist': [landmarks[15].x, landmarks[15].y],
        'right_wrist': [landmarks[16].x, landmarks[16].y],
        'left_hip': [landmarks[23].x, landmarks[23].y],
        'right_hip': [landmarks[24].x, landmarks[24].y],
        'left_knee': [landmarks[25].x, landmarks[25].y],
        'right_knee': [landmarks[26].x, landmarks[26].y],
        'left_ankle': [landmarks[27].x, landmarks[27].y],
        'right_ankle': [landmarks[28].x, landmarks[28].y]
    }

    angles = {
        'left_elbow': calculate_angle(points['left_shoulder'], points['left_elbow'], points['left_wrist']),
        'right_elbow': calculate_angle(points['right_shoulder'], points['right_elbow'], points['right_wrist']),
        'left_shoulder': calculate_angle(points['left_elbow'], points['left_shoulder'], points['left_hip']),
        'right_shoulder': calculate_angle(points['right_elbow'], points['right_shoulder'], points['right_hip']),
        'left_hip': calculate_angle(points['left_shoulder'], points['left_hip'], points['left_knee']),
        'right_hip': calculate_angle(points['right_shoulder'], points['right_hip'], points['right_knee']),
        'left_knee': calculate_angle(points['left_hip'], points['left_knee'], points['left_ankle']),
        'right_knee': calculate_angle(points['right_hip'], points['right_knee'], points['right_ankle']),
        'spine': calculate_angle(points['left_shoulder'], points['left_hip'], points['left_ankle'])
    }

    # Prepare image for CNN
    img_resized = cv2.resize(img, (224, 224))
    img_normalized = img_resized / 255.0

    return {
        'image': img_normalized,
        'angles': list(angles.values()),
        'landmarks': points
    }


def process_dataset(dataset_path):
    """Process entire dataset and save features"""
    data = []
    pose_classes = [d for d in os.listdir(dataset_path)
                    if os.path.isdir(os.path.join(dataset_path, d))]

    for pose_class in pose_classes:
        class_dir = os.path.join(dataset_path, pose_class)
        image_files = [f for f in os.listdir(class_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for img_file in tqdm(image_files, desc=f"Processing {pose_class}"):
            img_path = os.path.join(class_dir, img_file)
            features = extract_features(img_path)

            if features is not None:
                features['class'] = pose_class
                data.append(features)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Save processed data
    df.to_pickle('yoga_dataset_processed.pkl')

    # Calculate pose standards
    pose_standards = df.groupby('class')['angles'].apply(
        lambda x: {
            'median': np.median(np.vstack(x), axis=0),
            'std': np.std(np.vstack(x), axis=0)
        }
    ).to_dict()

    with open('../Model/pose_standards.pkl', 'wb') as f:
        pickle.dump(pose_standards, f)

    return df, pose_standards


def create_hybrid_model(num_classes=5):
    """Create a model that combines CNN and biomechanical features"""
    # Image branch (CNN)
    image_input = layers.Input(shape=(224, 224, 3), name='image_input')

    x = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Angle branch
    angle_input = layers.Input(shape=(9,), name='angle_input')
    a = layers.Dense(32, activation='relu')(angle_input)

    # Combined features
    combined = layers.concatenate([x, a])

    # Classifier
    z = layers.Dense(128, activation='relu')(combined)
    z = layers.Dropout(0.5)(z)
    output = layers.Dense(num_classes, activation='softmax')(z)

    return models.Model(inputs=[image_input, angle_input], outputs=output)


def prepare_data(df):
    """Prepare data for training"""
    # Convert images to array
    X_images = np.array([x for x in df['image']])

    # Convert angles to array
    X_angles = np.array([x for x in df['angles']])

    # Convert labels to one-hot
    class_to_idx = {cls: i for i, cls in enumerate(df['class'].unique())}
    y = tf.keras.utils.to_categorical(df['class'].map(class_to_idx))

    return X_images, X_angles, y, class_to_idx


def train_model():
    # Load processed data
    df = pd.read_pickle('../Model/yoga_dataset_processed.pkl')

    # Prepare data
    X_img, X_ang, y, class_to_idx = prepare_data(df)

    # Split data - USE THE SAME VARIABLE NAMES RETURNED FROM prepare_data()
    (X_img_train, X_img_val,
     X_ang_train, X_ang_val,
     y_train, y_val) = train_test_split(X_img, X_ang, y, test_size=0.2)

    # Create model
    model = create_hybrid_model(num_classes=len(class_to_idx))

    # Compile
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Define callbacks
    training_callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint('../Model/best_model.h5', save_best_only=True),
        ReduceLROnPlateau(factor=0.1, patience=3)
    ]

    # Train
    history = model.fit(
        x={'image_input': X_img_train, 'angle_input': X_ang_train},
        y=y_train,
        validation_data=({'image_input': X_img_val, 'angle_input': X_ang_val}, y_val),
        epochs=10,
        batch_size=32,
        callbacks=training_callbacks
    )

    # Save class mapping
    idx_to_class = {v: k for k, v in class_to_idx.items()}  # Reverse mapping
    with open('../Model/class_mapping.pkl', 'wb') as f:
        pickle.dump(idx_to_class, f)  # New: {0:'downdog', 1:'warrior'}

    return model, history


# Initialize MediaPipe components
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Configure TensorFlow for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# MediaPipe Pose Configuration
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# Threaded Voice Engine
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


voice = VoiceCoach()


# Core Functions
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def get_angles(landmarks, frame_shape):
    points = {}
    indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # Shoulders, elbows, wrists, hips, knees, ankles

    for idx in indices:
        landmark = landmarks.landmark[idx]
        points[idx] = [landmark.x * frame_shape[1], landmark.y * frame_shape[0]]

    try:
        angles = np.array([
            calculate_angle(points[11], points[13], points[15]),  # Left elbow
            calculate_angle(points[12], points[14], points[16]),  # Right elbow
            calculate_angle(points[13], points[11], points[23]),  # Left shoulder
            calculate_angle(points[14], points[12], points[24]),  # Right shoulder
            calculate_angle(points[11], points[23], points[25]),  # Left hip
            calculate_angle(points[12], points[24], points[26]),  # Right hip
            calculate_angle(points[23], points[25], points[27]),  # Left knee
            calculate_angle(points[24], points[26], points[28]),  # Right knee
            calculate_angle(points[11], points[23], points[27])  # Spine
        ])
        return points, angles
    except KeyError as e:
        print(f"Missing landmark: {e}")
        return None, None


def predict_pose(model, img_input, angle_input, idx_to_class):
    pred = model.predict({
        'image_input': np.expand_dims(img_input, axis=0),
        'angle_input': np.expand_dims(angle_input, axis=0)
    }, verbose=0)
    return idx_to_class[np.argmax(pred)], np.max(pred)


def get_feedback(pose_name, angles, confidence):
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


# Main Application
def main():
    # Load model
    model = tf.keras.models.load_model('../Model/best_model.h5')
    with open('../Model/class_mapping.pkl', 'rb') as f:
        idx_to_class = pickle.load(f)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    last_feedback = time.time()

    def process_frame(frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if not results.pose_landmarks:
            return frame, None, None, None
        points, angles = get_angles(results.pose_landmarks, frame.shape)
        if angles is None:
            return frame, None, None, None
        img_input = cv2.resize(rgb, (224, 224)) / 255.0
        pose_name, confidence = predict_pose(model, img_input, angles, idx_to_class)
        angle_text = " ".join([f"{a:.1f}°" for a in angles])
        cv2.putText(frame, angle_text, (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )
        cv2.putText(frame, f"{pose_name} ({confidence:.2f})", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        return frame, pose_name, angles, confidence

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, pose_name, angles, confidence = process_frame(frame)

        if pose_name is not None and angles is not None and confidence is not None:
            if time.time() - last_feedback > 3:
                feedback = get_feedback(pose_name, angles, confidence)
                if feedback:
                    voice.speak(feedback[0])
                    cv2.putText(frame, feedback[0], (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    last_feedback = time.time()

        cv2.imshow('Yogi Nova (Q to quit)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    os.system("python -m pip install --upgrade certifi")
    # Process dataset and train model
    ### dataset_path = kagglehub.dataset_download("niharika41298/yoga-poses-dataset")
    ### shutil.move(dataset_path, "./Dataset")
    dataset = "../Dataset/1/DATASET/TRAIN"
    df, standards = process_dataset(dataset)
    model, history = train_model()
    main()
