import os

# Clear the console and install required packages
if os.path.exists("requirements.txt"):
    # Verify if the operating system is Windows
    if os.name == 'nt':
        # Clear the console for Windows
        os.system("cls")
        os.system("python.exe -m pip install --upgrade pip")
        os.system("pip install -r requirements.txt")
        os.system("cls")
    else:
        # Clear the console for Unix/Linux/Mac
        os.system("clear")
        os.system("python.exe -m pip install --upgrade pip")
        os.system("pip install -r requirements.txt")
        os.system("clear")

from flask import Flask, request, render_template, redirect, url_for, flash, Response, jsonify
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from detection.pose_detector import gen_pose_frames
from detection.pose_detector import mp
from detection.pose_detector import YogaPoseCoach
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import pandas as pd
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    """Process the entire dataset and save features"""
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

    # Bar Chart: Number of Images per Pose Class
    class_counts = df['class'].value_counts()
    plt.figure(figsize=(8, 5))
    plt.bar(class_counts.index, class_counts.values, color='skyblue')
    plt.xlabel('Pose Class')
    plt.ylabel('Number of Images')
    plt.title('Images per Pose Class')
    plt.show()

    # Histogram: Distribution of a Specific Angle (e.g., left_knee)
    left_knee_angles = [angles[6] for angles in df['angles']]
    plt.figure(figsize=(8, 5))
    plt.hist(left_knee_angles, bins=20, color='orange', edgecolor='black')
    plt.xlabel('Left Knee Angle (degrees)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Left Knee Angles')
    plt.show()

    # Pie Chart: Proportion of Each Pose Class
    plt.figure(figsize=(6, 6))
    plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Pose Class Distribution')
    plt.axis('equal')
    plt.show()

    # Save processed data
    df.to_pickle("./Model/yoga_dataset_processed.pkl")

    # Calculate pose standards
    pose_standards = df.groupby('class')['angles'].apply(
        lambda x: {
            'median': np.median(np.vstack(x), axis=0),
            'std': np.std(np.vstack(x), axis=0)
        }
    ).to_dict()

    with open('./Model/pose_standards.pkl', 'wb') as f:
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
    df = pd.read_pickle('Model/yoga_dataset_processed.pkl')

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
        ModelCheckpoint('Model/best_model.h5', save_best_only=True),
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

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title('Accuracy')

    plt.show()

    # Save class mapping
    idx_to_class = {v: k for k, v in class_to_idx.items()}  # Reverse mapping
    with open('Model/class_mapping.pkl', 'wb') as f:
        pickle.dump(idx_to_class, f)  # New: {0:'downdog', 1:'warrior'}

    val_loss, val_acc = model.evaluate(
        {'image_input': X_img_val, 'angle_input': X_ang_val},
        y_val,
        verbose=0
    )
    print(f"Final Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"Final Validation Loss: {val_loss * 100:.2f}%")

    return model, history


app = Flask(__name__)
login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)

app.config['SECRET_KEY'] = 'your_secret_key'
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Process dataset and train model if not already done
if not os.path.exists('Model/best_model.h5') or not os.path.exists('Model/class_mapping.pkl'):
    df, standards = process_dataset("Dataset/1/DATASET/TRAIN")
    model, history = train_model()
coach = YogaPoseCoach(model_path='Model/best_model.h5', mapping_path='Model/class_mapping.pkl')

# Hardcoded users dictionary
users = {
    "admin": {
        "email": "admin@admin.com",
        "password": "admin"
    },
    "user1": {
        "email": "user1@example.com",
        "password": "user"
    }
}

positions = {
    "tree_pose": {
        "name": "Tree Pose",
        "purpose": "Improves balance, focus, and strengthens legs.",
        "image": "tree_pose.jpg",
        "steps": [
            "Stand tall with feet together and arms at your sides.",
            "Shift weight onto your left foot.",
            "Place the sole of your right foot on your left inner thigh or calf (avoid the knee).",
            "Bring your hands together in prayer position at your chest, or raise them overhead.",
            "Keep your gaze fixed on a point for balance.",
            "Hold for 30 seconds to 1 minute, then switch sides."
        ],
    },
    "downward_dog": {
        "name": "Downward Dog",
        "purpose": "Stretches the entire body, strengthens arms and legs.",
        "image": "downward_dog.jpg",
        "steps": [
            "Start on your hands and knees in a tabletop position.",
            "Spread your fingers wide and tuck your toes under.",
            "Lift your hips up and back, straightening your legs as much as comfortable.",
            "Press your heels toward the floor (knees can stay slightly bent).",
            "Keep your head between your arms and your back straight.",
            "Hold for 1–3 minutes, breathing deeply."
        ],
    },
    "warrior_pose": {
        "name": "Warrior Pose",
        "purpose": "Strengthens legs, opens hips, and improves focus.",
        "image": "warrior_pose.jpg",
        "steps": [
            "Stand tall, then step your left foot back about 3–4 feet.",
            "Turn your left foot slightly out, and bend your right knee over the ankle.",
            "Keep your hips square facing forward.",
            "Raise your arms overhead, palms facing each other.",
            "Gaze forward or slightly up.",
            "Hold for 30 seconds to 1 minute, then switch sides."
        ],
    },
    "goddess_pose": {
        "name": "Goddess Pose",
        "purpose": "Strengthens legs, opens hips, and improves balance.",
        "image": "goddess_pose.jpg",
        "steps": [
            "Stand with feet wide apart (about 3 feet), toes turned out at 45°.",
            "Bend your knees deeply, aligning them over your ankles.",
            "Keep your back straight and pelvis neutral.",
            "Bring your arms to shoulder height, bend the elbows, palms facing forward.",
            "Engage your core and hold for 30 seconds to 1 minute."
        ],
    },
    "plank_pose": {
        "name": "Plank Pose",
        "purpose": "Strengthens core, arms, and back.",
        "image": "plank_pose.jpg",
        "steps": [
            "Start in a tabletop position.",
            "Step both feet back, forming a straight line from head to heels.",
            "Hands should be shoulder-width apart, directly under shoulders.",
            "Engage your core and keep your hips level.",
            "Keep neck neutral, gaze down.",
            "Hold for 30 seconds to 1 minute, breathing evenly."
        ],
    }
}


@app.route('/video_feed')
@login_required
def video_feed():
    return Response(gen_pose_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def get_pose_info(pose_name):
    """Retrieve pose information by name."""
    return positions.get(pose_name, {"name": "", "purpose": "Pose not found", "image": "Yoga_splash.jpg", "steps": []})


class User(UserMixin):
    def __init__(self, username, email):
        self.id = username
        self.email = email

    @classmethod
    def get(cls, username):
        if username in users:
            user_data = users[username]
            return cls(username, user_data['email'])
        return None


@app.route('/compare_pose', methods=['POST'])
def compare_pose():
    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    if not results.pose_landmarks:
        return jsonify({'error': 'No pose detected'}), 400

    _, angles = coach.get_angles(results.pose_landmarks, img.shape)
    if angles is None:
        return jsonify({'error': 'Incomplete pose'}), 400

    img_input = cv2.resize(rgb, (224, 224)) / 255.0
    pose_name, confidence = coach.predict_pose(img_input, angles)
    accuracy = round(confidence * 100, 2)

    return jsonify({'pose': pose_name, 'accuracy': accuracy})


@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)


@app.route('/', methods=['GET'])
def index():
    return render_template('splashscreen.html', user=current_user)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Check if the user exists and the password matches
        if username in users and users[username]['password'] == password:
            user = User.get(username)
            login_user(user)
            flash("Logged in successfully!", "success")
            return redirect(url_for('home'))
        else:
            flash("Invalid username or password.", "danger")
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for('index'))


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        # Check if the user already exists
        if username in users:
            flash("Username already exists.", "danger")
        else:
            users[username] = {'email': email, 'password': password}
            flash("User created successfully!", "success")
            return redirect(url_for('login'))

    return render_template('signup.html')


@app.route('/home')
@login_required
def home():
    selected_pose = request.args.get('pose', 'tree_pose')
    is_camera = request.args.get('camera', 'false').lower() == 'true'
    return render_template('index.html', user=current_user,
                           pose_info=get_pose_info(selected_pose),
                           positions=positions,
                           camera=is_camera,
                           selected_pose=selected_pose
                           )


if __name__ == '__main__':
    app.run(debug=True)
