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
import numpy as np
import cv2

app = Flask(__name__)
login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)

app.config['SECRET_KEY'] = 'your_secret_key'
app.config['TEMPLATES_AUTO_RELOAD'] = True

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

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
