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

from flask import Flask, request, render_template, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from detection.pose_detector import YogaPoseDetector

app = Flask(__name__)
login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)

app.config['SECRET_KEY'] = 'your_secret_key'
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Hardcoded users dictionary
users = {
    "admin": {
        "email": "admin@admin.com",
        "password": "admin123"
    },
    "user1": {
        "email": "user1@example.com",
        "password": "user1123"
    }
}

positions = {
    "tree_pose": {
        "name": "Tree Pose",
        "purpose": "Improves balance, focus, and strengthens legs.",
        "image": "tree_pose.jpg",
        "steps": [
            "Stand tall with one foot on the inner thigh of the opposite leg.",
            "Raise your arms to the sides, forming a V shape.",
            "Engage your core and lift your hips off the ground.",
            "Hold the pose for 30 seconds, focusing on deep breathing."
        ],
    },
    "downward_dog": {
        "name": "Downward Dog",
        "purpose": "Stretches the entire body, strengthens arms and legs.",
        "image": "downward_dog.jpg",
        "steps": [
            "Start on all fours with your hands under your shoulders and knees under your hips.",
            "Spread your fingers wide and press into the ground.",
            "Lift your hips up and back, straightening your legs and arms.",
            "Hold the pose for 30 seconds, breathing deeply."
        ],
    },
    "warrior_pose": {
        "name": "Warrior Pose",
        "purpose": "Strengthens legs, opens hips, and improves focus.",
        "image": "warrior_pose.jpg",
        "steps": [
            "Stand with legs apart, feet wide apart.",
            "Turn one foot out 90 degrees, bending the knee.",
            "Extend arms parallel to the ground.",
            "Hold the pose for 30 seconds, focusing on deep breathing."
        ],
    },
    "goddess_pose": {
        "name": "Goddess Pose",
        "purpose": "Strengthens legs, opens hips, and improves balance.",
        "image": "goddess_pose.jpg",
        "steps": [
            "Stand with feet wide apart, toes turned out.",
            "Bend knees, lower hips, and raise arms to shoulder height.",
            "Hold the pose for 30 seconds, focusing on deep breathing."
        ],
    },
    "plank_pose": {
        "name": "Plank Pose",
        "purpose": "Strengthens core, arms, and back.",
        "image": "plank_pose.jpg",
        "steps": [
            "Start in a push-up position with your body straight.",
            "Engage your core and keep your elbows close to your body.",
            "Hold the pose for 30 seconds, focusing on deep breathing."
        ],
    }
}


def get_pose_info(pose_name):
    """Retrieve pose information by name."""
    return positions.get(pose_name, {"name": "", "purpose": "Pose not found", "image": "default.jpg", "steps": []})


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
    YogaPoseDetector()
    selected_pose = request.args.get('pose', 'tree_pose')
    is_camera = request.args.get('camera', 'false').lower() == 'true'
    if is_camera:
        flash("Camera mode is enabled. Please ensure your camera is connected.", "info")
    else:
        flash("Camera mode is disabled. You can enable it in the settings.", "info")
    return render_template('index.html', user=current_user,
                           pose_info=get_pose_info(selected_pose),
                           positions=positions,
                           is_camera=is_camera)


if __name__ == '__main__':
    app.run(debug=True)
