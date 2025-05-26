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
        "description": "Stand tall with one foot on the inner thigh of the opposite leg, hands in prayer position.",
        "image": "tree_pose.jpg"
    },
    "downward_dog": {
        "description": "Start on all fours, lift your hips up and back, forming an inverted V shape.",
        "image": "downward_dog.jpg"
    },
    "warrior_pose": {
        "description": "Stand with legs apart, turn one foot out, bend the knee, and extend arms parallel to the ground.",
        "image": "warrior_pose.jpg"
    },
    "godess_pose": {
        "description": "Stand with feet wide apart, toes turned out, bend knees, and lower hips while raising arms to shoulder height.",
        "image": "godess_pose.jpg"
    },
    "plank_pose": {
        "description": "Start in a push-up position, keeping your body straight from head to heels.",
        "image": "plank_pose.jpg"
    }
}


def get_pose_info(pose_name):
    """Retrieve pose information by name."""
    return positions.get(pose_name, {"description": "Pose not found", "image": "default.jpg"})


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
