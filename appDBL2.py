from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from flask_socketio import SocketIO
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import sqlite3
import hashlib
import os
import base64
from datetime import datetime
import pytz
from functools import wraps

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production' 
socketio = SocketIO(app)

model = load_model("tumor_otak.h5")

WIB = pytz.timezone('Asia/Jakarta') 

def get_wib_time():
    """Get current time in WIB timezone"""
    return datetime.now(WIB)

def format_wib_time(dt_string):
    """Convert UTC datetime string from database to WIB and format it"""
    try:
        
        dt_utc = datetime.strptime(dt_string, '%Y-%m-%d %H:%M:%S')
        dt_utc = pytz.utc.localize(dt_utc)
        
        dt_wib = dt_utc.astimezone(WIB)
        return dt_wib.strftime('%Y-%m-%d %H:%M:%S WIB')
    except:
        return dt_string

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
 
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS checkups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            result TEXT NOT NULL,
            confidence REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hash_password):
    return hashlib.sha256(password.encode()).hexdigest() == hash_password

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def save_checkup_to_db(user_id, image_path, result, confidence):
    """Save checkup result to database with WIB timestamp"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    
    wib_time = get_wib_time()
    utc_time = wib_time.astimezone(pytz.utc)
    
    cursor.execute('''
        INSERT INTO checkups (user_id, image_path, result, confidence, created_at)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, image_path, result, confidence, utc_time.strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()

def get_user_checkups(user_id):
    """Get all checkups for a specific user with WIB timestamps"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT image_path, result, confidence, created_at
        FROM checkups 
        WHERE user_id = ?
        ORDER BY created_at DESC
    ''', (user_id,))
    checkups = cursor.fetchall()
    conn.close()
    
    formatted_checkups = []
    for checkup in checkups:
        image_path, result, confidence, created_at = checkup
        formatted_time = format_wib_time(created_at)
        formatted_checkups.append((image_path, result, confidence, formatted_time))
    
    return formatted_checkups

@app.route("/")
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template("index.html")

@app.route("/history")
@login_required
def history():
    user_checkups = get_user_checkups(session['user_id'])
    return render_template("history.html", checkups=user_checkups, username=session.get('username'))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Please fill in all fields')
            return render_template("login.html")
        
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, password_hash FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user and verify_password(password, user[1]):
            session['user_id'] = user[0]
            session['username'] = username
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password')
    
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not all([username, email, password, confirm_password]):
            flash('Please fill in all fields')
            return render_template("signup.html")
        
        if password != confirm_password:
            flash('Passwords do not match')
            return render_template("signup.html")
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long')
            return render_template("signup.html")
        
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM users WHERE username = ? OR email = ?', (username, email))
        if cursor.fetchone():
            flash('Username or email already exists')
            conn.close()
            return render_template("signup.html")
        
       
        password_hash = hash_password(password)
        wib_time = get_wib_time()
        utc_time = wib_time.astimezone(pytz.utc)
        
        try:
            cursor.execute('INSERT INTO users (username, email, password_hash, created_at) VALUES (?, ?, ?, ?)',
                         (username, email, password_hash, utc_time.strftime('%Y-%m-%d %H:%M:%S')))
            conn.commit()
            flash('Account created successfully! Please log in.')
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.Error as e:
            flash('An error occurred while creating your account')
            conn.close()
            return render_template("signup.html")
    
    return render_template("signup.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if "image" not in request.files:
        return jsonify({"result": "No image file uploaded!"})

    file = request.files["image"]
    if file.filename == '':
        return jsonify({"result": "No image file selected!"})
    
    img_bytes = file.read()
    
    
    wib_time = get_wib_time()
    timestamp = wib_time.strftime("%Y%m%d_%H%M%S")
    filename = f"checkup_{session['user_id']}_{timestamp}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    with open(filepath, 'wb') as f:
        f.write(img_bytes)
    
   
    input_data = preprocess_image(img_bytes)
    prediction = model.predict(input_data)
    
    confidence = float(prediction[0][0])
    result = "Tumor Detected" if confidence > 0.5 else "No Tumor Detected"
 
    save_checkup_to_db(session['user_id'], filepath, result, confidence)
    
    return jsonify({
        "result": result,
        "confidence": f"{confidence:.2%}",
        "timestamp": wib_time.strftime('%Y-%m-%d %H:%M:%S WIB')
    })

if __name__ == "__main__":
    init_db()
    socketio.run(app, debug=True)