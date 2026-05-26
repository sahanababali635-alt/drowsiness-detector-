"""
DROWSINESS DETECTOR WEB APP
Flask-based web interface for real-time drowsiness detection
Access: http://localhost:5000
"""

from flask import Flask, render_template, Response, jsonify, request, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import cv2
import numpy as np
from collections import deque
import threading
import time
import json
import glob
import os
import sqlite3
from datetime import datetime
from config import DATABASE_URI

# Set up template and static folder paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ==================== DATABASE MODELS ====================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<User {self.username}>'

class DrowsinessLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    driver_id = db.Column(db.String(80), nullable=False)
    status = db.Column(db.String(10), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    duration_seconds = db.Column(db.Integer, nullable=True)

    def __repr__(self):
        return f'<DrowsinessLog {self.driver_id} {self.status}>'

class DriverSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    driver_id = db.Column(db.String(80), nullable=False)
    bus_id = db.Column(db.String(80), nullable=True)
    login_time = db.Column(db.DateTime, default=datetime.utcnow)
    logout_time = db.Column(db.DateTime, nullable=True)
    active = db.Column(db.Boolean, default=True)

    def __repr__(self):
        return f'<DriverSession {self.driver_id} {self.bus_id} {self.login_time}>'

class PeopleCount(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    bus_id = db.Column(db.String(20), nullable=False)
    entry_count = db.Column(db.Integer, default=0)
    exit_count = db.Column(db.Integer, default=0)
    total_passengers = db.Column(db.Integer, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<PeopleCount {self.bus_id} {self.total_passengers}>'

# ==================== DATABASE INITIALIZATION ====================
def init_db():
    """Initialize the database and create tables."""
    try:
        with app.app_context():
            db.create_all()
            # Create default admin user if not exists
            if not User.query.filter_by(username='admin').first():
                admin_user = User(username='admin', password='password123')
                db.session.add(admin_user)
                db.session.commit()
                print("Database initialized with default admin user.")
            print("Database connection successful!")
    except sqlite3.DatabaseError as e:
        print(f"[WARNING] SQLite database failed: {str(e)}")
        if DATABASE_URI.startswith('sqlite'):
            sqlite_path = DATABASE_URI.replace('sqlite:///', '')
            if os.path.exists(sqlite_path):
                print(f"[INFO] Deleting corrupted database file: {sqlite_path}")
                try:
                    os.remove(sqlite_path)
                except Exception as rm_err:
                    print(f"[ERROR] Could not delete corrupted database file: {rm_err}")
            try:
                with app.app_context():
                    db.create_all()
                    if not User.query.filter_by(username='admin').first():
                        admin_user = User(username='admin', password='password123')
                        db.session.add(admin_user)
                        db.session.commit()
                        print("Database recreated with default admin user.")
                    print("Database connection successful after recreate!")
                    return
            except Exception as e2:
                print(f"[ERROR] Failed to recreate database: {e2}")
        print("[INFO] The app will run without database features.")
        print("[INFO] Please verify your database settings in src/config.py")
    except Exception as e:
        print(f"[WARNING] Database initialization failed: {str(e)}")
        print("[INFO] The app will run without database features.")
        print("[INFO] Please verify your database settings in src/config.py")

# ==================== GLOBAL VARIABLES ====================
class DetectorState:
    def __init__(self):
        self.total_frames = 0
        self.drowsy_frames = 0
        self.people_count = 0
        self.beep_count = 0
        self.last_people_save_time = 0.0
        self.is_drowsy = False
        self.ear = 0.5
        self.recognized_name = 'Unknown'
        self.recognition_confidence = 0.0
        self.fps = 0
        self.running = False
        self.frame_buffer = None
        self.start_time = time.time()
        self.lock = threading.Lock()
        self.current_driver_id = None
        self.current_bus_id = None

detector_state = DetectorState()

# ==================== DATABASE HELPERS ====================
def save_drowsiness_log(driver_id=None, status='drowsy', duration_seconds=None):
    try:
        with app.app_context():
            selected_driver = driver_id or detector_state.current_driver_id or 'Unknown'
            log = DrowsinessLog(
                driver_id=selected_driver,
                status=status,
                duration_seconds=duration_seconds,
            )
            db.session.add(log)
            db.session.commit()
    except Exception as e:
        print(f"[WARNING] Could not save drowsiness log: {e}")


def create_driver_session(driver_id, bus_id=None):
    try:
        with app.app_context():
            session_record = DriverSession(
                driver_id=driver_id,
                bus_id=bus_id or 'Unknown Bus',
            )
            db.session.add(session_record)
            db.session.commit()
            return session_record.id
    except Exception as e:
        print(f"[WARNING] Could not create driver session: {e}")
        return None


def close_driver_session(driver_id=None, session_id=None):
    try:
        with app.app_context():
            if session_id:
                session_record = DriverSession.query.filter_by(id=session_id, active=True).first()
            else:
                session_record = DriverSession.query.filter_by(driver_id=driver_id, active=True).order_by(DriverSession.login_time.desc()).first()
            if session_record:
                session_record.logout_time = datetime.utcnow()
                session_record.active = False
                db.session.commit()
    except Exception as e:
        print(f"[WARNING] Could not close driver session: {e}")


def save_people_count(bus_id, people_count):
    try:
        with app.app_context():
            record = PeopleCount(
                bus_id=bus_id,
                entry_count=people_count,
                exit_count=0,
                total_passengers=people_count,
            )
            db.session.add(record)
            db.session.commit()
    except Exception as e:
        print(f"[WARNING] Could not save people count: {e}")

# ==================== CONSTANTS ====================
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20
RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
USE_DEMO = False
KNOWN_FACES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'known_faces'))
RECOGNITION_CONFIDENCE_THRESHOLD = 80

# ==================== LOGIN SETUP ====================
app.secret_key = "yoursecretkey"

# ==================== ALARM SYSTEM ====================
class AlarmManager:
    def __init__(self):
        self.is_playing = False

    def play_alarm(self):
        if self.is_playing:
            return
        self.is_playing = True
        try:
            import winsound
            for _ in range(3):
                winsound.Beep(1000, 500)
                time.sleep(0.2)
        except ImportError:
            print('\n⚠️  DROWSINESS ALERT! ⚠️')
        finally:
            self.is_playing = False

# ==================== DETECTION CLASS ====================
class DrowsinessDetectorWeb:
    def __init__(self, use_demo=True):
        self.use_demo = use_demo
        self.eye_closed_counter = 0
        self.frame_count = 0
        self.alarm = AlarmManager()
        self.recognized_name = 'Unknown'
        self.recognition_confidence = 0.0
        self.face_recognizer_ready = False
        self.face_id_to_name = {}
        self.face_recognizer = None
        
        if not use_demo:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        else:
            self.cap = None
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.load_face_recognizer()
    
    def load_face_recognizer(self):
        """Load or train LBPH face recognizer using labeled images."""
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        face_samples = []
        labels = []
        label_ids = {}
        current_id = 0

        for person_name in sorted(os.listdir(KNOWN_FACES_DIR)):
            person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
            if not os.path.isdir(person_dir):
                continue
            label_ids[person_name] = current_id
            for image_path in glob.glob(os.path.join(person_dir, '*.*')):
                image = cv2.imread(image_path)
                if image is None:
                    continue
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
                if len(faces) != 1:
                    continue
                (x, y, w, h) = faces[0]
                face_samples.append(gray[y:y+h, x:x+w])
                labels.append(current_id)
            current_id += 1

        recognizer_supported = hasattr(cv2, 'face') and hasattr(cv2.face, 'LBPHFaceRecognizer_create')
        if len(face_samples) >= 2 and recognizer_supported:
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.face_recognizer.train(face_samples, np.array(labels))
            self.face_id_to_name = {v: k for k, v in label_ids.items()}
            self.face_recognizer_ready = True
        else:
            self.face_recognizer_ready = False
            print('[INFO] Face recognizer not ready. Add data/known_faces/<name>/<images> or install OpenCV contrib.')

    def recognize_face(self, face_gray):
        """Return recognized name and confidence for a face ROI."""
        if not self.face_recognizer_ready:
            return 'Unknown', 0.0
        try:
            label, confidence = self.face_recognizer.predict(face_gray)
            name = self.face_id_to_name.get(label, 'Unknown')
            if confidence <= RECOGNITION_CONFIDENCE_THRESHOLD:
                return name, confidence
            return 'Unknown', confidence
        except Exception:
            return 'Unknown', 0.0

    def create_demo_frame(self):
        """Create synthetic demo frame"""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
        
        # Cycle through states
        cycle = self.frame_count % 150
        
        if cycle < 30:
            state = "NORMAL"
            eye_openness = 0.8
            color = GREEN
        elif cycle < 60:
            state = "CLOSING"
            eye_openness = 0.5 - (cycle - 30) / 30 * 0.4
            color = YELLOW
        elif cycle < 90:
            state = "DROWSY"
            eye_openness = 0.1
            color = RED
        elif cycle < 120:
            state = "RECOVERING"
            eye_openness = 0.1 + (cycle - 90) / 30 * 0.7
            color = YELLOW
        else:
            state = "NORMAL"
            eye_openness = 0.8
            color = GREEN
        
        # Draw face
        cv2.rectangle(frame, (150, 100), (490, 400), color, 3)
        cv2.putText(frame, "Face Detected", (160, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw eyes
        left_eye_height = int(20 * eye_openness)
        right_eye_height = int(20 * eye_openness)
        
        cv2.ellipse(frame, (240, 180), (30, left_eye_height), 0, 0, 360, GREEN, -1)
        cv2.circle(frame, (240, 180), 8, BLACK, -1)
        
        cv2.ellipse(frame, (400, 180), (30, right_eye_height), 0, 0, 360, GREEN, -1)
        cv2.circle(frame, (400, 180), 8, BLACK, -1)
        
        return frame, state, eye_openness
    
    def get_frame(self):
        """Get next frame - either from camera or demo"""
        if self.use_demo:
            frame, state, ear = self.create_demo_frame()
            is_drowsy = (state == "DROWSY")
            return frame, is_drowsy, ear, 1, False
        else:
            ret, frame = self.cap.read()
            if not ret:
                return None, False, 0.5, 0, False
            
            frame = cv2.resize(frame, (640, 480))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            is_drowsy = False
            ear = 0.5
            people_count = len(faces)
            self.recognized_name = 'Unknown'
            self.recognition_confidence = 0.0
            beep_triggered = False
            
            if len(faces) > 0:
                face = max(faces, key=lambda f: f[2] * f[3])
                (x, y, w, h) = face
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]

                if self.face_recognizer_ready and roi_gray.size:
                    face_gray = cv2.resize(roi_gray, (200, 200))
                    name, confidence = self.recognize_face(face_gray)
                    self.recognized_name = name
                    self.recognition_confidence = confidence
                
                eyes = self.eye_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(20, 20)
                )
                eyes_detected = len(eyes)

                if eyes_detected >= 1:
                    eyes = sorted(eyes, key=lambda e: e[0])
                    for (ex, ey, ew, eh) in eyes[:2]:
                        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), GREEN, 2)

                    eye_heights = [e[3] for e in eyes[:2]]
                    eye_widths = [e[2] for e in eyes[:2]]
                    avg_width = np.mean(eye_widths)
                    avg_height = np.mean(eye_heights)
                    ear = avg_height / max(avg_width, 1)

                    if ear < EYE_AR_THRESH:
                        self.eye_closed_counter += 1
                    else:
                        self.eye_closed_counter = 0
                else:
                    self.eye_closed_counter += 1

                if self.eye_closed_counter >= EYE_AR_CONSEC_FRAMES:
                    is_drowsy = True

                beep_triggered = False
                if is_drowsy and not self.alarm.is_playing:
                    alarm_thread = threading.Thread(target=self.alarm.play_alarm, daemon=True)
                    alarm_thread.start()
                    beep_triggered = True

                color = RED if is_drowsy else GREEN
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, self.recognized_name, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)
            else:
                self.eye_closed_counter = 0

            return frame, is_drowsy, ear, people_count, beep_triggered
    
    def draw_dashboard(self, frame, is_drowsy, ear, people_count):
        """Draw dashboard info on frame"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (460, 190), BLACK, -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        y_pos = 35
        cv2.putText(frame, f"Frame: {detector_state.total_frames}", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
        
        cv2.putText(frame, f"EAR: {ear:.3f}", (20, y_pos + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
        
        cv2.putText(frame, f"People: {people_count}", (20, y_pos + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
        
        cv2.putText(frame, f"Driver: {detector_state.recognized_name}", (20, y_pos + 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)
        
        status_color = RED if is_drowsy else GREEN
        status_text = "🚨 DROWSY!" if is_drowsy else "✓ ALERT"
        cv2.putText(frame, status_text, (20, y_pos + 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        return frame

detector = DrowsinessDetectorWeb(use_demo=USE_DEMO)

# ==================== DETECTION LOOP ====================
def run_detection():
    """Main detection loop running in background"""
    fps_counter = 0
    fps_start_time = time.time()
    
    detector_state.running = True
    
    while detector_state.running:
        previous_people_count = detector_state.people_count
        frame, is_drowsy, ear, people_count, beep_event = detector.get_frame()

        if frame is None:
            continue

        detector_state.total_frames += 1
        if is_drowsy:
            detector_state.drowsy_frames += 1

        detector_state.is_drowsy = is_drowsy
        detector_state.ear = ear
        detector_state.people_count = people_count
        detector_state.recognized_name = detector.recognized_name
        detector_state.recognition_confidence = detector.recognition_confidence

        if beep_event:
            detector_state.beep_count += 1
            save_drowsiness_log(detector.recognized_name or 'Unknown', status='drowsy', duration_seconds=0)

        if people_count != previous_people_count or time.time() - detector_state.last_people_save_time > 10.0:
            save_people_count('bus-101', people_count)
            detector_state.last_people_save_time = time.time()

        # Draw dashboard
        frame = detector.draw_dashboard(frame, is_drowsy, ear, people_count)

        # Store frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        with detector_state.lock:
            detector_state.frame_buffer = frame_bytes
        
        detector.frame_count += 1
        
        # Update FPS
        fps_counter += 1
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            detector_state.fps = fps_counter / elapsed
            fps_counter = 0
            fps_start_time = time.time()
        
        time.sleep(0.033)  # ~30 FPS

# ==================== LOGIN ROUTES ====================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        bus_id = request.form.get('bus_id', '').strip() or 'Unknown Bus'
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            session['user'] = username
            session['bus_id'] = bus_id
            session['driver_session_id'] = create_driver_session(username, bus_id)
            detector_state.current_driver_id = username
            detector_state.current_bus_id = bus_id
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid credentials', username=username, bus_id=bus_id)
    return render_template('login.html')

@app.route('/logout')
def logout():
    current_user = session.get('user')
    session_id = session.get('driver_session_id')
    if current_user:
        close_driver_session(driver_id=current_user, session_id=session_id)
    session.pop('user', None)
    session.pop('bus_id', None)
    session.pop('driver_session_id', None)
    detector_state.current_driver_id = None
    detector_state.current_bus_id = None
    return redirect(url_for('login'))

# ==================== FLASK ROUTES ====================
@app.route('/')
def index():
    """Home page"""
    if 'user' not in session:
        return redirect(url_for('login'))

    driver_name = session['user']
    driver_logs = DrowsinessLog.query.filter_by(driver_id=driver_name).order_by(DrowsinessLog.timestamp.desc()).limit(10).all()
    sessions = DriverSession.query.filter_by(driver_id=driver_name).order_by(DriverSession.login_time.desc()).limit(10).all()
    return render_template('index.html', driver_name=driver_name, bus_id=session.get('bus_id'), driver_logs=driver_logs, sessions=sessions)

@app.route('/video_feed')
def video_feed():
    """Video stream endpoint"""
    def generate():
        while True:
            with detector_state.lock:
                if detector_state.frame_buffer is not None:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' +
                           detector_state.frame_buffer + b'\r\n\r\n')
            time.sleep(0.033)
    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    """Get current statistics"""
    elapsed = time.time() - detector_state.start_time
    drowsy_pct = (detector_state.drowsy_frames / max(detector_state.total_frames, 1)) * 100
    
    database_beep_count = DrowsinessLog.query.filter_by(status='drowsy').count()
    latest_people = PeopleCount.query.order_by(PeopleCount.timestamp.desc()).first()
    stats = {
        'total_frames': detector_state.total_frames,
        'drowsy_frames': detector_state.drowsy_frames,
        'drowsy_percentage': drowsy_pct,
        'current_ear': detector_state.ear,
        'people_count': detector_state.people_count,
        'beep_count': detector_state.beep_count,
        'recognized_name': detector_state.recognized_name,
        'recognition_confidence': detector_state.recognition_confidence,
        'fps': detector_state.fps,
        'is_drowsy': detector_state.is_drowsy,
        'elapsed_time': elapsed,
        'database_beep_count': database_beep_count,
        'database_latest_people': latest_people.total_passengers if latest_people else 0,
        'database_latest_bus_id': latest_people.bus_id if latest_people else None,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return jsonify(stats)

@app.route('/start')
def start_detection():
    """Start detection"""
    if not detector_state.running:
        detector_state.running = True
        thread = threading.Thread(target=run_detection, daemon=True)
        thread.start()
    return jsonify({'status': 'started'})

@app.route('/stop')
def stop_detection():
    """Stop detection"""
    detector_state.running = False
    return jsonify({'status': 'stopped'})

@app.route('/reset')
def reset_stats():
    """Reset statistics"""
    detector_state.total_frames = 0
    detector_state.drowsy_frames = 0
    detector_state.people_count = 0
    detector_state.is_drowsy = False
    detector_state.ear = 0.5
    detector_state.recognized_name = 'Unknown'
    detector_state.recognition_confidence = 0.0
    detector_state.start_time = time.time()
    detector.eye_closed_counter = 0
    return jsonify({'status': 'reset'})

@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'running' if detector_state.running else 'stopped',
        'frames': detector_state.total_frames
    })

# ==================== MAIN ====================
if __name__ == '__main__':
    # Initialize database
    init_db()
    
    print("\n" + "="*60)
    print("DROWSINESS DETECTOR - WEB APPLICATION")
    print("="*60)
    print("\n🌐 Starting web server...")
    print("\n📍 Access the application at:")
    print("   ➜ http://localhost:5000")
    print("\n💡 Features:")
    print("   ✓ Real-time video stream")
    print("   ✓ Live statistics dashboard")
    print("   ✓ Drowsiness detection alerts")
    print("   ✓ Start/Stop controls")
    print("   ✓ Performance metrics")
    print("\n⚠️  Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    # Start detection thread
    detection_thread = threading.Thread(target=run_detection, daemon=True)
    detection_thread.start()
    
    # Start Flask app
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n\n[INFO] Server stopped by user")
        detector_state.running = False
