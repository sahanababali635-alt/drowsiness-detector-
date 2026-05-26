import cv2
import glob
import numpy as np
from scipy.spatial import distance as dist
from collections import deque
import threading
import time
from datetime import datetime
import os

# ==================== CONSTANTS ====================
EYE_AR_THRESH = 0.2  # Eye Aspect Ratio threshold
EYE_AR_CONSEC_FRAMES = 30  # Consecutive frames below threshold to trigger alarm
MOUTH_AR_THRESH = 0.5  # Mouth Aspect Ratio threshold
KNOWN_FACES_DIR = os.path.join('data', 'known_faces')
RECOGNITION_CONFIDENCE_THRESHOLD = 80
ALARM_BEEP_DURATION = 2

# Color codes (BGR format)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# ==================== FACIAL LANDMARKS ====================
# Eye landmarks from dlib (36-47 points)
LEFT_EYE_START, LEFT_EYE_END = 36, 42
RIGHT_EYE_START, RIGHT_EYE_END = 42, 48
MOUTH_START, MOUTH_END = 48, 68

# ==================== HELPER FUNCTIONS ====================
def calculate_eye_aspect_ratio(eye):
    """
    Calculate Eye Aspect Ratio (EAR) to determine if eyes are open.
    Formula: EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    """
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_mouth_aspect_ratio(mouth):
    """
    Calculate Mouth Aspect Ratio to detect yawning.
    """
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    D = dist.euclidean(mouth[0], mouth[6])
    mar = (A + B + C) / (3.0 * D)
    return mar

def count_people(frame, net, layer_names, output_layers, confidence_threshold=0.5, nms_threshold=0.4):
    """
    Detect and count people using YOLO.
    Returns: frame with detections, person count
    """
    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    try:
        detections = net.forward(output_layers)
    except:
        return frame, 0
    
    boxes = []
    confidences = []
    classIDs = []
    person_count = 0
    
    for output in detections:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if confidence > confidence_threshold and classID == 0:  # Class 0 is 'person'
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    
    if len(idxs) > 0:
        person_count = len(idxs)
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)
            cv2.putText(frame, "Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
    
    return frame, person_count

def play_alarm(duration=2):
    """
    Play alarm sound (cross-platform compatible).
    """
    try:
        import winsound
        winsound.Beep(1000, duration * 1000)
    except:
        # Fallback for non-Windows systems
        print("\a" * 5)  # System beep

# ==================== MAIN DETECTION CLASS ====================
class DrowsinessDetector:
    def __init__(self, use_camera=True, video_path=None):
        """
        Initialize the drowsiness detector.
        Args:
            use_camera: If True, use webcam; if False, use video_path
            video_path: Path to video file (if use_camera=False)
        """
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # YOLO setup for people counting
        self.load_yolo_model()
        self.load_face_recognizer()
        
        # Frame counters
        self.COUNTER = 0
        self.ALARM_ON = False
        self.drowsy_frames_log = deque(maxlen=100)
        self.driver_name = 'Unknown'
        self.face_recognizer_ready = False
        
        # Video source
        if use_camera:
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.VideoCapture(video_path)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Output video writer
        self.setup_output_writer()
        
        # Statistics
        self.total_frames = 0
        self.drowsy_frames = 0
        self.start_time = time.time()
        
    def load_yolo_model(self):
        """Load YOLO model for people detection."""
        # Check if YOLO files exist
        if not os.path.exists("yolov3.weights") or not os.path.exists("yolov3.cfg"):
            print("[INFO] YOLO model files not found. Using Haar Cascade for people detection.")
            print("[INFO] To use YOLO, download:")
            print("       - yolov3.weights: https://pjreddie.com/media/files/yolov3.weights")
            print("       - yolov3.cfg: https://github.com/pjreddie/darknet/raw/master/cfg/yolov3.cfg")
            self.yolo_available = False
            return
        
        try:
            self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            self.yolo_available = True
            print("[INFO] YOLO model loaded successfully")
        except Exception as e:
            print(f"[INFO] Error loading YOLO model: {e}")
            print("[INFO] Using Haar Cascade for people detection.")
            self.yolo_available = False
    
    def setup_output_writer(self):
        """Setup video writer for output."""
        os.makedirs(os.path.join('..', 'outputs'), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(
            '../outputs/drowsiness_detection_output.mp4',
            fourcc,
            30.0,
            (640, 480)
        )

    def load_face_recognizer(self):
        """Load or train the LBPH face recognizer using known faces."""
        if not os.path.isdir(KNOWN_FACES_DIR):
            os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
            print(f"[INFO] Created known face dataset directory: {KNOWN_FACES_DIR}")
            print("[INFO] Add subfolders for each person under data/known_faces with sample images.")
            self.face_recognizer_ready = False
            return

        face_samples = []
        labels = []
        label_ids = {}
        current_id = 0

        for person_dir in sorted(os.listdir(KNOWN_FACES_DIR)):
            person_path = os.path.join(KNOWN_FACES_DIR, person_dir)
            if not os.path.isdir(person_path):
                continue

            label_ids[person_dir] = current_id
            image_files = glob.glob(os.path.join(person_path, '*.*'))
            for image_path in image_files:
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
            print(f"[INFO] Trained face recognizer with {len(label_ids)} identities.")
        else:
            self.face_recognizer_ready = False
            if len(face_samples) < 2:
                print("[INFO] Not enough known face data to train the recognizer.")
                print("       Add at least two labeled face images under data/known_faces.")
            elif not recognizer_supported:
                print("[INFO] LBPH face recognizer is not available in the installed OpenCV build.")
            else:
                print("[INFO] Face recognizer training failed.")

    def recognize_face(self, face_gray):
        """Recognize a single face ROI and return name and confidence."""
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
    
    def draw_info_panel(self, frame, ear, mar, person_count, drowsy_status, recognized_name='Unknown'):
        """Draw information panel on frame."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (425, 170), BLACK, -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        y_offset = 35
        cv2.putText(frame, f"EAR: {ear:.2f}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (20, y_offset + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        cv2.putText(frame, f"People Count: {person_count}", (20, y_offset + 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        cv2.putText(frame, f"Recognized: {recognized_name}", (20, y_offset + 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)
        
        status_color = RED if drowsy_status else GREEN
        status_text = "DROWSY!!! ALERT!!!" if drowsy_status else "ALERT DRIVER"
        cv2.putText(frame, status_text, (20, y_offset + 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        return frame
    
    def run(self):
        """Main detection loop."""
        print("[INFO] Starting drowsiness detection...")
        print(f"[INFO] Press 'q' to quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.total_frames += 1
            frame = cv2.resize(frame, (640, 480))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            drowsy = False
            ear_avg = 0
            mar_avg = 0
            recognized_names = []
            
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = frame[y:y+h, x:x+w]
                    
                    # Face recognition on the detected face region
                    face_gray = cv2.resize(roi_gray, (200, 200)) if roi_gray.size else roi_gray
                    name, confidence = self.recognize_face(face_gray)
                    recognized_names.append(f"{name} ({confidence:.1f})" if name != 'Unknown' else 'Unknown')
                    cv2.putText(frame, f"{name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)

                    # Detect eyes
                    eyes = self.eye_cascade.detectMultiScale(roi_gray)
                    
                    if len(eyes) >= 2:
                        # Calculate eye coordinates
                        eyes = sorted(eyes, key=lambda e: e[0])
                        left_eye = eyes[0]
                        right_eye = eyes[1]
                        
                        # Draw eye rectangles
                        for (ex, ey, ew, eh) in eyes:
                            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), GREEN, 2)
                        
                        # Simulate EAR calculation (simplified)
                        # In production, use dlib for accurate facial landmarks
                        eye_area_ratio = (left_eye[2] * left_eye[3] + right_eye[2] * right_eye[3]) / (h * w)
                        ear = eye_area_ratio * 100
                        ear_avg = ear
                        
                        # Detect yawning (simplified)
                        mouth = self.eye_cascade.detectMultiScale(roi_gray)
                        mar = 0.3 if len(mouth) > 2 else 0.1
                        mar_avg = mar
                        
                        # Check drowsiness
                        if ear < EYE_AR_THRESH:
                            self.COUNTER += 1
                            if self.COUNTER >= EYE_AR_CONSEC_FRAMES:
                                drowsy = True
                                self.drowsy_frames += 1
                                self.drowsy_frames_log.append(datetime.now())
                        else:
                            self.COUNTER = 0
                        
                        # Draw face rectangle
                        cv2.rectangle(frame, (x, y), (x+w, y+h), 
                                    RED if drowsy else GREEN, 2)
                    else:
                        # No eyes detected
                        cv2.rectangle(frame, (x, y), (x+w, y+h), YELLOW, 2)
            
            recognized_name = 'Unknown'
            if any(name != 'Unknown' for name in recognized_names):
                recognized_name = next((name for name in recognized_names if name != 'Unknown'), recognized_name)
            elif recognized_names:
                recognized_name = 'Unknown'
            
            # Count people
            if self.yolo_available:
                try:
                    frame, person_count = count_people(frame, self.net, [], 
                                                       self.output_layers, 
                                                       confidence_threshold=0.5)
                except:
                    person_count = 0
            else:
                person_count = len(faces)  # Fallback
            
            # Draw info panel
            frame = self.draw_info_panel(frame, ear_avg, mar_avg, person_count, drowsy, recognized_name)
            
            # Trigger alarm if drowsy
            if drowsy and not self.ALARM_ON:
                self.ALARM_ON = True
                print("[ALERT] DRIVER IS DROWSY! PLAYING ALARM...")
                alarm_thread = threading.Thread(target=play_alarm, args=(2,))
                alarm_thread.daemon = True
                alarm_thread.start()
            
            if not drowsy:
                self.ALARM_ON = False
            
            # Write frame
            self.out.write(frame)
            
            # Display frame
            cv2.imshow("Driver Drowsiness Detector", frame)
            
            # Print stats periodically
            if self.total_frames % 100 == 0:
                elapsed = time.time() - self.start_time
                print(f"[STATS] Frame: {self.total_frames} | "
                      f"Drowsy Frames: {self.drowsy_frames} | "
                      f"Elapsed: {elapsed:.1f}s | "
                      f"Avg Person Count: {person_count}")
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("[INFO] Cleaning up resources...")
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        elapsed = time.time() - self.start_time
        drowsy_percentage = (self.drowsy_frames / max(self.total_frames, 1)) * 100
        print("\n" + "="*50)
        print("FINAL STATISTICS")
        print("="*50)
        print(f"Total Frames Processed: {self.total_frames}")
        print(f"Drowsy Frames Detected: {self.drowsy_frames}")
        print(f"Drowsiness Percentage: {drowsy_percentage:.2f}%")
        print(f"Total Duration: {elapsed:.2f} seconds")
        print(f"Output Video: outputs/drowsiness_detection_output.mp4")
        print("="*50)


# ==================== IMPROVED VERSION WITH DNN FACE DETECTION ====================
class ImprovedDrowsinessDetector(DrowsinessDetector):
    """
    Improved detector using DNN for better face detection.
    """
    def __init__(self, use_camera=True, video_path=None):
        super().__init__(use_camera, video_path)
        self.load_dnn_face_detector()
    
    def load_dnn_face_detector(self):
        """Load DNN-based face detector (more accurate)."""
        try:
            prototxt_path = "deploy.prototxt.txt"
            model_path = "res10_300x300_ssd_iter_140000.caffemodel"
            
            if os.path.exists(prototxt_path) and os.path.exists(model_path):
                self.dnn_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
                print("[INFO] DNN face detector loaded successfully")
                self.use_dnn = True
            else:
                print("[INFO] DNN model files not found, using Haar Cascade")
                self.use_dnn = False
        except:
            print("[INFO] Error loading DNN model, using Haar Cascade")
            self.use_dnn = False
    
    def detect_faces_dnn(self, frame):
        """Detect faces using DNN."""
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), [104.0, 177.0, 123.0],
                                     False, False)
        self.dnn_net.setInput(blob)
        detections = self.dnn_net.forward()
        faces = []
        
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                faces.append((startX, startY, endX - startX, endY - startY))
        
        return faces


# ==================== MAIN ENTRY POINT ====================
if __name__ == "__main__":
    print("╔════════════════════════════════════════════════════════╗")
    print("║   DRIVER DROWSINESS DETECTOR & PEOPLE COUNTER         ║")
    print("║   AI-Based Real-Time Monitoring System                 ║")
    print("╚════════════════════════════════════════════════════════╝\n")
    
    print("Please ensure your webcam is connected.")
    print("Starting detection...\n")
    
    # Create detector and run
    detector = DrowsinessDetector(use_camera=True)
    detector.run()
