"""
TEST & DIAGNOSTICS UTILITY
Use this script to test camera, verify dependencies, and troubleshoot issues
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
END = '\033[0m'

# ==================== TEST FUNCTIONS ====================

def test_python_version():
    """Test Python version."""
    print(f"\n{BLUE}[TEST 1] Python Version{END}")
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print(f"{GREEN}✓ PASS: Python 3.8+{END}")
        return True
    else:
        print(f"{RED}✗ FAIL: Python 3.8+ required{END}")
        return False


def test_opencv():
    """Test OpenCV installation."""
    print(f"\n{BLUE}[TEST 2] OpenCV Installation{END}")
    try:
        print(f"OpenCV Version: {cv2.__version__}")
        print(f"{GREEN}✓ PASS: OpenCV is installed{END}")
        return True
    except ImportError:
        print(f"{RED}✗ FAIL: OpenCV not installed{END}")
        print("Install with: pip install opencv-python")
        return False


def test_dependencies():
    """Test all required dependencies."""
    print(f"\n{BLUE}[TEST 3] Dependencies{END}")
    dependencies = {
        'numpy': 'numpy',
        'scipy': 'scipy',
        'PIL': 'PIL',
    }
    
    all_ok = True
    for package, import_name in dependencies.items():
        try:
            __import__(import_name)
            print(f"{GREEN}✓{END} {package}")
        except ImportError:
            print(f"{RED}✗{END} {package} - NOT INSTALLED")
            all_ok = False
    
    if all_ok:
        print(f"{GREEN}✓ PASS: All dependencies installed{END}")
    else:
        print(f"{RED}✗ FAIL: Some dependencies missing{END}")
        print("Install with: pip install -r requirements.txt")
    
    return all_ok


def test_camera():
    """Test camera connectivity."""
    print(f"\n{BLUE}[TEST 4] Camera Access{END}")
    
    # Try different backends
    backends = [cv2.CAP_WINRT, cv2.CAP_DSHOW, cv2.CAP_MSMF]
    backend_names = ['WINRT', 'DSHOW', 'MSMF']
    
    cap = None
    working_backend = None
    
    for i, backend in enumerate(backends):
        try:
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                working_backend = backend_names[i]
                print(f"✓ PASS: Camera accessible (using {working_backend})")
                break
        except:
            continue
    
    if not cap or not cap.isOpened():
        print(f"{RED}✗ FAIL: Cannot access camera{END}")
        print("Troubleshooting:")
        print("  - Check if camera is connected")
        print("  - Check if camera is in use by another application")
        print("  - Try restarting camera application")
        return False
    
    # Get camera properties
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera Resolution: {int(width)}x{int(height)}")
    print(f"Camera FPS: {fps:.1f}")
    
    cap.release()
    return True


def test_camera_feed(duration=5):
    """Test camera feed quality."""
    print(f"\n{BLUE}[TEST 5] Camera Feed Quality ({duration}s){END}")
    
    # Try different backends
    backends = [cv2.CAP_WINRT, cv2.CAP_DSHOW, cv2.CAP_MSMF]
    backend_names = ['WINRT', 'DSHOW', 'MSMF']
    
    cap = None
    for i, backend in enumerate(backends):
        try:
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                print(f"Using {backend_names[i]} backend")
                break
        except:
            continue
    
    if not cap or not cap.isOpened():
        print(f"{RED}✗ FAIL: Cannot open camera{END}")
        return False
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Reading camera feed...")
    
    frame_count = 0
    try:
        while frame_count < duration * 30:  # ~30 FPS
            ret, frame = cap.read()
            if not ret:
                print(f"{RED}✗ FAIL: Cannot read frame{END}")
                return False
            frame_count += 1
        
        print(f"{GREEN}✓ PASS: Captured {frame_count} frames successfully{END}")
        print(f"Average brightness: {np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)):.1f}")
        
        if np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) < 50:
            print(f"{YELLOW}⚠ WARNING: Low lighting detected{END}")
            print("Recommendation: Improve lighting for better detection")
        
        cap.release()
        return True
    except Exception as e:
        print(f"{RED}✗ FAIL: {e}{END}")
        cap.release()
        return False


def test_face_detection():
    """Test face detection on camera feed."""
    print(f"\n{BLUE}[TEST 6] Face Detection{END}")
    
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Test on camera
    # Try different backends
    backends = [cv2.CAP_WINRT, cv2.CAP_DSHOW, cv2.CAP_MSMF]
    backend_names = ['WINRT', 'DSHOW', 'MSMF']
    
    cap = None
    for i, backend in enumerate(backends):
        try:
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                print(f"Using {backend_names[i]} backend for face detection")
                break
        except:
            continue
    
    if not cap or not cap.isOpened():
        print(f"{RED}✗ FAIL: Cannot open camera{END}")
        return False
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Testing face detection for 10 seconds...")
    print("(Position your face in front of camera)")
    
    detected_count = 0
    frame_count = 0
    max_frames = 300  # ~10 seconds at 30 FPS
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            detected_count += 1
        
        # Display frame with detections
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.imshow("Face Detection Test", frame)
        
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    cap.release()
    
    detection_rate = (detected_count / max_frames) * 100
    
    if detection_rate > 50:
        print(f"{GREEN}✓ PASS: Face detection working ({detection_rate:.1f}% detection rate){END}")
        return True
    elif detection_rate > 20:
        print(f"{YELLOW}⚠ WARNING: Low face detection rate ({detection_rate:.1f}%){END}")
        print("Tips:")
        print("  - Improve lighting")
        print("  - Position face properly")
        print("  - Move closer to camera")
        return True
    else:
        print(f"{RED}✗ FAIL: Face not detected ({detection_rate:.1f}%){END}")
        return False


def test_cascades():
    """Test cascade classifier availability."""
    print(f"\n{BLUE}[TEST 7] Cascade Classifiers{END}")
    
    cascade_path = cv2.data.haarcascades
    cascades = {
        'Face': 'haarcascade_frontalface_default.xml',
        'Eye': 'haarcascade_eye.xml',
        'Smile': 'haarcascade_smile.xml',
        'Upper Body': 'haarcascade_upperbody.xml',
    }
    
    all_ok = True
    for name, filename in cascades.items():
        try:
            cascade = cv2.CascadeClassifier(cascade_path + filename)
            if cascade.empty():
                print(f"{RED}✗{END} {name} - Failed to load")
                all_ok = False
            else:
                print(f"{GREEN}✓{END} {name}")
        except Exception as e:
            print(f"{RED}✗{END} {name} - {e}")
            all_ok = False
    
    if all_ok:
        print(f"{GREEN}✓ PASS: All cascades available{END}")
    else:
        print(f"{RED}✗ FAIL: Some cascades missing{END}")
    
    return all_ok


def test_output_directory():
    """Test output directory creation."""
    print(f"\n{BLUE}[TEST 8] Output Directory{END}")
    
    try:
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Test write permission
        test_file = output_dir / "test.txt"
        test_file.write_text("test")
        test_file.unlink()
        
        print(f"{GREEN}✓ PASS: Output directory writable{END}")
        print(f"Location: {output_dir.absolute()}")
        return True
    except Exception as e:
        print(f"{RED}✗ FAIL: Cannot create/write to output directory{END}")
        print(f"Error: {e}")
        return False


def test_config_file():
    """Test if config file can be imported."""
    print(f"\n{BLUE}[TEST 9] Configuration File{END}")
    
    try:
        import sys
        import os
        # Add src directory to path
        src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
        sys.path.insert(0, src_path)
        import config
        print(f"{GREEN}✓ PASS: Config file loaded{END}")
        
        # Show current config
        required_attrs = ['EYE_AR_THRESH', 'EYE_AR_CONSEC_FRAMES', 'CAMERA_INDEX']
        missing = []
        
        for attr in required_attrs:
            if not hasattr(config, attr):
                missing.append(attr)
        
        if missing:
            print(f"{YELLOW}⚠ WARNING: Missing config attributes: {missing}{END}")
        else:
            print(f"{GREEN}✓ All required config attributes present{END}")
        
        return len(missing) == 0
    except ImportError:
        print(f"{RED}✗ FAIL: Config file not found{END}")
        return False


def run_full_test():
    """Run all tests."""
    print("\n" + "="*60)
    print("DROWSINESS DETECTOR - DIAGNOSTIC TEST SUITE")
    print("="*60)
    
    tests = [
        ("Python Version", test_python_version),
        ("OpenCV Installation", test_opencv),
        ("Dependencies", test_dependencies),
        ("Camera Access", test_camera),
        ("Camera Feed", test_camera_feed),
        ("Face Detection", test_face_detection),
        ("Cascade Classifiers", test_cascades),
        ("Output Directory", test_output_directory),
        ("Configuration File", test_config_file),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"{RED}✗ ERROR in {name}: {e}{END}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = f"{GREEN}✓ PASS{END}" if result else f"{RED}✗ FAIL{END}"
        print(f"{status} - {name}")
    
    print("-"*60)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print(f"\n{GREEN}✓ All tests passed! System is ready to use.{END}")
        print(f"Run: python simple_detector.py\n")
        return True
    else:
        print(f"\n{RED}✗ Some tests failed. Please fix issues and try again.{END}\n")
        return False


def quick_camera_test():
    """Quick camera test without other dependencies."""
    print(f"\n{BLUE}Quick Camera Test{END}")
    
    # Try different backends
    backends = [cv2.CAP_WINRT, cv2.CAP_DSHOW, cv2.CAP_MSMF]
    backend_names = ['WINRT', 'DSHOW', 'MSMF']
    
    cap = None
    for i, backend in enumerate(backends):
        try:
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                print(f"Using {backend_names[i]} backend")
                break
        except:
            continue
    
    if not cap:
        print(f"{RED}✗ Camera not available{END}")
        return
    
    print("Camera open. Press Q to quit.")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        cv2.putText(frame, "Camera Test - Press Q to exit",
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Camera Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    cap.release()


# ==================== MAIN ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Drowsiness Detector Diagnostic Tool")
    parser.add_argument("--quick", action="store_true", help="Quick camera test")
    parser.add_argument("--face", action="store_true", help="Test face detection only")
    parser.add_argument("--all", action="store_true", help="Run all tests (default)")
    
    args = parser.parse_args()
    
    if args.quick:
        quick_camera_test()
    elif args.face:
        test_face_detection()
    else:
        success = run_full_test()
        sys.exit(0 if success else 1)
