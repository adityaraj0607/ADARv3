"""
============================================================
 PROJECT ADAR — IRON MAN / JARVIS CINEMATIC HUD
 Complete Standalone Demo with Zero-Lag Architecture
 
 Features:
 - MODULE A: Multi-threaded camera stream (zero lag)
 - MODULE B: Real-time EAR drowsiness detection
 - MODULE C: Async GPT-4o worker (optional, respects API key setting)
 - MODULE D: Cinematic Jarvis HUD with all visual effects
 
 Hardware: NVIDIA RTX 2050, Ryzen 5 5600H
 Target: 30+ FPS professional-grade HUD
============================================================
"""

import cv2
import numpy as np
import time
import math
import threading
import queue
from collections import deque
from typing import Optional, Tuple
import mediapipe as mp

# ═════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════

# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# EAR thresholds for drowsiness detection
EAR_THRESHOLD = 0.25  # Below this = drowsy
EAR_DANGER_DURATION = 2.0  # Seconds of low EAR to trigger danger

# HUD colors (BGR format for OpenCV)
COLOR_SAFE_CYAN = (255, 255, 0)      # Cyan for safe mode
COLOR_DANGER_RED = (50, 50, 255)      # Red for danger mode
COLOR_HUD_DIM = (100, 100, 100)       # Dim gray
COLOR_HUD_BRIGHT = (255, 255, 255)    # Bright white
COLOR_ORANGE = (0, 165, 255)          # Orange accent
COLOR_GREEN = (0, 255, 0)             # Status OK

# API key control (set to False to disable all API calls)
ENABLE_GPT4O = False  # User requested API key be disabled

# GPT-4o settings (only used if ENABLE_GPT4O = True)
GPT4O_SCAN_INTERVAL = 10.0  # Seconds between scans
OPENAI_API_KEY = ""  # Will be loaded from config if enabled


# ═════════════════════════════════════════════════════════════
# MODULE A: ZERO-LAG CAMERA STREAM (Threading)
# ═════════════════════════════════════════════════════════════

class CameraStream:
    """
    Multi-threaded camera capture for zero-lag video processing.
    Continuously reads frames in background thread and provides
    the latest frame on-demand without blocking.
    """
    
    def __init__(self, src=CAMERA_INDEX, width=CAMERA_WIDTH, height=CAMERA_HEIGHT):
        self.src = src
        self.width = width
        self.height = height
        
        # Thread-safe frame storage
        self.frame = None
        self.frame_lock = threading.Lock()
        
        # Thread control
        self.running = False
        self.thread = None
        
        # Performance metrics
        self.fps = 0.0
        self.frame_count = 0
        self.fps_start_time = time.time()
        
    def start(self):
        """Initialize camera and start capture thread."""
        # Open camera with DirectShow for best performance on Windows
        try:
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
        except:
            self.cap = cv2.VideoCapture(self.src)
            
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.src}")
        
        # Configure camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for low latency
        
        # Try to enable CUDA if available
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except:
            pass
        
        # Start capture thread
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        
        print(f"[CAMERA] Started capture thread (source {self.src})")
        return self
    
    def _update_loop(self):
        """Background thread: continuously grab frames."""
        while self.running:
            if not self.cap.isOpened():
                time.sleep(0.1)
                continue
            
            # Grab frame (discard buffer)
            self.cap.grab()
            ret, frame = self.cap.read()
            
            if ret and frame is not None:
                # Store latest frame atomically
                with self.frame_lock:
                    self.frame = frame
                    self.frame_count += 1
                
                # Calculate FPS
                elapsed = time.time() - self.fps_start_time
                if elapsed >= 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.fps_start_time = time.time()
    
    def read(self) -> Optional[np.ndarray]:
        """Get the latest frame (non-blocking)."""
        with self.frame_lock:
            return self.frame.copy() if self.frame is not None else None
    
    def stop(self):
        """Stop capture thread and release camera."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if hasattr(self, 'cap'):
            self.cap.release()
        print("[CAMERA] Stopped")


# ═════════════════════════════════════════════════════════════
# MODULE B: SAFETY HOOK - DROWSINESS DETECTION (EAR)
# ═════════════════════════════════════════════════════════════

class DrowsinessDetector:
    """
    Real-time Eye Aspect Ratio (EAR) calculation using MediaPipe Face Mesh.
    Triggers danger mode when EAR falls below threshold for sustained period.
    """
    
    # MediaPipe eye landmark indices
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # State
        self.ear = 1.0
        self.danger_detected = False
        self.drowsy_start_time = 0.0
        self.face_detected = False
        self.landmarks = None
        
        print("[DROWSINESS] MediaPipe Face Mesh initialized")
    
    def calculate_ear(self, eye_points: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio from 6 eye landmarks.
        
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        
        Higher EAR = more open eye
        Lower EAR = more closed eye
        """
        # Vertical distances
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Horizontal distance
        h = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # EAR formula
        ear = (v1 + v2) / (2.0 * h + 1e-6)
        return ear
    
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process frame and detect drowsiness.
        
        Returns:
            dict: {
                'ear': float,
                'danger_detected': bool,
                'face_detected': bool,
                'landmarks': array or None
            }
        """
        h, w = frame.shape[:2]
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        if results.multi_face_landmarks:
            self.face_detected = True
            landmarks = results.multi_face_landmarks[0]
            
            # Extract landmark coordinates
            points = np.array([
                [lm.x * w, lm.y * h, lm.z * w] 
                for lm in landmarks.landmark
            ])
            self.landmarks = points
            
            # Get eye points
            left_eye = points[self.LEFT_EYE]
            right_eye = points[self.RIGHT_EYE]
            
            # Calculate EAR for both eyes
            left_ear = self.calculate_ear(left_eye[:, :2])
            right_ear = self.calculate_ear(right_eye[:, :2])
            self.ear = (left_ear + right_ear) / 2.0
            
            # Check for danger condition
            if self.ear < EAR_THRESHOLD:
                if self.drowsy_start_time == 0.0:
                    self.drowsy_start_time = time.time()
                elif time.time() - self.drowsy_start_time >= EAR_DANGER_DURATION:
                    self.danger_detected = True
            else:
                self.drowsy_start_time = 0.0
                self.danger_detected = False
        else:
            self.face_detected = False
            self.landmarks = None
            self.drowsy_start_time = 0.0
        
        return {
            'ear': self.ear,
            'danger_detected': self.danger_detected,
            'face_detected': self.face_detected,
            'landmarks': self.landmarks
        }


# ═════════════════════════════════════════════════════════════
# MODULE C: OPENAI GPT-4o INTELLIGENCE BRIDGE (Async)
# ═════════════════════════════════════════════════════════════

class GPT4oWorker:
    """
    Background thread that periodically sends frames to GPT-4o Vision API.
    Completely optional - respects API key disable setting.
    """
    
    def __init__(self):
        self.enabled = ENABLE_GPT4O
        self.response = "J.A.R.V.I.S. OFFLINE - API key disabled"
        self.response_lock = threading.Lock()
        self.last_scan_time = 0.0
        self.running = False
        self.thread = None
        
        if self.enabled:
            try:
                import openai
                self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
                print("[GPT-4o] Worker initialized")
            except Exception as e:
                print(f"[GPT-4o] Failed to initialize: {e}")
                self.enabled = False
        else:
            print("[GPT-4o] Worker DISABLED (API key protection active)")
    
    def start(self):
        """Start background worker thread."""
        if not self.enabled:
            return self
        
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()
        return self
    
    def _worker_loop(self):
        """Background thread: periodically analyze frames."""
        while self.running:
            time.sleep(0.5)  # Check every 500ms
    
    def analyze_frame(self, frame: np.ndarray, danger_detected: bool):
        """
        Send frame to GPT-4o for analysis (non-blocking).
        Only triggers if enough time has passed since last scan.
        """
        if not self.enabled:
            return
        
        now = time.time()
        if now - self.last_scan_time < GPT4O_SCAN_INTERVAL:
            return
        
        self.last_scan_time = now
        
        # Spawn thread to avoid blocking
        thread = threading.Thread(
            target=self._analyze_async,
            args=(frame.copy(), danger_detected),
            daemon=True
        )
        thread.start()
    
    def _analyze_async(self, frame: np.ndarray, danger_detected: bool):
        """Actual API call (runs in background thread)."""
        try:
            import base64
            
            # Encode frame
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Build prompt
            prompt = (
                "You are JARVIS, an AI safety system. Analyze this driver's face. "
                "Is there a safety risk? Reply in 1 short sentence."
            )
            
            # Call API
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{frame_b64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=100
            )
            
            text = response.choices[0].message.content.strip()
            
            with self.response_lock:
                self.response = text
                
        except Exception as e:
            with self.response_lock:
                self.response = f"Analysis failed: {str(e)}"
    
    def get_response(self) -> str:
        """Get latest GPT-4o response (thread-safe)."""
        with self.response_lock:
            return self.response
    
    def stop(self):
        """Stop worker thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)


# ═════════════════════════════════════════════════════════════
# MODULE D: JARVIS UI OVERLAY (The Masterpiece)
# ═════════════════════════════════════════════════════════════

class JarvisHUD:
    """
    Cinematic Iron Man / Jarvis Heads-Up Display.
    
    Features:
    - Central rotating reticle
    - Face targeting with corner brackets
    - Vertical scanning line
    - System log window
    - Color-coded danger mode (cyan → red)
    """
    
    def __init__(self, width=CAMERA_WIDTH, height=CAMERA_HEIGHT):
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height // 2
        
        # Animation state
        self.reticle_angle = 0.0
        self.scanline_y = 0
        self.scanline_direction = 1
        
        # System logs
        self.system_logs = deque(maxlen=5)
        self.add_log("SYSTEM ONLINE")
        self.add_log("JARVIS DEFENSE ACTIVE")
        self.add_log("MONITORING DRIVER")
        
        # Start time
        self.start_time = time.time()
        
    def add_log(self, message: str):
        """Add message to system log."""
        timestamp = time.strftime("%H:%M:%S")
        self.system_logs.append(f"[{timestamp}] {message}")
    
    def draw_reticle(self, frame: np.ndarray, color: tuple):
        """Draw rotating triple-ring targeting reticle at center."""
        cx, cy = self.center_x, self.center_y
        
        # Rotate reticle
        self.reticle_angle += 1.5
        if self.reticle_angle >= 360:
            self.reticle_angle = 0
        
        angle_rad = math.radians(self.reticle_angle)
        
        # Three concentric rings
        for radius in [30, 50, 70]:
            # Main circle
            cv2.circle(frame, (cx, cy), radius, color, 1, cv2.LINE_AA)
            
            # Rotating tick marks (4 marks at 90° intervals)
            for i in range(4):
                tick_angle = angle_rad + (i * math.pi / 2)
                x1 = int(cx + radius * math.cos(tick_angle))
                y1 = int(cy + radius * math.sin(tick_angle))
                x2 = int(cx + (radius + 10) * math.cos(tick_angle))
                y2 = int(cy + (radius + 10) * math.sin(tick_angle))
                cv2.line(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        
        # Center crosshair
        cross_len = 15
        cv2.line(frame, (cx - cross_len, cy), (cx + cross_len, cy), 
                 color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - cross_len), (cx, cy + cross_len), 
                 color, 1, cv2.LINE_AA)
    
    def draw_face_brackets(self, frame: np.ndarray, landmarks: np.ndarray, color: tuple):
        """Draw corner brackets around detected face."""
        if landmarks is None or len(landmarks) == 0:
            return
        
        # Get face bounding box
        xs = landmarks[:, 0]
        ys = landmarks[:, 1]
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())
        
        # Add padding
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(self.width, x2 + padding)
        y2 = min(self.height, y2 + padding)
        
        # Corner bracket length
        bracket_len = 30
        thickness = 2
        
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + bracket_len, y1), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + bracket_len), color, thickness, cv2.LINE_AA)
        
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - bracket_len, y1), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y1 + bracket_len), color, thickness, cv2.LINE_AA)
        
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + bracket_len, y2), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - bracket_len), color, thickness, cv2.LINE_AA)
        
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - bracket_len, y2), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - bracket_len), color, thickness, cv2.LINE_AA)
        
        # Add glow effect
        glow_color = tuple(int(c * 0.3) for c in color)
        cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), glow_color, 1, cv2.LINE_AA)
    
    def draw_scanline(self, frame: np.ndarray, color: tuple):
        """Draw animated vertical scanning line."""
        # Move scanline
        self.scanline_y += self.scanline_direction * 3
        
        if self.scanline_y >= self.height:
            self.scanline_y = self.height
            self.scanline_direction = -1
        elif self.scanline_y <= 0:
            self.scanline_y = 0
            self.scanline_direction = 1
        
        # Draw line with gradient effect
        y = self.scanline_y
        for offset in range(-5, 6):
            alpha = 1.0 - abs(offset) / 5.0
            line_color = tuple(int(c * alpha) for c in color)
            cv2.line(frame, (0, y + offset), (self.width, y + offset), 
                    line_color, 1, cv2.LINE_AA)
    
    def draw_system_log(self, frame: np.ndarray, color: tuple, 
                       ear: float, fps: float, latency_ms: float,
                       jarvis_response: str):
        """Draw system log window in bottom-left."""
        # Background panel
        panel_x1, panel_y1 = 10, self.height - 180
        panel_x2, panel_y2 = 400, self.height - 10
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), 
                     (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Border
        cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), 
                     color, 2, cv2.LINE_AA)
        
        # Title
        cv2.putText(frame, "SYSTEM STATUS", (panel_x1 + 10, panel_y1 + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
        # Metrics
        y_offset = panel_y1 + 50
        line_height = 20
        
        metrics = [
            f"> FPS: {fps:.1f}",
            f"> LATENCY: {latency_ms:.1f} ms",
            f"> EAR: {ear:.3f}",
            f"> J.A.R.V.I.S.: {jarvis_response[:40]}..."
        ]
        
        for metric in metrics:
            cv2.putText(frame, metric, (panel_x1 + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_HUD_BRIGHT, 
                       1, cv2.LINE_AA)
            y_offset += line_height
    
    def draw_danger_alert(self, frame: np.ndarray):
        """Flash critical danger alert in center of screen."""
        # Pulsing effect
        pulse = int(abs(math.sin(time.time() * 5)) * 255)
        alert_color = (50, 50, pulse)
        
        # Alert box
        box_w, box_h = 500, 80
        x1 = self.center_x - box_w // 2
        y1 = self.center_y - box_h // 2
        x2 = x1 + box_w
        y2 = y1 + box_h
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Border
        cv2.rectangle(frame, (x1, y1), (x2, y2), alert_color, 3, cv2.LINE_AA)
        
        # Text
        text = "CRITICAL ALERT"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_BOLD, 1.2, 2)[0]
        text_x = self.center_x - text_size[0] // 2
        text_y = self.center_y - 10
        
        cv2.putText(frame, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_BOLD, 1.2, alert_color, 2, cv2.LINE_AA)
        
        # Subtitle
        subtitle = "DRIVER UNRESPONSIVE"
        sub_size = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        sub_x = self.center_x - sub_size[0] // 2
        sub_y = self.center_y + 20
        
        cv2.putText(frame, subtitle, (sub_x, sub_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    def draw_top_bar(self, frame: np.ndarray, color: tuple):
        """Draw top status bar."""
        # Background
        cv2.rectangle(frame, (0, 0), (self.width, 40), (20, 20, 20), -1)
        cv2.line(frame, (0, 40), (self.width, 40), color, 1)
        
        # Title
        title = "PROJECT ADAR - ADVANCED DRIVER MONITORING"
        cv2.putText(frame, title, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        
        # Timestamp
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (self.width - 100, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    def render(self, frame: np.ndarray, detector_result: dict, 
              fps: float, latency_ms: float, jarvis_response: str) -> np.ndarray:
        """
        Render complete HUD overlay on frame.
        
        Args:
            frame: Input camera frame
            detector_result: Results from DrowsinessDetector
            fps: Current FPS
            latency_ms: Processing latency
            jarvis_response: Latest GPT-4o response
            
        Returns:
            Frame with HUD overlay
        """
        # Determine HUD color based on danger status
        if detector_result['danger_detected']:
            primary_color = COLOR_DANGER_RED
        else:
            primary_color = COLOR_SAFE_CYAN
        
        # Layer 1: Top bar
        self.draw_top_bar(frame, primary_color)
        
        # Layer 2: Scanline
        self.draw_scanline(frame, primary_color)
        
        # Layer 3: Central reticle
        self.draw_reticle(frame, primary_color)
        
        # Layer 4: Face targeting
        if detector_result['face_detected'] and detector_result['landmarks'] is not None:
            self.draw_face_brackets(frame, detector_result['landmarks'], primary_color)
        
        # Layer 5: System log
        self.draw_system_log(
            frame, 
            primary_color,
            detector_result['ear'],
            fps,
            latency_ms,
            jarvis_response
        )
        
        # Layer 6: Danger alert (if triggered)
        if detector_result['danger_detected']:
            self.draw_danger_alert(frame)
        
        return frame


# ═════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═════════════════════════════════════════════════════════════

def main():
    """Main application loop."""
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║   PROJECT ADAR — IRON MAN JARVIS HUD DEMO           ║
    ║   Advanced Driver Attention & Response System         ║
    ║                                                       ║
    ║   Press 'Q' or ESC to exit                           ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    # Initialize modules
    print("\n[INIT] Starting modules...")
    
    try:
        # MODULE A: Camera stream
        camera = CameraStream().start()
        
        # MODULE B: Drowsiness detector
        detector = DrowsinessDetector()
        
        # MODULE C: GPT-4o worker (optional)
        gpt_worker = GPT4oWorker().start()
        
        # MODULE D: HUD renderer
        hud = JarvisHUD()
        
        print("[INIT] All modules ready\n")
        print("[STATUS] System operational - monitoring driver...")
        print(f"[STATUS] API Integration: {'ENABLED' if ENABLE_GPT4O else 'DISABLED (API key protected)'}")
        print()
        
        # Performance tracking
        frame_times = deque(maxlen=30)
        last_time = time.time()
        
        # Main render loop
        while True:
            loop_start = time.time()
            
            # Get latest frame (non-blocking)
            frame = camera.read()
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Process frame for drowsiness
            detector_result = detector.process_frame(frame)
            
            # Trigger GPT-4o analysis if danger detected
            if detector_result['danger_detected']:
                gpt_worker.analyze_frame(frame, True)
                hud.add_log("DANGER: Driver drowsy!")
            
            # Get latest Jarvis response
            jarvis_response = gpt_worker.get_response()
            
            # Calculate performance metrics
            processing_time = time.time() - loop_start
            latency_ms = processing_time * 1000
            
            frame_times.append(processing_time)
            current_fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0
            
            # Render HUD
            display_frame = hud.render(
                frame,
                detector_result,
                current_fps,
                latency_ms,
                jarvis_response
            )
            
            # Display
            cv2.imshow("ADAR - JARVIS HUD", display_frame)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                break
            
            # Adaptive sleep to maintain target FPS
            elapsed = time.time() - loop_start
            target_frame_time = 1.0 / CAMERA_FPS
            if elapsed < target_frame_time:
                time.sleep(target_frame_time - elapsed)
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\n[SHUTDOWN] Stopping modules...")
        camera.stop()
        gpt_worker.stop()
        cv2.destroyAllWindows()
        print("[SHUTDOWN] Complete")


if __name__ == "__main__":
    main()
