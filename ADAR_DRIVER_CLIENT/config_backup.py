"""
============================================================
 ADAR V3.0 — Configuration & Thresholds
 Central control for sensitivity, paths, and settings.
============================================================
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─── HARDWARE SETTINGS ───
CAMERA_INDEX = 0          # 0 for webcam, 1 for external
CAMERA_WIDTH = 640        # Keep 640x480 for 60 FPS
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_RECONNECT_DELAY = 2.0

# ─── AI MODEL PATHS ───
# Ensure these files exist in the root folder!
FACE_LANDMARKER_MODEL_PATH = "face_landmarker.task"
YOLO_MODEL_PATH = "yolov8n.pt"

# ─── SAFETY THRESHOLDS (The "Sensitivity" Dial) ───
EAR_THRESHOLD = 0.22      # Eyes Closed < 0.22
EAR_CONSEC_FRAMES = 15    # ~0.5 seconds of eyes closed triggers drowsy

MAR_THRESHOLD = 0.65      # Mouth Open > 0.65
MAR_CONSEC_FRAMES = 20    # ~0.7 seconds triggers yawn

HEAD_YAW_THRESHOLD = 25.0   # Degrees left/right
HEAD_PITCH_THRESHOLD = 20.0 # Degrees up/down

# ─── YOLO SETTINGS ───
YOLO_CONFIDENCE = 0.5
# COCO Class IDs: 67=cell phone, 39=bottle, 77=teddy bear(test)
# You can map standard COCO classes here
YOLO_CLASSES_OF_INTEREST = {
    67: "cell phone",
    39: "bottle",
}

# ─── APP SETTINGS ───
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = False
DATABASE_URI = "sqlite:///adar_logs.db"
SOCKETIO_EMIT_INTERVAL = 0.1  # Update dashboard every 100ms

# ─── JARVIS SETTINGS ───
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-your-openai-api-key-here")
OPENAI_MODEL = "gpt-4o"
OPENAI_TTS_MODEL = "tts-1"
OPENAI_TTS_VOICE = "onyx"  # Options: alloy, echo, fable, onyx, nova, shimmer
JARVIS_COOLDOWN = 10.0     # Seconds between voice alerts
DANGER_FRAME_THRESHOLD = 10 # Frames of DANGER before Jarvis speaks

# ─── CONSTANTS ───
STATUS_SAFE = "SAFE"
STATUS_WARNING = "WARNING"
STATUS_DANGER = "DANGER"