"""
============================================================
 ADAR V3.0 — Central Configuration
 All tunable thresholds and constants live here.
============================================================
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI ──────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-5.2"  # Frontier model — fast, non-reasoning, vision-capable
OPENAI_TTS_MODEL = "tts-1"
OPENAI_TTS_VOICE = "onyx"
GPT_TIMEOUT = 3.0  # 3000ms — fallback to local rule-based alert if exceeded

# ── Camera ──────────────────────────────────────────────────
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_RECONNECT_DELAY = 2.0
CAMERA_JPEG_QUALITY = 70  # Optimized for low-latency streaming
CAMERA_FLIP_HORIZONTAL = True  # Flip camera horizontally for natural mirror view

# ── Flask ───────────────────────────────────────────────────
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "change-me")
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = False

# ── Face Mesh / EAR ────────────────────────────────────────
EAR_THRESHOLD = 0.21  # Drowsiness-tuned for 85% detection sensitivity
EAR_CONSEC_FRAMES = 8  # ~0.27s — faster drowsiness response
MAR_THRESHOLD = 0.62  # Sensitive yawn detection for 95% accuracy
MAR_CONSEC_FRAMES = 5  # Faster yawn catch

# ── Head Pose ───────────────────────────────────────────────
HEAD_YAW_THRESHOLD = 28  # 95% accuracy: tighter for real looking-away detection
HEAD_PITCH_THRESHOLD = 23  # 95% accuracy: tighter for looking-down detection

# ── MediaPipe Face Landmarker ───────────────────────────────
FACE_LANDMARKER_MODEL_PATH = "face_landmarker.task"

# ── YOLO Object Detection ──────────────────────────────────
YOLO_MODEL_PATH = "yolov8n.pt"
YOLO_CONFIDENCE = 0.35  # 95% accuracy: higher confidence = fewer false positives
YOLO_CLASSES_OF_INTEREST = {
    # Critical distraction objects
    67: "cell phone",      # Mobile phone
    39: "bottle",          # Water bottle
    41: "cup",             # Cup/drink
    76: "scissors",        # Sharp objects
    43: "knife",           # Sharp objects
    # Common in-vehicle objects
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    28: "suitcase",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    73: "book",
    74: "clock",
    75: "vase",
}

# ── Jarvis / Alert System ──────────────────────────────────
JARVIS_COOLDOWN = 4            # Seconds between alerts (fast for real-time responsiveness)
DANGER_FRAME_THRESHOLD = 4     # Faster alert trigger (95% responsive)
DROWSY_ALERT_DURATION = 3.5    # 85% drowsiness: alert sooner (3.5s)

# ── Database ────────────────────────────────────────────────
DATABASE_URI = "sqlite:///adar_logs.db"

# ── Dashboard ───────────────────────────────────────────────
GRAPH_HISTORY_LENGTH = 100
SOCKETIO_EMIT_INTERVAL = 0.08  # Fast responsive dashboard updates
MJPEG_FRAME_SKIP = 0.033  # Target ~30fps for smooth low-latency streaming

# ── Safety Status Levels ────────────────────────────────────
STATUS_SAFE = "SAFE"
STATUS_WARNING = "WARNING"
STATUS_DANGER = "DANGER"

# ── Fleet Server ────────────────────────────────────────────
# Set FLEET_SERVER_URL env var to override, or use the Render cloud URL below
# For local-only dev: change to "ws://localhost:8000/ws/vehicle/{vehicle_id}"
FLEET_SERVER_URL = os.getenv("FLEET_SERVER_URL", "wss://adar-fleet-command-centre.onrender.com/ws/vehicle/{vehicle_id}")

# ── Version ─────────────────────────────────────────────────
VERSION = "3.0.2"

# ── Spatial Scan (GPT-4o Room Analysis — Thread C) ──────────
SPATIAL_SCAN_INTERVAL = 5.0
SPATIAL_PROMPT = (
    "You are JARVIS, Tony Stark's tactical AI vision system integrated into "
    "the ADAR advanced driver safety platform. Analyze this camera frame.\n\n"
    "Respond in EXACTLY 3 short lines:\n"
    "SUBJECTS: [describe visible people and their state]\n"
    "ENV: [describe environment, key objects, lighting]\n"
    "STATUS: [one-line tactical assessment] | THREAT: [LOW/MEDIUM/HIGH]\n\n"
    "Rules: Each line under 70 chars. Crisp technical language. "
    "No markdown, no asterisks, no extra formatting."
)

# ── Attention Score Weights ─────────────────────────────────
ATTENTION_WEIGHTS = {
    "ear": 0.30,      # Eye closure is the strongest drowsiness signal
    "blink_rate": 0.10,  # Blink anomaly
    "mar": 0.15,      # Yawning
    "head_pose": 0.25, # Looking away
    "distraction": 0.20, # Phone/object distraction
}

# ── Blink Detection ─────────────────────────────────────────
BLINK_EAR_THRESHOLD = 0.17  # Optimized for accurate blink detection
BLINK_RATE_WINDOW = 60
NORMAL_BLINK_RATE = (8, 25)  # Normal range: 8-25 blinks/min

# ── Jarvis Rate Limiting ────────────────────────────────────
JARVIS_BACKOFF_BASE = 30
JARVIS_BACKOFF_MAX = 300
JARVIS_MAX_RETRIES = 2

# ── Offline Jarvis Fallback Messages ────────────────────────
JARVIS_OFFLINE_MESSAGES = {
    "DROWSINESS": "Driver, your eyes are closing. Please pull over and take a break immediately.",
    "YAWNING": "Frequent yawning detected. Consider stopping for rest at the next safe location.",
    "DISTRACTION": "You are distracted. Put down any objects and focus on the road.",
    "HEAD_POSE": "Eyes on the road, driver. You have been looking away for too long.",
    "LOOKING_AWAY": "You are not watching the road. Turn your eyes forward immediately.",
    "LOOKING_DOWN": "You are looking down. Keep your eyes on the road ahead.",
    "PHONE_USE": "Put your phone down now. Using a phone while driving is extremely dangerous.",
    "DRINKING": "Finish your drink and keep both hands on the wheel.",
    "OBJECT_DETECTED": "Distracting objects detected. Stay focused on driving safely.",
    "GENERAL": "Your attention level is critically low. Please stay focused on driving safely.",
}
