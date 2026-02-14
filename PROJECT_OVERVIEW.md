# 🚗 PROJECT ADAR V3.0 — Complete Technical Overview

> **Advanced Driver Attention & Response System**
> Built for the **OpenAI Buildathon Grand Finale 2026**
> Version: 3.0.2 | Python 3.10.11 | Windows 11

---

## 📋 Table of Contents

1. [What Is ADAR?](#1-what-is-adar)
2. [Project Structure](#2-project-structure)
3. [System Architecture](#3-system-architecture)
4. [Technology Stack](#4-technology-stack)
5. [Core Modules (Detailed)](#5-core-modules-detailed)
6. [AI Detection Pipeline](#6-ai-detection-pipeline)
7. [Alert System — 2-Path Architecture](#7-alert-system--2-path-architecture)
8. [Dashboard & Frontend](#8-dashboard--frontend)
9. [Database & Logging](#9-database--logging)
10. [Configuration & Thresholds](#10-configuration--thresholds)
11. [Threading Model](#11-threading-model)
12. [Key Features Summary](#12-key-features-summary)
13. [How To Run](#13-how-to-run)

---

## 1. What Is ADAR?

ADAR (Advanced Driver Attention & Response) is a **real-time AI-powered driver safety monitoring system** that uses computer vision and frontier AI models to detect dangerous driving behaviors and alert the driver with spoken voice warnings.

The system watches the driver through a webcam and detects:
- **Drowsiness** (eyes closing, micro-sleep)
- **Yawning** (mouth opening)
- **Distraction** (phone use, drinking, objects in hand)
- **Head pose deviation** (looking away from road)
- **Hand-to-face behaviors** (phone near ear, hand on head)

When danger is detected, the system:
1. Analyzes the situation using **OpenAI GPT-5.2 Vision** (frontier model)
2. Generates a **spoken voice alert** using OpenAI TTS
3. Plays the warning through speakers in real-time
4. Logs every incident to an SQLite database
5. Shows everything on a live **Iron Man / JARVIS-themed web dashboard**

The AI assistant is named **J.A.R.V.I.S.** (inspired by Iron Man's AI), and all the UI/UX follows that design language.

---

## 2. Project Structure

```
E:\ADAR V3.0\
│
├── main.py                    # Entry point — starts Flask + SocketIO server
├── config.py                  # Central configuration (all thresholds, API keys, model settings)
├── requirements.txt           # Python dependencies
├── face_landmarker.task       # MediaPipe Face Landmarker model file (pre-trained)
├── yolov8n.pt                 # YOLOv8 Nano model weights (object detection)
├── pyrightconfig.json         # Type checker config
├── adar_logs.db               # SQLite database (auto-created at runtime)
│
├── app/                       # Flask application package
│   ├── __init__.py            # Flask app factory + SocketIO init
│   ├── ai_core.py             # AI detection engine (1559 lines) — MediaPipe + YOLO + all detection logic
│   ├── camera.py              # Lock-free threaded camera capture
│   ├── database.py            # SQLAlchemy models + incident logging
│   ├── jarvis.py              # GPT-5.2 Vision + TTS alert pipeline
│   └── routes.py              # Flask routes, engine controller, background threads
│
├── templates/
│   └── dashboard.html         # JARVIS-themed web dashboard (single page)
│
├── static/
│   ├── css/
│   │   └── style.css          # Full JARVIS/Iron Man themed CSS
│   └── js/
│       └── dashboard.js       # SocketIO client, Chart.js graphs, real-time UI updates
│
└── .venv/                     # Python virtual environment
```

---

## 3. System Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   CAMERA     │────▶│   AI CORE    │────▶│   TELEMETRY     │
│  (camera.py) │     │ (ai_core.py) │     │  via SocketIO   │
│  Lock-free   │     │  MediaPipe   │     │  to Dashboard   │
│  30fps       │     │  + YOLOv8    │     └─────────────────┘
└─────────────┘     └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  ALERT LOGIC │
                    │ (routes.py)  │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼                         ▼
    ┌─────────────────┐      ┌──────────────────┐
    │  PATH A: LOCAL   │      │  PATH B: GPT-5.2  │
    │  Drowsy 4s+      │      │  Vision Analysis   │
    │  Instant alert   │      │  + JSON response   │
    │  (no API call)   │      │  + TTS audio       │
    └────────┬────────┘      └────────┬──────────┘
             │                        │
             ▼                        ▼
    ┌─────────────────┐      ┌──────────────────┐
    │  OpenAI TTS      │      │  OpenAI TTS       │
    │  → pygame audio  │      │  → pygame audio   │
    └────────┬────────┘      └────────┬──────────┘
             │                        │
             ▼                        ▼
    ┌──────────────────────────────────┐
    │        SQLite Database           │
    │     + Dashboard SocketIO         │
    └──────────────────────────────────┘
```

### Thread Model

| Thread | Name | Purpose |
|--------|------|---------|
| **Main** | Flask/SocketIO | Serves web dashboard, handles HTTP & WebSocket |
| **Camera** | Camera Thread | Captures frames at 30fps, lock-free atomic swap |
| **Thread A** | Processing Loop | Reads camera → AI detection → emit telemetry → trigger alerts |
| **Thread C** | Spatial Scan | Every 5s, sends a frame to GPT-5.2 for room/environment analysis |
| **Alert Threads** | Jarvis Workers | Spawned on-demand for GPT-5.2 calls + TTS + audio playback |

---

## 4. Technology Stack

### Backend (Python 3.10.11)
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Web Framework | **Flask 3.1+** | Serves dashboard, API routes |
| Real-time | **Flask-SocketIO 5.5+** | WebSocket telemetry streaming |
| Face Detection | **MediaPipe Face Landmarker** | 478 facial landmarks, eye/mouth tracking |
| Object Detection | **YOLOv8 Nano** (Ultralytics) | Detects phone, bottle, cup, etc. |
| Hand Detection | **MediaPipe Hands** | Detects hand position relative to face |
| AI Vision | **OpenAI GPT-5.2** (Vision) | Analyzes driver frames, returns JSON danger assessment |
| Text-to-Speech | **OpenAI TTS-1** (voice: onyx) | Generates spoken voice warnings |
| Audio Playback | **pygame-ce 2.5+** | Plays TTS audio through speakers |
| Computer Vision | **OpenCV 4.11+** | Frame capture, encoding, image processing |
| Database | **SQLAlchemy 2.0+** + SQLite | Logs every incident with full telemetry |
| Math | **NumPy, SciPy** | EAR/MAR calculations, head pose estimation |

### Frontend
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Dashboard | **HTML5 / CSS3 / JavaScript** | Single-page JARVIS-themed command center |
| Real-time Updates | **Socket.IO 4.7** | Live telemetry from backend |
| Charts | **Chart.js 4.4** | EAR/MAR timeline graph |
| Fonts | **Orbitron, Rajdhani, Share Tech Mono** | Iron Man / sci-fi aesthetic |

### Hardware Requirements
| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 2050 (4GB VRAM) — used by YOLO |
| CPU | AMD Ryzen 5 5600H |
| Camera | Any USB webcam (configured for 640×480 @ 30fps) |
| OS | Windows 11 |

---

## 5. Core Modules (Detailed)

### 5.1 `main.py` — Entry Point
- Prints the ADAR ASCII art banner
- Shows system status (dashboard URL, camera source, Jarvis online/offline)
- Creates the Flask app via `create_app()`
- Runs SocketIO server on `0.0.0.0:5000`
- Handles Ctrl+C graceful shutdown

### 5.2 `config.py` — Central Configuration
All tunable parameters live here. Key settings:

| Setting | Value | Purpose |
|---------|-------|---------|
| `OPENAI_MODEL` | `"gpt-5.2"` | Frontier model — fast, non-reasoning, vision-capable |
| `GPT_TIMEOUT` | `3.0` seconds | If GPT-5.2 exceeds this, local fallback fires |
| `EAR_THRESHOLD` | `0.25` | Eye Aspect Ratio below this = eyes closing |
| `EAR_CONSEC_FRAMES` | `15` | Frames below threshold to confirm drowsiness |
| `MAR_THRESHOLD` | `0.70` | Mouth Aspect Ratio above this = yawning |
| `HEAD_YAW_THRESHOLD` | `25°` | Looking sideways beyond this = looking away |
| `HEAD_PITCH_THRESHOLD` | `20°` | Looking up/down beyond this = looking away |
| `JARVIS_COOLDOWN` | `5` seconds | Minimum time between consecutive alerts |
| `DROWSY_ALERT_DURATION` | `4.0` seconds | Drowsy timer must reach this to trigger DANGER |
| `DANGER_FRAME_THRESHOLD` | `5` | Consecutive DANGER frames before GPT-5.2 alert |
| `SPATIAL_SCAN_INTERVAL` | `5.0` seconds | How often Thread C scans the room |
| `YOLO_CONFIDENCE` | `0.45` | YOLO detection threshold |
| `OPENAI_TTS_VOICE` | `"onyx"` | Deep male voice for JARVIS |

Safety status levels: `SAFE`, `WARNING`, `DANGER`

### 5.3 `app/camera.py` — Lock-Free Threaded Camera
- Opens webcam via OpenCV (DirectShow backend on Windows for lowest latency)
- Runs a dedicated background thread that continuously captures frames
- Uses **atomic reference swap** (no threading.Lock) for zero-latency frame access
- Configures: 640×480, 30fps, MJPG codec, manual exposure, buffer size 1
- Auto-reconnects if camera connection is lost
- Flips frame horizontally for natural mirror view
- Tracks real-time FPS

### 5.4 `app/ai_core.py` — AI Detection Engine (1559 lines)
This is the heart of the system. It contains:

#### Face Detection (MediaPipe Face Landmarker)
- Extracts 478 facial landmarks per frame
- Calculates **EAR** (Eye Aspect Ratio) — measures eye openness
- Calculates **MAR** (Mouth Aspect Ratio) — measures mouth openness
- Estimates **head pose** (yaw + pitch) using solvePnP with 3D model points
- Detects **blinks** by tracking EAR transitions (falling edge detection)
- Calculates **blink rate** (blinks per minute) over a rolling 60-second window

#### Drowsiness Detection (Multi-Factor, ~95% Accuracy)
- **Primary**: EAR below 0.25 for 15+ consecutive frames
- **Secondary**: Very low EAR (< 0.25 × 0.85) for 5+ frames
- **Tertiary**: Base drowsy + abnormal blink rate (< 5 or > 30 blinks/min)
- **Drowsy timer**: Tracks how long drowsiness has been sustained
  - 10-frame grace period prevents brief EAR fluctuations from resetting the timer
  - Timer reaching 4 seconds triggers DANGER state + instant local alert

#### Yawning Detection
- **Primary**: MAR above 0.70 for 8+ consecutive frames
- **Secondary**: Extreme MAR (> 0.70 × 1.2) for 3+ frames

#### Looking Away Detection (with hysteresis)
- Yaw > 25° or Pitch > 20° triggers "looking away"
- Uses frame counter with asymmetric increase/decrease to prevent false positives
- Must be looking away for 3+ frames to confirm

#### Object Detection (YOLOv8 Nano)
- Runs every 10th frame (to save GPU resources)
- Detects: cell phone, bottle, cup, scissors, knife, laptop, backpack, etc.
- Objects in `_DISTRACTION_OBJECTS` set trigger distraction alerts

#### Hand Detection (MediaPipe Hands)
- Detects up to 2 hands per frame
- Calculates hand position relative to face
- Detects: hand near face, hand on head, phone near ear

#### Advanced Behavior Analysis
- **Phone near ear**: Hand holding phone close to ear region
- **Looking down**: Pitch angle below -20° sustained
- **Drinking**: Bottle/cup detected near face region
- **Tiredness level**: Composite 0-100 score combining EAR history, blink rate, yawn frequency
- **Affective state**: ALERT / TIRED / DROWSY / DISTRACTED

#### Attention Score (0-100 Composite)
Weighted combination of:
- EAR (30%) — eye openness
- Blink rate (15%) — abnormal = lower score
- MAR (15%) — yawning = lower score
- Head pose (20%) — looking away = lower score
- Distraction (20%) — objects/behaviors = lower score
- Temporal smoothing (70% current + 30% previous) for stability

#### Safety Status (3 Levels)
- **DANGER**: 2+ danger factors, attention < 20, critical drowsy, extreme distraction, or dangerous behavior (phone/drinking)
- **WARNING**: 1 danger factor, attention < 45, or severe looking away
- **SAFE**: No danger factors detected

#### JARVIS HUD Overlay (Built-in but NOT shown on camera feed)
The ai_core contains a full Iron Man-style HUD overlay system with 12 visual layers including:
- Helmet visor vignette
- 3D depth-shaded face mesh
- Rotating targeting reticle
- Iron Man object detection boxes
- System integrity bars
- Mini radar, process monitor
- Corner brackets with neon glow

> **Note**: The HUD overlay is NOT drawn on the live camera feed (kept clean by design). It exists in the code and can be enabled if desired.

### 5.5 `app/jarvis.py` — GPT-5.2 Vision + TTS Alert Pipeline
The voice alert assistant. Key features:

#### Two Alert Methods

**1. `trigger_alert(frame, telemetry)` — GPT-5.2 Path**
Used for: EAR below threshold, danger frames, distraction
Pipeline:
1. Encode camera frame to base64 JPEG
2. Build context prompt with sensor readings (EAR, MAR, Yaw, Pitch, Attention, BlinkRate)
3. Send to GPT-5.2 Vision with image + ask for JSON response
4. GPT-5.2 returns: `{"status": "DANGER"/"SAFE", "reason": "...", "confidence": 0.0-1.0}`
5. If GPT says SAFE with ≥80% confidence → suppress the alert (false positive override)
6. If DANGER → generate TTS spoken warning → play audio → log to DB → emit to dashboard
7. Timeout (>3s) → falls back to local rule-based alert
8. Rate limited → exponential backoff (30s base, 300s max) + local fallback

**2. `trigger_drowsy_alert(telemetry)` — Local Path (No GPT)**
Used for: Drowsy timer reaching 4+ seconds
Pipeline:
1. Immediately fires local rule-based alert (no API call)
2. Uses pre-written fallback messages from config
3. Generates TTS → plays audio → logs to DB → emits to dashboard
4. **Guaranteed** to fire — no API dependency

#### Cooldown & Override
- Normal cooldown: 5 seconds between alerts
- Critical drowsy override (8s+): `force=True` bypasses cooldown (only checks if currently speaking)
- Rate limit backoff: exponential, 30s base → 300s max

#### OpenAI Client Configuration
- Model: `gpt-5.2`
- `max_retries=0` — no SDK retries, system handles fallback manually
- `max_completion_tokens=100` — short responses for speed
- `temperature=0.3` — focused, deterministic output
- `timeout=3.0` seconds — enforced via `openai.APITimeoutError`
- TTS model: `tts-1`, voice: `onyx` (deep male), format: `mp3`
- Audio: pygame-ce mixer at 24kHz mono

### 5.6 `app/routes.py` — Flask Routes & Engine Controller

#### Routes
| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Serves the JARVIS dashboard HTML |
| `/video_feed` | GET | MJPEG video stream (30fps target) |
| `/api/stats` | GET | JSON incident statistics for current session |
| `/api/incidents` | GET | JSON list of recent 50 incidents |

#### Engine Controller
- `_start_engine()`: Initializes Camera, AICore, Jarvis, starts Thread A + Thread C
- `stop_engine()`: Gracefully shuts down all threads and resources
- SocketIO `connect` event: Sends system status to newly connected dashboard clients

#### Thread A — Processing Loop (Main Loop)
Runs continuously at camera speed (~30fps):
1. Read frame from camera
2. Run AI detection (`ai_core.process_frame()`) — YOLO every 10th frame
3. Encode frame as JPEG for MJPEG streaming
4. Emit telemetry via SocketIO (throttled to 10Hz)
5. Check alert conditions:
   - **Drowsy timer ≥ 4s** → Local alert (PATH A) — no GPT
   - **Drowsy timer ≥ 8s** → Local alert with force override (bypasses cooldown)
   - **EAR below threshold / danger frames / distraction** → GPT-5.2 alert (PATH B)

#### Thread C — Spatial Analysis
Every 5 seconds:
1. Grab camera frame
2. Send to GPT-5.2 with spatial analysis prompt
3. GPT-5.2 returns 3-line tactical assessment:
   - SUBJECTS: (people and their state)
   - ENV: (environment, objects, lighting)
   - STATUS: (assessment) | THREAT: (LOW/MEDIUM/HIGH)
4. Result stored in ai_core for the HUD spatial panel

### 5.7 `app/database.py` — SQLAlchemy Models

#### Incident Table
| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Auto-increment primary key |
| timestamp | DateTime | When the incident occurred |
| alert_type | String | DROWSINESS, YAWNING, DISTRACTION, HEAD_POSE, GENERAL |
| severity | String | WARNING or DANGER |
| ear_value | Float | Eye Aspect Ratio at time of incident |
| mar_value | Float | Mouth Aspect Ratio |
| yaw_angle | Float | Head yaw in degrees |
| pitch_angle | Float | Head pitch in degrees |
| detected_objects | String | Comma-separated list of objects |
| jarvis_response | Text | Full GPT-5.2 response or local fallback message |
| attention_score | Float | 0-100 composite score |
| blink_rate | Float | Blinks per minute |

#### Session Table
| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Auto-increment |
| start_time | DateTime | Session start |
| end_time | DateTime | Session end |
| total_incidents | Integer | Count of incidents |
| max_severity | String | Worst severity reached |

Functions: `log_incident()`, `get_incident_stats()`, `get_recent_incidents()`

---

## 6. AI Detection Pipeline

```
Camera Frame (640×480, 30fps)
       │
       ▼
┌──────────────────────────────────┐
│  MediaPipe Face Landmarker       │
│  478 landmarks → EAR, MAR,      │
│  head pose (yaw/pitch),         │
│  blink detection, iris tracking  │
└──────────────────┬───────────────┘
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
  ┌─────────┐ ┌─────────┐ ┌─────────────┐
  │ Drowsy? │ │ Yawning?│ │ Looking     │
  │ EAR<0.25│ │ MAR>0.70│ │ Away?       │
  │ 15frames│ │ 8 frames│ │ Yaw>25°     │
  └────┬────┘ └────┬────┘ └──────┬──────┘
       │           │              │
       ▼           ▼              ▼
┌──────────────────────────────────┐
│  YOLOv8 Nano (every 10 frames)  │
│  → phone, bottle, cup, knife    │
└──────────────────┬───────────────┘
                   │
                   ▼
┌──────────────────────────────────┐
│  MediaPipe Hands                 │
│  → hand near face, phone at ear │
└──────────────────┬───────────────┘
                   │
                   ▼
┌──────────────────────────────────┐
│  Behavior Analysis               │
│  → phone_near_ear, drinking,     │
│    looking_down, hand_on_head    │
└──────────────────┬───────────────┘
                   │
                   ▼
┌──────────────────────────────────┐
│  Attention Score (0-100)         │
│  EAR(30%) + Blink(15%) +        │
│  MAR(15%) + Head(20%) +         │
│  Distraction(20%)               │
└──────────────────┬───────────────┘
                   │
                   ▼
┌──────────────────────────────────┐
│  Safety Status                   │
│  SAFE / WARNING / DANGER         │
└──────────────────────────────────┘
```

---

## 7. Alert System — 2-Path Architecture

### PATH A: Drowsy Timer → Instant Local Alert
```
Drowsy Timer hits 4 seconds
       │
       ▼
[JARVIS] 💤 Drowsy timer triggered — instant local alert (no GPT)
       │
       ▼
Pre-written message: "Driver, your eyes are closing. Please pull over
and take a break immediately."
       │
       ▼
OpenAI TTS → pygame audio → Speaker
       │
       ▼
Log to SQLite + Emit to Dashboard
```
**Why local?** GPT-5.2 can fail (timeout, rate limit, misjudge). Drowsiness is life-critical, so the 4-second alert is **guaranteed** to fire with no API dependency.

At 8+ seconds: `force=True` bypasses the 5-second cooldown entirely.

### PATH B: Other Dangers → GPT-5.2 Vision Analysis
```
EAR below threshold / Danger frames / Distraction detected
       │
       ▼
Encode frame → base64 JPEG (quality 70)
       │
       ▼
Send to GPT-5.2 Vision with sensor context:
  "EAR=0.180, MAR=0.120, Yaw=5.2°, Pitch=-3.1°,
   Attention=45/100, BlinkRate=8/min"
       │
       ▼
GPT-5.2 responds (within 3s timeout):
  {"status": "DANGER", "reason": "Micro-sleep detected", "confidence": 0.93}
       │
       ├── If SAFE + confidence ≥ 80% → Suppress alert (false positive override)
       │
       ├── If DANGER → Generate TTS warning → Play audio
       │
       ├── If Timeout (>3s) → Local fallback alert
       │
       └── If Rate Limited → Exponential backoff + local fallback
```

### Offline Fallback Messages
| Alert Type | Message |
|-----------|---------|
| DROWSINESS | "Driver, your eyes are closing. Please pull over and take a break immediately." |
| YAWNING | "Frequent yawning detected. Consider stopping for rest at the next safe location." |
| DISTRACTION | "Put your phone down and focus on the road. Your life depends on it." |
| HEAD_POSE | "Eyes on the road, driver. You have been looking away for too long." |
| GENERAL | "Your attention level is critically low. Please stay focused on driving safely." |

---

## 8. Dashboard & Frontend

### Layout (4-Panel Grid)
```
┌────────────────────────┬─────────────────────┐
│                        │     TELEMETRY        │
│     LIVE FEED          │  ┌─────────────┐    │
│     (MJPEG Stream)     │  │  ATTENTION   │    │
│                        │  │   GAUGE      │    │
│     Camera 01          │  │   (0-100)    │    │
│     30fps              │  └─────────────┘    │
│                        │  ┌──┬──┬──┬──┐      │
│                        │  │DR│YW│DI│HP│      │
│                        │  └──┴──┴──┴──┘      │
│                        │  EAR / MAR Chart    │
├────────────────────────┼─────────────────────┤
│     J.A.R.V.I.S.       │   SESSION STATS     │
│     Feed / Logs        │   Total | Drowsy    │
│                        │   Yawning | Distract│
│                        │   AI Latency        │
└────────────────────────┴─────────────────────┘
```

### Top Header Bar — 3-Tier Drowsiness Status
The top bar shows the current drowsiness state only:

| State | Display | Color |
|-------|---------|-------|
| No drowsiness detected | 🟢 **SAFE** | Green |
| Drowsiness detected, timer < 4s | 🟠 **WARNING** | Orange/Amber |
| Drowsiness ≥ 4s (alert firing) | 🔴 **DANGER** | Red (+ red overlay flash) |

### Real-Time Elements
- **Attention Gauge**: SVG ring showing 0-100 score, color transitions (green → orange → red)
- **Status Cards**: 4 cards for Drowsiness, Yawning, Distraction, Head Pose — each with active/inactive indicators
- **Drowsiness Card**: Shows tier labels — ✅ SAFE / ⚠️ WARNING 2.3s / 🔴 DANGER 5.1s (client-side smooth 20Hz timer)
- **EAR/MAR Chart**: Chart.js line graph with 100-point rolling history, threshold lines drawn
- **JARVIS Feed**: Last 3 alert messages with timestamps
- **Session Stats**: Total alerts, drowsiness count, yawning count, distraction count
- **AI Latency**: Real-time processing time in milliseconds
- **Danger Overlay**: Full-screen red flash when in DANGER state
- **System Clock**: HH:MM:SS live clock
- **Session Uptime**: Running time since dashboard connected
- **FPS Counter**: Camera frames per second

### Design Theme
- **Iron Man / JARVIS** aesthetic
- Colors: Dark background (#08090d), Orange accents (#ff8c00), White text
- Fonts: Orbitron (headers), Rajdhani (body), Share Tech Mono (data)
- Neon glow effects, animated beacon, arc reactor logo element
- Scanline effect on video feed

---

## 9. Database & Logging

- **Engine**: SQLite via SQLAlchemy 2.0
- **File**: `adar_logs.db` (auto-created in project root)
- **Auto-migration**: Adds new columns to existing tables on startup
- **Thread-safe**: Uses `scoped_session` for safe multi-thread access
- Every alert (both GPT-5.2 and local fallback) is logged with full telemetry
- Dashboard fetches stats via `/api/stats` every 5 seconds

---

## 10. Configuration & Thresholds

### OpenAI API
```python
OPENAI_MODEL = "gpt-5.2"           # Frontier model (NOT gpt-5.2-instant — doesn't exist)
OPENAI_TTS_MODEL = "tts-1"          # Text-to-speech
OPENAI_TTS_VOICE = "onyx"           # Deep male voice
GPT_TIMEOUT = 3.0                   # 3 second hard timeout
```

### Detection Thresholds
```python
EAR_THRESHOLD = 0.25                # Eye Aspect Ratio — below = drowsy
EAR_CONSEC_FRAMES = 15              # Frames to confirm drowsiness
MAR_THRESHOLD = 0.70                # Mouth Aspect Ratio — above = yawning
MAR_CONSEC_FRAMES = 8               # Frames to confirm yawning
HEAD_YAW_THRESHOLD = 25             # Degrees — looking sideways
HEAD_PITCH_THRESHOLD = 20           # Degrees — looking up/down
BLINK_EAR_THRESHOLD = 0.18          # EAR threshold for blink detection
YOLO_CONFIDENCE = 0.45              # YOLO detection confidence
```

### Alert System
```python
JARVIS_COOLDOWN = 5                 # Seconds between alerts
DANGER_FRAME_THRESHOLD = 5          # Danger frames before GPT alert
DROWSY_ALERT_DURATION = 4.0         # Seconds of drowsiness before DANGER
JARVIS_BACKOFF_BASE = 30            # Rate limit backoff base (seconds)
JARVIS_BACKOFF_MAX = 300            # Max backoff (5 minutes)
```

### Camera
```python
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_JPEG_QUALITY = 80
CAMERA_FLIP_HORIZONTAL = True       # Mirror view
```

---

## 11. Threading Model

```
Main Thread (Flask/SocketIO Server)
    │
    ├── Camera Thread (continuous)
    │   └── Captures frames at 30fps, atomic reference swap
    │
    ├── Thread A — Processing Loop (continuous)
    │   └── Camera → AI detection → JPEG encode → SocketIO emit → Alert trigger
    │       │
    │       ├── Spawns: Jarvis GPT-5.2 alert thread (on-demand, daemon)
    │       └── Spawns: Jarvis Local drowsy alert thread (on-demand, daemon)
    │
    └── Thread C — Spatial Scan (every 5 seconds)
        └── Camera → GPT-5.2 Vision → Spatial analysis text → Stored in AICore
```

### Concurrency Controls
- Camera: Lock-free atomic reference (no mutex)
- Frame for MJPEG: `threading.Lock` protects `_latest_frame_bytes`
- Jarvis: `threading.Lock` protects `is_speaking` flag
- Jarvis cooldown: `is_ready` property checks elapsed time + backoff
- Force override: Bypasses cooldown, only checks `is_speaking` and `_backoff_until`

---

## 12. Key Features Summary

### Core AI Detection
- ✅ **Drowsiness detection** — Multi-factor EAR analysis with 10-frame grace period
- ✅ **Yawning detection** — MAR threshold with consecutive frame confirmation
- ✅ **Head pose tracking** — Yaw + Pitch with hysteresis to prevent false positives
- ✅ **Object detection** — YOLOv8 Nano detects phone, bottle, cup, knife, etc.
- ✅ **Hand tracking** — MediaPipe Hands detects hand-to-face behaviors
- ✅ **Blink rate monitoring** — Rolling 60-second window
- ✅ **Attention score** — Weighted composite 0-100 with temporal smoothing
- ✅ **3-level safety status** — SAFE / WARNING / DANGER

### Alert System
- ✅ **GPT-5.2 Vision analysis** — Sends frame + sensor data, gets JSON response
- ✅ **GPT-5.2 false positive override** — SAFE with ≥80% confidence suppresses alert
- ✅ **Instant local drowsy alerts** — 4s+ drowsiness fires without API call
- ✅ **Critical drowsy override** — 8s+ bypasses cooldown entirely
- ✅ **OpenAI TTS spoken warnings** — Natural voice through speakers
- ✅ **Timeout fallback** — >3s GPT response → local rule-based alert
- ✅ **Rate limit handling** — Exponential backoff + local fallback
- ✅ **Pre-written fallback messages** — Works even when API is down

### Dashboard
- ✅ **JARVIS/Iron Man themed** — Sci-fi command center aesthetic
- ✅ **Live MJPEG video feed** — 30fps camera stream
- ✅ **Real-time telemetry** — SocketIO at 10Hz
- ✅ **3-tier status bar** — SAFE → WARNING → DANGER based on drowsiness
- ✅ **Attention gauge** — SVG ring with color transitions
- ✅ **EAR/MAR chart** — Chart.js rolling timeline
- ✅ **Smooth drowsy timer** — Client-side 20Hz counter
- ✅ **Alert log** — Last 3 JARVIS messages
- ✅ **Session statistics** — Incident counts by type
- ✅ **Danger overlay** — Full-screen red flash

### Infrastructure
- ✅ **SQLite incident logging** — Every alert recorded with full telemetry
- ✅ **Lock-free camera** — Atomic reference swap, zero-latency
- ✅ **Multi-threaded architecture** — Camera, Processing, Spatial, Alert threads
- ✅ **Graceful shutdown** — Signal handler, resource cleanup
- ✅ **Auto DB migration** — New columns added safely to existing tables
- ✅ **Spatial environment scanning** — GPT-5.2 room analysis every 5s

---

## 13. How To Run

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA (for YOLOv8)
- Webcam connected
- OpenAI API key with GPT-5.2 and TTS access

### Setup
```bash
cd "E:\ADAR V3.0"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Set API Key
Create `.env` file or set environment variable:
```
OPENAI_API_KEY=sk-your-key-here
```

### Run
```bash
python main.py
```

### Access Dashboard
Open browser: **http://localhost:5000**

---

## Dependencies (requirements.txt)

```
flask>=3.1.0
flask-socketio>=5.5.1
eventlet>=0.37.0
opencv-python>=4.11.0
mediapipe>=0.10.30
ultralytics>=8.3.0
numpy>=1.26.4
openai>=1.61.0
pygame-ce>=2.5.3
sqlalchemy>=2.0.37
python-dotenv>=1.0.1
Pillow>=11.1.0
scipy>=1.15.0
```

---

*Document generated: February 12, 2026*
*ADAR V3.0 — Advanced Driver Attention & Response System*
*Built for the OpenAI Buildathon Grand Finale*
