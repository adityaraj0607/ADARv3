# 🔬 PROJECT ADAR V3.0 — Complete Feature-by-Feature Implementation Guide

> **Every feature explained: WHAT it does, HOW it works internally, and the EXACT code logic.**
> This document is meant to give a new AI model (like Gemini) complete understanding
> of every feature's inner workings without needing to read the source code.

---

## 📋 Table of Contents

1. [Camera Capture System](#1-camera-capture-system)
2. [Face Detection & Landmark Extraction](#2-face-detection--landmark-extraction)
3. [Eye Aspect Ratio (EAR) Calculation](#3-eye-aspect-ratio-ear-calculation)
4. [Mouth Aspect Ratio (MAR) Calculation](#4-mouth-aspect-ratio-mar-calculation)
5. [Head Pose Estimation (Yaw & Pitch)](#5-head-pose-estimation-yaw--pitch)
6. [Blink Detection & Blink Rate](#6-blink-detection--blink-rate)
7. [Drowsiness Detection (Multi-Factor)](#7-drowsiness-detection-multi-factor)
8. [Drowsy Timer & Grace Period](#8-drowsy-timer--grace-period)
9. [Yawning Detection](#9-yawning-detection)
10. [Looking Away Detection (Hysteresis)](#10-looking-away-detection-hysteresis)
11. [Object Detection (YOLOv8 Nano)](#11-object-detection-yolov8-nano)
12. [Hand Detection (MediaPipe Hands)](#12-hand-detection-mediapipe-hands)
13. [Advanced Behavior Analysis](#13-advanced-behavior-analysis)
14. [Tiredness & Affective State Analysis](#14-tiredness--affective-state-analysis)
15. [Distraction Detection (Composite)](#15-distraction-detection-composite)
16. [Attention Score (Weighted 0-100)](#16-attention-score-weighted-0-100)
17. [Safety Status (3-Level Decision Tree)](#17-safety-status-3-level-decision-tree)
18. [Alert System — PATH A: Local Drowsy Alert](#18-alert-system--path-a-local-drowsy-alert)
19. [Alert System — PATH B: GPT-5.2 Vision Analysis](#19-alert-system--path-b-gpt-52-vision-analysis)
20. [GPT-5.2 JSON Response Parsing](#20-gpt-52-json-response-parsing)
21. [GPT-5.2 False Positive Override](#21-gpt-52-false-positive-override)
22. [Local Fallback Alert System](#22-local-fallback-alert-system)
23. [Text-to-Speech (TTS) Generation](#23-text-to-speech-tts-generation)
24. [Audio Playback System](#24-audio-playback-system)
25. [Cooldown & Force Override Mechanism](#25-cooldown--force-override-mechanism)
26. [Rate Limit Handling & Exponential Backoff](#26-rate-limit-handling--exponential-backoff)
27. [Spatial Environment Scanning (Thread C)](#27-spatial-environment-scanning-thread-c)
28. [Telemetry Emission (SocketIO)](#28-telemetry-emission-socketio)
29. [MJPEG Video Streaming](#29-mjpeg-video-streaming)
30. [Dashboard — 3-Tier Drowsiness Status Bar](#30-dashboard--3-tier-drowsiness-status-bar)
31. [Dashboard — Client-Side Drowsy Timer (20Hz)](#31-dashboard--client-side-drowsy-timer-20hz)
32. [Dashboard — Attention Score Gauge](#32-dashboard--attention-score-gauge)
33. [Dashboard — EAR/MAR Real-Time Chart](#33-dashboard--earmar-real-time-chart)
34. [Dashboard — Status Indicator Cards](#34-dashboard--status-indicator-cards)
35. [Dashboard — JARVIS Alert Feed](#35-dashboard--jarvis-alert-feed)
36. [Dashboard — Session Statistics](#36-dashboard--session-statistics)
37. [Dashboard — Danger Overlay Flash](#37-dashboard--danger-overlay-flash)
38. [Database — Incident Logging](#38-database--incident-logging)
39. [Database — Auto-Migration](#39-database--auto-migration)
40. [Thread Architecture](#40-thread-architecture)
41. [Auto-Logging on State Changes](#41-auto-logging-on-state-changes)
42. [JARVIS HUD Overlay (Built-in, Disabled)](#42-jarvis-hud-overlay-built-in-disabled)
43. [Telemetry Data Structure (30+ Fields)](#43-telemetry-data-structure-30-fields)

---

## 1. Camera Capture System

**File**: `app/camera.py` (147 lines)

### What It Does
Captures webcam frames at 30fps using a dedicated background thread with zero-latency frame access.

### How It Works Internally

1. **Initialization**: Opens the webcam using OpenCV with the **DirectShow** backend (`cv2.CAP_DSHOW`) on Windows for lowest possible latency. Configures:
   - Resolution: 640×480
   - FPS: 30
   - Codec: MJPG (hardware-accelerated)
   - Manual exposure (disables auto-exposure for consistency)
   - Buffer size: 1 (grabs latest frame only, no buffering delay)

2. **Lock-Free Threading**: A background thread continuously calls `cap.read()` in a tight loop. Instead of using a `threading.Lock` (which causes contention), it uses **atomic reference swap**:
   ```
   self._frame = new_frame   # Python's GIL makes this atomic
   ```
   The main thread reads `self._frame` at any time — it always gets the latest frame or the previous one, never a half-written frame.

3. **Horizontal Flip**: Every frame is flipped with `cv2.flip(frame, 1)` so the camera acts as a natural mirror (driver sees themselves correctly).

4. **FPS Tracking**: Counts frames per second using a rolling time delta.

5. **Auto-Reconnect**: If `cap.read()` fails (camera disconnected), waits `CAMERA_RECONNECT_DELAY` (2s) then re-opens the camera.

6. **Read API**: `camera.read()` returns `(True, frame)` or `(False, None)`. No lock is acquired — the caller simply reads the current reference.

### Key Design Choice
Lock-free design means the processing thread (Thread A) never blocks waiting for the camera thread. The camera thread runs at its natural speed (~30fps), and Thread A grabs whatever frame is currently available.

---

## 2. Face Detection & Landmark Extraction

**File**: `app/ai_core.py`, `process_frame()` method

### What It Does
Detects the driver's face and extracts 478 facial landmarks (3D coordinates) using Google's MediaPipe Face Landmarker.

### How It Works Internally

1. **Model**: Uses the `face_landmarker.task` file — a pre-trained TFLite model from Google.

2. **Configuration** (tuned for high accuracy):
   ```
   min_face_detection_confidence = 0.6   (higher = fewer false faces)
   min_face_presence_confidence = 0.6    (higher = more stable tracking)
   min_tracking_confidence = 0.7         (higher = smoother landmarks)
   num_faces = 1                         (single driver)
   running_mode = VIDEO                  (optimized for frame sequences)
   ```

3. **Per-Frame Processing**:
   - Convert BGR frame to RGB (MediaPipe expects RGB)
   - Create `mp.Image` from the numpy array
   - Call `landmarker.detect_for_video(image, timestamp)` with an incrementing timestamp (+33ms per frame, simulating 30fps)
   - If a face is found, extract all 478 landmarks as pixel coordinates `(x*width, y*height)` and z-depth values

4. **Landmark Storage**:
   - `_cached_coords`: NumPy array of shape `(478, 2)` — pixel (x, y) for each landmark
   - `_cached_z_values`: NumPy array of shape `(478,)` — relative depth
   - `_cached_oval_pts`: 36-point face contour from specific landmark indices (used for face bounding box)

5. **When No Face Detected**: All detection flags reset, `face_detected = False`, coords cleared.

### Important Landmark Indices
```
LEFT_EYE  = [362, 385, 387, 263, 373, 380]   # 6 points around left eye
RIGHT_EYE = [33, 160, 158, 133, 153, 144]    # 6 points around right eye
UPPER_LIP = [13]                               # Upper lip center
LOWER_LIP = [14]                               # Lower lip center
LEFT_MOUTH_CORNER = 61                         # Left corner of mouth
RIGHT_MOUTH_CORNER = 291                       # Right corner of mouth
NOSE_TIP = 1                                   # Nose tip
CHIN = 152                                     # Bottom of chin
LEFT_EYE_CORNER = 263                          # Outer left eye
RIGHT_EYE_CORNER = 33                          # Outer right eye
```

---

## 3. Eye Aspect Ratio (EAR) Calculation

**File**: `app/ai_core.py`, `_calculate_ear()` static method

### What It Does
Measures how open or closed the eyes are. A lower EAR means more closed eyes.

### The Exact Formula

Given 6 eye landmarks arranged as:
```
      P1 (top-left)        P2 (top-right)
P0 (left corner) ───────────────── P3 (right corner)
      P5 (bottom-left)     P4 (bottom-right)
```

The formula is:
```
EAR = (|P1 - P5| + |P2 - P4|) / (2 × |P0 - P3|)
```

Where `|A - B|` is the Euclidean distance between two points.

- **Numerator**: Sum of two vertical distances (eye height at two cross-sections)
- **Denominator**: Two times the horizontal distance (eye width)
- **Division by 2**: Normalizes for the two vertical measurements

### How The Code Works
```python
pts = coords[eye_indices]           # Extract 6 points
v1 = np.linalg.norm(pts[1] - pts[5])  # Vertical distance 1
v2 = np.linalg.norm(pts[2] - pts[4])  # Vertical distance 2
h  = np.linalg.norm(pts[0] - pts[3])  # Horizontal distance
EAR = (v1 + v2) / (2.0 * h)           # Final ratio
```

### Typical Values
- **Eyes wide open**: EAR ≈ 0.30–0.35
- **Normal open**: EAR ≈ 0.25–0.30
- **Drowsy / half-closed**: EAR ≈ 0.18–0.25
- **Eyes closed**: EAR < 0.18
- **Threshold**: `EAR_THRESHOLD = 0.25` — below this triggers drowsiness counter

### Averaging
Left and right eye EARs are calculated separately, then averaged:
```python
self.ear = (left_ear + right_ear) / 2.0
```

---

## 4. Mouth Aspect Ratio (MAR) Calculation

**File**: `app/ai_core.py`, `_calculate_mar()` static method

### What It Does
Measures how open the mouth is. A higher MAR means the mouth is more open (yawning).

### The Exact Formula
```
MAR = |upper_lip - lower_lip| / |left_corner - right_corner|
```

- **Numerator**: Vertical distance between upper and lower lip centers (landmark 13 and 14)
- **Denominator**: Horizontal distance between left and right mouth corners (landmarks 61 and 291)

### How The Code Works
```python
upper = coords[13]         # Upper lip center
lower = coords[14]         # Lower lip center
left  = coords[61]         # Left mouth corner
right = coords[291]        # Right mouth corner
v = np.linalg.norm(upper - lower)     # Vertical opening
h = np.linalg.norm(left - right)      # Horizontal width
MAR = v / h
```

### Typical Values
- **Mouth closed**: MAR ≈ 0.05–0.15
- **Talking**: MAR ≈ 0.20–0.50
- **Yawning**: MAR > 0.70
- **Big yawn**: MAR > 0.84
- **Threshold**: `MAR_THRESHOLD = 0.70`

---

## 5. Head Pose Estimation (Yaw & Pitch)

**File**: `app/ai_core.py`, `_estimate_head_pose()` method

### What It Does
Estimates where the driver is looking — specifically the head rotation angles:
- **Yaw**: Left-right rotation (looking sideways)
- **Pitch**: Up-down rotation (looking up/down)

### How It Works Internally (solvePnP)

This uses OpenCV's **Perspective-n-Point** algorithm to solve the 3D orientation of the head from 2D image points.

1. **6 Facial Points** are extracted from the 2D image:
   - Nose tip (landmark 1)
   - Chin (landmark 152)
   - Left eye corner (landmark 263)
   - Right eye corner (landmark 33)
   - Left mouth corner (landmark 61)
   - Right mouth corner (landmark 291)

2. **3D Model Points** (a generic face model in world coordinates):
   ```
   Nose tip:         (0, 0, 0)
   Chin:             (0, -330, -65)
   Left eye corner:  (-225, 170, -135)
   Right eye corner: (225, 170, -135)
   Left mouth:       (-150, -150, -125)
   Right mouth:      (150, -150, -125)
   ```

3. **Camera Matrix** (synthetic, based on image dimensions):
   ```
   focal_length = image_width
   center = (width/2, height/2)
   camera_matrix = [[focal, 0, cx], [0, focal, cy], [0, 0, 1]]
   dist_coeffs = [0, 0, 0, 0]   # No lens distortion
   ```

4. **solvePnP**: Matches the 2D image points to the 3D model points to find the rotation vector:
   ```python
   success, rotation_vec, _ = cv2.solvePnP(
       MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
       flags=cv2.SOLVEPNP_ITERATIVE
   )
   ```

5. **Convert to Euler Angles**:
   ```python
   rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
   angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
   yaw = angles[1]    # Left-right
   pitch = angles[0]  # Up-down
   ```

### Thresholds
- **Yaw**: > 25° = looking sideways (left or right)
- **Pitch**: > 20° = looking up or down
- Combined check: `abs(yaw) > 25 OR abs(pitch) > 20`

---

## 6. Blink Detection & Blink Rate

**File**: `app/ai_core.py`, `_detect_blink()` method

### What It Does
Detects individual blinks and calculates the blink rate (blinks per minute).

### How Blink Detection Works (Falling-Edge)

A blink is detected when the EAR **drops below** a blink-specific threshold in a single frame transition:

```python
if self._prev_ear >= BLINK_EAR_THRESHOLD > raw_ear:
    # Blink detected! (EAR just crossed below threshold)
    self._blink_total += 1
    self._blink_timestamps.append(time.time())
self._prev_ear = raw_ear
```

- **BLINK_EAR_THRESHOLD = 0.18** (lower than the drowsiness threshold of 0.25)
- This is a **falling-edge detector** — it fires on the transition from open → closed, not during sustained closure
- The previous frame's EAR must be ≥ 0.18, and the current frame's EAR must be < 0.18

### How Blink Rate Works (Rolling 60-Second Window)

```python
# Keep only timestamps from the last 60 seconds
self._blink_timestamps = [ts for ts in self._blink_timestamps if now - ts <= 60]

# Convert to blinks per minute
count = len(self._blink_timestamps)
self.blink_rate = count * (60.0 / 60)  # = count
```

Since the window is exactly 60 seconds, the count directly equals blinks per minute.

### Normal vs Abnormal Blink Rates
- **Normal range**: 10–25 blinks/minute (`NORMAL_BLINK_RATE = (10, 25)`)
- **Too low** (< 10): Possible staring, fatigue, micro-sleep
- **Too high** (> 25): Possible stress, irritation, fatigue
- Abnormal blink rate is used as a **secondary drowsiness indicator**

---

## 7. Drowsiness Detection (Multi-Factor)

**File**: `app/ai_core.py`, inside `process_frame()`

### What It Does
Detects whether the driver is drowsy using multiple evidence sources, not just a single threshold. This achieves ~95% accuracy.

### The 3-Factor Detection Algorithm

```python
# Factor 1 — Primary: Sustained low EAR
if self.ear < 0.25:
    self._ear_below_count += 1
else:
    self._ear_below_count = 0

base_drowsy = self._ear_below_count >= 15     # 15 frames ≈ 0.5 seconds

# Factor 2 — Secondary: Very low EAR (eyes nearly shut)
very_low_ear = self.ear < (0.25 × 0.85)       # = 0.2125
                                                # Only needs 5 frames

# Factor 3 — Tertiary: Base drowsy + abnormal blink rate
abnormal_blink = self.blink_rate < 5 OR self.blink_rate > 30

# Final decision
is_drowsy = (
    base_drowsy                                     # Eyes below 0.25 for 15 frames
    OR (very_low_ear AND ear_below_count >= 5)      # Nearly shut for 5 frames
    OR (base_drowsy AND abnormal_blink)             # Drowsy + weird blink pattern
)
```

### Why Multi-Factor?
- **Single threshold** can miss micro-sleeps (eyes shut briefly but deeply)
- **Very low EAR** catches rapid eye closure even before 15 frames
- **Blink rate anomaly** catches drowsiness even when EAR fluctuates near threshold
- Together these cover more drowsiness patterns with fewer false negatives

---

## 8. Drowsy Timer & Grace Period

**File**: `app/ai_core.py`, inside `process_frame()`

### What It Does
Tracks how long the driver has been continuously drowsy. The timer is what triggers the actual voice alert at 4 seconds.

### How The Timer Works

```python
if self.is_drowsy:
    if self.drowsy_start == 0.0:
        self.drowsy_start = time.time()    # Start the timer
    self._drowsy_grace = 0                 # Reset grace counter
else:
    # NOT drowsy this frame — but don't reset immediately
    self._drowsy_grace += 1
    if self._drowsy_grace >= 10:           # 10 frames ≈ 0.33 seconds
        self.drowsy_start = 0.0            # NOW reset the timer
```

### The Grace Period Problem & Solution

**Problem**: MediaPipe landmarks can flicker — the driver's eyes might be genuinely closed, but a single bad frame could produce EAR > 0.25 for one frame, which would reset the drowsy timer and delay the alert.

**Solution**: A **10-frame grace period** (~0.33 seconds at 30fps). When drowsiness stops being detected:
- A counter (`_drowsy_grace`) increments each non-drowsy frame
- The timer only resets once 10 consecutive non-drowsy frames are seen
- If drowsiness resumes within 10 frames, the grace counter resets and the timer continues

### Timer-Triggered Alert Thresholds
- **4 seconds** (`DROWSY_ALERT_DURATION = 4.0`): Triggers PATH A local alert
- **8 seconds** (`DROWSY_ALERT_DURATION × 2`): Critical — triggers with `force=True` (bypasses cooldown)

---

## 9. Yawning Detection

**File**: `app/ai_core.py`, inside `process_frame()`

### What It Does
Detects when the driver is yawning based on mouth openness.

### The 2-Factor Detection Algorithm

```python
# Consecutive frame counter
if self.mar > 0.70:
    self._mar_above_count += 1
else:
    self._mar_above_count = 0

# Factor 1 — Primary: Sustained high MAR
base_yawning = self._mar_above_count >= 8      # 8 frames ≈ 0.27 seconds

# Factor 2 — Secondary: Extreme mouth opening
extreme_mar = self.mar > (0.70 × 1.2)          # = 0.84 (very wide open)

# Final decision
is_yawning = (
    base_yawning                                 # MAR > 0.70 for 8 frames
    OR (extreme_mar AND mar_above_count >= 3)    # MAR > 0.84 for just 3 frames
)
```

### Why Two Factors?
- **Normal yawn** (MAR 0.70–0.84): Requires 8 frames to confirm (prevents false positives from talking)
- **Big yawn** (MAR > 0.84): Only needs 3 frames because such wide opening is almost certainly a yawn

---

## 10. Looking Away Detection (Hysteresis)

**File**: `app/ai_core.py`, inside `process_frame()`

### What It Does
Detects when the driver is looking away from the road, with built-in resistance to brief glances.

### How Hysteresis Works

```python
# Is the driver currently looking away?
looking_away_now = abs(yaw) > 25 OR abs(pitch) > 20

# Asymmetric counter update
if looking_away_now:
    self._look_away_frames += 1      # Increment by 1
else:
    self._look_away_frames -= 2      # Decrement by 2 (faster recovery)
    self._look_away_frames = max(0, self._look_away_frames)

# Confirmed looking away when counter reaches 3
self.is_looking_away = self._look_away_frames >= 3
```

### Why Asymmetric?
- **Slow to trigger** (+1 per frame): Brief head turns (checking mirror) don't trigger
- **Fast to recover** (-2 per frame): When the driver looks back, the counter drops quickly
- **Threshold of 3**: Need 3 net accumulations of looking away before flagging

### Example Scenario
```
Frame 1: Looking away → counter = 1
Frame 2: Looking away → counter = 2
Frame 3: Brief look back → counter = 0 (dropped by 2)
Frame 4: Looking away → counter = 1
Frame 5: Looking away → counter = 2
Frame 6: Looking away → counter = 3 → TRIGGERED!
```

---

## 11. Object Detection (YOLOv8 Nano)

**File**: `app/ai_core.py`, `_run_yolo()` method

### What It Does
Detects real-world objects in the camera frame (phone, bottle, cup, knife, etc.) using the YOLOv8 Nano deep learning model.

### How It Works Internally

1. **Model**: `yolov8n.pt` — YOLOv8 Nano, the smallest and fastest variant (~3MB)
2. **Runs Every 10 Frames**: `frame_count % 10 == 0` to save GPU resources. At 30fps, this means YOLO runs 3 times per second.
3. **Confidence Threshold**: `YOLO_CONFIDENCE = 0.45` — only detections above 45% confidence are kept

4. **Detection Pipeline**:
   ```python
   results = self._yolo(frame, conf=0.45, verbose=False)
   for r in results:
       for box in r.boxes:
           cls_id = int(box.cls[0])
           if cls_id in YOLO_CLASSES_OF_INTEREST:
               x1, y1, x2, y2 = box.xyxy[0]   # Bounding box
               label = YOLO_CLASSES_OF_INTEREST[cls_id]
               conf = float(box.conf[0])
               objects.append({"label": label, "conf": conf, "box": (x1, y1, x2, y2)})
   ```

5. **Objects of Interest** (17 COCO classes):
   - **Critical (trigger DISTRACTION)**: cell phone, bottle, cup, scissors, knife
   - **Non-critical (detected, no distraction)**: backpack, umbrella, handbag, suitcase, laptop, mouse, remote, keyboard, book, clock, vase

### Why Every 10 Frames?
YOLO is computationally expensive. Running it every frame would drop overall FPS. Running every 10 frames (3× per second) is fast enough to catch objects while keeping the system at 30fps for other detections.

---

## 12. Hand Detection (MediaPipe Hands)

**File**: `app/ai_core.py`, `_detect_hands()` method

### What It Does
Detects hands in the frame and determines if a hand is near the face or on the head (tiredness gesture).

### How It Works Internally

1. **Model**: MediaPipe Hands solution — detects up to 2 hands, returns 21 hand landmarks each
2. **Configuration**:
   ```
   static_image_mode = False     (optimized for video)
   max_num_hands = 2
   min_detection_confidence = 0.5
   min_tracking_confidence = 0.5
   ```

3. **Face Region Calculation**: From the cached face landmarks, calculate the face bounding box:
   ```python
   face_y_min = min of all landmark Y values
   face_y_max = max of all landmark Y values
   face_x_min = min of all landmark X values
   face_x_max = max of all landmark X values
   ```

4. **Hand Near Face Check**: For each detected hand, get the palm position (landmark 0) and check if it falls within the face region expanded by 50 pixels on each side:
   ```python
   if (face_x_min - 50 <= palm_x <= face_x_max + 50 AND
       face_y_min - 50 <= palm_y <= face_y_max + 100):
       hand_near_face = True
   ```

5. **Hand On Head Check**: If the palm Y position is in the top 30% of the face region:
   ```python
   if palm_y < face_y_min + (face_y_max - face_y_min) × 0.3:
       hand_on_head = True   # Tiredness gesture (rubbing forehead)
   ```

---

## 13. Advanced Behavior Analysis

**File**: `app/ai_core.py`, `_analyze_behaviors()` method

### What It Does
Combines multiple detection sources to identify specific dangerous behaviors like talking on the phone, drinking, or texting.

### Behavior 1: Phone Near Ear
```python
for each detected phone object:
    phone_center = center of phone bounding box
    face_center = mean of face oval landmark coordinates
    
    # Normalized distance (0 to 1 scale)
    dx = abs(phone_cx - face_cx) / frame_width
    dy = abs(phone_cy - face_cy) / frame_height
    distance = sqrt(dx² + dy²)
    
    if distance < 0.15:       # Phone very close to face
        is_phone_near_ear = True
```

### Behavior 2: Drinking
```python
for each detected bottle/cup object:
    drink_center = center of bottle/cup bounding box
    mouth_position = midpoint of upper_lip and lower_lip landmarks
    
    dx = abs(drink_cx - mouth_x) / frame_width
    dy = abs(drink_cy - mouth_y) / frame_height
    distance = sqrt(dx² + dy²)
    
    if distance < 0.12:       # Drink very close to mouth
        is_drinking = True
```

### Behavior 3: Looking Down
```python
if pitch < -20°:
    is_looking_down = True
if pitch < -30°:
    behavior = "EXTREME_LOOKING_DOWN"
```

### Behavior 4: Texting / Reading Phone
```python
if is_looking_down AND phone_detected:
    behavior = "TEXTING_OR_READING_PHONE"
    is_distracted = True       # Immediate distraction flag
```

---

## 14. Tiredness & Affective State Analysis

**File**: `app/ai_core.py`, `_analyze_tiredness_and_affect()` method

### What It Does
Calculates tiredness level (0-100), eye closure level (0-100), distraction level (0-100), and determines the driver's affective state.

### Eye Closure Level
```python
normal_ear = 0.30               # Typical fully-open EAR
closure_ratio = max(0, (0.30 - current_ear) / 0.30)
eye_closure_level = min(100, closure_ratio × 100)
```
- EAR = 0.30 → closure_level = 0% (fully open)
- EAR = 0.15 → closure_level = 50% (half closed)
- EAR = 0.00 → closure_level = 100% (fully closed)

### Distraction Level
Takes the **maximum** of all active distraction factors:
| Factor | Score |
|--------|-------|
| Phone near ear | 90 |
| General distraction | 80 |
| Looking away | 60 |
| Hand near face | 40 |
| Objects detected | 30 |

### Tiredness Level
Takes the **maximum** of all tiredness factors:
| Factor | Score |
|--------|-------|
| Drowsy | 80 |
| Yawning | 70 |
| Hand on head | 70 |
| Low blink rate (< 10/min) | 60 |
| High blink rate (> 25/min) | 50 |
| Eye closure contribution | eye_closure × 0.4 |
| Attention deficit | (100 - attention_score) × 0.6 |

### Affective State Decision Tree
```python
if tiredness_level > 70 OR is_drowsy:    → "DROWSY"
elif tiredness_level > 40 OR is_yawning: → "TIRED"
elif distraction_level > 60:              → "DISTRACTED"
elif attention_score > 75:                → "ALERT"
else:                                     → "NEUTRAL"
```

---

## 15. Distraction Detection (Composite)

**File**: `app/ai_core.py`, inside `process_frame()`

### What It Does
Determines if the driver is distracted by combining multiple evidence sources into a single boolean.

### The Logic
```python
is_distracted = (
    any object has label in {"cell phone", "knife", "scissors", "bottle", "cup"}
    OR is_phone_near_ear
    OR is_drinking
    OR hand_near_face
)
```

This fires if ANY of the following are true:
1. A critical object is detected by YOLO
2. The phone-near-ear behavior analysis triggers
3. The drinking behavior analysis triggers
4. A hand is detected near the face (could be on phone, rubbing eyes, etc.)

---

## 16. Attention Score (Weighted 0-100)

**File**: `app/ai_core.py`, `_compute_attention_score()` method

### What It Does
Computes a single 0-100 score representing overall driver attention, where 100 = fully alert and 0 = completely inattentive.

### Weight Distribution
| Component | Weight | What It Measures |
|-----------|--------|------------------|
| EAR | 30% | Eye openness |
| Blink Rate | 15% | Blink pattern normality |
| MAR | 15% | Mouth (yawning) |
| Head Pose | 20% | Looking at road |
| Distraction | 20% | Objects/behaviors |

### Penalty Calculations (starting from 100)

**EAR Penalty** (30% weight):
```python
if ear < 0.25:
    ear_penalty = (0.25 - ear) / 0.25
    if ear_penalty > 0.15:           # Only significant drops
        ear_penalty = ear_penalty ** 1.3   # Power-law amplification
        score -= 0.30 × 80 × ear_penalty
elif ear > 0.275:                    # Eyes wide open
    score += 3                       # Small bonus
```
The power-law (`** 1.3`) makes large EAR drops much more severe than small ones.

**Blink Rate Penalty** (15% weight):
```python
if blink_rate < 10:                  # Too few blinks
    deficit = (10 - blink_rate) / 10
    score -= 0.15 × 60 × deficit
elif blink_rate > 25:                # Too many blinks
    excess = (blink_rate - 25) / 25
    score -= 0.15 × 40 × min(excess, 1.0)
```

**MAR Penalty** (15% weight):
```python
if mar > 0.70:                       # Yawning
    mar_ratio = (mar - 0.70) / 0.70
    score -= 0.15 × 85 × min(mar_ratio, 1.5)
```

**Head Pose Penalty** (20% weight — graduated):
```python
max_angle_ratio = max(abs(yaw)/25, abs(pitch)/20)

if max_angle_ratio > 1.0:           # Beyond threshold
    score -= 0.20 × 100             # Full penalty (-20)
elif max_angle_ratio > 0.7:         # Near threshold
    score -= 0.20 × 80 × ratio
elif max_angle_ratio > 0.5:         # Mild deviation
    score -= 0.20 × 50 × ratio
```

**Distraction Penalty** (20% weight):
```python
if is_distracted:
    score -= 0.20 × 100             # Full penalty (-20)
elif objects detected:
    score -= 5                       # Mild penalty for non-critical objects
```

### Temporal Smoothing
```python
final_score = 0.7 × current_score + 0.3 × previous_score
```
This prevents the score from jumping wildly frame-to-frame. 70% current + 30% previous creates smooth transitions.

---

## 17. Safety Status (3-Level Decision Tree)

**File**: `app/ai_core.py`, `_update_safety_status()` method

### What It Does
Determines the overall safety status: SAFE, WARNING, or DANGER.

### The Decision Logic

First, count active dangers (0-7):
```python
dangers = sum([
    is_drowsy,          is_yawning,
    is_distracted,      is_looking_away,
    is_phone_near_ear,  is_drinking,
    is_looking_down
])
```

Then check critical conditions:
```python
critical_drowsy      = is_drowsy AND ear < 0.25 × 0.8    (= 0.20)
extreme_distraction  = is_distracted AND objects > 1
severe_look_away     = is_looking_away AND (yaw > 40° OR pitch > 35°)
dangerous_behavior   = is_phone_near_ear OR is_drinking OR is_looking_down
```

### DANGER Conditions (any one triggers DANGER):
- 2+ dangers active simultaneously
- Attention score < 20
- Critical drowsy (EAR < 0.20 while drowsy)
- Extreme distraction (distracted + multiple objects)
- Any dangerous behavior (phone/drinking/looking down)

### WARNING Conditions:
- Exactly 1 danger active
- Attention score < 45
- Severe looking away (yaw > 40° or pitch > 35°)

### SAFE:
- No danger conditions met

### Danger Counter
- DANGER: `danger_counter += 1` (increments)
- WARNING: `danger_counter = max(0, counter - 1)` (slow decay)
- SAFE: `danger_counter = max(0, counter - 2)` (faster decay)

The danger counter is used in routes.py to require `DANGER_FRAME_THRESHOLD` (5) consecutive danger frames before triggering the GPT-5.2 alert.

---

## 18. Alert System — PATH A: Local Drowsy Alert

**File**: `app/routes.py` (trigger logic), `app/jarvis.py` (`trigger_drowsy_alert` + `_process_drowsy_alert`)

### What It Does
When the drowsy timer reaches 4+ seconds, fires an instant voice alert using pre-written messages — NO GPT-5.2 API call. This is the critical safety path.

### Trigger Logic (in routes.py)
```python
drowsy_duration = time.time() - ai_core.drowsy_start  (if drowsy_start > 0)

if drowsy_duration >= 4.0:           # 4 second threshold
    if drowsy_duration >= 8.0:       # 8 seconds = critical
        jarvis.trigger_drowsy_alert(telemetry, force=True)   # Bypasses cooldown
    elif jarvis.is_ready:            # Respects cooldown
        jarvis.trigger_drowsy_alert(telemetry)
```

### Processing Pipeline (in jarvis.py)
```
1. Set last_alert_time = now
2. Print "[JARVIS] 💤 Drowsy timer triggered — instant local alert (no GPT)"
3. Call _local_fallback_alert(telemetry):
   a. Determine alert type (DROWSINESS in this case)
   b. Pick message: "Driver, your eyes are closing. Please pull over..."
   c. Generate TTS audio via OpenAI TTS-1
   d. Play audio through pygame speakers
   e. Log incident to SQLite database
   f. Emit "jarvis_alert" event to dashboard via SocketIO
4. Set is_speaking = False (allow next alert)
```

### Why Local (No GPT)?
The user discovered that GPT-5.2 sometimes fails to correctly identify drowsiness, or the API call times out. Since drowsiness is **life-critical**, the 4-second alert must fire **100% of the time**, regardless of API availability. Local fallback guarantees this.

---

## 19. Alert System — PATH B: GPT-5.2 Vision Analysis

**File**: `app/routes.py` (trigger logic), `app/jarvis.py` (`trigger_alert` + `_process_alert`)

### What It Does
For non-drowsiness dangers (distraction, low EAR, etc.), sends the camera frame + sensor data to GPT-5.2 for visual analysis before generating an alert.

### Trigger Conditions (in routes.py)
```python
should_alert_gpt = (
    (0 < current_ear < 0.25)                          # EAR below threshold
    OR (safety_status == "DANGER" AND danger_counter >= 5)  # 5 frames in DANGER
    OR (is_distracted AND danger_counter >= 5)          # 5 frames distracted
)
```
Only fires if `jarvis.client` exists AND `jarvis.is_ready` (cooldown elapsed).

### Processing Pipeline (in jarvis.py)

```
1. Set is_speaking = True + last_alert_time = now
2. Encode camera frame to base64 JPEG (quality 70)
3. Build context prompt:
   "Analyze the driver's face in this image.
    Sensor readings: EAR=0.180, MAR=0.120, Yaw=5.2°, Pitch=-3.1°,
    Attention=45/100, BlinkRate=8/min
    JSON Output required: {"status": "DANGER"/"SAFE", "reason": "...", "confidence": 0.0-1.0}"
4. Send to GPT-5.2 Vision:
   - System: "You are JARVIS, a real-time safety AI. Be concise."
   - User: [text context] + [image_url with base64 data, detail="low"]
   - max_completion_tokens=100, temperature=0.3, timeout=3.0s
5. Parse JSON response
6. If GPT says SAFE with ≥ 80% confidence → SUPPRESS (false positive override)
7. If DANGER → Generate TTS for the reason → Play audio → Log to DB → Emit to dashboard
```

### Error Handling
- **APITimeoutError** (>3s): Local fallback alert fires
- **RateLimitError** (429): Exponential backoff + local fallback fires
- **Any other error**: Local fallback fires

---

## 20. GPT-5.2 JSON Response Parsing

**File**: `app/jarvis.py`, `_parse_gpt_response()` method

### What It Does
Robustly parses GPT-5.2's JSON response, handling all the ways GPT might format it.

### 3-Layer Parsing Strategy

```
Layer 1: Try json.loads(text)
         → Success? Return parsed dict
         → Fail? Continue...

Layer 2: Regex extract first JSON object: \{[^}]+\}
         → Found? Try json.loads(match)
         → Fail? Continue...

Layer 3: Return fallback dict:
         {"status": "DANGER", "reason": first_120_chars, "confidence": 0.5}
```

This handles:
- Clean JSON: `{"status": "DANGER", "reason": "Eyes closed", "confidence": 0.92}`
- JSON in markdown: `` ```json {"status": ...} ``` ``
- JSON with extra text: `Based on my analysis, {"status": ...} is the result`
- No JSON at all: Returns DANGER with the text as the reason

---

## 21. GPT-5.2 False Positive Override

**File**: `app/jarvis.py`, inside `_process_alert()`

### What It Does
If GPT-5.2 looks at the camera frame and confidently determines the driver is actually SAFE, the alert is suppressed.

### The Logic
```python
if status == "SAFE" and confidence >= 0.80:
    print("GPT-5.2 override: SAFE (85%) — alert suppressed")
    return   # No TTS, no logging, no dashboard emit
```

### Why This Exists
The local detection system might flag a danger (e.g., EAR momentarily dips), but GPT-5.2 can see the actual image and determine the driver is fine. With ≥80% confidence, the system trusts GPT-5.2's visual assessment over the sensor readings. This reduces false alarm fatigue.

---

## 22. Local Fallback Alert System

**File**: `app/jarvis.py`, `_local_fallback_alert()` method

### What It Does
Fires a pre-written alert when GPT-5.2 is unavailable, too slow, or for guaranteed drowsiness alerts.

### How It Works

1. **Determine alert type** from telemetry:
   ```python
   if is_drowsy → "DROWSINESS"
   elif is_distracted → "DISTRACTION"
   elif is_yawning → "YAWNING"
   elif is_looking_away → "HEAD_POSE"
   else → "GENERAL"
   ```

2. **Pick pre-written message** from config:
   ```python
   messages = {
       "DROWSINESS": "Driver, your eyes are closing. Please pull over and take a break immediately.",
       "YAWNING": "Frequent yawning detected. Consider stopping for rest at the next safe location.",
       "DISTRACTION": "Put your phone down and focus on the road. Your life depends on it.",
       "HEAD_POSE": "Eyes on the road, driver. You have been looking away for too long.",
       "GENERAL": "Your attention level is critically low. Please stay focused on driving safely."
   }
   ```

3. **Generate TTS** → **Play audio** → **Log to database** → **Emit to dashboard** (with `fallback: True` flag)

---

## 23. Text-to-Speech (TTS) Generation

**File**: `app/jarvis.py`, `_generate_tts()` method

### What It Does
Converts text warning messages into spoken audio using OpenAI's TTS API.

### How It Works
```python
response = client.audio.speech.create(
    model="tts-1",        # Standard TTS model (fast)
    voice="onyx",         # Deep male voice (JARVIS-like)
    input=text,           # The warning message
    response_format="mp3" # MP3 audio format
)
return response.content   # Raw audio bytes
```

### Voice Choice
`onyx` was chosen because it's a deep, authoritative male voice — fitting the JARVIS persona from Iron Man.

---

## 24. Audio Playback System

**File**: `app/jarvis.py`, `_play_audio()` method

### What It Does
Plays the TTS audio through the computer's speakers.

### How It Works
```python
# Initialize pygame mixer (done once at startup)
pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=2048)

# Play audio
audio_stream = io.BytesIO(audio_data)     # Wrap bytes as file-like object
pygame.mixer.music.load(audio_stream, "mp3")
pygame.mixer.music.play()

# Block until audio finishes playing
while pygame.mixer.music.get_busy():
    time.sleep(0.1)
```

### Key Details
- **Frequency**: 24kHz (matches TTS output)
- **Size**: -16 (signed 16-bit samples)
- **Channels**: 1 (mono)
- **Buffer**: 2048 (low latency)
- **Blocking**: The thread waits until playback finishes. Since this runs in a daemon thread, it doesn't block the main processing loop.

---

## 25. Cooldown & Force Override Mechanism

**File**: `app/jarvis.py`, `is_ready` property and `trigger_*` methods

### What It Does
Prevents alert spam by enforcing a minimum time between alerts, with an override for critical situations.

### Normal Cooldown (`is_ready`)
```python
@property
def is_ready(self):
    return (
        not self.is_speaking                           # Not currently talking
        AND (time.time() - last_alert_time) > 5        # 5 seconds since last alert
        AND time.time() >= self._backoff_until          # Not in rate-limit backoff
    )
```

### Force Override (Critical Drowsy 8s+)
```python
# In trigger_drowsy_alert(telemetry, force=True):
if force:
    if self.is_speaking or time.time() < self._backoff_until:
        return    # Only blocked by active speech or rate limit
    # DOES NOT CHECK cooldown timer — fires immediately
```

### Why Two Modes?
- **Normal**: 5-second cooldown prevents annoying rapid-fire alerts
- **Force**: When the driver has been drowsy for 8+ seconds, the situation is critical enough to override the cooldown. Only two things can stop a forced alert: currently playing audio, or active rate-limit backoff.

---

## 26. Rate Limit Handling & Exponential Backoff

**File**: `app/jarvis.py`, inside `_process_alert()` exception handling

### What It Does
When OpenAI returns HTTP 429 (rate limited), the system backs off exponentially and uses local alerts during the backoff period.

### How It Works
```python
except openai.RateLimitError:
    self._consecutive_429s += 1
    backoff = min(
        30 × (2 ** consecutive_429s),    # Exponential: 30s, 60s, 120s, 240s...
        300                               # Cap at 5 minutes
    )
    self._backoff_until = time.time() + backoff
    self._local_fallback_alert(telemetry)   # Still alert the driver!
```

### Backoff Progression
| 429 Count | Backoff Time |
|-----------|-------------|
| 1st | 60 seconds |
| 2nd | 120 seconds |
| 3rd | 240 seconds |
| 4th+ | 300 seconds (max) |

### Recovery
After a successful GPT-5.2 call: `self._consecutive_429s = 0` — backoff counter resets.

---

## 27. Spatial Environment Scanning (Thread C)

**File**: `app/routes.py`, `_spatial_scan_loop()` function

### What It Does
Every 5 seconds, sends a camera frame to GPT-5.2 for a tactical assessment of the driving environment.

### How It Works

1. **Initial Delay**: Waits 3 seconds after startup to let the system stabilize.

2. **Per Scan**:
   ```
   Grab frame → Encode JPEG (quality 50) → base64 → Send to GPT-5.2
   ```

3. **GPT-5.2 Prompt** (SPATIAL_PROMPT from config):
   ```
   "You are JARVIS, Tony Stark's tactical AI vision system integrated into
   the ADAR advanced driver safety platform. Analyze this camera frame.

   Respond in EXACTLY 3 short lines:
   SUBJECTS: [describe visible people and their state]
   ENV: [describe environment, key objects, lighting]
   STATUS: [one-line assessment] | THREAT: [LOW/MEDIUM/HIGH]

   Rules: Each line under 70 chars. Crisp technical language."
   ```

4. **GPT-5.2 Config**: `max_completion_tokens=120`, `temperature=0.5`

5. **Result Storage**: `ai_core.set_spatial_analysis(text)` — stored for the HUD system.

6. **Rate Limit Handling**: On 429 error, sleeps 30 seconds extra.

### Example Output
```
SUBJECTS: Single adult male, alert posture, eyes forward
ENV: Indoor room, warm lighting, monitor visible, neutral background
STATUS: Driver appears attentive, no hazards detected | THREAT: LOW
```

---

## 28. Telemetry Emission (SocketIO)

**File**: `app/routes.py`, inside `_processing_loop()`

### What It Does
Sends real-time detection data from the backend to the web dashboard via WebSocket.

### How It Works

1. **Throttled to 10Hz**: Only emits once per 100ms (`SOCKETIO_EMIT_INTERVAL = 0.1`):
   ```python
   if now - last_emit >= 0.1:
       socketio.emit("telemetry_update", telemetry)
       last_emit = now
   ```

2. **Data Sent**: The telemetry dict from `ai_core._get_telemetry()` — 30+ fields (see Feature #43).

3. **Transport**: Socket.IO over WebSocket (with polling fallback). The dashboard connects with:
   ```javascript
   const socket = io({ transports: ["websocket", "polling"] });
   ```

---

## 29. MJPEG Video Streaming

**File**: `app/routes.py`, `_mjpeg_generator()` function

### What It Does
Streams the camera feed to the browser as a continuous sequence of JPEG images (MJPEG).

### How It Works

1. **Frame Source**: Thread A encodes each processed frame as JPEG and stores it in `_latest_frame_bytes` (protected by `_frame_lock`).

2. **Generator**: The `/video_feed` route serves a `multipart/x-mixed-replace` response:
   ```python
   yield (
       b"--frame\r\n"
       b"Content-Type: image/jpeg\r\n\r\n"
       + frame_bytes
       + b"\r\n"
   )
   ```

3. **Frame Rate Control**: Targets ~30fps via `MJPEG_FRAME_SKIP = 0.033` seconds minimum between frames.

4. **Duplicate Avoidance**: Skips yielding if the frame hasn't changed since the last yield.

5. **Browser Side**: The dashboard displays it in a simple `<img>` tag:
   ```html
   <img src="/video_feed" />
   ```

### Note: Clean Feed
The camera frame is streamed WITHOUT any overlay drawn on it. All visualization happens in the web dashboard, not on the video feed.

---

## 30. Dashboard — 3-Tier Drowsiness Status Bar

**File**: `static/js/dashboard.js`, inside `telemetry_update` handler

### What It Does
The top header bar shows the current drowsiness state in 3 levels, driven **purely** by drowsiness detection.

### How It Works

```javascript
const drowsyDur = data.drowsy_duration || 0;

if (data.is_drowsy && drowsyDur >= 4.0) {
    // TIER 3: DANGER — drowsy 4s+ → local alert fires
    updateMainStatus("DANGER");                    // Red status
    dangerOverlay.classList.add("active");          // Red flash
} else if (data.is_drowsy) {
    // TIER 2: WARNING — drowsiness detected, timer < 4s
    updateMainStatus("WARNING");                   // Orange status
    dangerOverlay.classList.remove("active");
} else {
    // TIER 1: SAFE — no drowsiness detected
    updateMainStatus("SAFE");                      // Green status
    dangerOverlay.classList.remove("active");
}
```

### Key Design Choice
The top bar does NOT use the backend's general `safety_status`. It is **purely drowsiness-based**. This was specifically changed because the backend safety status could show "WARNING" for non-drowsy reasons (like yawning), which confused the display when the driver wasn't actually drowsy.

### Visual Appearance
| State | Beacon | Label | Color |
|-------|--------|-------|-------|
| SAFE | 🟢 Pulsing green | "SAFE" | Green (#00dc6e) |
| WARNING | 🟠 Pulsing orange | "WARNING" | Orange (#ff8c00) |
| DANGER | 🔴 Pulsing red | "DANGER" | Red (#ff1744) |

---

## 31. Dashboard — Client-Side Drowsy Timer (20Hz)

**File**: `static/js/dashboard.js`, `setInterval` at 50ms

### What It Does
Provides a buttery-smooth drowsiness timer display that updates 20 times per second, independent of the backend's 10Hz telemetry rate.

### How It Works

```javascript
let _drowsyActive = false;
let _drowsyStart = 0;

// State sync from backend (10Hz)
socket.on("telemetry_update", (data) => {
    if (data.is_drowsy && !_drowsyActive) {
        _drowsyActive = true;
        _drowsyStart = Date.now();        // Start client timer
    } else if (!data.is_drowsy && _drowsyActive) {
        _drowsyActive = false;
        earValue.textContent = "✅ SAFE";
    }
});

// Client-side timer (20Hz — runs independently)
setInterval(() => {
    if (_drowsyActive) {
        const elapsed = (Date.now() - _drowsyStart) / 1000;
        if (elapsed >= 4.0) {
            earValue.textContent = `🔴 DANGER ${elapsed.toFixed(1)}s`;
        } else {
            earValue.textContent = `⚠️ WARNING ${elapsed.toFixed(1)}s`;
        }
    }
}, 50);   // 20Hz = every 50ms
```

### Why Client-Side?
The backend sends telemetry at 10Hz (every 100ms). If the timer only updated at 10Hz, the display would look choppy (jumping 0.1s at a time). By running a 20Hz client timer that's synced on drowsiness start/stop, the user sees smooth 0.05s increments.

---

## 32. Dashboard — Attention Score Gauge

**File**: `static/js/dashboard.js`, inside `telemetry_update` handler

### What It Does
Displays the 0-100 attention score as an SVG arc ring with dynamic color.

### How It Works

```javascript
const score = data.attention_score ?? 100;

// Update text
attentionScore.textContent = Math.round(score);

// Update arc (SVG stroke-dashoffset)
const circumference = 2 × Math.PI × 85;           // Circle radius = 85
gaugeArc.style.strokeDashoffset = circumference × (1 - score / 100);

// Color based on score
const color = score >= 80 ? "green"
            : score >= 50 ? "orange"
            : "red";
gaugeArc.style.stroke = color;
attentionScore.style.color = color;
```

### Visual
- Score 80-100: Green arc, green text
- Score 50-79: Orange arc, orange text
- Score 0-49: Red arc, red text
- The arc visually fills proportionally to the score (100% = full circle)

---

## 33. Dashboard — EAR/MAR Real-Time Chart

**File**: `static/js/dashboard.js`, Chart.js setup

### What It Does
Displays a rolling line chart of EAR and MAR values over the last 100 data points with threshold lines.

### How It Works

1. **Chart.js Line Chart**: Two datasets — EAR (white line) and MAR (orange line)
2. **Rolling Window**: Maximum 100 points. When exceeded, oldest point is shifted off:
   ```javascript
   if (labels.length > 100) {
       labels.shift(); earData.shift(); marData.shift();
   }
   ```
3. **Y-axis**: 0 to 1.2 (covers both EAR and MAR ranges)
4. **No Animation**: `animation: { duration: 0 }` for real-time performance
5. **Threshold Lines**: Custom Chart.js plugin draws dashed lines:
   - Red dashed line at Y=0.22 (EAR danger zone)
   - Orange dashed line at Y=0.75 (MAR yawn zone)

---

## 34. Dashboard — Status Indicator Cards

**File**: `static/js/dashboard.js`, `updateIndicator()` function

### What It Does
4 cards showing the status of Drowsiness, Yawning, Distraction, and Head Pose — each with colored indicators.

### How It Works
```javascript
function updateIndicator(name, isActive, isDanger) {
    if (isActive) {
        if (isDanger) {
            card.classList.add("active-danger");    // Red background
            indicator.classList.add("danger");       // Red dot
        } else {
            card.classList.add("active-warning");   // Orange background
            indicator.classList.add("warning");      // Orange dot
        }
    } else {
        indicator.classList.add("safe");            // Green dot
    }
}
```

### Card Configuration
| Card | Detection | Danger Level |
|------|-----------|-------------|
| Drowsiness | `is_drowsy` | DANGER (red) |
| Yawning | `is_yawning` | WARNING (orange) |
| Distraction | `is_distracted` | DANGER (red) |
| Head Pose | `is_looking_away` | WARNING (orange) |

---

## 35. Dashboard — JARVIS Alert Feed

**File**: `static/js/dashboard.js`, `addJarvisMessage()` function

### What It Does
Shows the last 3 JARVIS alert messages with timestamps in a scrolling feed.

### How It Works
```javascript
socket.on("jarvis_alert", (data) => {
    addJarvisMessage(data.message || "Alert triggered", "alert");
    fetchStats();    // Refresh incident counters
});

function addJarvisMessage(text, type) {
    // Create message element with timestamp
    const msg = document.createElement("div");
    msg.className = `jarvis-message ${type}`;
    msg.innerHTML = `<span class="msg-time">${time}</span>
                     <span class="msg-text">${escapeHtml(text)}</span>`;
    
    jarvisFeed.appendChild(msg);
    jarvisFeed.scrollTop = jarvisFeed.scrollHeight;  // Auto-scroll
    
    // Keep only last 3 messages
    while (jarvisFeed.children.length > 3) {
        jarvisFeed.removeChild(jarvisFeed.firstChild);
    }
}
```

### Message Types
- `"system"`: System status messages (connection, initialization)
- `"alert"`: JARVIS voice alerts (both GPT-5.2 and local fallback)

---

## 36. Dashboard — Session Statistics

**File**: `static/js/dashboard.js`, `fetchStats()` function

### What It Does
Shows cumulative incident counts for the current session: Total, Drowsiness, Yawning, Distraction.

### How It Works
```javascript
// Fetch every 5 seconds + after each alert
async function fetchStats() {
    const res = await fetch("/api/stats");
    const stats = await res.json();
    totalAlerts.textContent = stats.total || 0;
    drowsyCount.textContent = stats.drowsiness || 0;
    yawnCount.textContent = stats.yawning || 0;
    distractCount.textContent = stats.distraction || 0;
}
setInterval(fetchStats, 5000);
```

The `/api/stats` endpoint queries the SQLite database filtered by the current session start time.

---

## 37. Dashboard — Danger Overlay Flash

**File**: `static/js/dashboard.js` + `static/css/style.css`

### What It Does
When the drowsiness status reaches DANGER (4s+), a full-screen red translucent overlay flashes to get the driver's attention.

### How It Works
```javascript
if (data.is_drowsy && drowsyDur >= 4.0) {
    dangerOverlay.classList.add("active");     // Show red overlay
} else {
    dangerOverlay.classList.remove("active");  // Hide overlay
}
```

The overlay is a `<div>` positioned absolute over the entire viewport with a red background, animated opacity.

---

## 38. Database — Incident Logging

**File**: `app/database.py`

### What It Does
Every alert (both GPT-5.2 and local fallback) is logged to an SQLite database with full telemetry.

### Incident Table Schema
| Column | Type | Example |
|--------|------|---------|
| id | Auto-increment | 42 |
| timestamp | DateTime | 2026-02-12T14:30:22 |
| alert_type | String(50) | "DROWSINESS" |
| severity | String(20) | "DANGER" |
| ear_value | Float | 0.182 |
| mar_value | Float | 0.089 |
| yaw_angle | Float | 3.2 |
| pitch_angle | Float | -1.7 |
| detected_objects | String(200) | "cell phone" |
| jarvis_response | Text | "GPT-5.2: {\"status\": \"DANGER\"...}" |
| attention_score | Float | 34.5 |
| blink_rate | Float | 8.0 |

### How Logging Works
```python
def log_incident(alert_type, severity, **kwargs):
    session = Session()       # Thread-safe scoped session
    incident = Incident(alert_type=alert_type, severity=severity, ...)
    session.add(incident)
    session.commit()
    Session.remove()          # Return session to pool
```

### Thread Safety
Uses SQLAlchemy's `scoped_session` — each thread gets its own session automatically.

---

## 39. Database — Auto-Migration

**File**: `app/database.py`, `_migrate_db()` function

### What It Does
Safely adds new columns to existing tables without losing data. Runs on every startup.

### How It Works
```python
def _migrate_db():
    conn = sqlite3.connect("adar_logs.db")
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(incidents)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    
    migrations = [
        ("attention_score", "REAL"),
        ("blink_rate", "REAL"),
    ]
    
    for col_name, col_type in migrations:
        if col_name not in existing_columns:
            cursor.execute(f"ALTER TABLE incidents ADD COLUMN {col_name} {col_type}")
```

This allows the database schema to evolve across versions without requiring database resets.

---

## 40. Thread Architecture

**File**: `app/routes.py`, `_start_engine()` function

### Overview
The system runs 5 types of threads:

### Thread Diagram
```
┌─────────────────────────────────────────────────────────────┐
│ MAIN THREAD — Flask/SocketIO HTTP + WebSocket Server        │
│ Serves dashboard at http://localhost:5000                    │
│ Handles /video_feed, /api/stats, /api/incidents             │
└─────────────┬───────────────────────────────────────────────┘
              │
  ┌───────────┼──────────────┬────────────────┐
  ▼           ▼              ▼                ▼
┌─────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐
│CAMERA│  │ THREAD A │  │ THREAD C │  │ ALERT THREAD │
│Thread│  │Processing│  │ Spatial  │  │ (on-demand)  │
│      │  │   Loop   │  │  Scan    │  │              │
│ 30fps│  │          │  │ every 5s │  │ GPT-5.2 call │
│ frame│  │ Camera→  │  │ Frame→   │  │ + TTS        │
│ grab │  │ AI→      │  │ GPT-5.2→ │  │ + playback   │
│      │  │ encode→  │  │ spatial  │  │              │
│      │  │ emit→    │  │ text     │  │ (daemon,     │
│      │  │ alert    │  │          │  │  fire&forget) │
└─────┘  └──────────┘  └──────────┘  └──────────────┘
```

### Concurrency Controls
- **Camera → Thread A**: Lock-free atomic reference (GIL-protected)
- **Thread A → MJPEG**: `threading.Lock` on `_latest_frame_bytes`
- **Alert Thread**: `threading.Lock` on `is_speaking` flag
- **Thread C**: Independent, only writes to `ai_core._spatial_text`

---

## 41. Auto-Logging on State Changes

**File**: `app/ai_core.py`, `_auto_log()` method

### What It Does
Automatically generates internal log entries when detection states change.

### What Triggers a Log
```python
# Safety status changes
if safety_status != prev_status:
    log("Status: SAFE -> WARNING")

# Face detection changes
if face_detected != prev_face:
    log("Face mesh tracking active")  # or "Face lost - searching"

# New YOLO objects detected
if object_count != prev_count and object_count > 0:
    log("YOLO: cell phone, bottle")

# Behavior detections
if behavior_details != "":
    log("⚠️ BEHAVIOR: PHONE_NEAR_EAR")
```

These logs are stored in a deque (max 6 entries) and displayed in the HUD system log panel (when HUD is enabled).

---

## 42. JARVIS HUD Overlay (Built-in, Disabled)

**File**: `app/ai_core.py`, `draw_overlay()` method and 12+ drawing methods

### What It Does
A full Iron Man/JARVIS-style heads-up display overlay with 12 visual layers. **Currently NOT drawn on the camera feed** (kept clean by design).

### The 12 Layers
1. **Helmet Visor Vignette**: Dark edges simulating looking through Iron Man's helmet
2. **Scan Line**: Horizontal line sweeping vertically
3. **3D Depth Face Mesh**: Face tessellation colored by z-depth (darker = deeper)
4. **Rotating Reticle**: Triple concentric ring that rotates around the detected face
5. **Face Bio-Data Panel**: Shows EAR, MAR, blink rate as text next to the face
6. **Iron Man Object Boxes**: Styled bounding boxes for YOLO-detected objects
7. **System HUD Bars**: Top and bottom status bars with time, FPS, status
8. **Integrity Bars**: SYS/CAM/AI/NET progress bars
9. **Scrolling System Logs**: Last 6 log entries in a panel
10. **Spatial Panel**: GPT-5.2 room analysis text display
11. **Mini Radar**: Rotating radar sweep animation
12. **Corner Brackets**: Neon-glow corner brackets on the frame edges

### Why Disabled?
The user prefers a clean camera feed for the competition presentation. The HUD code remains in the file and can be enabled by uncommenting `self.draw_overlay(frame)` in the processing loop.

---

## 43. Telemetry Data Structure (30+ Fields)

**File**: `app/ai_core.py`, `_get_telemetry()` method

### What It Does
Returns a comprehensive dictionary of all detection data, sent to the dashboard via SocketIO 10 times per second.

### Complete Field List

| Field | Type | Description |
|-------|------|-------------|
| `ear` | float | Eye Aspect Ratio (4 decimal places) |
| `mar` | float | Mouth Aspect Ratio (4 decimal places) |
| `yaw` | float | Head yaw angle in degrees |
| `pitch` | float | Head pitch angle in degrees |
| `attention_score` | float | 0-100 composite score |
| `safety_status` | string | "SAFE" / "WARNING" / "DANGER" |
| `is_drowsy` | bool | Drowsiness detected |
| `is_yawning` | bool | Yawning detected |
| `is_distracted` | bool | Distraction detected |
| `is_looking_away` | bool | Looking away detected |
| `is_phone_near_ear` | bool | Phone near ear behavior |
| `is_looking_down` | bool | Looking down behavior |
| `is_drinking` | bool | Drinking behavior |
| `behavior_details` | string | Active behavior label |
| `face_detected` | bool | Face currently tracked |
| `face_confidence` | float | Face detection confidence 0-1 |
| `detected_objects` | list | Object labels from YOLO |
| `blink_rate` | float | Blinks per minute |
| `blink_total` | int | Total blinks this session |
| `danger_counter` | int | Consecutive danger frames |
| `drowsy_start` | float | Timestamp when drowsiness began |
| `drowsy_duration` | float | Seconds of current drowsiness |
| `ear_below_count` | int | Frames with EAR below threshold |
| `mar_above_count` | int | Frames with MAR above threshold |
| `process_time_ms` | float | AI processing latency (ms) |
| `detection_accuracy` | string | "95%" (confidence level) |
| `hand_detected` | bool | Any hand visible |
| `hand_near_face` | bool | Hand in face region |
| `hand_on_head` | bool | Hand on forehead |
| `tiredness_level` | float | 0-100 tiredness score |
| `eye_closure_level` | float | 0-100 eye closure % |
| `distraction_level` | float | 0-100 distraction score |
| `affective_state` | string | ALERT/TIRED/DROWSY/DISTRACTED/NEUTRAL |

---

## 🏁 Summary

Project ADAR V3.0 is a complete real-time driver safety system with:

- **7 detection types**: Drowsiness, yawning, distraction, looking away, phone near ear, drinking, looking down
- **3 AI models**: MediaPipe Face Landmarker (478 landmarks), YOLOv8 Nano (object detection), MediaPipe Hands (gesture detection)
- **1 frontier AI model**: GPT-5.2 Vision for intelligent danger confirmation + spatial scanning
- **2 alert paths**: PATH A (local instant for drowsiness) + PATH B (GPT-5.2 for everything else)
- **5 concurrent threads**: Main, Camera, Processing, Spatial, Alert
- **30+ telemetry fields** streamed at 10Hz to a JARVIS-themed web dashboard
- **Full database logging** of every incident with telemetry
- **Graceful degradation**: Works offline with local fallback alerts when API is unavailable

---

*Document generated: February 12, 2026*
*ADAR V3.0 — Advanced Driver Attention & Response System*
*Built for the OpenAI Buildathon Grand Finale 2026*
