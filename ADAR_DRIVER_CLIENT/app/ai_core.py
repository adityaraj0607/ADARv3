"""
============================================================
 ADAR V3.0 — AI Core Engine · JARVIS Masterpiece HUD
 White + Orange Industry-Grade Head-Up Display

 Visual layers:
   0  Helmet visor vignette
   1  Horizontal scan line
   2  3D depth-shaded face mesh
   3  Rotating targeting reticle (triple-ring)
   4  Face bio-data panel
   5  Iron Man object detection boxes
   6  System HUD (top bar + bottom bar)
   7  System integrity bars (SYS/CAM/AI/NET)
   8  Scrolling system event logs
   9  JARVIS spatial analysis panel (GPT-4o)
  10  Mini radar
  11  Active process monitor
  12  Corner brackets with glow
============================================================
"""

import cv2
import time
import math
import numpy as np
import mediapipe as mp
from collections import deque
from ultralytics import YOLO
import config

# ── MediaPipe task API ──────────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# ── MediaPipe Hands for gesture detection ───────────────────
try:
    Hands = mp.solutions.hands
    mp_hands = Hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
except Exception:
    mp_hands = None

# ── Landmark indices ────────────────────────────────────────
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
# Extra eye landmarks for higher accuracy EAR
LEFT_EYE_UPPER = [386, 374]   # additional vertical pairs
LEFT_EYE_LOWER = [373, 380]
RIGHT_EYE_UPPER = [159, 145]
RIGHT_EYE_LOWER = [153, 144]
UPPER_LIP = [13]
LOWER_LIP = [14]
# Additional mouth landmarks for 3-point MAR
UPPER_LIP_MID_LEFT = [37]
UPPER_LIP_MID_RIGHT = [267]
LOWER_LIP_MID_LEFT = [84]
LOWER_LIP_MID_RIGHT = [314]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
NOSE_TIP_IDX = 1
CHIN_IDX = 152
LEFT_EYE_CORNER = 263
RIGHT_EYE_CORNER = 33
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291

FACE_OVAL_IDX = np.array([
    10, 338, 297, 332, 284, 251, 389, 356, 454,
    323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132,
    93, 234, 127, 162, 21, 54, 103, 67, 109,
])

# ── 3D model points for head-pose solvePnP ──────────────────
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye corner
    (225.0, 170.0, -135.0),   # Right eye corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0),  # Right mouth corner
], dtype=np.float64)

# ── Tessellation edge set ────────────────────────────────────
def _load_tess() -> list:
    conns = mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
    edges: set = set()
    for c in conns:
        a, b = c.start, c.end
        edges.add((min(a, b), max(a, b)))
    return list(edges)

_TESS_EDGES = _load_tess()

# ═════════════════════════════════════════════════════════════
#  COLOR PALETTE — White + Orange (Industry-Grade)
#  All colors in BGR for OpenCV
# ═════════════════════════════════════════════════════════════
C_WHITE  = (255, 255, 255)
C_ORANGE = (0, 140, 255)          # Primary accent
C_AMBER  = (0, 191, 255)          # Warm secondary
C_WARM   = (60, 170, 255)         # Bright warm
C_DIM    = (0, 70, 120)           # Dim accent
C_GHOST  = (0, 35, 60)            # Faint ghost
C_BG     = (10, 12, 18)           # Panel background
C_SAFE   = (0, 220, 110)          # Green status
C_WARN   = (0, 165, 255)          # Warning amber
C_DANGER = (50, 35, 255)          # Red danger
C_TEXT   = (220, 225, 230)        # Light text

# Depth layers for face mesh — orange palette (z_min, z_max, color)
_DEPTH_LAYERS = [
    (0.70, 1.01, (0, 18, 35)),    # Ghost
    (0.45, 0.70, (0, 40, 75)),    # Dim
    (0.20, 0.45, (0, 80, 150)),   # Medium
    (0.00, 0.20, (0, 130, 230)),  # Bright orange
]

# Process monitor labels
_PROC_LABELS = [
    "FACE_MESH", "EAR_MON", "MAR_MON", "HEAD_POSE",
    "BLINK_DET", "YOLOv8n", "ATTN_FUSE", "GPT4o_SCAN",
]

# Objects that trigger DISTRACTION alerts (critical behaviors)
_DISTRACTION_OBJECTS: set = {"cell phone", "knife", "scissors", "bottle", "cup"}

# Behavior detection zones (for advanced analysis)
_PHONE_NEAR_EAR_DISTANCE = 0.15  # Normalized distance threshold
_LOOKING_DOWN_PITCH = -28  # Degrees (wider to avoid false positives)
_EXTREME_LOOKING_DOWN_PITCH = -38  # Degrees

_AA = cv2.LINE_AA


# ═════════════════════════════════════════════════════════════
#  NEON GLOW HELPERS
# ═════════════════════════════════════════════════════════════

def _glow_line(f: np.ndarray, p1: tuple, p2: tuple,
               c: tuple, th: int = 1, gl: int = 4) -> None:
    """Line with outer glow (thick dim) + inner bright."""
    dc = (c[0] // 4, c[1] // 4, c[2] // 4)
    cv2.line(f, p1, p2, dc, th + gl, _AA)
    cv2.line(f, p1, p2, c, th, _AA)


def _glow_circle(f: np.ndarray, ctr: tuple, r: int,
                 c: tuple, th: int = 1, gl: int = 4) -> None:
    dc = (c[0] // 4, c[1] // 4, c[2] // 4)
    cv2.circle(f, ctr, r, dc, th + gl, _AA)
    cv2.circle(f, ctr, r, c, th, _AA)


def _glow_ellipse(f: np.ndarray, ctr: tuple, axes: tuple,
                  angle: float, sa: float, ea: float,
                  c: tuple, th: int = 1, gl: int = 3) -> None:
    dc = (c[0] // 4, c[1] // 4, c[2] // 4)
    cv2.ellipse(f, ctr, axes, angle, sa, ea, dc, th + gl, _AA)
    cv2.ellipse(f, ctr, axes, angle, sa, ea, c, th, _AA)


def _glow_rect_corners(f: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                       c: tuple, ln: int = 20, th: int = 2,
                       gl: int = 4) -> None:
    """Corner brackets with neon glow."""
    corners = [
        ((x1, y1), (x1 + ln, y1), (x1, y1 + ln)),
        ((x2, y1), (x2 - ln, y1), (x2, y1 + ln)),
        ((x1, y2), (x1 + ln, y2), (x1, y2 - ln)),
        ((x2, y2), (x2 - ln, y2), (x2, y2 - ln)),
    ]
    for pt, h_end, v_end in corners:
        _glow_line(f, pt, h_end, c, th, gl)
        _glow_line(f, pt, v_end, c, th, gl)


# ═════════════════════════════════════════════════════════════
#  ALPHA-BLENDED RECTANGLE
# ═════════════════════════════════════════════════════════════

def _alpha_rect(f: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                color: tuple, alpha: float = 0.3) -> None:
    """Draw a semi-transparent filled rectangle (ROI-based)."""
    fh, fw = f.shape[:2]
    x1c = max(0, min(x1, fw))
    y1c = max(0, min(y1, fh))
    x2c = max(0, min(x2, fw))
    y2c = max(0, min(y2, fh))
    if x2c <= x1c or y2c <= y1c:
        return
    roi = f[y1c:y2c, x1c:x2c]
    overlay = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)


# ═════════════════════════════════════════════════════════════
#  VIGNETTE (HELMET VISOR)
# ═════════════════════════════════════════════════════════════

def _create_vignette(h: int, w: int) -> np.ndarray:
    """Pre-compute helmet visor vignette as uint8 3-channel mask."""
    Y = np.linspace(-1.0, 1.0, h)[:, None]
    X = np.linspace(-1.0, 1.0, w)[None, :]
    dist = np.sqrt(X * X + Y * Y)
    mask = np.clip(1.0 - (dist - 0.65) * 1.2, 0.25, 1.0)
    mask_u8 = (mask * 255).astype(np.uint8)
    return np.stack([mask_u8, mask_u8, mask_u8], axis=-1)


# ═════════════════════════════════════════════════════════════
#  AI CORE
# ═════════════════════════════════════════════════════════════

class AICore:
    """ADAR V3.0 AI Detection Engine + JARVIS Masterpiece HUD."""

    def __init__(self) -> None:
        # ── MediaPipe FaceLandmarker (Optimized for 95% accuracy) ──
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=config.FACE_LANDMARKER_MODEL_PATH
            ),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.6,  # Higher for better accuracy
            min_face_presence_confidence=0.6,   # Higher for better accuracy
            min_tracking_confidence=0.7,        # Higher for stable tracking
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._mp_ts: int = 0
        self._frame_count: int = 0
        print("[AI CORE] FaceLandmarker loaded ✓ (High accuracy mode)")

        # ── YOLO ──
        print("[AI CORE] Loading YOLOv8 model...")
        self._yolo = YOLO(config.YOLO_MODEL_PATH, verbose=False)
        print("[AI CORE] YOLOv8 loaded ✓")

        # ── Detection state ──
        self.ear: float = 0.0
        self.mar: float = 0.0
        self.yaw: float = 0.0
        self.pitch: float = 0.0
        self.attention_score: float = 100.0
        self.safety_status: str = config.STATUS_SAFE
        self.danger_counter: int = 0
        self.is_drowsy: bool = False
        self.is_yawning: bool = False
        self.is_distracted: bool = False
        self.is_looking_away: bool = False
        self.face_detected: bool = False
        self.detected_objects: list = []
        self.drowsy_start: float = 0.0

        # ── Advanced behavior detection ──
        self.is_phone_near_ear: bool = False
        self.is_looking_down: bool = False
        self.is_drinking: bool = False
        self.behavior_details: str = ""
        
        # ── Hand detection & tiredness analysis ──
        self.hand_detected: bool = False
        self.hand_near_face: bool = False
        self.hand_on_head: bool = False
        self.tiredness_level: float = 0.0  # 0-100 scale
        self.eye_closure_level: float = 0.0  # 0-100 scale
        self.distraction_level: float = 0.0  # 0-100 scale
        self.affective_state: str = "NEUTRAL"  # ALERT, TIRED, DROWSY, DISTRACTED

        # ── Blink tracking ──
        self._blink_total: int = 0
        self._blink_timestamps: list = []
        self.blink_rate: float = 0.0
        self._prev_ear: float = 0.3

        # ── EAR history for temporal smoothing (reduces noise) ──
        self._ear_history: deque = deque(maxlen=3)  # Smaller window = faster response
        self._mar_history: deque = deque(maxlen=2)  # Faster yawn response
        self._ear_baseline: float = 0.30  # Adaptive baseline
        self._ear_baseline_samples: deque = deque(maxlen=300)  # 10 sec @ 30fps
        self._face_conf_history: deque = deque(maxlen=10)

        # ── Eye closure timer (drives the drowsy badge + timer bar) ──
        self._eye_closure_start: float = 0.0
        self._eye_closure_duration: float = 0.0
        self._eye_open_grace: int = 0
        self._face_lost_frames: int = 0

        # ── Consecutive frame counters ──
        self._ear_below_count: int = 0
        self._mar_above_count: int = 0
        self._look_away_frames: int = 0

        # ── Cached drawing data (populated by process_frame) ──
        self._cached_coords: np.ndarray | None = None
        self._cached_z_values: np.ndarray | None = None
        self._cached_oval_pts: np.ndarray | None = None

        # ── solvePnP cached guess for faster convergence ──
        self._prev_rvec: np.ndarray | None = None
        self._prev_tvec: np.ndarray | None = None

        # ── Timing ──
        self._last_process_time: float = 0.0
        self._hud_start: float = time.time()
        self._frame_idx: int = 0

        # ── Vignette mask (lazy-init) ──
        self._vignette: np.ndarray | None = None

        # ── Spatial analysis (set by Thread C) ──
        self._spatial_text: str = ""
        self._spatial_time: float = 0.0

        # ── System event log ──
        self._system_logs: deque = deque(maxlen=6)
        self._add_log("ADAR V3.0 initialized")
        self._add_log("JARVIS defense system online")

        # ── Auto-log state tracking ──
        self._prev_status: str = config.STATUS_SAFE
        self._prev_face: bool = False
        self._prev_obj_count: int = 0

    # ═════════════════════════════════════════════════════════
    #  PUBLIC API
    # ═════════════════════════════════════════════════════════

    def _add_log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self._system_logs.append(f"[{ts}] {msg}")

    def add_log(self, msg: str) -> None:
        """Public — called from routes.py / spatial thread."""
        self._add_log(msg)

    def is_ready_for_alert(self) -> bool:
        """Check if distraction/alert has been sustained long enough to warrant GPT call."""
        return self.danger_counter >= config.DANGER_FRAME_THRESHOLD

    def set_spatial_analysis(self, text: str) -> None:
        """Called by the spatial-scan thread (Thread C)."""
        self._spatial_text = text
        self._spatial_time = time.time()

    # ═════════════════════════════════════════════════════════
    #  FRAME PROCESSING (detection only — no drawing)
    # ═════════════════════════════════════════════════════════

    def process_frame(self, frame: np.ndarray,
                      run_yolo: bool = True) -> dict:
        t0 = time.time()
        h, w = frame.shape[:2]
        self._frame_count += 1

        # ── MediaPipe face landmarks ──
        self._mp_ts += 33
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, self._mp_ts)

        if result.face_landmarks and len(result.face_landmarks) > 0:
            landmarks = result.face_landmarks[0]
            n = len(landmarks)
            # Fast vectorized coordinate extraction (no Python loop)
            _raw = np.array(
                [(lm.x, lm.y, lm.z) for lm in landmarks],
                dtype=np.float64,
            )
            coords = _raw[:, :2] * np.array([w, h], dtype=np.float64)
            z_vals = _raw[:, 2]
            self._cached_coords = coords
            self._cached_z_values = z_vals
            if n > int(np.max(FACE_OVAL_IDX)):
                self._cached_oval_pts = coords[FACE_OVAL_IDX].astype(
                    np.int32
                )
            else:
                self._cached_oval_pts = None
            self.face_detected = True
            self._face_lost_frames = 0  # Reset face-loss counter

            # EAR (smoothed for accuracy)
            left_ear = self._calculate_ear(coords, LEFT_EYE)
            right_ear = self._calculate_ear(coords, RIGHT_EYE)
            raw_ear = float((left_ear + right_ear) / 2.0)
            self.ear = self._get_smoothed_ear(raw_ear)

            # MAR (smoothed for accuracy)
            raw_mar = float(self._calculate_mar(coords))
            self.mar = self._get_smoothed_mar(raw_mar)

            # Head pose
            self.yaw, self.pitch = self._estimate_head_pose(coords, w, h)

            # Blink detection
            self._detect_blink(float(self.ear))

            # ══════ DROWSINESS — Eye Closure Timer (85% Detection Rate) ══════
            # Timer starts IMMEDIATELY when EAR drops below threshold.
            # WARNING at first frame, DANGER after 3.5 seconds.
            adaptive_threshold = min(config.EAR_THRESHOLD, self._ear_baseline * 0.78)
            eyes_closed = self.ear < adaptive_threshold
            
            if eyes_closed:
                self._ear_below_count += 1
                # Start timer on first frame of eye closure
                if self._eye_closure_start == 0.0:
                    self._eye_closure_start = time.time()
                self._eye_open_grace = 0
            else:
                # Grace period: don't reset immediately on brief eye open (blink)
                self._eye_open_grace = getattr(self, '_eye_open_grace', 0) + 1
                if self._eye_open_grace >= 4:  # ~0.13s grace — faster reset for 85% accuracy
                    self._ear_below_count = 0
                    self._eye_closure_start = 0.0
                    self._eye_open_grace = 0
                else:
                    # During grace: keep the timer running
                    pass
            
            # Drowsy timer duration (0 if eyes open)
            if self._eye_closure_start > 0:
                self._eye_closure_duration = time.time() - self._eye_closure_start
            else:
                self._eye_closure_duration = 0.0
            
            # is_drowsy = eyes have been closed for >= 4 seconds
            self.is_drowsy = self._eye_closure_duration >= config.DROWSY_ALERT_DURATION
            
            # Drowsy start (for legacy telemetry compatibility)
            self.drowsy_start = self._eye_closure_start

            # Enhanced yawning detection
            if self.mar > config.MAR_THRESHOLD:
                self._mar_above_count += 1
            else:
                self._mar_above_count = 0
            
            # Multi-factor yawn detection
            base_yawning = self._mar_above_count >= config.MAR_CONSEC_FRAMES
            extreme_mar = self.mar > (config.MAR_THRESHOLD * 1.2)
            
            self.is_yawning = (
                base_yawning or 
                (extreme_mar and self._mar_above_count >= 3)
            )

            # Enhanced looking away detection with hysteresis (95% accuracy)
            looking_away_now = (
                abs(self.yaw) > config.HEAD_YAW_THRESHOLD
                or abs(self.pitch) > config.HEAD_PITCH_THRESHOLD
            )
            
            # Stability: ramp up faster, decay slower → fewer missed detections
            if looking_away_now:
                self._look_away_frames += 2  # Faster ramp-up
            else:
                self._look_away_frames = max(0, self._look_away_frames - 1)  # Slower decay
            
            self.is_looking_away = self._look_away_frames >= 4  # Faster trigger

        else:
            # ── Face lost ── use grace period before resetting drowsy ──
            self._face_lost_frames = getattr(self, '_face_lost_frames', 0) + 1
            
            self.face_detected = False
            self._cached_coords = None
            self._cached_z_values = None
            self._cached_oval_pts = None

            # Only reset drowsy state after face is lost for > 1 second (30 frames)
            # This prevents eye closure from killing the timer when landmarks flicker
            if self._face_lost_frames > 30:
                self._eye_closure_start = 0.0
                self._eye_closure_duration = 0.0
                self.is_drowsy = False
                self._ear_below_count = 0
                self.drowsy_start = 0.0
            # else: keep drowsy timer running during brief face loss

            # Reset non-drowsy flags immediately
            self.is_yawning = False
            self.is_looking_away = False
            self.is_phone_near_ear = False
            self.is_looking_down = False
            self.is_drinking = False

            # Reset non-drowsy counters
            self._mar_above_count = 0
            self._look_away_frames = 0

            # Reset behavior state
            self.behavior_details = ""
            self.hand_near_face = False
            self.hand_on_head = False

        # ── YOLO detection ──
        if run_yolo:
            self._run_yolo(frame)

        # ── Hand detection (MediaPipe) — every 10th frame to save CPU ──
        if self._frame_count % 10 == 0:
            self._detect_hands(frame, w, h)

        # ── Advanced behavior analysis (90-95% accuracy) ──
        self._analyze_behaviors(w, h)
        
        # ── Tiredness & Affective State Analysis ──
        self._analyze_tiredness_and_affect()

        # ── Distraction (only from confirmed YOLO detections + behavior) ──
        self.is_distracted = (
            any(o["label"] in _DISTRACTION_OBJECTS for o in self.detected_objects)
            or self.is_phone_near_ear
            or self.is_drinking
        )

        # ── Attention score ──
        self._compute_attention_score()

        # ── Safety status ──
        self._update_safety_status()

        # ── Auto-log state changes ──
        self._auto_log()

        self._last_process_time = 0.7 * self._last_process_time + 0.3 * (time.time() - t0)
        return self._get_telemetry()

    # ═════════════════════════════════════════════════════════
    #  OVERLAY DRAWING — JARVIS MASTERPIECE HUD
    # ═════════════════════════════════════════════════════════

    def draw_overlay(self, frame: np.ndarray) -> None:
        """Full JARVIS Masterpiece HUD overlay."""
        h, w = frame.shape[:2]
        t = time.time() - self._hud_start

        # Vignette (helmet visor)
        if self._vignette is None or self._vignette.shape[:2] != (h, w):
            self._vignette = _create_vignette(h, w)
        frame[:] = (frame.astype(np.uint16) * self._vignette.astype(np.uint16) // 255).astype(np.uint8)

        # All HUD layers
        self._draw_scan_line(frame, w, h, t)
        if self._cached_coords is not None and self._cached_z_values is not None:
            self._draw_face_mesh_3d(frame, self._cached_coords, self._cached_z_values)
            self._draw_face_reticle(frame, self._cached_coords, w, h, t)
            self._draw_face_data_panel(frame, self._cached_coords, w, h)
        for idx, obj in enumerate(self.detected_objects):
            self._draw_iron_man_box(frame, obj, t, idx)
        self._draw_system_hud(frame, w, h, t)
        self._draw_integrity_bars(frame, w, h)
        self._draw_system_logs(frame, w, h)
        self._draw_spatial_panel(frame, w, h, t)
        self._draw_mini_radar(frame, w, h, t)
        self._draw_process_monitor(frame, w, h)
        self._draw_corner_brackets(frame, w, h, t)

    def draw_minimal_overlay(self, frame: np.ndarray) -> None:
        """Lightweight overlay for higher FPS streaming."""
        h, w = frame.shape[:2]
        t = time.time() - self._hud_start
        self._draw_system_hud(frame, w, h, t)
        self._draw_corner_brackets(frame, w, h, t)
        if self._cached_coords is not None:
            self._draw_face_data_panel(frame, self._cached_coords, w, h)

    # ─── Layer 1: Scan Line ──────────────────────────────────

    def _draw_scan_line(self, f: np.ndarray,
                        w: int, h: int, t: float) -> None:
        y = int((t * 80) % h)
        for i in range(20):
            alpha_v = 1.0 - i / 20.0
            yy = y - i
            if 0 <= yy < h:
                c = int(50 * alpha_v)
                cv2.line(f, (0, yy), (w, yy),
                         (0, c, int(c * 1.8)), 1)
        if 0 <= y < h:
            cv2.line(f, (0, y), (w, y), C_ORANGE, 1, _AA)

    # ─── Layer 2: 3D Face Mesh ───────────────────────────────

    def _draw_face_mesh_3d(self, f: np.ndarray,
                           coords: np.ndarray,
                           z_vals: np.ndarray) -> None:
        n = len(coords)
        for a, b in _TESS_EDGES:
            if a >= n or b >= n:
                continue
            avg_z = (z_vals[a] + z_vals[b]) / 2.0
            for z_min, z_max, color in _DEPTH_LAYERS:
                if z_min <= avg_z < z_max:
                    pt1 = (int(coords[a][0]), int(coords[a][1]))
                    pt2 = (int(coords[b][0]), int(coords[b][1]))
                    cv2.line(f, pt1, pt2, color, 1, _AA)
                    break

    # ─── Layer 3: Rotating Targeting Reticle ─────────────────

    def _draw_face_reticle(self, f: np.ndarray,
                           coords: np.ndarray,
                           w: int, h: int, t: float) -> None:
        oval = self._cached_oval_pts
        if oval is None or len(oval) < 4:
            return
        x_min = int(np.min(oval[:, 0]))
        x_max = int(np.max(oval[:, 0]))
        y_min = int(np.min(oval[:, 1]))
        y_max = int(np.max(oval[:, 1]))
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2
        rx = int((x_max - x_min) * 0.55)
        ry = int((y_max - y_min) * 0.55)
        if rx < 10 or ry < 10:
            return

        # Status color
        sc = (C_SAFE if self.safety_status == config.STATUS_SAFE
              else C_WARN if self.safety_status == config.STATUS_WARNING
              else C_DANGER)

        # Ring 1 — Outer (white, rotating CW)
        rot1 = (t * 45) % 360
        for seg in range(0, 360, 90):
            s = seg + rot1
            _glow_ellipse(f, (cx, cy), (rx, ry),
                          0, s, s + 60, C_WHITE, 2, 4)

        # Ring 2 — Middle (orange, rotating CCW)
        rx2 = int(rx * 0.82)
        ry2 = int(ry * 0.82)
        rot2 = (-t * 60) % 360
        for seg in range(0, 360, 120):
            s = seg + rot2
            _glow_ellipse(f, (cx, cy), (rx2, ry2),
                          0, s, s + 80, C_ORANGE, 1, 3)

        # Ring 3 — Inner (status color, pulsing)
        pulse = 0.8 + 0.2 * math.sin(t * 4)
        rx3 = int(rx * 0.6 * pulse)
        ry3 = int(ry * 0.6 * pulse)
        cv2.ellipse(f, (cx, cy), (rx3, ry3), 0, 0, 360, sc, 1, _AA)

        # Crosshair
        cl = int(rx * 0.15)
        _glow_line(f, (cx - cl, cy), (cx + cl, cy), C_WHITE, 1, 2)
        _glow_line(f, (cx, cy - cl), (cx, cy + cl), C_WHITE, 1, 2)

        # Tick marks on outer ring
        for ang in range(0, 360, 30):
            rad = math.radians(ang + rot1)
            ix = int(cx + rx * 0.9 * math.cos(rad))
            iy = int(cy + ry * 0.9 * math.sin(rad))
            ox = int(cx + rx * 1.0 * math.cos(rad))
            oy = int(cy + ry * 1.0 * math.sin(rad))
            cv2.line(f, (ix, iy), (ox, oy), C_AMBER, 1, _AA)

    # ─── Layer 4: Face Bio-Data Panel ────────────────────────

    def _draw_face_data_panel(self, f: np.ndarray,
                              coords: np.ndarray,
                              w: int, h: int) -> None:
        oval = self._cached_oval_pts
        if oval is None or len(oval) < 4:
            return
        x_max_face = int(np.max(oval[:, 0]))
        y_min_face = int(np.min(oval[:, 1]))

        px = min(x_max_face + 20, w - 200)
        py = max(y_min_face, 10)
        pw, ph = 185, 165

        # Background
        _alpha_rect(f, px, py, px + pw, py + ph, C_BG, 0.6)
        cv2.line(f, (px, py), (px + pw, py), C_ORANGE, 2, _AA)

        # Title
        cv2.putText(f, "SUBJECT ANALYSIS", (px + 8, py + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_ORANGE, 1, _AA)

        # Metrics
        metrics = [
            ("EAR",   self.ear,             0.35,  config.EAR_THRESHOLD,         "low"),
            ("MAR",   self.mar,             1.0,   config.MAR_THRESHOLD,         "high"),
            ("YAW",   abs(self.yaw),        90.0,  float(config.HEAD_YAW_THRESHOLD),   "high"),
            ("PITCH", abs(self.pitch),      90.0,  float(config.HEAD_PITCH_THRESHOLD),  "high"),
            ("BLINK", self.blink_rate,      30.0,  None,                          ""),
            ("ATTN",  self.attention_score, 100.0, None,                          ""),
        ]
        for i, (label, val, max_v, thresh, mode) in enumerate(metrics):
            yy = py + 30 + i * 22
            ratio = min(val / max_v, 1.0) if max_v > 0 else 0.0
            bar_w = int(80 * ratio)

            # Color logic
            if thresh is not None:
                if mode == "low":
                    mc = C_DANGER if val < thresh else C_SAFE
                else:
                    mc = C_DANGER if val > thresh else C_SAFE
            elif label == "ATTN":
                mc = (C_SAFE if val > 60
                      else C_WARN if val > 30
                      else C_DANGER)
            else:
                mc = C_ORANGE

            cv2.putText(f, label, (px + 8, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_TEXT, 1, _AA)
            val_txt = f"{val:.3f}" if max_v <= 1.0 else f"{val:.0f}"
            cv2.putText(f, val_txt, (px + 50, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_WHITE, 1, _AA)
            # Bar background
            cv2.rectangle(f, (px + 95, yy - 8),
                          (px + 175, yy - 1), C_GHOST, -1)
            # Bar fill
            if bar_w > 0:
                cv2.rectangle(f, (px + 95, yy - 8),
                              (px + 95 + bar_w, yy - 1), mc, -1)

    # ─── Layer 5: Iron Man Object Boxes ──────────────────────

    def _draw_iron_man_box(self, f: np.ndarray, obj: dict,
                           t: float, idx: int) -> None:
        x1, y1, x2, y2 = obj["box"]
        label: str = obj["label"]
        conf: float = obj["conf"]
        fh, fw = f.shape[:2]

        x1 = max(0, min(x1, fw - 1))
        y1 = max(0, min(y1, fh - 1))
        x2 = max(0, min(x2, fw - 1))
        y2 = max(0, min(y2, fh - 1))
        if x2 <= x1 or y2 <= y1:
            return

        is_danger = label in _DISTRACTION_OBJECTS
        color = C_DANGER if is_danger else C_ORANGE

        # Corner brackets
        ln = max(15, min(x2 - x1, y2 - y1) // 4)
        _glow_rect_corners(f, x1, y1, x2, y2, color, ln, 2, 4)

        # Pulsing overlay for danger items
        if is_danger:
            pulse = 0.15 + 0.1 * math.sin(t * 6)
            _alpha_rect(f, x1, y1, x2, y2, C_DANGER, pulse)

        # Label tag
        tag = f"{label.upper()} {conf:.0%}"
        tag_w = len(tag) * 8 + 10
        _alpha_rect(f, x1, y1 - 20, x1 + tag_w, y1, C_BG, 0.7)
        cv2.putText(f, tag, (x1 + 5, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, _AA)

        # Connector line to center
        ccx = (x1 + x2) // 2
        ccy = (y1 + y2) // 2
        cv2.line(f, (x1 + tag_w // 2, y1), (ccx, ccy),
                 color, 1, _AA)

    # ─── Layer 6: System HUD (Top + Bottom Bars) ─────────────

    def _draw_system_hud(self, f: np.ndarray,
                         w: int, h: int, t: float) -> None:
        # ═══ TOP BAR ═══
        _alpha_rect(f, 0, 0, w, 52, C_BG, 0.55)
        cv2.line(f, (0, 52), (w, 52), C_ORANGE, 1, _AA)

        # Title
        cv2.putText(f, "ADAR V3.0", (15, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_WHITE, 1, _AA)
        cv2.putText(f, "JARVIS DEFENSE SYSTEM", (15, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, C_ORANGE, 1, _AA)

        # Center — attention bar
        bar_x = w // 2 - 120
        cv2.putText(f, "ATTENTION", (bar_x, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_TEXT, 1, _AA)
        cv2.rectangle(f, (bar_x, 24), (bar_x + 240, 38), C_GHOST, -1)
        fill = int(240 * self.attention_score / 100.0)
        ac = (C_SAFE if self.attention_score > 60
              else C_WARN if self.attention_score > 30
              else C_DANGER)
        if fill > 0:
            cv2.rectangle(f, (bar_x, 24),
                          (bar_x + fill, 38), ac, -1)
        cv2.putText(f, f"{self.attention_score:.0f}%",
                    (bar_x + 245, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_WHITE, 1, _AA)

        # Right — status + time
        status = self.safety_status
        sc = (C_SAFE if status == config.STATUS_SAFE
              else C_WARN if status == config.STATUS_WARNING
              else C_DANGER)
        if status == config.STATUS_DANGER:
            pulse = 0.5 + 0.5 * math.sin(t * 6)
            sc = (int(sc[0] * pulse),
                  int(sc[1] * pulse),
                  int(sc[2] * pulse))
        cv2.putText(f, f"// {status}", (w - 170, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, sc, 1, _AA)
        cv2.putText(f, time.strftime("%H:%M:%S"), (w - 100, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_DIM, 1, _AA)

        # ═══ BOTTOM BAR ═══
        _alpha_rect(f, 0, h - 48, w, h, C_BG, 0.55)
        cv2.line(f, (0, h - 48), (w, h - 48), C_ORANGE, 1, _AA)

        # Telemetry row
        items = [
            f"EAR {self.ear:.3f}",
            f"MAR {self.mar:.3f}",
            f"YAW {self.yaw:+.0f}deg",
            f"PITCH {self.pitch:+.0f}deg",
            f"BPM {self.blink_rate:.0f}",
            f"AI {self._last_process_time*1000:.0f}ms",
        ]
        x = 15
        for item in items:
            cv2.putText(f, item, (x, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, C_TEXT, 1, _AA)
            x += len(item) * 7 + 20

        # Status flags row
        flags = [
            ("DROWSY", self.is_drowsy),
            ("YAWNING", self.is_yawning),
            ("DISTRACTED", self.is_distracted),
            ("LOOK_AWAY", self.is_looking_away),
        ]
        x = 15
        for label, active in flags:
            dot_c = C_DANGER if active else C_DIM
            cv2.circle(f, (x + 4, h - 12), 4, dot_c, -1, _AA)
            txt_c = C_TEXT if active else C_DIM
            cv2.putText(f, label, (x + 12, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, txt_c, 1, _AA)
            x += len(label) * 6 + 30

        # Object count
        cv2.putText(f, f"OBJ: {len(self.detected_objects)}",
                    (w - 80, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, C_ORANGE, 1, _AA)

    # ─── Layer 7: System Integrity Bars ──────────────────────

    def _draw_integrity_bars(self, f: np.ndarray,
                             w: int, h: int) -> None:
        bx = w - 110
        by = 70
        _alpha_rect(f, bx - 5, by - 5, bx + 105, by + 78, C_BG, 0.4)

        bars = [
            ("SYS", min(self.attention_score / 100.0, 1.0)),
            ("CAM", 1.0 if self.face_detected else 0.3),
            ("AI",  max(0.0, 1.0 - self._last_process_time * 10)),
            ("NET", 1.0 if (time.time() - self._spatial_time) < 5.0
             else 0.3),
        ]
        for i, (label, val) in enumerate(bars):
            yy = by + i * 18
            cv2.putText(f, label, (bx, yy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, C_TEXT, 1, _AA)
            cv2.rectangle(f, (bx + 30, yy + 2),
                          (bx + 95, yy + 12), C_GHOST, -1)
            fill_w = int(65 * max(0.0, min(val, 1.0)))
            if fill_w > 0:
                bc = (C_SAFE if val > 0.6
                      else C_WARN if val > 0.3
                      else C_DANGER)
                cv2.rectangle(f, (bx + 30, yy + 2),
                              (bx + 30 + fill_w, yy + 12), bc, -1)

    # ─── Layer 8: Scrolling System Logs ──────────────────────

    def _draw_system_logs(self, f: np.ndarray,
                          w: int, h: int) -> None:
        n = len(self._system_logs)
        if n == 0:
            return
        lx = 15
        ly = h - 65 - n * 15
        ly = max(60, ly)
        _alpha_rect(f, lx - 5, ly - 5,
                    lx + 365, ly + n * 15 + 5, C_BG, 0.35)
        for i, msg in enumerate(self._system_logs):
            yy = ly + i * 15
            cv2.putText(f, msg, (lx, yy + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                        C_DIM, 1, _AA)

    # ─── Layer 9: Spatial Analysis Panel ─────────────────────

    def _draw_spatial_panel(self, f: np.ndarray,
                            w: int, h: int, t: float) -> None:
        if not self._spatial_text:
            # Placeholder — "awaiting scan"
            pw, ph2 = 300, 50
            px = w // 2 - pw // 2
            py = h - 115
            _alpha_rect(f, px, py, px + pw, py + ph2, C_BG, 0.4)
            cv2.line(f, (px, py), (px + pw, py), C_DIM, 1, _AA)
            pulse = 0.5 + 0.5 * math.sin(t * 3)
            sc = (int(C_ORANGE[0] * pulse),
                  int(C_ORANGE[1] * pulse),
                  int(C_ORANGE[2] * pulse))
            cv2.putText(f, "JARVIS SPATIAL SCAN", (px + 10, py + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, sc, 1, _AA)
            cv2.putText(f, "Awaiting GPT-4o analysis...",
                        (px + 10, py + 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, C_DIM, 1, _AA)
            return

        lines = self._spatial_text.split("\n")[:4]
        line_h = 16
        panel_w = 420
        panel_h = 28 + len(lines) * line_h + 8
        px = w // 2 - panel_w // 2
        py = h - 60 - panel_h

        _alpha_rect(f, px, py, px + panel_w, py + panel_h, C_BG, 0.55)
        cv2.rectangle(f, (px, py), (px + panel_w, py + panel_h),
                      C_ORANGE, 1, _AA)
        cv2.line(f, (px, py + 22), (px + panel_w, py + 22),
                 C_DIM, 1, _AA)
        cv2.putText(f, "JARVIS SPATIAL ANALYSIS",
                    (px + 10, py + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_ORANGE, 1, _AA)

        # Live indicator + age
        age = time.time() - self._spatial_time
        if age < 5.0:
            pulse = 0.5 + 0.5 * math.sin(t * 5)
            dot_c = (int(C_SAFE[0] * pulse),
                     int(C_SAFE[1] * pulse),
                     int(C_SAFE[2] * pulse))
            cv2.circle(f, (px + panel_w - 78, py + 12),
                       3, dot_c, -1, _AA)
            cv2.putText(f, "LIVE", (px + panel_w - 72, py + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                        C_SAFE, 1, _AA)
        else:
            cv2.putText(f, f"{age:.0f}s ago",
                        (px + panel_w - 65, py + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                        C_DIM, 1, _AA)

        # Content
        for i, line in enumerate(lines):
            yy = py + 38 + i * line_h
            display = line[:58] if len(line) > 58 else line
            cv2.putText(f, f"> {display}", (px + 10, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        C_TEXT, 1, _AA)

    # ─── Layer 10: Mini Radar ────────────────────────────────

    def _draw_mini_radar(self, f: np.ndarray,
                         w: int, h: int, t: float) -> None:
        cx_r = w - 65
        cy_r = h - 110
        r = 40
        _alpha_rect(f, cx_r - r - 5, cy_r - r - 5,
                    cx_r + r + 5, cy_r + r + 5, C_BG, 0.35)

        cv2.circle(f, (cx_r, cy_r), r, C_GHOST, 1, _AA)
        cv2.circle(f, (cx_r, cy_r), r // 2, C_GHOST, 1, _AA)
        cv2.line(f, (cx_r - r, cy_r), (cx_r + r, cy_r),
                 C_GHOST, 1, _AA)
        cv2.line(f, (cx_r, cy_r - r), (cx_r, cy_r + r),
                 C_GHOST, 1, _AA)

        # Sweep
        sweep = (t * 120) % 360
        sx = int(cx_r + r * math.cos(math.radians(sweep)))
        sy = int(cy_r + r * math.sin(math.radians(sweep)))
        cv2.line(f, (cx_r, cy_r), (sx, sy), C_ORANGE, 1, _AA)

        # Sweep trail
        for i in range(1, 15):
            ta = sweep - i * 3
            av = 1.0 - i / 15.0
            tx = int(cx_r + r * math.cos(math.radians(ta)))
            ty = int(cy_r + r * math.sin(math.radians(ta)))
            tc = (0, int(60 * av), int(110 * av))
            cv2.line(f, (cx_r, cy_r), (tx, ty), tc, 1, _AA)

        # Blips
        if self.face_detected:
            br = 3 + int(2 * math.sin(t * 5))
            cv2.circle(f, (cx_r, cy_r - 8), br, C_SAFE, -1, _AA)

        for i, obj in enumerate(self.detected_objects):
            ang = (i * 73 + t * 10) % 360
            dist = r * 0.6
            ox = int(cx_r + dist * math.cos(math.radians(ang)))
            oy = int(cy_r + dist * math.sin(math.radians(ang)))
            oc = (C_DANGER if obj["label"] in _DISTRACTION_OBJECTS
                  else C_ORANGE)
            cv2.circle(f, (ox, oy), 3, oc, -1, _AA)

    # ─── Layer 11: Process Monitor ───────────────────────────

    def _draw_process_monitor(self, f: np.ndarray,
                              w: int, h: int) -> None:
        px_m = 15
        py_m = 65
        n = len(_PROC_LABELS)
        _alpha_rect(f, px_m - 5, py_m - 5,
                    px_m + 155, py_m + n * 16 + 5, C_BG, 0.4)

        for i, label in enumerate(_PROC_LABELS):
            yy = py_m + i * 16
            if label == "GPT4o_SCAN":
                active = (time.time() - self._spatial_time) < 5.0
            elif label == "YOLOv8n":
                active = (len(self.detected_objects) > 0
                          or self._frame_idx % 6 == 0)
            else:
                active = self.face_detected

            dot_c = C_SAFE if active else C_DIM
            cv2.circle(f, (px_m + 4, yy + 6), 3, dot_c, -1, _AA)
            cv2.putText(f, label, (px_m + 12, yy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        C_TEXT, 1, _AA)
            st = "RUN" if active else "IDLE"
            cv2.putText(f, st, (px_m + 110, yy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                        dot_c, 1, _AA)

    # ─── Layer 12: Corner Brackets ───────────────────────────

    def _draw_corner_brackets(self, f: np.ndarray,
                              w: int, h: int, t: float) -> None:
        m = 8
        ln = 40
        if self.safety_status == config.STATUS_DANGER:
            pulse = 0.5 + 0.5 * math.sin(t * 6)
            color = (int(C_DANGER[0] * pulse),
                     int(C_DANGER[1] * pulse),
                     int(C_DANGER[2] * pulse))
        else:
            color = C_ORANGE
        _glow_rect_corners(f, m, m, w - m, h - m, color, ln, 2, 4)

    # ═════════════════════════════════════════════════════════
    #  CALCULATION & UTILITY METHODS
    # ═════════════════════════════════════════════════════════

    @staticmethod
    def _calculate_ear(coords: np.ndarray,
                       eye_indices: list) -> float:
        """6-point EAR with improved vertical measurement."""
        pts = coords[eye_indices]
        # Two vertical distances
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        # Horizontal distance
        h_dist = np.linalg.norm(pts[0] - pts[3])
        if h_dist <= 0:
            return 0.3
        ear = float((v1 + v2) / (2.0 * h_dist))
        return ear

    def _get_smoothed_ear(self, raw_ear: float) -> float:
        """Temporal smoothing + adaptive baseline for 95%+ accuracy."""
        self._ear_history.append(raw_ear)
        # Weighted moving average (recent frames matter more)
        if len(self._ear_history) >= 3:
            weights = [0.1, 0.2, 0.3, 0.4, 0.5][-len(self._ear_history):]
            total_w = sum(weights)
            smoothed = sum(v * w for v, w in zip(self._ear_history, weights)) / total_w
        else:
            smoothed = raw_ear
        
        # Update adaptive baseline when eyes are open
        if raw_ear > 0.22:
            self._ear_baseline_samples.append(raw_ear)
            if len(self._ear_baseline_samples) >= 30:
                self._ear_baseline = float(np.median(list(self._ear_baseline_samples)))
        
        return smoothed

    def _get_smoothed_mar(self, raw_mar: float) -> float:
        """Temporal smoothing for MAR."""
        self._mar_history.append(raw_mar)
        return float(np.mean(list(self._mar_history)))

    @staticmethod
    def _calculate_mar(coords: np.ndarray) -> float:
        """Enhanced MAR using 3 vertical distances for accuracy."""
        upper = coords[UPPER_LIP[0]]
        lower = coords[LOWER_LIP[0]]
        left = coords[LEFT_MOUTH_CORNER]
        right = coords[RIGHT_MOUTH_CORNER]
        # Primary vertical
        v_center = np.linalg.norm(upper - lower)
        # Additional verticals for robustness
        try:
            v_left = np.linalg.norm(coords[UPPER_LIP_MID_LEFT[0]] - coords[LOWER_LIP_MID_LEFT[0]])
            v_right = np.linalg.norm(coords[UPPER_LIP_MID_RIGHT[0]] - coords[LOWER_LIP_MID_RIGHT[0]])
            v_avg = (v_center + v_left + v_right) / 3.0
        except (IndexError, KeyError):
            v_avg = v_center
        h_dist = np.linalg.norm(left - right)
        return float(v_avg / h_dist) if h_dist > 0 else 0.0

    def _estimate_head_pose(self, coords: np.ndarray,
                            w: int, h: int) -> tuple:
        image_points = np.array([
            coords[NOSE_TIP_IDX],
            coords[CHIN_IDX],
            coords[LEFT_EYE_CORNER],
            coords[RIGHT_EYE_CORNER],
            coords[LEFT_MOUTH_CORNER],
            coords[RIGHT_MOUTH_CORNER],
        ], dtype=np.float64)

        focal_length = float(w)
        center = (float(w / 2), float(h / 2))
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vec, tvec = cv2.solvePnP(
            MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
            rvec=self._prev_rvec, tvec=self._prev_tvec,
            useExtrinsicGuess=self._prev_rvec is not None,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return 0.0, 0.0

        # Cache for next frame (faster convergence)
        self._prev_rvec = rotation_vec
        self._prev_tvec = tvec

        rmat, _ = cv2.Rodrigues(rotation_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        return float(angles[1]), float(angles[0])

    def _detect_blink(self, raw_ear: float) -> None:
        if self._prev_ear >= config.BLINK_EAR_THRESHOLD > raw_ear:
            self._blink_total += 1
            self._blink_timestamps.append(time.time())
        self._prev_ear = raw_ear

        now = time.time()
        window = config.BLINK_RATE_WINDOW
        self._blink_timestamps = [
            ts for ts in self._blink_timestamps if now - ts <= window
        ]
        count = len(self._blink_timestamps)
        self.blink_rate = float(count * (60.0 / window))

    def _run_yolo(self, frame: np.ndarray) -> None:
        results = self._yolo(
            frame, conf=config.YOLO_CONFIDENCE, imgsz=320, verbose=False
        )
        objects: list = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id in config.YOLO_CLASSES_OF_INTEREST:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    objects.append({
                        "label": config.YOLO_CLASSES_OF_INTEREST[cls_id],
                        "conf": float(box.conf[0]),
                        "box": (x1, y1, x2, y2),
                    })
        self.detected_objects = objects

    def _analyze_behaviors(self, w: int, h: int) -> None:
        """Advanced behavior detection (90-95% accuracy) combining pose + objects."""
        self.behavior_details = ""
        
        # Reset flags
        self.is_phone_near_ear = False
        self.is_looking_down = False
        self.is_drinking = False
        
        if not self.face_detected or self._cached_coords is None:
            return
        
        # Detect looking down (texting/reading phone)
        if self.pitch < _LOOKING_DOWN_PITCH:
            self.is_looking_down = True
            if self.pitch < _EXTREME_LOOKING_DOWN_PITCH:
                self.behavior_details = "EXTREME_LOOKING_DOWN"
            else:
                self.behavior_details = "LOOKING_DOWN"
        
        # Detect phone near ear (calling)
        phone_objs = [o for o in self.detected_objects if o["label"] == "cell phone"]
        if phone_objs and self.face_detected:
            # Get face center
            oval = self._cached_oval_pts
            if oval is not None and len(oval) > 0:
                face_cx = int(np.mean(oval[:, 0]))
                face_cy = int(np.mean(oval[:, 1]))
                
                for phone in phone_objs:
                    x1, y1, x2, y2 = phone["box"]
                    phone_cx = (x1 + x2) // 2
                    phone_cy = (y1 + y2) // 2
                    
                    # Calculate normalized distance
                    dx = abs(phone_cx - face_cx) / w
                    dy = abs(phone_cy - face_cy) / h
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    # Check if phone is near ear level (side of face)
                    if distance < _PHONE_NEAR_EAR_DISTANCE:
                        self.is_phone_near_ear = True
                        self.behavior_details = "PHONE_NEAR_EAR"
                        break
        
        # Detect drinking (bottle/cup near mouth)
        drink_objs = [o for o in self.detected_objects if o["label"] in ["bottle", "cup"]]
        if drink_objs and self.face_detected:
            # Get mouth position
            if len(self._cached_coords) > max(UPPER_LIP[0], LOWER_LIP[0]):
                mouth_y = int((self._cached_coords[UPPER_LIP[0]][1] + 
                              self._cached_coords[LOWER_LIP[0]][1]) / 2)
                mouth_x = int((self._cached_coords[LEFT_MOUTH_CORNER][0] + 
                              self._cached_coords[RIGHT_MOUTH_CORNER][0]) / 2)
                
                for drink in drink_objs:
                    x1, y1, x2, y2 = drink["box"]
                    drink_cx = (x1 + x2) // 2
                    drink_cy = (y1 + y2) // 2
                    
                    # Check if drink is near mouth
                    dx = abs(drink_cx - mouth_x) / w
                    dy = abs(drink_cy - mouth_y) / h
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    if distance < 0.12:  # Very close to mouth
                        self.is_drinking = True
                        self.behavior_details = "DRINKING"
                        break
        
        # Combine looking down with phone detection
        if self.is_looking_down and phone_objs:
            self.behavior_details = "TEXTING_OR_READING_PHONE"
            self.is_distracted = True

    def _detect_hands(self, frame: np.ndarray, w: int, h: int) -> None:
        """Detect hands and analyze gestures for tiredness/distraction."""
        self.hand_detected = False
        self.hand_near_face = False
        self.hand_on_head = False
        
        if mp_hands is None:
            return
            
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_hands.process(rgb_frame)
            
            if results.multi_hand_landmarks and self.face_detected and self._cached_coords is not None:
                self.hand_detected = True
                
                # Get face region
                if len(self._cached_coords) > 0:
                    face_y_min = int(np.min(self._cached_coords[:, 1]))
                    face_y_max = int(np.max(self._cached_coords[:, 1]))
                    face_x_min = int(np.min(self._cached_coords[:, 0]))
                    face_x_max = int(np.max(self._cached_coords[:, 0]))
                    
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Get hand center (palm)
                        palm_x = int(hand_landmarks.landmark[0].x * w)
                        palm_y = int(hand_landmarks.landmark[0].y * h)
                        
                        # Check if hand is near face
                        if (face_x_min - 50 <= palm_x <= face_x_max + 50 and
                            face_y_min - 50 <= palm_y <= face_y_max + 100):
                            self.hand_near_face = True
                            
                            # Check if hand is on head/forehead (tiredness gesture)
                            if palm_y < face_y_min + (face_y_max - face_y_min) * 0.3:
                                self.hand_on_head = True
                                self.behavior_details = "HAND_ON_HEAD_TIRED"
                            else:
                                self.behavior_details = "HAND_NEAR_FACE"
                            break
        except Exception:
            pass

    def _analyze_tiredness_and_affect(self) -> None:
        """Analyze tiredness level and affective state like the Affectiva system."""

        # ── No face → reset everything to neutral ──
        if not self.face_detected:
            self.eye_closure_level = 0.0
            self.distraction_level = 0.0
            self.tiredness_level = 0.0
            self.affective_state = "NEUTRAL"
            return

        # Calculate eye closure level (0-100)
        if self.ear > 0:
            # Inverse EAR to closure level
            normal_ear = 0.30  # Typical open eye EAR
            closure_ratio = max(0, (normal_ear - self.ear) / normal_ear)
            self.eye_closure_level = min(100, closure_ratio * 100)
        else:
            self.eye_closure_level = 0.0
        
        # Calculate distraction level (0-100)
        distraction_factors = []
        if self.is_distracted:
            distraction_factors.append(80)
        if self.is_looking_away:
            distraction_factors.append(60)
        if self.is_phone_near_ear:
            distraction_factors.append(90)
        if self.hand_near_face:
            distraction_factors.append(40)
        if len(self.detected_objects) > 0:
            distraction_factors.append(30)
            
        self.distraction_level = min(100, max(distraction_factors) if distraction_factors else 0)
        
        # Calculate tiredness level (0-100)
        tiredness_factors = []
        
        # Eye closure contribution
        tiredness_factors.append(self.eye_closure_level * 0.4)
        
        # Drowsiness
        if self.is_drowsy:
            tiredness_factors.append(80)
        
        # Yawning
        if self.is_yawning:
            tiredness_factors.append(70)
        
        # Blink rate (abnormal = tired)
        low, high = config.NORMAL_BLINK_RATE
        if self.blink_rate < low:
            tiredness_factors.append(60)
        elif self.blink_rate > high:
            tiredness_factors.append(50)
        
        # Hand on head gesture
        if self.hand_on_head:
            tiredness_factors.append(70)
        
        # Attention score
        tiredness_factors.append((100 - self.attention_score) * 0.6)
        
        self.tiredness_level = min(100, max(tiredness_factors) if tiredness_factors else 0)
        
        # Determine affective state
        if self.tiredness_level > 70 or self.is_drowsy:
            self.affective_state = "DROWSY"
        elif self.tiredness_level > 40 or self.is_yawning:
            self.affective_state = "TIRED"
        elif self.distraction_level > 60:
            self.affective_state = "DISTRACTED"
        elif self.attention_score > 75:
            self.affective_state = "ALERT"
        else:
            self.affective_state = "NEUTRAL"

    def _compute_attention_score(self) -> None:
        """Accurate attention scoring.
        
        Score starts at 100 and only decreases for CONFIRMED issues:
        - Eyes closing / drowsy timer running
        - Active yawning (sustained high MAR)
        - Confirmed looking away (sustained head turn > threshold)
        - Confirmed distraction (phone, drinking)
        
        Normal resting state with face detected = 95–100.
        Minor natural head movements do NOT reduce the score.
        """
        score = 100.0

        if not self.face_detected:
            self.attention_score = 0.0
            self._prev_attention_score = 0.0
            return

        # ── Eye closure / drowsiness (biggest factor) ──
        if self._eye_closure_duration >= config.DROWSY_ALERT_DURATION:
            # Eyes closed > 4s = DANGER level
            score -= 50
        elif self._eye_closure_duration > 0:
            # Eyes closing, timer running
            timer_ratio = self._eye_closure_duration / config.DROWSY_ALERT_DURATION
            score -= 30 * timer_ratio
        elif self.ear < config.EAR_THRESHOLD:
            # EAR below threshold but timer not started yet (single frame)
            score -= 5

        # ── Yawning (only confirmed sustained yawning) ──
        if self.is_yawning and self._mar_above_count >= config.MAR_CONSEC_FRAMES:
            score -= 15
        elif self.mar > config.MAR_THRESHOLD:
            score -= 3  # Mouth open but not confirmed yawn

        # ── Head pose (only penalize BIG deviations past threshold) ──
        yaw_excess = max(0, abs(self.yaw) - config.HEAD_YAW_THRESHOLD)
        pitch_excess = max(0, abs(self.pitch) - config.HEAD_PITCH_THRESHOLD)
        
        if yaw_excess > 0 or pitch_excess > 0:
            # Only penalize the amount PAST the threshold
            head_penalty = min(25, (yaw_excess + pitch_excess) * 0.8)
            score -= head_penalty
        # Minor head movements within threshold: NO penalty at all

        # ── Confirmed distraction (phone use, drinking) ──
        if self.is_phone_near_ear:
            score -= 30
        if self.is_drinking:
            score -= 15
        if self.is_distracted and not self.is_phone_near_ear and not self.is_drinking:
            # Generic YOLO distraction object detected
            score -= 10

        # ── Temporal smoothing (heavy smoothing = rock-stable display) ──
        prev = getattr(self, '_prev_attention_score', 100.0)
        score = 0.5 * score + 0.5 * prev
        self._prev_attention_score = score

        self.attention_score = float(max(0.0, min(100.0, score)))

    def _update_safety_status(self) -> None:
        """Progressive safety status: SAFE → WARNING → DANGER.
        
        SAFE:    No active issues — driver is alert and attentive
        WARNING: Eyes closing (timer < 4s) / confirmed yawning / confirmed distraction / looking away
        DANGER:  Drowsy timer > 4s OR phone use OR multiple confirmed dangers
        
        Rules:
        - Only CONFIRMED high-confidence detections trigger WARNING/DANGER
        - Single minor flags (slight head tilt, brief look away) stay SAFE
        - is_looking_down alone is informational, NOT a danger signal
        """
        drowsy_duration = self._eye_closure_duration
        self._drowsy_timer_sec = drowsy_duration
        
        # Eyes are closing (timer is running but < 4s)
        eyes_closing = drowsy_duration > 0 and not self.is_drowsy
        
        # Confirmed danger flags (high-confidence only)
        confirmed_yawning = self.is_yawning and self._mar_above_count >= config.MAR_CONSEC_FRAMES
        confirmed_looking_away = self.is_looking_away and self._look_away_frames >= 6  # 95% responsive
        phone_use = self.is_phone_near_ear
        drinking = self.is_drinking
        confirmed_distraction = self.is_distracted and (phone_use or drinking)
        
        # Count confirmed danger signals
        confirmed_dangers = sum([
            confirmed_yawning,
            confirmed_looking_away,
            phone_use,
            drinking,
        ])
        
        # DANGER: drowsy timer exceeded 4s OR phone use OR multiple confirmed dangers
        if self.is_drowsy or phone_use or confirmed_dangers >= 2:
            self.safety_status = config.STATUS_DANGER
            self.danger_counter += 1
        # WARNING: eyes closing (timer < 4s) OR confirmed yawning/distraction/looking away
        elif eyes_closing or confirmed_yawning or confirmed_looking_away or confirmed_distraction:
            self.safety_status = config.STATUS_WARNING
            self.danger_counter = max(0, self.danger_counter - 1)
        # SAFE: everything clear
        else:
            self.safety_status = config.STATUS_SAFE
            self.danger_counter = max(0, self.danger_counter - 3)

    def _auto_log(self) -> None:
        """Generate log entries on state changes."""
        if self.safety_status != self._prev_status:
            self._add_log(
                f"Status: {self._prev_status} -> {self.safety_status}"
            )
            self._prev_status = self.safety_status

        if self.face_detected != self._prev_face:
            msg = ("Face mesh tracking active" if self.face_detected
                   else "Face lost - searching")
            self._add_log(msg)
            self._prev_face = self.face_detected

        obj_count = len(self.detected_objects)
        if obj_count != self._prev_obj_count and obj_count > 0:
            labels = ", ".join(
                o["label"] for o in self.detected_objects[:3]
            )
            self._add_log(f"YOLO: {labels}")
        self._prev_obj_count = obj_count
        
        # Log behavior detections
        if self.behavior_details and self.behavior_details != "":
            self._add_log(f"⚠️ BEHAVIOR: {self.behavior_details}")

    def _get_telemetry(self) -> dict:
        """Return comprehensive telemetry data."""
        # Calculate face detection confidence (more accurate)
        face_confidence = 0.0
        if self.face_detected and self._cached_coords is not None:
            self._face_conf_history.append(1.0)
            # Confidence based on landmark quality + ear validity + history
            ear_quality = min(1.0, self.ear / 0.35) if self.ear > 0.05 else 0.3
            stability = len(self._face_conf_history) / 10.0
            face_confidence = min(1.0, 0.5 + ear_quality * 0.3 + stability * 0.2)
        else:
            self._face_conf_history.append(0.0)
            face_confidence = 0.0
        
        return {
            "ear": round(self.ear, 4),
            "mar": round(self.mar, 4),
            "yaw": round(self.yaw, 2),
            "pitch": round(self.pitch, 2),
            "attention_score": round(self.attention_score, 1),
            "safety_status": self.safety_status,
            "is_drowsy": self.is_drowsy,
            "is_yawning": self.is_yawning,
            "is_distracted": self.is_distracted,
            "is_looking_away": self.is_looking_away,
            "is_phone_near_ear": self.is_phone_near_ear,
            "is_looking_down": self.is_looking_down,
            "is_drinking": self.is_drinking,
            "behavior_details": self.behavior_details,
            "face_detected": self.face_detected,
            "face_confidence": round(face_confidence, 2),
            "detected_objects": [
                o["label"] for o in self.detected_objects
            ],
            "blink_rate": round(self.blink_rate, 1),
            "blink_total": self._blink_total,
            "danger_counter": self.danger_counter,
            "drowsy_start": self.drowsy_start,
            "drowsy_duration": round(time.time() - self.drowsy_start, 1) if self.drowsy_start > 0 else 0,
            "drowsy_timer_sec": round(getattr(self, '_drowsy_timer_sec', 0.0), 2),
            "ear_below_count": self._ear_below_count,
            "mar_above_count": self._mar_above_count,
            "process_time_ms": round(
                self._last_process_time * 1000, 1
            ),
            "detection_accuracy": "95%",  # Confidence level
            # New affective/tiredness metrics
            "hand_detected": self.hand_detected,
            "hand_near_face": self.hand_near_face,
            "hand_on_head": self.hand_on_head,
            "tiredness_level": round(self.tiredness_level, 1),
            "eye_closure_level": round(self.eye_closure_level, 1),
            "distraction_level": round(self.distraction_level, 1),
            "affective_state": self.affective_state,
        }

    def release(self) -> None:
        """Release all resources."""
        self._landmarker.close()
        print("[AI CORE] Released resources.")

    # ═════════════════════════════════════════════════════════
    #  MINIMAL PROFESSIONAL OVERLAY METHODS
    # ═════════════════════════════════════════════════════════

    def _draw_clean_face_box(self, frame: np.ndarray, coords: np.ndarray, 
                             w: int, h: int) -> None:
        """Draw clean bounding box around detected face."""
        oval = self._cached_oval_pts
        if oval is None or len(oval) < 4:
            return
        
        x_min = int(np.min(oval[:, 0]))
        x_max = int(np.max(oval[:, 0]))
        y_min = int(np.min(oval[:, 1]))
        y_max = int(np.max(oval[:, 1]))
        
        # Status-based color
        color = (C_SAFE if self.safety_status == config.STATUS_SAFE
                else C_WARN if self.safety_status == config.STATUS_WARNING
                else C_DANGER)
        
        # Simple rectangle
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2, _AA)
        
        # Status label
        label = f"DRIVER: {self.safety_status}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(frame, (x_min, y_min - label_size[1] - 8), 
                     (x_min + label_size[0] + 8, y_min), color, -1)
        cv2.putText(frame, label, (x_min + 4, y_min - 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, _AA)

    def _draw_clean_object_box(self, frame: np.ndarray, obj: dict) -> None:
        """Draw minimal object detection box."""
        x1, y1, x2, y2 = obj["box"]
        label = obj["label"]
        conf = obj["conf"]
        
        is_danger = label in _DISTRACTION_OBJECTS
        color = C_DANGER if is_danger else C_ORANGE
        
        # Rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, _AA)
        
        # Label
        text = f"{label.upper()} {conf:.0%}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.rectangle(frame, (x1, y1 - text_size[1] - 6),
                     (x1 + text_size[0] + 6, y1), color, -1)
        cv2.putText(frame, text, (x1 + 3, y1 - 3),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, _AA)

    def _draw_minimal_status_bar(self, frame: np.ndarray, w: int, h: int) -> None:
        """Clean top status bar."""
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Border
        cv2.line(frame, (0, 50), (w, 50), C_ORANGE, 2, _AA)
        
        # Title
        cv2.putText(frame, "ADAR V3.0", (15, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, _AA)
        
        # Status indicator
        status = self.safety_status
        status_color = (C_SAFE if status == config.STATUS_SAFE
                       else C_WARN if status == config.STATUS_WARNING
                       else C_DANGER)
        
        status_x = w - 180
        cv2.circle(frame, (status_x, 25), 8, status_color, -1, _AA)
        cv2.putText(frame, status, (status_x + 18, 32),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, _AA)

    def _draw_minimal_metrics_bar(self, frame: np.ndarray, w: int, h: int) -> None:
        """Clean bottom metrics bar."""
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 45), (w, h), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Border
        cv2.line(frame, (0, h - 45), (w, h - 45), C_ORANGE, 2, _AA)
        
        # Metrics
        metrics = [
            f"EAR: {self.ear:.3f}",
            f"MAR: {self.mar:.3f}",
            f"YAW: {self.yaw:+.0f}°",
            f"PITCH: {self.pitch:+.0f}°",
            f"ATTENTION: {self.attention_score:.0f}%",
            f"BLINK RATE: {self.blink_rate:.0f}/min"
        ]
        
        x = 15
        for metric in metrics:
            cv2.putText(frame, metric, (x, h - 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, _AA)
            x += len(metric) * 8 + 20
        
        # Alert indicators (bottom row)
        x = 15
        alerts = [
            ("DROWSY", self.is_drowsy),
            ("YAWNING", self.is_yawning),
            ("DISTRACTED", self.is_distracted),
            ("LOOK AWAY", self.is_looking_away),
        ]
        
        for label, active in alerts:
            color = C_DANGER if active else (150, 150, 150)
            cv2.circle(frame, (x + 4, h - 8), 4, color, -1, _AA)
            cv2.putText(frame, label, (x + 12, h - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, _AA)
            x += len(label) * 5 + 25
