"""
============================================================
 ADAR V3.0 — Routes & Engine Controller
 Handles Flask routes, MJPEG video streaming,
 SocketIO telemetry broadcasting, and background threads
 (Camera + AI, Jarvis alerts, Spatial scanning).
============================================================
"""

import cv2
import time
import threading
import base64
import openai
from flask import Response, render_template, jsonify
from datetime import datetime

import config
from app.camera import Camera
from app.ai_core import AICore
from app.jarvis import Jarvis
from app.database import get_incident_stats, get_recent_incidents
from bridge import FleetConnector

# ── Module-level references (populated by register_routes) ──
_camera: Camera | None = None
_ai_core: AICore | None = None
_jarvis: Jarvis | None = None
_fleet: FleetConnector | None = None
_socketio = None

_engine_running = False
_session_start: datetime | None = None

# MJPEG frames now served directly from Camera thread (zero-copy)


# ═════════════════════════════════════════════════════════════
#  PUBLIC API — called from __init__.py and main.py
# ═════════════════════════════════════════════════════════════

def register_routes(app, socketio):
    """Register all Flask routes and start the engine threads."""
    global _socketio
    _socketio = socketio

    # ── Dashboard page ──
    @app.route("/")
    def dashboard():
        return render_template("hud.html")

    # ── MJPEG video feed ──
    @app.route("/video_feed")
    def video_feed():
        return Response(
            _mjpeg_generator(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    # ── Stats API ──
    @app.route("/api/stats")
    def api_stats():
        stats = get_incident_stats(since=_session_start)
        return jsonify(stats)

    # ── Recent incidents API ──
    @app.route("/api/incidents")
    def api_incidents():
        incidents = get_recent_incidents(limit=50)
        return jsonify(incidents)

    # ── SocketIO connect event ──
    @socketio.on("connect")
    def on_connect():
        jarvis_online = (
            _jarvis is not None
            and _jarvis.client is not None
        )
        socketio.emit("system_status", {
            "message": "ADAR V3.0 online — all systems nominal.",
            "jarvis_online": jarvis_online,
        })
        # Load historical session stats from DB and send to client
        try:
            stats = get_incident_stats(since=_session_start)
            socketio.emit("session_stats_init", {
                "total_alerts": stats.get("total", 0),
                "drowsiness": stats.get("drowsiness", 0),
                "yawning": stats.get("yawning", 0),
                "distraction": stats.get("distraction", 0),
            })
        except Exception:
            pass
    # ── Start the engine ──
    _start_engine(socketio)


def stop_engine():
    """Gracefully shut down all background threads and resources."""
    global _engine_running, _camera, _ai_core, _jarvis

    _engine_running = False
    time.sleep(0.3)  # Let threads exit their loops

    if _camera is not None:
        _camera.stop()
        _camera = None

    if _ai_core is not None:
        _ai_core.release()
        _ai_core = None

    if _jarvis is not None:
        _jarvis.shutdown()
        _jarvis = None

    if _fleet is not None:
        _fleet.disconnect()

    print("[ENGINE] All resources released.")


# ═════════════════════════════════════════════════════════════
#  ENGINE START — Initialize components and launch threads
# ═════════════════════════════════════════════════════════════

def _start_engine(socketio):
    """Initialize camera, AI core, Jarvis, and start worker threads."""
    global _camera, _ai_core, _jarvis, _engine_running, _session_start

    _session_start = datetime.utcnow()

    # Initialize camera
    _camera = Camera()
    _camera.start()

    # Initialize AI core
    _ai_core = AICore()

    # Initialize Jarvis
    _jarvis = Jarvis(socketio=socketio)

    _engine_running = True

    # ── Fleet Bridge (background connect to Fleet Server :8000) ──
    global _fleet
    _fleet = FleetConnector(vehicle_id="ADAR-01", server_url=config.FLEET_SERVER_URL, verbose=True)
    _fleet.connect(timeout=3.0)  # Non-blocking — retries in background

    # Thread A — Main processing loop (camera + AI + telemetry emit)
    thread_a = threading.Thread(
        target=_processing_loop,
        args=(socketio,),
        daemon=True,
        name="Thread-A-Processing",
    )
    thread_a.start()
    print("[ENGINE] Thread A (Processing) started ✓")

    # Thread C — Spatial analysis (GPT-4o room scan)
    if _jarvis.client is not None:
        thread_c = threading.Thread(
            target=_spatial_scan_loop,
            daemon=True,
            name="Thread-C-Spatial",
        )
        thread_c.start()
        print("[ENGINE] Thread C (Spatial Scan) started ✓")
    else:
        print("[ENGINE] Thread C (Spatial Scan) skipped — no API key")

    print("[ENGINE] All threads started — system active.")


# ═════════════════════════════════════════════════════════════
#  THREAD A — Main Processing Loop
# ═════════════════════════════════════════════════════════════

def _processing_loop(socketio):
    """Read camera → AI process → draw overlay → emit telemetry."""
    yolo_interval = 5  # Run YOLO every 5 frames for better object detection
    frame_count = 0
    last_emit = 0.0

    # Wait for camera to be ready
    print("[Thread A] Waiting for camera...")
    for _ in range(100):
        if not _engine_running:
            return
        ok, _ = _camera.read()
        if ok:
            break
        time.sleep(0.1)
    print("[Thread A] Camera ready — entering main loop.")

    while _engine_running:
        # Read frame
        ok, frame = _camera.read()
        if not ok or frame is None:
            time.sleep(0.001)
            continue

        frame_count += 1
        run_yolo = (frame_count % yolo_interval == 0)

        # AI processing (always run)
        telemetry = _ai_core.process_frame(frame, run_yolo=run_yolo)

        # MJPEG frames are now encoded in the camera thread at full 30fps,
        # completely decoupled from AI processing latency.

        # Emit telemetry via SocketIO (throttled)
        now = time.time()
        if now - last_emit >= config.SOCKETIO_EMIT_INTERVAL:
            telemetry["camera_fps"] = int(_camera.fps)
            try:
                socketio.emit("telemetry_update", telemetry)
            except Exception:
                pass

            # ── Send to Fleet Server (non-blocking) ──
            if _fleet is not None and _fleet.is_connected:
                try:
                    # Build real drowsiness alerts list (only genuine detections)
                    real_alerts = []
                    if telemetry.get('is_drowsy'):
                        dur = telemetry.get('drowsy_duration', 0)
                        real_alerts.append(f"DROWSINESS DETECTED — eyes closed {dur:.1f}s")
                    if telemetry.get('is_yawning'):
                        real_alerts.append("YAWNING DETECTED")
                    if telemetry.get('is_distracted'):
                        real_alerts.append("DISTRACTION DETECTED")
                    if telemetry.get('is_looking_away'):
                        real_alerts.append("LOOKING AWAY FROM ROAD")
                    if telemetry.get('is_phone_near_ear'):
                        real_alerts.append("PHONE USE DETECTED")
                    if telemetry.get('is_looking_down'):
                        real_alerts.append("LOOKING DOWN")

                    _fleet.send_telemetry(
                        ear=telemetry.get('ear', 0),
                        mar=telemetry.get('mar', 0),
                        co2=telemetry.get('co2', 0),
                        status=telemetry.get('safety_status', 'SAFE'),
                        attention_score=telemetry.get('attention_score', 100),
                        alerts=real_alerts,
                        extra={
                            "is_drowsy": telemetry.get('is_drowsy', False),
                            "is_yawning": telemetry.get('is_yawning', False),
                            "is_distracted": telemetry.get('is_distracted', False),
                            "is_looking_away": telemetry.get('is_looking_away', False),
                            "is_phone_near_ear": telemetry.get('is_phone_near_ear', False),
                            "is_looking_down": telemetry.get('is_looking_down', False),
                            "is_drinking": telemetry.get('is_drinking', False),
                            "drowsy_duration": telemetry.get('drowsy_duration', 0),
                            "drowsy_timer_sec": telemetry.get('drowsy_timer_sec', 0),
                            "face_detected": telemetry.get('face_detected', False),
                            "face_confidence": telemetry.get('face_confidence', 0),
                            "affective_state": telemetry.get('affective_state', ''),
                            "blink_rate": telemetry.get('blink_rate', 0),
                            "danger_counter": telemetry.get('danger_counter', 0),
                            "detected_objects": telemetry.get('detected_objects', []),
                            "behavior_details": telemetry.get('behavior_details', ''),
                            "yaw": telemetry.get('yaw', 0),
                            "pitch": telemetry.get('pitch', 0),
                            "tiredness_level": telemetry.get('tiredness_level', 0),
                            "process_time_ms": telemetry.get('process_time_ms', 0),
                        },
                    )
                except Exception:
                    pass

            last_emit = now

        # ═══ Jarvis Alert Routing ═══
        # PATH A: Drowsy timer >= 3.5s → OFFLINE instant (no GPT call, zero latency)
        # PATH B: ALL other alerts     → GPT-5.2 instant (live OpenAI vision analysis)
        drowsy_duration = (
            time.time() - _ai_core.drowsy_start
            if _ai_core.drowsy_start > 0 else 0.0
        )
        is_drowsy_timer = drowsy_duration >= config.DROWSY_ALERT_DURATION
        is_critical_drowsy = drowsy_duration >= (config.DROWSY_ALERT_DURATION * 2)

        if is_drowsy_timer:
            # ── PATH A: DROWSINESS → Offline instant alert (fast response) ──
            if _jarvis.client is not None:
                if is_critical_drowsy:
                    _jarvis.trigger_drowsy_alert(telemetry, force=True)
                elif _jarvis.is_ready:
                    _jarvis.trigger_drowsy_alert(telemetry)
        elif _jarvis.client is not None and _jarvis.is_ready:
            # ── PATH B: ALL other dangers → GPT-5.2 with specific context ──
            alert_reason = None

            if _ai_core.is_phone_near_ear:
                alert_reason = "PHONE_USE"
            elif _ai_core.is_distracted and _ai_core.is_ready_for_alert():
                alert_reason = "DISTRACTION"
            elif _ai_core.is_yawning and _ai_core._mar_above_count >= config.MAR_CONSEC_FRAMES:
                alert_reason = "YAWNING"
            elif _ai_core.is_looking_away and _ai_core._look_away_frames >= 4:
                alert_reason = "LOOKING_AWAY"
            elif _ai_core.is_looking_down and _ai_core.pitch < -28:
                alert_reason = "LOOKING_DOWN"
            elif _ai_core.is_drinking:
                alert_reason = "DRINKING"
            elif len(_ai_core.detected_objects) > 0 and _ai_core.is_distracted:
                alert_reason = "OBJECT_DETECTED"
            elif _ai_core.safety_status == config.STATUS_DANGER:
                alert_reason = "DANGER"
            elif (_ai_core.safety_status == config.STATUS_WARNING
                  and _ai_core.danger_counter >= config.DANGER_FRAME_THRESHOLD):
                alert_reason = "WARNING_SUSTAINED"

            if alert_reason:
                _jarvis.trigger_alert(frame, telemetry, alert_reason=alert_reason)

        # No artificial frame limiting - run at camera speed


# ═════════════════════════════════════════════════════════════
#  THREAD C — Spatial Analysis (GPT-4o Room Scan)
# ═════════════════════════════════════════════════════════════

def _spatial_scan_loop():
    """Periodically send a frame to GPT-4o for spatial analysis."""
    time.sleep(3.0)  # Initial delay to let the system warm up

    while _engine_running:
        try:
            if _camera is None or _ai_core is None or _jarvis is None:
                time.sleep(1.0)
                continue

            if _jarvis.client is None:
                time.sleep(5.0)
                continue

            # Grab current frame
            ok, frame = _camera.read()
            if not ok or frame is None:
                time.sleep(1.0)
                continue

            # Encode frame
            _, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50]
            )
            frame_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

            # Call GPT-4o for spatial analysis
            response = _jarvis.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": config.SPATIAL_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{frame_b64}",
                                    "detail": "low",
                                },
                            },
                        ],
                    }
                ],
                max_completion_tokens=120,
                temperature=0.5,
            )

            text = response.choices[0].message.content
            if text:
                _ai_core.set_spatial_analysis(text.strip())
                _ai_core.add_log("Spatial scan updated")

        except openai.RateLimitError:
            _ai_core.add_log("Spatial: rate limited")
            time.sleep(30.0)
        except Exception as e:
            _ai_core.add_log(f"Spatial: error")
            print(f"[Thread C] Spatial scan error: {e}")

        time.sleep(config.SPATIAL_SCAN_INTERVAL)


# ═════════════════════════════════════════════════════════════
#  MJPEG GENERATOR
# ═════════════════════════════════════════════════════════════

def _mjpeg_generator():
    """Yield MJPEG frames from camera thread (decoupled from AI processing)."""
    last_frame_id = None
    target_interval = 1.0 / 30.0  # Target 30fps
    last_yield_time = 0.0

    while True:
        # Read pre-encoded JPEG directly from camera thread
        jpeg_bytes = _camera._current_jpeg if _camera else None

        if jpeg_bytes is None:
            time.sleep(0.005)
            continue

        # Skip if same frame (identity check — fast)
        frame_id = id(jpeg_bytes)
        if frame_id == last_frame_id:
            time.sleep(0.003)
            continue

        # Pace output for smooth delivery
        now = time.time()
        wait = target_interval - (now - last_yield_time)
        if wait > 0.001:
            time.sleep(wait)

        last_frame_id = frame_id
        last_yield_time = time.time()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg_bytes
            + b"\r\n"
        )
