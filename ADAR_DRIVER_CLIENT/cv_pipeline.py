"""
============================================================
 ADAR V3.0 — CV Pipeline (The Brain → Dashboard Connector)

 Connects the OpenCV drowsiness-detection engine (AICore)
 to the Fleet Command Center dashboard via WebSocket.

 How it works:
   Camera → AICore.process_frame() → real EAR/MAR/status →
   ADARBridge → server.py WS → dashboard.html

 Run:
   python cv_pipeline.py
   python cv_pipeline.py --show           (local HUD window)
   python cv_pipeline.py --show --clean   (clean feed, no HUD)
   python cv_pipeline.py --url ws://192.168.1.5:8000/ws/vehicle/ADAR-001

 Requirements:
   - server.py must be running (python server.py)
   - Camera must be available (laptop webcam or USB)
============================================================
"""

import argparse
import signal
import sys
import time

import cv2
import numpy as np

import config
from app.camera import Camera
from app.ai_core import AICore
from local_bridge import ADARBridge


# ═════════════════════════════════════════════════════════════
#  CLEAN LOCAL DISPLAY (lightweight info overlay)
# ═════════════════════════════════════════════════════════════

def _draw_clean_info(frame: np.ndarray, telemetry: dict,
                     fps: int, connected: bool) -> None:
    """Minimal local monitor overlay — no HUD, just key metrics."""
    h, w = frame.shape[:2]
    aa = cv2.LINE_AA

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 62), (15, 15, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Title + FPS
    cv2.putText(frame, "ADAR V3.0", (12, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, aa)
    cv2.putText(frame, f"FPS: {fps}", (12, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 100), 1, aa)

    # Connection status
    conn_color = (0, 220, 110) if connected else (0, 0, 220)
    conn_text = "LIVE" if connected else "DISCONNECTED"
    cv2.circle(frame, (w - 115, 15), 5, conn_color, -1, aa)
    cv2.putText(frame, conn_text, (w - 105, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, conn_color, 1, aa)

    # Face status
    face = telemetry.get("face_detected", False)
    face_color = (0, 220, 110) if face else (0, 100, 220)
    cv2.putText(frame, f"Face: {'YES' if face else 'NO'}",
                (w - 105, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                face_color, 1, aa)

    # Key metrics bar at bottom
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 38), (w, h), (15, 15, 20), -1)
    cv2.addWeighted(overlay2, 0.65, frame, 0.35, 0, frame)

    ear = telemetry.get("ear", 0)
    mar = telemetry.get("mar", 0)
    status = telemetry.get("safety_status", "SAFE")
    attn = telemetry.get("attention_score", 100)
    state = telemetry.get("affective_state", "NEUTRAL")

    ear_c = (0, 0, 240) if ear < config.EAR_THRESHOLD else (200, 200, 200)
    mar_c = (0, 0, 240) if mar > config.MAR_THRESHOLD else (200, 200, 200)
    status_c = (0, 0, 240) if status == "DANGER" else (0, 220, 110)

    metrics = (
        f"EAR:{ear:.3f}  MAR:{mar:.3f}  "
        f"ATTN:{attn:.0f}%  [{state}]  {status}"
    )
    cv2.putText(frame, metrics, (12, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, aa)

    # Status color indicator
    cv2.circle(frame, (w - 20, h - 19), 8, status_c, -1, aa)


# ═════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ADAR V3.0 — CV Pipeline → Fleet Dashboard"
    )
    parser.add_argument(
        "--url",
        default="ws://localhost:8000/ws/vehicle/ADAR-001",
        help="WebSocket URL of the Fleet Command Center",
    )
    parser.add_argument(
        "--vehicle-id", default="ADAR-001",
        help="Vehicle identifier sent to the server",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Show local camera window (with JARVIS HUD overlay)",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Use clean display instead of full HUD (requires --show)",
    )
    parser.add_argument(
        "--camera", type=int, default=None,
        help="Camera index (default: config.CAMERA_INDEX)",
    )
    parser.add_argument(
        "--yolo-interval", type=int, default=8,
        help="Run YOLO every N frames (default: 8)",
    )
    args = parser.parse_args()

    print(r"""
    ╔═══════════════════════════════════════════════════════════╗
    ║         ADAR V3.0 — CV Pipeline                          ║
    ║         Camera → AI → Fleet Dashboard                    ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    # ── Initialize Camera ──
    cam_src = args.camera if args.camera is not None else config.CAMERA_INDEX
    print(f"[PIPELINE] Opening camera (source={cam_src})...")
    camera = Camera(src=cam_src)
    camera.start()

    # Wait for first frame
    print("[PIPELINE] Waiting for camera...")
    cam_ready = False
    for _ in range(100):
        ok, _ = camera.read()
        if ok:
            cam_ready = True
            break
        time.sleep(0.1)

    if not cam_ready:
        print("[PIPELINE] ✗ Camera failed to open! Check connection.")
        camera.stop()
        sys.exit(1)
    print("[PIPELINE] ✓ Camera ready")

    # ── Initialize AI Core ──
    print("[PIPELINE] Loading AI models...")
    ai_core = AICore()
    print("[PIPELINE] ✓ AI Core ready")

    # ── Initialize Bridge ──
    bridge = ADARBridge(vehicle_id=args.vehicle_id, verbose=True)
    print(f"[PIPELINE] Connecting to server: {args.url}")
    connected = bridge.connect(args.url, timeout=8.0)
    if connected:
        print("[PIPELINE] ✓ Connected to Fleet Command Center")
    else:
        print("[PIPELINE] ⚠ Connection pending — will retry in background")

    # ── Signal Handling ──
    running = True

    def _on_exit(sig, frame):
        nonlocal running
        print("\n[PIPELINE] Shutting down...")
        running = False

    signal.signal(signal.SIGINT, _on_exit)
    signal.signal(signal.SIGTERM, _on_exit)

    # ── Main Loop ──
    frame_count = 0
    fps_count = 0
    fps_timer = time.time()
    display_fps = 0
    send_interval = 0.05  # Send telemetry at ~20 Hz max
    last_send = 0.0

    print("[PIPELINE] ═══════════════════════════════════")
    print("[PIPELINE]  Pipeline ACTIVE — press Ctrl+C or 'q' to stop")
    print("[PIPELINE] ═══════════════════════════════════")

    try:
        while running:
            # Read frame
            ok, frame = camera.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue

            frame_count += 1
            run_yolo = (frame_count % args.yolo_interval == 0)

            # ── AI Processing ──
            telemetry = ai_core.process_frame(frame, run_yolo=run_yolo)

            # ── Build Alerts ──
            alerts = []
            if telemetry["is_drowsy"]:
                drowsy_dur = telemetry.get("drowsy_duration", 0)
                if drowsy_dur >= config.DROWSY_ALERT_DURATION:
                    alerts.append(f"⚠️ DROWSINESS — {drowsy_dur:.0f}s")
                else:
                    alerts.append("Microsleep Detected")
            if telemetry["is_yawning"]:
                alerts.append("Excessive Yawning")
            if telemetry["is_distracted"]:
                alerts.append("Driver Distracted")
            if telemetry["is_looking_away"]:
                alerts.append("Looking Away From Road")
            if telemetry["is_phone_near_ear"]:
                alerts.append("Phone Near Ear")
            if telemetry["is_drinking"]:
                alerts.append("Drinking While Driving")
            if telemetry["is_looking_down"]:
                alerts.append("Looking Down")

            status = telemetry["safety_status"]
            # Map WARNING → SAFE for the dashboard (it only knows SAFE/DANGER)
            if status == "WARNING":
                status = "SAFE"

            # ── Send to Server ──
            now = time.time()
            if now - last_send >= send_interval:
                bridge.send_data(
                    ear=telemetry["ear"],
                    mar=telemetry["mar"],
                    co2=0.0,          # ESP32 handles CO2 separately
                    status=status,
                    alerts=alerts,
                    extra={
                        # Detection flags
                        "face_detected": telemetry["face_detected"],
                        "is_drowsy": telemetry["is_drowsy"],
                        "is_yawning": telemetry["is_yawning"],
                        "is_distracted": telemetry["is_distracted"],
                        "is_looking_away": telemetry["is_looking_away"],
                        "is_phone_near_ear": telemetry["is_phone_near_ear"],
                        "is_looking_down": telemetry["is_looking_down"],
                        "is_drinking": telemetry["is_drinking"],
                        # Scores
                        "attention_score": telemetry["attention_score"],
                        "tiredness_level": telemetry["tiredness_level"],
                        "eye_closure_level": telemetry["eye_closure_level"],
                        "distraction_level": telemetry["distraction_level"],
                        "affective_state": telemetry["affective_state"],
                        # Head pose
                        "yaw": telemetry["yaw"],
                        "pitch": telemetry["pitch"],
                        # Blink metrics
                        "blink_rate": telemetry["blink_rate"],
                        "blink_total": telemetry["blink_total"],
                        # Objects
                        "detected_objects": telemetry["detected_objects"],
                        # Performance
                        "process_time_ms": telemetry["process_time_ms"],
                        "camera_fps": int(camera.fps),
                    },
                )
                last_send = now

            # ── FPS Counter ──
            fps_count += 1
            if now - fps_timer >= 1.0:
                display_fps = fps_count
                fps_count = 0
                fps_timer = now

            # ── Local Display (optional) ──
            if args.show:
                display_frame = frame.copy()
                if args.clean:
                    _draw_clean_info(
                        display_frame, telemetry,
                        display_fps, bridge.is_connected,
                    )
                else:
                    ai_core.draw_overlay(display_frame)

                cv2.imshow("ADAR V3.0", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    except Exception as e:
        print(f"[PIPELINE] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # ── Cleanup ──
        print("[PIPELINE] Releasing resources...")
        bridge.disconnect()
        camera.stop()
        ai_core.release()
        if args.show:
            cv2.destroyAllWindows()
        print("[PIPELINE] ✓ Shutdown complete.")


if __name__ == "__main__":
    main()
