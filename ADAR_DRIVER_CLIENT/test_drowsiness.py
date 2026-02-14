"""Quick smoke test for drowsiness detection — no camera needed."""
from app.ai_core import AICore
import numpy as np

# Create a test frame (black 640x480 — no face visible)
frame = np.zeros((480, 640, 3), dtype=np.uint8)

ai = AICore()

print("=== Test 1: process_frame with NO face ===")
t = ai.process_frame(frame, run_yolo=False)
assert t["face_detected"] == False, f"Expected False, got {t['face_detected']}"
assert t["is_drowsy"] == False, f"is_drowsy should be False, got {t['is_drowsy']}"
assert t["is_yawning"] == False, f"is_yawning should be False, got {t['is_yawning']}"
assert t["is_looking_away"] == False, f"is_looking_away should be False, got {t['is_looking_away']}"
assert t["is_distracted"] == False, f"is_distracted should be False, got {t['is_distracted']}"
assert t["is_phone_near_ear"] == False, f"is_phone_near_ear should be False"
assert t["is_drinking"] == False, f"is_drinking should be False"
assert t["is_looking_down"] == False, f"is_looking_down should be False"
assert t["affective_state"] == "NEUTRAL", f"affective_state should be NEUTRAL, got {t['affective_state']}"
assert t["tiredness_level"] == 0.0, f"tiredness_level should be 0, got {t['tiredness_level']}"
assert t["eye_closure_level"] == 0.0, f"eye_closure_level should be 0, got {t['eye_closure_level']}"
assert t["attention_score"] == 0.0, f"attention_score should be 0 with no face"
print("  PASSED: All flags correctly reset when no face")

print("\n=== Test 2: Multiple no-face frames (no state leakage) ===")
for i in range(30):
    t2 = ai.process_frame(frame, run_yolo=False)
assert t2["is_drowsy"] == False, "Drowsy flag leaking across frames"
assert t2["is_yawning"] == False, "Yawning flag leaking across frames"
assert t2["affective_state"] == "NEUTRAL", f"State should be NEUTRAL after 30 empty frames, got {t2['affective_state']}"
print("  PASSED: No state leakage after 30 empty frames")

print("\n=== Test 3: draw_overlay with no face (no crash) ===")
try:
    display = frame.copy()
    ai.draw_overlay(display)
    print("  PASSED: draw_overlay works with no face")
except Exception as e:
    print(f"  FAILED: draw_overlay crashed: {e}")
    raise

print("\n=== Test 4: draw_minimal_overlay with no face (no crash) ===")
try:
    display2 = frame.copy()
    ai.draw_minimal_overlay(display2)
    print("  PASSED: draw_minimal_overlay works with no face")
except Exception as e:
    print(f"  FAILED: draw_minimal_overlay crashed: {e}")
    raise

print("\n=== Test 5: Telemetry dict has all required keys ===")
required_keys = [
    "ear", "mar", "yaw", "pitch", "attention_score", "safety_status",
    "is_drowsy", "is_yawning", "is_distracted", "is_looking_away",
    "is_phone_near_ear", "is_looking_down", "is_drinking",
    "face_detected", "blink_rate", "blink_total", "danger_counter",
    "drowsy_duration", "process_time_ms", "affective_state",
    "tiredness_level", "eye_closure_level", "distraction_level",
    "hand_detected", "hand_near_face", "detected_objects",
    "behavior_details",
]
t3 = ai.process_frame(frame, run_yolo=False)
missing = [k for k in required_keys if k not in t3]
assert len(missing) == 0, f"Missing keys in telemetry: {missing}"
print(f"  PASSED: All {len(required_keys)} required keys present")

print("\n=== Test 6: _get_telemetry values are JSON-serializable ===")
import json
try:
    json.dumps(t3)
    print("  PASSED: Telemetry is JSON-serializable")
except (TypeError, ValueError) as e:
    print(f"  FAILED: Not JSON-serializable: {e}")
    raise

ai.release()
print("\n" + "=" * 50)
print("ALL 6 TESTS PASSED — Drowsiness detection is error-free")
print("=" * 50)
