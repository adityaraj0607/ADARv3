"""
============================================================
 ADAR V3.0 — Firebase Realtime Database Integration
 Provides async-safe, non-blocking writes to Firebase
 Firestore for persistent telemetry, alerts, sensors,
 and vehicle status tracking.

 Firestore Structure:
   vehicles/{vehicle_id}          — online/offline status + latest telemetry
   telemetry/{vehicle_id}/{docId} — historical telemetry frames (throttled)
   alerts/{vehicle_id}/{docId}    — real CV-detected safety alerts
   sensors/{vehicle_id}           — latest sensor snapshot (CO2, ALC, HR, etc.)
   incidents/{vehicle_id}/{docId} — driver client incidents (dual-write)
   emergencies/{vehicle_id}/{docId} — SOS / emergency events
   sessions/{vehicle_id}/{docId}  — driving session logs
============================================================
"""

import asyncio
import json
import os
import threading
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# ── Firebase Admin SDK ─────────────────────────────────────────────
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    print("[FIREBASE] ⚠ firebase-admin not installed — pip install firebase-admin")


class FirebaseDB:
    """
    Non-blocking Firebase Firestore wrapper.
    All writes happen on a background thread to never block
    the FastAPI event loop or WebSocket handlers.
    """

    # Throttle settings — avoid excessive Firestore writes
    TELEMETRY_WRITE_INTERVAL = 5.0    # Write telemetry snapshot every 5s (not every 1s)
    SENSOR_WRITE_INTERVAL = 10.0      # Write sensor snapshot every 10s
    MAX_TELEMETRY_HISTORY = 5000      # Max docs in telemetry sub-collection per vehicle

    def __init__(self) -> None:
        self._db = None
        self._initialized = False
        self._last_telemetry_write: Dict[str, float] = {}
        self._last_sensor_write: Dict[str, float] = {}
        self._write_count = 0
        self._error_count = 0

    def initialize(self, cred_path: Optional[str] = None) -> bool:
        """
        Initialize Firebase Admin SDK with service account credentials.
        Returns True if successful, False otherwise.
        """
        if not FIREBASE_AVAILABLE:
            print("[FIREBASE] ✗ firebase-admin package not installed")
            return False

        if self._initialized:
            print("[FIREBASE] Already initialized")
            return True

        # Find credentials file
        if cred_path is None:
            # Search common locations
            search_paths = [
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adar-driver-safety-firebase-adminsdk-fbsvc-dc6fc03b02.json"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "firebase-credentials.json"),
                os.getenv("FIREBASE_CREDENTIALS", ""),
            ]
            for p in search_paths:
                if p and os.path.isfile(p):
                    cred_path = p
                    break

        if not cred_path or not os.path.isfile(cred_path):
            print("[FIREBASE] ✗ No credentials file found")
            return False

        try:
            cred = credentials.Certificate(cred_path)
            # Check if already initialized
            try:
                firebase_admin.get_app()
            except ValueError:
                firebase_admin.initialize_app(cred)

            self._db = firestore.client()
            self._initialized = True
            print(f"[FIREBASE] ✓ Connected to Firestore project: adar-driver-safety")
            return True
        except Exception as e:
            print(f"[FIREBASE] ✗ Initialization failed: {e}")
            traceback.print_exc()
            return False

    @property
    def is_active(self) -> bool:
        return self._initialized and self._db is not None

    # ──────────────────────────────────────────────────────────
    #  WRITE HELPERS (fire-and-forget on background thread)
    # ──────────────────────────────────────────────────────────

    def _bg_write(self, func, *args, **kwargs):
        """Run a Firestore write on a background thread (non-blocking)."""
        if not self.is_active:
            return
        def _worker():
            try:
                func(*args, **kwargs)
                self._write_count += 1
            except Exception as e:
                self._error_count += 1
                if self._error_count <= 10 or self._error_count % 100 == 0:
                    print(f"[FIREBASE] Write error #{self._error_count}: {e}")
        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def _safe_data(self, data: dict) -> dict:
        """Clean data for Firestore (remove None, convert datetime, limit depth)."""
        clean = {}
        for k, v in data.items():
            if v is None:
                continue
            if isinstance(v, datetime):
                clean[k] = v.isoformat()
            elif isinstance(v, dict):
                clean[k] = self._safe_data(v)
            elif isinstance(v, (list, tuple)):
                clean[k] = [self._safe_data(i) if isinstance(i, dict) else i for i in v]
            elif isinstance(v, float) and (v != v):  # NaN check
                clean[k] = 0.0
            else:
                clean[k] = v
        return clean

    # ──────────────────────────────────────────────────────────
    #  VEHICLE STATUS
    # ──────────────────────────────────────────────────────────

    def set_vehicle_online(self, vehicle_id: str, ip_address: str = "") -> None:
        """Mark a vehicle as online in Firestore."""
        def _write():
            doc_ref = self._db.collection("vehicles").document(vehicle_id)
            doc_ref.set({
                "status": "online",
                "connected_at": datetime.now(timezone.utc).isoformat(),
                "last_seen": datetime.now(timezone.utc).isoformat(),
                "ip_address": ip_address,
                "vehicle_id": vehicle_id,
            }, merge=True)
        self._bg_write(_write)

    def set_vehicle_offline(self, vehicle_id: str) -> None:
        """Mark a vehicle as offline in Firestore."""
        def _write():
            doc_ref = self._db.collection("vehicles").document(vehicle_id)
            doc_ref.set({
                "status": "offline",
                "disconnected_at": datetime.now(timezone.utc).isoformat(),
                "last_seen": datetime.now(timezone.utc).isoformat(),
            }, merge=True)
        self._bg_write(_write)

    # ──────────────────────────────────────────────────────────
    #  TELEMETRY (throttled — every 5s per vehicle)
    # ──────────────────────────────────────────────────────────

    def write_telemetry(self, vehicle_id: str, data: dict) -> None:
        """
        Write a telemetry frame to Firestore.
        Throttled to TELEMETRY_WRITE_INTERVAL to avoid excessive writes.
        Also updates vehicles/{vehicle_id} with latest snapshot.
        """
        now = time.time()
        last = self._last_telemetry_write.get(vehicle_id, 0)
        if now - last < self.TELEMETRY_WRITE_INTERVAL:
            return  # Throttled
        self._last_telemetry_write[vehicle_id] = now

        safe = self._safe_data(data)
        safe["_written_at"] = datetime.now(timezone.utc).isoformat()

        def _write():
            # 1. Update latest snapshot on vehicles/{vehicle_id}
            self._db.collection("vehicles").document(vehicle_id).set({
                "latest_telemetry": safe,
                "last_seen": datetime.now(timezone.utc).isoformat(),
                "status": "online",
            }, merge=True)

            # 2. Add to telemetry history sub-collection
            self._db.collection("telemetry").document(vehicle_id) \
                .collection("history").add(safe)

        self._bg_write(_write)

    # ──────────────────────────────────────────────────────────
    #  ALERTS (every real alert — not throttled)
    # ──────────────────────────────────────────────────────────

    def write_alert(self, vehicle_id: str, alert_data: dict) -> None:
        """Write a safety alert to Firestore (drowsiness, yawning, etc.)."""
        safe = self._safe_data(alert_data)
        safe["_written_at"] = datetime.now(timezone.utc).isoformat()
        safe["vehicle_id"] = vehicle_id

        def _write():
            self._db.collection("alerts").document(vehicle_id) \
                .collection("history").add(safe)

            # Update vehicle's alert_count
            self._db.collection("vehicles").document(vehicle_id).set({
                "last_alert": safe,
                "last_alert_time": datetime.now(timezone.utc).isoformat(),
            }, merge=True)

        self._bg_write(_write)

    # ──────────────────────────────────────────────────────────
    #  SENSOR DATA (throttled — every 10s per vehicle)
    # ──────────────────────────────────────────────────────────

    def write_sensor_data(self, vehicle_id: str, sensor_data: dict) -> None:
        """Write sensor snapshot (CO2, alcohol, HR, SpO2, IMU, presence)."""
        now = time.time()
        last = self._last_sensor_write.get(vehicle_id, 0)
        if now - last < self.SENSOR_WRITE_INTERVAL:
            return
        self._last_sensor_write[vehicle_id] = now

        safe = self._safe_data(sensor_data)
        safe["_written_at"] = datetime.now(timezone.utc).isoformat()

        def _write():
            # Latest snapshot (overwrite)
            self._db.collection("sensors").document(vehicle_id).set(safe, merge=True)

            # Historical record
            self._db.collection("sensors").document(vehicle_id) \
                .collection("history").add(safe)

        self._bg_write(_write)

    # ──────────────────────────────────────────────────────────
    #  GPS (update latest position)
    # ──────────────────────────────────────────────────────────

    def write_gps(self, vehicle_id: str, lat: float, lon: float,
                  accuracy: float = None, heading: float = None,
                  speed_mps: float = None) -> None:
        """Write GPS position to Firestore."""
        gps_data = {
            "lat": lat, "lon": lon,
            "accuracy": accuracy, "heading": heading,
            "speed_mps": speed_mps,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        safe = self._safe_data(gps_data)

        def _write():
            self._db.collection("vehicles").document(vehicle_id).set({
                "location": safe,
                "last_gps_update": datetime.now(timezone.utc).isoformat(),
            }, merge=True)

        self._bg_write(_write)

    # ──────────────────────────────────────────────────────────
    #  EMERGENCY / SOS
    # ──────────────────────────────────────────────────────────

    def write_emergency(self, vehicle_id: str, emergency_data: dict) -> None:
        """Write an SOS/emergency event to Firestore."""
        safe = self._safe_data(emergency_data)
        safe["_written_at"] = datetime.now(timezone.utc).isoformat()
        safe["vehicle_id"] = vehicle_id

        def _write():
            self._db.collection("emergencies").document(vehicle_id) \
                .collection("history").add(safe)

            self._db.collection("vehicles").document(vehicle_id).set({
                "last_emergency": safe,
                "last_emergency_time": datetime.now(timezone.utc).isoformat(),
            }, merge=True)

        self._bg_write(_write)

    # ──────────────────────────────────────────────────────────
    #  INCIDENTS (from Driver Client — dual-write with SQLite)
    # ──────────────────────────────────────────────────────────

    def write_incident(self, vehicle_id: str, incident_data: dict) -> None:
        """Write an incident from the Driver Client to Firestore."""
        safe = self._safe_data(incident_data)
        safe["_written_at"] = datetime.now(timezone.utc).isoformat()
        safe["vehicle_id"] = vehicle_id

        def _write():
            self._db.collection("incidents").document(vehicle_id) \
                .collection("history").add(safe)

        self._bg_write(_write)

    # ──────────────────────────────────────────────────────────
    #  SESSIONS (driving session logs)
    # ──────────────────────────────────────────────────────────

    def write_session(self, vehicle_id: str, session_data: dict) -> None:
        """Log a driving session to Firestore."""
        safe = self._safe_data(session_data)
        safe["_written_at"] = datetime.now(timezone.utc).isoformat()

        def _write():
            self._db.collection("sessions").document(vehicle_id) \
                .collection("history").add(safe)

        self._bg_write(_write)

    # ──────────────────────────────────────────────────────────
    #  STATS / STATUS
    # ──────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return Firebase write statistics."""
        return {
            "firebase_active": self.is_active,
            "total_writes": self._write_count,
            "total_errors": self._error_count,
        }


# ── Module-level singleton ────────────────────────────────────────
firebase = FirebaseDB()
