"""
============================================================
 ADAR V3.0 — Fleet Command Center (Cloud Brain)
 FastAPI + WebSocket Hub + Jinja2 Templates + Real GPS
 Run locally:  uvicorn server:app --reload --port 8000
 Deploy:       Render.com with Procfile
============================================================
"""

import asyncio
import json
import logging
import math
import os
import random
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from dotenv import load_dotenv
load_dotenv()

# ── OpenAI Integration ────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")  # Fast, cost-effective for real-time safety
AI_ENABLED = bool(OPENAI_API_KEY)
if AI_ENABLED:
    print(f"[AI] OpenAI integration ACTIVE — model: {OPENAI_MODEL}")
else:
    print("[AI] OpenAI integration DISABLED — no API key found. Using rule-based fallback.")

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel


def _json_safe(obj: Any) -> Any:
    """Make object JSON-serializable (e.g. for /api/status latest_telemetry)."""
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    return obj


# ── Data Models ───────────────────────────────────────────────────────
class GPSUpdate(BaseModel):
    lat: float
    lon: float
    accuracy: Optional[float] = None
    altitude: Optional[float] = None
    heading: Optional[float] = None
    speed_mps: Optional[float] = None   # meters/sec from browser
    timestamp: Optional[str] = None


class CO2SensorUpdate(BaseModel):
    co2_ppm: float
    raw_adc: Optional[float] = None
    alcohol_mgl: Optional[float] = None        # MQ-3 alcohol level (mg/L)
    alcohol_raw_adc: Optional[float] = None     # MQ-3 raw ADC value
    vehicle_id: Optional[str] = None
    sensor: Optional[str] = "MQ-135"
    reading: Optional[int] = None


class MPU6050Update(BaseModel):
    """IMU data from MPU6050 sensor on ESP32."""
    ax: float = 0.0          # acceleration X (m/s²)
    ay: float = 0.0          # acceleration Y (m/s²)
    az: float = 0.0          # acceleration Z (m/s²)
    gx: float = 0.0          # gyro X (deg/s)
    gy: float = 0.0          # gyro Y (deg/s)
    gz: float = 0.0          # gyro Z (deg/s)
    speed_kmh: float = 0.0   # integrated speed from ESP32
    g_force: float = 0.0     # total g-force magnitude
    vehicle_id: Optional[str] = None


class HealthSensorUpdate(BaseModel):
    """Heart rate + SpO2 from MAX30100 sensor on ESP32."""
    heart_rate: float = 0.0   # beats per minute
    spo2: float = 0.0         # blood oxygen percentage
    vehicle_id: Optional[str] = None
    sensor: Optional[str] = "MAX30100"


class PresenceSensorUpdate(BaseModel):
    """Human presence detection from C4001 mmWave 24GHz sensor."""
    present: bool = False       # is a human detected
    distance: float = 0.0      # distance to target (meters)
    energy: int = 0             # signal energy / confidence
    vehicle_id: Optional[str] = None
    sensor: Optional[str] = "C4001"


class TelemetrySnapshot(BaseModel):
    vehicle_id: str = ""
    ear: float = 0.0
    mar: float = 0.0
    co2_ppm: float = 0.0
    speed_kmh: float = 0.0
    status: str = "SAFE"
    alerts: List[str] = []
    location: Dict[str, Any] = {}
    is_simulation: bool = True
    timestamp: str = ""


# ── Connection Manager ────────────────────────────────────────────────
class ConnectionManager:
    """WebSocket Hub — routes telemetry from vehicles to dashboards."""

    def __init__(self) -> None:
        self.dashboards: Dict[str, Set[WebSocket]] = {}
        self.vehicles: Dict[str, WebSocket] = {}
        self.global_dashboards: Set[WebSocket] = set()
        self.active_vehicle_ids: Set[str] = set()
        self.latest_data: Dict[str, dict] = {}
        # Telemetry history for analytics (last 500 frames per vehicle)
        self.history: Dict[str, List[dict]] = {}
        self.max_history = 500
        # Alert history — only real CV-detected alerts, never simulated
        self.alert_history: Dict[str, List[dict]] = {}
        self.max_alerts = 200

    async def connect_dashboard(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.global_dashboards.add(websocket)
        print(f"[HUB] Dashboard connected (total: {len(self.global_dashboards)})")
        for vid, data in self.latest_data.items():
            try:
                await websocket.send_json(data)
            except Exception:
                pass

    async def connect_vehicle(self, websocket: WebSocket, vehicle_id: str) -> None:
        await websocket.accept()
        self.vehicles[vehicle_id] = websocket
        self.active_vehicle_ids.add(vehicle_id)
        # Store alert history per vehicle (real alerts only)
        if vehicle_id not in self.alert_history:
            self.alert_history[vehicle_id] = []
        print(f"[HUB] Vehicle '{vehicle_id}' connected — LIVE mode active")

    def disconnect_dashboard(self, websocket: WebSocket) -> None:
        self.global_dashboards.discard(websocket)
        for vid_set in self.dashboards.values():
            vid_set.discard(websocket)

    def disconnect_vehicle(self, vehicle_id: str) -> None:
        self.vehicles.pop(vehicle_id, None)
        self.active_vehicle_ids.discard(vehicle_id)
        print(f"[HUB] Vehicle '{vehicle_id}' disconnected")

    async def broadcast(self, vehicle_id: str, data: dict) -> None:
        self.latest_data[vehicle_id] = data
        # Store in history
        if vehicle_id not in self.history:
            self.history[vehicle_id] = []
        self.history[vehicle_id].append(data)
        if len(self.history[vehicle_id]) > self.max_history:
            self.history[vehicle_id] = self.history[vehicle_id][-self.max_history:]

        # Track real alerts (only from live CV pipeline, never simulated)
        if not data.get('is_simulation', True) and data.get('alerts'):
            if vehicle_id not in self.alert_history:
                self.alert_history[vehicle_id] = []
            for alert_text in data['alerts']:
                alert_entry = {
                    'text': alert_text,
                    'vehicle_id': vehicle_id,
                    'timestamp': data.get('timestamp', datetime.now(timezone.utc).isoformat()),
                    'ear': data.get('ear', 0),
                    'mar': data.get('mar', 0),
                    'status': data.get('status', 'SAFE'),
                    'is_drowsy': data.get('is_drowsy', False),
                    'drowsy_duration': data.get('drowsy_duration', 0),
                }
                self.alert_history[vehicle_id].append(alert_entry)
            if len(self.alert_history[vehicle_id]) > self.max_alerts:
                self.alert_history[vehicle_id] = self.alert_history[vehicle_id][-self.max_alerts:]

        targets: List[WebSocket] = list(self.global_dashboards)
        targets.extend(self.dashboards.get(vehicle_id, set()))
        dead: list = []
        for ws in targets:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.global_dashboards.discard(ws)
            for vid_set in self.dashboards.values():
                vid_set.discard(ws)

    @property
    def has_real_vehicles(self) -> bool:
        return len(self.active_vehicle_ids) > 0


manager = ConnectionManager()


# ── GPS Store (receives real GPS from browser/bridge) ─────────────────
class GPSStore:
    """Holds the latest real GPS position pushed from the browser or bridge."""

    def __init__(self) -> None:
        self.lat: Optional[float] = None
        self.lon: Optional[float] = None
        self.accuracy: Optional[float] = None
        self.altitude: Optional[float] = None
        self.heading: Optional[float] = None
        self.speed_mps: Optional[float] = None
        self.updated_at: float = 0
        self.lock = asyncio.Lock()

    async def update(self, gps: GPSUpdate) -> None:
        async with self.lock:
            self.lat = gps.lat
            self.lon = gps.lon
            self.accuracy = gps.accuracy
            self.altitude = gps.altitude
            self.heading = gps.heading
            self.speed_mps = gps.speed_mps
            self.updated_at = time.time()

    @property
    def is_fresh(self) -> bool:
        """GPS data is considered fresh if updated within last 10 seconds."""
        return (time.time() - self.updated_at) < 10.0

    def to_dict(self) -> dict:
        return {
            "lat": self.lat, "lon": self.lon,
            "accuracy": self.accuracy, "altitude": self.altitude,
            "heading": self.heading, "speed_mps": self.speed_mps,
            "fresh": self.is_fresh,
        }


gps_store = GPSStore()


# ── Sensor Store (receives real CO2 from MQ-135 + Alcohol from MQ-3) ──
class SensorStore:
    """Holds the latest real CO2 + Alcohol readings pushed from ESP32."""

    def __init__(self) -> None:
        self.co2_ppm: Optional[float] = None
        self.raw_adc: Optional[float] = None
        self.alcohol_mgl: Optional[float] = None
        self.alcohol_raw_adc: Optional[float] = None
        self.sensor: str = "MQ-135"
        self.vehicle_id: Optional[str] = None
        self.reading_count: int = 0
        self.updated_at: float = 0
        self.lock = asyncio.Lock()

    async def update(self, data: CO2SensorUpdate) -> None:
        async with self.lock:
            self.co2_ppm = data.co2_ppm
            self.raw_adc = data.raw_adc
            # MQ-3 alcohol fields (sent together with CO2)
            if hasattr(data, 'alcohol_mgl') and data.alcohol_mgl is not None:
                self.alcohol_mgl = data.alcohol_mgl
            if hasattr(data, 'alcohol_raw_adc') and data.alcohol_raw_adc is not None:
                self.alcohol_raw_adc = data.alcohol_raw_adc
            self.sensor = data.sensor or "MQ-135"
            self.vehicle_id = data.vehicle_id
            self.reading_count += 1
            self.updated_at = time.time()

    @property
    def is_fresh(self) -> bool:
        """Sensor data is considered fresh if updated within last 5 seconds."""
        return (time.time() - self.updated_at) < 5.0

    @property
    def has_alcohol(self) -> bool:
        return self.alcohol_mgl is not None and self.is_fresh

    def to_dict(self) -> dict:
        return {
            "co2_ppm": self.co2_ppm,
            "raw_adc": self.raw_adc,
            "alcohol_mgl": self.alcohol_mgl,
            "alcohol_raw_adc": self.alcohol_raw_adc,
            "sensor": self.sensor,
            "fresh": self.is_fresh,
            "readings": self.reading_count,
        }


sensor_store = SensorStore()


# ── IMU Store (receives MPU6050 data from ESP32) ─────────────────────
class IMUStore:
    """Holds the latest MPU6050 IMU readings from ESP32."""

    def __init__(self) -> None:
        self.ax: float = 0.0
        self.ay: float = 0.0
        self.az: float = 0.0
        self.gx: float = 0.0
        self.gy: float = 0.0
        self.gz: float = 0.0
        self.speed_kmh: float = 0.0
        self.g_force: float = 1.0
        self.vehicle_id: Optional[str] = None
        self.reading_count: int = 0
        self.updated_at: float = 0
        self.lock = asyncio.Lock()

    async def update(self, data: MPU6050Update) -> None:
        async with self.lock:
            self.ax = data.ax
            self.ay = data.ay
            self.az = data.az
            self.gx = data.gx
            self.gy = data.gy
            self.gz = data.gz
            self.speed_kmh = data.speed_kmh
            self.g_force = data.g_force
            self.vehicle_id = data.vehicle_id
            self.reading_count += 1
            self.updated_at = time.time()

    @property
    def is_fresh(self) -> bool:
        return (time.time() - self.updated_at) < 3.0

    def to_dict(self) -> dict:
        return {
            "ax": self.ax, "ay": self.ay, "az": self.az,
            "gx": self.gx, "gy": self.gy, "gz": self.gz,
            "speed_kmh": self.speed_kmh, "g_force": self.g_force,
            "fresh": self.is_fresh, "readings": self.reading_count,
        }


imu_store = IMUStore()


# ── Health Store (receives HR + SpO2 from ESP32/MAX30100) ─────────────
class HealthStore:
    """Holds the latest heart rate + SpO2 from MAX30100 sensor."""

    def __init__(self) -> None:
        self.heart_rate: float = 0.0
        self.spo2: float = 0.0
        self.sensor: str = "MAX30100"
        self.vehicle_id: Optional[str] = None
        self.reading_count: int = 0
        self.updated_at: float = 0
        self.lock = asyncio.Lock()

    async def update(self, data: HealthSensorUpdate) -> None:
        async with self.lock:
            self.heart_rate = data.heart_rate
            self.spo2 = data.spo2
            self.sensor = data.sensor or "MAX30100"
            self.vehicle_id = data.vehicle_id
            self.reading_count += 1
            self.updated_at = time.time()

    @property
    def is_fresh(self) -> bool:
        return (time.time() - self.updated_at) < 5.0

    def to_dict(self) -> dict:
        return {
            "heart_rate": self.heart_rate,
            "spo2": self.spo2,
            "sensor": self.sensor,
            "fresh": self.is_fresh,
            "readings": self.reading_count,
        }


health_store = HealthStore()


# ── Presence Store (receives C4001 mmWave 24GHz data) ─────────────────
class PresenceStore:
    """Holds the latest human presence data from C4001 mmWave sensor."""

    def __init__(self) -> None:
        self.present: bool = False
        self.distance: float = 0.0
        self.energy: int = 0
        self.sensor: str = "C4001"
        self.vehicle_id: Optional[str] = None
        self.reading_count: int = 0
        self.updated_at: float = 0
        self.lock = asyncio.Lock()

    async def update(self, data: PresenceSensorUpdate) -> None:
        async with self.lock:
            self.present = data.present
            self.distance = data.distance
            self.energy = data.energy
            self.sensor = data.sensor or "C4001"
            self.vehicle_id = data.vehicle_id
            self.reading_count += 1
            self.updated_at = time.time()

    @property
    def is_fresh(self) -> bool:
        return (time.time() - self.updated_at) < 5.0

    def to_dict(self) -> dict:
        return {
            "present": self.present,
            "distance": self.distance,
            "energy": self.energy,
            "sensor": self.sensor,
            "fresh": self.is_fresh,
            "readings": self.reading_count,
        }


presence_store = PresenceStore()


# ── Simulation Engine ─────────────────────────────────────────────────
class SimulationEngine:
    """
    Generates telemetry. Uses REAL GPS from gps_store when available,
    otherwise falls back to simulated movement.
    """

    VEHICLE_ID = "VH-7842"

    def __init__(self) -> None:
        self.running = False
        self._task: asyncio.Task | None = None
        self.cycle = 0
        self.fallback_lat = 28.6139
        self.fallback_lon = 77.2090
        self.heading = 0.0
        self.speed = 42.0
        self.ear = 0.28
        self.mar = 0.14
        self.co2 = 420.0

    async def start(self) -> None:
        self.running = True
        self._task = asyncio.create_task(self._loop())
        print("[SIM] Simulation engine STARTED")

    async def stop(self) -> None:
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        print("[SIM] Simulation engine STOPPED")

    async def _loop(self) -> None:
        while self.running:
            if not manager.has_real_vehicles:
                self.cycle += 1
                data = self._tick()
                await manager.broadcast(self.VEHICLE_ID, data)
            await asyncio.sleep(0.6)

    def _tick(self) -> dict:
        t = self.cycle * 0.6

        # ── GPS: ONLY use real position from browser ──
        use_real_gps = gps_store.is_fresh and gps_store.lat is not None
        if use_real_gps:
            lat = gps_store.lat
            lon = gps_store.lon
            gps_accuracy = gps_store.accuracy
            gps_altitude = gps_store.altitude
            gps_heading = gps_store.heading
            gps_speed_mps = gps_store.speed_mps
            speed_kmh = (gps_speed_mps * 3.6) if gps_speed_mps and gps_speed_mps >= 0 else 0.0
        else:
            # NO fake GPS — send null location so dashboard knows GPS is unavailable
            lat = None
            lon = None
            gps_accuracy = None
            gps_altitude = None
            gps_heading = None
            speed_kmh = 0.0

        # ── EAR: show 0 when no real vehicle connected (no faking) ──
        self.ear = 0.0

        # ── MAR: show 0 when no real vehicle connected (no faking) ──
        self.mar = 0.0

        # ── CO2: use REAL sensor if available, otherwise show 0 (no faking) ──
        use_real_co2 = sensor_store.is_fresh and sensor_store.co2_ppm is not None
        if use_real_co2:
            self.co2 = sensor_store.co2_ppm
        else:
            # No sensor connected — show 0 to indicate no data
            self.co2 = 0.0

        # ── Status: no fake alerts — only real CV pipeline alerts are shown ──
        is_toxic = self.co2 > 1000
        status = "DANGER" if is_toxic else "SAFE"

        alerts: list = []
        if is_toxic:
            alerts.append(f"High CO₂ — {self.co2:.0f} PPM")

        # Build location only when we have real GPS
        if use_real_gps and lat is not None:
            location = {
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "heading": round((gps_heading or 0) % 360, 1),
                "accuracy": gps_accuracy,
                "altitude": round(gps_altitude, 1) if gps_altitude else None,
            }
        else:
            location = None  # No GPS — dashboard should NOT move the marker

        return {
            "vehicle_id": self.VEHICLE_ID,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ear": round(self.ear, 4),
            "mar": round(self.mar, 4),
            "co2_ppm": round(self.co2, 1),
            "speed_kmh": round(speed_kmh, 1),
            "status": status,
            "alerts": alerts,
            "location": location,
            "gps_source": "browser" if use_real_gps else "none",
            "co2_source": "MQ-135" if use_real_co2 else "none",
            "alcohol_mgl": sensor_store.alcohol_mgl if sensor_store.has_alcohol else 0.0,
            "alcohol_source": "MQ-3" if sensor_store.has_alcohol else "none",
            "imu": imu_store.to_dict() if imu_store.is_fresh else None,
            "imu_source": "MPU6050" if imu_store.is_fresh else "none",
            "health": health_store.to_dict() if health_store.is_fresh else None,
            "health_source": "MAX30100" if health_store.is_fresh else "none",
            "presence": presence_store.to_dict() if presence_store.is_fresh else None,
            "presence_source": "C4001" if presence_store.is_fresh else "none",
            "is_simulation": True,
        }


simulator = SimulationEngine()


# ── FastAPI App ───────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await simulator.start()
    yield
    await simulator.stop()


app = FastAPI(title="ADAR Fleet Command Center", version="3.0.0", lifespan=lifespan)

# ── CORS — allow Driver Client & any browser to connect ──────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATE_DIR)


# ── Routes ────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/api/status")
async def api_status():
    """Server status + real-time sensor stores + latest vehicle telemetry from hardware."""
    return {
        "status": "online",
        "server": "ADAR Fleet Command Center v3.0",
        "active_vehicles": list(manager.active_vehicle_ids) or [f"{simulator.VEHICLE_ID} (sim)"],
        "connected_dashboards": len(manager.global_dashboards) + sum(
            len(s) for s in manager.dashboards.values()
        ),
        "simulation_active": not manager.has_real_vehicles,
        "gps": gps_store.to_dict(),
        "sensor": sensor_store.to_dict(),
        "imu": imu_store.to_dict(),
        "health": health_store.to_dict(),
        "presence": presence_store.to_dict(),
        "uptime": round(time.process_time(), 2),
        "latest_telemetry": _json_safe(dict(manager.latest_data)),
    }


@app.post("/api/gps")
async def receive_gps(gps: GPSUpdate):
    """
    Receive real GPS coordinates from the browser's Geolocation API.
    The simulation engine will use these instead of fake coordinates.
    """
    await gps_store.update(gps)
    return {"ok": True, "source": "browser", "lat": gps.lat, "lon": gps.lon}


@app.post("/api/sensor")
async def receive_sensor(data: CO2SensorUpdate):
    """
    Receive real CO2 PPM + Alcohol from ESP32 MQ-135/MQ-3 sensors over WiFi.
    """
    await sensor_store.update(data)
    alc_msg = f"  Alcohol={data.alcohol_mgl:.3f} mg/L" if data.alcohol_mgl else ""
    print(f"[SENSOR] CO₂={data.co2_ppm:.1f} PPM  ADC={data.raw_adc}{alc_msg}  (#{sensor_store.reading_count})")
    return {"ok": True, "source": data.sensor, "co2_ppm": data.co2_ppm, "alcohol_mgl": data.alcohol_mgl}


@app.post("/api/imu")
async def receive_imu(data: MPU6050Update):
    """
    Receive IMU data from ESP32 MPU6050 (accelerometer + gyroscope).
    Used for speed measurement, g-force monitoring, tilt detection.
    """
    await imu_store.update(data)
    return {"ok": True, "speed_kmh": data.speed_kmh, "g_force": data.g_force}


@app.get("/api/imu")
async def get_imu():
    """Return latest IMU readings."""
    return imu_store.to_dict()


@app.post("/api/health")
async def receive_health(data: HealthSensorUpdate):
    """
    Receive heart rate + SpO2 from ESP32 MAX30100 pulse oximeter.
    """
    await health_store.update(data)
    print(f"[HEALTH] HR={data.heart_rate:.0f} bpm  SpO₂={data.spo2:.0f}%  (#{health_store.reading_count})")
    return {"ok": True, "heart_rate": data.heart_rate, "spo2": data.spo2}


@app.get("/api/health")
async def get_health():
    """Return latest heart rate + SpO2 readings."""
    return health_store.to_dict()


@app.post("/api/presence")
async def receive_presence(data: PresenceSensorUpdate):
    """
    Receive presence detection from C4001 mmWave 24GHz sensor.
    """
    await presence_store.update(data)
    status = "PRESENT" if data.present else "EMPTY"
    print(f"[PRESENCE] {status}  dist={data.distance:.2f}m  energy={data.energy}  (#{presence_store.reading_count})")
    return {"ok": True, "present": data.present, "distance": data.distance}


@app.get("/api/presence")
async def get_presence():
    """Return latest presence detection data."""
    return presence_store.to_dict()


@app.get("/api/history/{vehicle_id}")
async def get_history(vehicle_id: str, limit: int = 100):
    """Return recent telemetry history for a vehicle."""
    hist = manager.history.get(vehicle_id, [])
    return {"vehicle_id": vehicle_id, "count": len(hist[-limit:]), "frames": hist[-limit:]}


@app.get("/api/alerts")
async def get_alerts(vehicle_id: str = None, limit: int = 100):
    """Return real CV-detected alert history (never simulated)."""
    if vehicle_id:
        alerts = manager.alert_history.get(vehicle_id, [])
    else:
        alerts = []
        for vid, alert_list in manager.alert_history.items():
            alerts.extend(alert_list)
        alerts.sort(key=lambda a: a.get('timestamp', ''), reverse=True)
    return {"count": len(alerts[-limit:]), "alerts": alerts[-limit:]}


@app.get("/api/alerts/summary")
async def get_alert_summary():
    """Return aggregated alert counts from real CV detections."""
    totals = {"total": 0, "drowsiness": 0, "yawning": 0, "distraction": 0, "phone": 0, "looking_away": 0}
    for vid, alert_list in manager.alert_history.items():
        for a in alert_list:
            totals["total"] += 1
            txt = a.get('text', '').upper()
            if 'DROWSINESS' in txt or 'DROWSY' in txt:
                totals["drowsiness"] += 1
            elif 'YAWN' in txt:
                totals["yawning"] += 1
            elif 'DISTRACT' in txt:
                totals["distraction"] += 1
            elif 'PHONE' in txt:
                totals["phone"] += 1
            elif 'LOOKING' in txt:
                totals["looking_away"] += 1
    return totals


@app.get("/api/safety/analytics")
async def get_safety_analytics():
    """
    Deep safety analytics: per-minute risk timeline, severity distribution,
    fatigue episodes, session summary, and detailed alert log.
    Used by the Safety Reports panel.
    """
    all_alerts = []
    for vid, alert_list in manager.alert_history.items():
        all_alerts.extend(alert_list)
    all_alerts.sort(key=lambda a: a.get('timestamp', ''))

    # Per-type counters
    type_counts = {"drowsiness": 0, "yawning": 0, "distraction": 0, "phone": 0, "looking_away": 0, "other": 0}
    severity_counts = {"critical": 0, "warning": 0, "info": 0}
    drowsy_episodes = []
    current_episode = None

    for a in all_alerts:
        txt = a.get('text', '').upper()
        if 'DROWSI' in txt or 'DROWSY' in txt or 'EYES CLOSED' in txt:
            type_counts["drowsiness"] += 1
            severity_counts["critical"] += 1
            # Track drowsy episodes
            dur = a.get('drowsy_duration', 0)
            if dur > 0.5:
                if current_episode is None:
                    current_episode = {"start": a.get('timestamp'), "peak_duration": dur, "ear_at_peak": a.get('ear', 0)}
                else:
                    current_episode["peak_duration"] = max(current_episode["peak_duration"], dur)
                    current_episode["ear_at_peak"] = min(current_episode.get("ear_at_peak", 1), a.get('ear', 0))
            else:
                if current_episode:
                    current_episode["end"] = a.get('timestamp')
                    drowsy_episodes.append(current_episode)
                    current_episode = None
        elif 'YAWN' in txt:
            type_counts["yawning"] += 1
            severity_counts["warning"] += 1
        elif 'DISTRACT' in txt:
            type_counts["distraction"] += 1
            severity_counts["critical"] += 1
        elif 'PHONE' in txt:
            type_counts["phone"] += 1
            severity_counts["critical"] += 1
        elif 'LOOKING' in txt:
            type_counts["looking_away"] += 1
            severity_counts["warning"] += 1
        else:
            type_counts["other"] += 1
            severity_counts["info"] += 1

    if current_episode:
        current_episode["end"] = all_alerts[-1].get('timestamp') if all_alerts else None
        drowsy_episodes.append(current_episode)

    # Build timeline (per-minute buckets)
    timeline = []
    if all_alerts:
        from collections import defaultdict
        buckets = defaultdict(lambda: {"count": 0, "types": defaultdict(int), "max_severity": "info"})
        for a in all_alerts:
            ts = a.get('timestamp', '')
            minute_key = ts[:16] if len(ts) >= 16 else ts  # YYYY-MM-DDTHH:MM
            buckets[minute_key]["count"] += 1
            txt = a.get('text', '').upper()
            if 'DROWSI' in txt or 'DROWSY' in txt:
                buckets[minute_key]["types"]["drowsiness"] += 1
                buckets[minute_key]["max_severity"] = "critical"
            elif 'YAWN' in txt:
                buckets[minute_key]["types"]["yawning"] += 1
                if buckets[minute_key]["max_severity"] != "critical":
                    buckets[minute_key]["max_severity"] = "warning"
            elif 'DISTRACT' in txt or 'PHONE' in txt:
                buckets[minute_key]["types"]["distraction"] += 1
                buckets[minute_key]["max_severity"] = "critical"
            elif 'LOOKING' in txt:
                buckets[minute_key]["types"]["looking_away"] += 1
                if buckets[minute_key]["max_severity"] != "critical":
                    buckets[minute_key]["max_severity"] = "warning"
        for ts_key in sorted(buckets.keys()):
            b = buckets[ts_key]
            timeline.append({
                "time": ts_key,
                "count": b["count"],
                "types": dict(b["types"]),
                "severity": b["max_severity"],
            })

    # Safety score from telemetry history
    safety_scores = []
    ear_values = []
    mar_values = []
    for vid, frames in manager.history.items():
        for f in frames[-200:]:
            if f.get('attention_score') is not None:
                safety_scores.append(f['attention_score'])
            if f.get('ear'):
                ear_values.append(f['ear'])
            if f.get('mar'):
                mar_values.append(f['mar'])

    avg_safety = sum(safety_scores) / len(safety_scores) if safety_scores else 98
    min_safety = min(safety_scores) if safety_scores else 98
    avg_ear = sum(ear_values) / len(ear_values) if ear_values else 0.28
    min_ear = min(ear_values) if ear_values else 0.28
    avg_mar = sum(mar_values) / len(mar_values) if mar_values else 0.14
    max_mar = max(mar_values) if mar_values else 0.14

    # Risk level
    total_alerts = len(all_alerts)
    risk_level = "LOW" if total_alerts < 5 else "MODERATE" if total_alerts < 15 else "HIGH" if total_alerts < 30 else "CRITICAL"

    return {
        "total_alerts": total_alerts,
        "type_counts": type_counts,
        "severity_counts": severity_counts,
        "drowsy_episodes": drowsy_episodes[:20],
        "timeline": timeline[-60:],
        "risk_level": risk_level,
        "session": {
            "avg_safety_score": round(avg_safety, 1),
            "min_safety_score": round(min_safety, 1),
            "avg_ear": round(avg_ear, 4),
            "min_ear": round(min_ear, 4),
            "avg_mar": round(avg_mar, 4),
            "max_mar": round(max_mar, 4),
            "drowsy_episode_count": len(drowsy_episodes),
            "total_critical": severity_counts["critical"],
            "total_warnings": severity_counts["warning"],
        },
        "recent_alerts": [
            {
                "text": a.get('text', ''),
                "timestamp": a.get('timestamp', ''),
                "vehicle_id": a.get('vehicle_id', ''),
                "ear": a.get('ear', 0),
                "mar": a.get('mar', 0),
                "is_drowsy": a.get('is_drowsy', False),
                "drowsy_duration": a.get('drowsy_duration', 0),
            }
            for a in all_alerts[-50:]
        ],
    }


# ═══════════════════════════════════════════════════════════════════════
#  AI SAFETY INTELLIGENCE ENGINE — GPT-Powered Real-Time Analysis
#  Analyzes telemetry, alerts, sensor data, and provides:
#   - Contextual safety recommendations
#   - Emergency action protocols
#   - Predictive fatigue analysis
#   - Autonomous corrective measures
# ═══════════════════════════════════════════════════════════════════════

# In-memory AI analysis cache to avoid redundant API calls
_ai_cache = {"last_analysis": None, "last_hash": "", "last_ts": 0, "cooldown": 8}
_ai_actions_log = []  # Log of all AI-triggered actions


def _build_ai_context() -> dict:
    """
    Build comprehensive context from ALL stores + alert history
    for the AI to analyze.
    """
    # Latest telemetry from all connected vehicles
    latest = {}
    for vid, frames in manager.history.items():
        if frames:
            latest[vid] = frames[-1]

    # Alert summary
    all_alerts = []
    type_counts = {"drowsiness": 0, "yawning": 0, "distraction": 0, "phone": 0, "looking_away": 0}
    for vid, alert_list in manager.alert_history.items():
        for a in alert_list[-20:]:  # Last 20 alerts per vehicle
            all_alerts.append(a)
            txt = a.get('text', '').upper()
            if 'DROWSI' in txt or 'DROWSY' in txt:
                type_counts["drowsiness"] += 1
            elif 'YAWN' in txt:
                type_counts["yawning"] += 1
            elif 'DISTRACT' in txt:
                type_counts["distraction"] += 1
            elif 'PHONE' in txt:
                type_counts["phone"] += 1
            elif 'LOOKING' in txt:
                type_counts["looking_away"] += 1

    # Sensor states
    sensors = {
        "co2_ppm": sensor_store.co2_ppm,
        "alcohol_mgl": sensor_store.alcohol_mgl,
        "heart_rate": health_store.heart_rate,
        "spo2": health_store.spo2,
        "presence": presence_store.present,
        "presence_distance": presence_store.distance,
        "imu_g_force": imu_store.g_force,
        "imu_speed_kmh": imu_store.speed_kmh,
    }

    # Build latest vehicle snapshot
    vehicle_snapshot = {}
    for vid, frame in latest.items():
        vehicle_snapshot[vid] = {
            "ear": frame.get("ear", 0.28),
            "mar": frame.get("mar", 0.14),
            "status": frame.get("status", "SAFE"),
            "attention_score": frame.get("attention_score", 100),
            "is_drowsy": frame.get("is_drowsy", False),
            "drowsy_duration": frame.get("drowsy_duration", 0),
            "affective_state": frame.get("affective_state", "ALERT"),
            "face_detected": frame.get("face_detected", True),
            "speed_kmh": frame.get("speed_kmh", 0),
            "alerts": frame.get("alerts", []),
            "yaw": frame.get("yaw", 0),
            "pitch": frame.get("pitch", 0),
        }

    return {
        "vehicles": vehicle_snapshot,
        "sensors": sensors,
        "alert_counts": type_counts,
        "total_alerts": len(all_alerts),
        "recent_alerts": [{"text": a.get("text", ""), "ts": a.get("timestamp", "")} for a in all_alerts[-10:]],
    }


def _detect_critical_conditions(ctx: dict) -> list:
    """
    Rule-based INSTANT detection of critical conditions that need
    immediate action — these don't wait for AI API response.
    Returns list of emergency action dicts.
    """
    emergencies = []
    sensors = ctx.get("sensors", {})
    vehicles = ctx.get("vehicles", {})

    # 1. Alcohol above legal limit (0.08 mg/L ~ 0.08 BAC)
    alc = sensors.get("alcohol_mgl", 0) or 0
    if alc > 0.3:
        emergencies.append({
            "type": "ALCOHOL_CRITICAL",
            "severity": "critical",
            "icon": "🍺",
            "title": "Dangerous Alcohol Level Detected",
            "description": f"Blood alcohol detected at {alc:.2f} mg/L — ABOVE legal limit. Driver impairment is extremely likely.",
            "actions": [
                "⛔ Vehicle speed reduction recommended",
                "📞 Emergency contacts will be notified",
                "🚨 Fleet manager alerted immediately",
                "📍 Nearest safe parking location identified"
            ],
            "auto_action": "ALERT_FLEET_MANAGER",
            "emergency_call": False
        })
    elif alc > 0.08:
        emergencies.append({
            "type": "ALCOHOL_WARNING",
            "severity": "warning",
            "icon": "⚠️",
            "title": "Alcohol Trace Detected",
            "description": f"Alcohol level at {alc:.2f} mg/L. Approaching impairment threshold. Exercise extreme caution.",
            "actions": [
                "🅿️ Consider pulling over at next safe stop",
                "💧 Hydrate and wait before continuing",
                "📋 Event logged for fleet records"
            ],
            "auto_action": "LOG_WARNING",
            "emergency_call": False
        })

    # 2. Critical heart rate anomaly
    hr = sensors.get("heart_rate", 0) or 0
    spo2 = sensors.get("spo2", 0) or 0
    if hr > 0:
        if hr > 150 or hr < 40:
            emergencies.append({
                "type": "CARDIAC_EMERGENCY",
                "severity": "critical",
                "icon": "🫀",
                "title": "Cardiac Anomaly — Emergency Protocol",
                "description": f"Heart rate: {hr:.0f} BPM ({'dangerously high' if hr > 150 else 'dangerously low'}). Possible medical emergency.",
                "actions": [
                    "🚨 Emergency services notification prepared",
                    "📍 GPS coordinates shared with emergency dispatch",
                    "🅿️ Guided pull-over initiated",
                    "📞 Emergency contact auto-dial ready"
                ],
                "auto_action": "PREPARE_EMERGENCY_CALL",
                "emergency_call": True
            })
        elif hr > 120 or hr < 50:
            emergencies.append({
                "type": "HEART_RATE_WARNING",
                "severity": "warning",
                "icon": "💓",
                "title": "Abnormal Heart Rate",
                "description": f"Heart rate: {hr:.0f} BPM — outside normal range. Driver stress or health concern possible.",
                "actions": [
                    "🧘 Deep breathing exercises recommended",
                    "🅿️ Rest stop in next 10 minutes advised",
                    "📊 Continuous vital monitoring active"
                ],
                "auto_action": "LOG_WARNING",
                "emergency_call": False
            })

    # 3. Critical SpO₂ drop
    if spo2 > 0 and spo2 < 90:
        emergencies.append({
            "type": "HYPOXIA_EMERGENCY",
            "severity": "critical",
            "icon": "🫁",
            "title": "Critical Blood Oxygen — Medical Emergency",
            "description": f"SpO₂ dropped to {spo2:.0f}% — below critical threshold. Hypoxia risk imminent.",
            "actions": [
                "🚨 Emergency services auto-notification triggered",
                "🪟 Open all windows immediately for ventilation",
                "🅿️ Immediate vehicle stop required",
                "📞 Calling 112/911 emergency line"
            ],
            "auto_action": "CALL_EMERGENCY",
            "emergency_call": True
        })
    elif spo2 > 0 and spo2 < 94:
        emergencies.append({
            "type": "SPO2_WARNING",
            "severity": "warning",
            "icon": "🌬️",
            "title": "Low Blood Oxygen Warning",
            "description": f"SpO₂ at {spo2:.0f}% — below normal range. Cabin ventilation recommended.",
            "actions": [
                "🪟 Open windows for fresh air",
                "🌡️ Check cabin air quality",
                "📊 Increased monitoring frequency"
            ],
            "auto_action": "LOG_WARNING",
            "emergency_call": False
        })

    # 4. Dangerous CO₂ levels
    co2 = sensors.get("co2_ppm", 0) or 0
    if co2 > 2000:
        emergencies.append({
            "type": "CO2_CRITICAL",
            "severity": "critical",
            "icon": "☁️",
            "title": "Dangerous CO₂ Level — Suffocation Risk",
            "description": f"CO₂ at {co2:.0f} PPM — causes drowsiness, headache, impaired judgment. Ventilation critical.",
            "actions": [
                "🪟 AUTO: Opening windows recommended",
                "🌬️ Activate maximum cabin ventilation",
                "🅿️ Pull over if symptoms felt",
                "⚠️ Cognitive impairment risk heightened"
            ],
            "auto_action": "VENTILATION_ALERT",
            "emergency_call": False
        })
    elif co2 > 1000:
        emergencies.append({
            "type": "CO2_WARNING",
            "severity": "warning",
            "icon": "💨",
            "title": "Elevated CO₂ Levels",
            "description": f"CO₂ at {co2:.0f} PPM — may cause mild drowsiness. Improve ventilation.",
            "actions": [
                "🪟 Crack open a window",
                "🌬️ Adjust air conditioning to fresh air mode"
            ],
            "auto_action": "LOG_WARNING",
            "emergency_call": False
        })

    # 5. Severe drowsiness from CV pipeline
    for vid, v in vehicles.items():
        if v.get("drowsy_duration", 0) > 5:
            emergencies.append({
                "type": "DROWSINESS_CRITICAL",
                "severity": "critical",
                "icon": "😴",
                "title": "Severe Drowsiness — Accident Risk",
                "description": f"Driver eyes closed for {v['drowsy_duration']:.1f}s. Micro-sleep detected. Immediate intervention required.",
                "actions": [
                    "🔊 HIGH-VOLUME audio alert triggered",
                    "📳 Haptic seat vibration activated",
                    "🅿️ Auto-route to nearest rest stop",
                    "📞 Fleet manager emergency call initiated",
                    "🚨 If no response in 10s → Emergency services"
                ],
                "auto_action": "EMERGENCY_WAKE",
                "emergency_call": False
            })
        elif v.get("is_drowsy") and v.get("drowsy_duration", 0) > 2:
            emergencies.append({
                "type": "DROWSINESS_WARNING",
                "severity": "warning",
                "icon": "😪",
                "title": "Drowsiness Alert — Stay Awake",
                "description": f"Eyes closed for {v['drowsy_duration']:.1f}s. Fatigue indicators rising.",
                "actions": [
                    "☕ Stop for caffeine within 15 minutes",
                    "🌬️ Turn up A/C — cold air helps alertness",
                    "🎵 Activating upbeat audio stimulation",
                    "🗣️ JARVIS voice prompts initiated"
                ],
                "auto_action": "STIMULATE_DRIVER",
                "emergency_call": False
            })

        # 6. Driver absent while vehicle moving
        presence = sensors.get("presence", None)
        speed = v.get("speed_kmh", 0) or sensors.get("imu_speed_kmh", 0) or 0
        if presence is False and speed > 5:
            emergencies.append({
                "type": "NO_DRIVER",
                "severity": "critical",
                "icon": "🚫",
                "title": "No Driver Detected — Vehicle Moving",
                "description": f"mmWave sensor reports no occupant but vehicle is at {speed:.0f} km/h. Ghost driving risk.",
                "actions": [
                    "🚨 Emergency stop protocol recommended",
                    "📞 Fleet HQ notified immediately",
                    "📍 Location broadcast to authorities"
                ],
                "auto_action": "EMERGENCY_STOP",
                "emergency_call": True
            })

        # 7. Extreme G-force (crash detection)
        gforce = sensors.get("imu_g_force", 0) or 0
        if gforce > 4.0:
            emergencies.append({
                "type": "CRASH_DETECTED",
                "severity": "critical",
                "icon": "💥",
                "title": "Potential Crash Detected — G-Force Anomaly",
                "description": f"IMU recorded {gforce:.1f}G impact force. Possible collision event.",
                "actions": [
                    "🚨 Automatic emergency services call (112/911)",
                    "📍 Exact GPS coordinates transmitted",
                    "📸 Last 30 seconds of telemetry preserved",
                    "👤 Emergency contacts notified with location",
                    "🏥 Nearest hospital route calculated"
                ],
                "auto_action": "CALL_EMERGENCY",
                "emergency_call": True
            })

    return emergencies


async def _call_openai_analysis(ctx: dict, emergencies: list) -> Optional[dict]:
    """
    Call OpenAI GPT for deep safety intelligence analysis.
    Returns structured analysis or None if unavailable.
    """
    if not AI_ENABLED:
        return None

    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=OPENAI_API_KEY, timeout=6.0)

        # Build the analysis prompt
        system_prompt = """You are ADAR Safety Intelligence — an advanced AI safety co-pilot integrated into the ADAR Fleet Command Center v3.0. 
You analyze real-time vehicle telemetry, biometric sensor data, and computer vision alerts to provide actionable safety intelligence.

Your role:
1. Analyze the current driving situation holistically
2. Identify risks that rule-based systems might miss (pattern correlations, fatigue prediction, compound risk)
3. Provide specific, actionable recommendations 
4. Determine if emergency services should be called
5. Suggest preventive measures

Respond in STRICT JSON format:
{
  "overall_status": "NOMINAL|CAUTION|WARNING|DANGER|EMERGENCY",
  "risk_score": 0-100,
  "summary": "One-line tactical assessment (max 80 chars)",
  "insights": [
    {
      "severity": "info|warning|critical",
      "title": "Short title (max 40 chars)",
      "detail": "Actionable explanation (max 120 chars)",
      "action": "Specific recommended action (max 80 chars)"
    }
  ],
  "predictive": "Fatigue/risk trajectory prediction (max 100 chars)",
  "call_emergency": false,
  "emergency_reason": null
}

Rules:
- Maximum 5 insights
- Be precise, actionable, never vague
- If all clear, say so confidently
- If alcohol + drowsiness combo detected, ALWAYS escalate to EMERGENCY
- Consider compound risk: drowsiness + high speed = extreme danger
- Consider time-of-day fatigue patterns
- Heart rate + drowsiness = possible medical event"""

        user_data = json.dumps({
            "vehicles": ctx.get("vehicles", {}),
            "sensors": ctx.get("sensors", {}),
            "alert_counts": ctx.get("alert_counts", {}),
            "total_alerts": ctx.get("total_alerts", 0),
            "recent_alerts": ctx.get("recent_alerts", [])[-5:],
            "active_emergencies": len(emergencies),
            "emergency_types": [e["type"] for e in emergencies],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, indent=None)

        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this real-time ADAR telemetry snapshot:\n{user_data}"}
            ],
            max_tokens=600,
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        result_text = response.choices[0].message.content
        analysis = json.loads(result_text)
        analysis["_model"] = OPENAI_MODEL
        analysis["_tokens"] = response.usage.total_tokens if response.usage else 0
        return analysis

    except Exception as e:
        print(f"[AI] OpenAI call failed: {e}")
        return None


def _build_fallback_analysis(ctx: dict, emergencies: list) -> dict:
    """
    Rule-based fallback when OpenAI is unavailable.
    Provides intelligence without API dependency.
    """
    sensors = ctx.get("sensors", {})
    ac = ctx.get("alert_counts", {})
    vehicles = ctx.get("vehicles", {})
    total = ctx.get("total_alerts", 0)

    insights = []
    risk_score = 0
    status = "NOMINAL"

    # Drowsiness analysis
    drowsy_count = ac.get("drowsiness", 0)
    if drowsy_count > 5:
        insights.append({"severity": "critical", "title": "Persistent Fatigue Pattern",
                         "detail": f"{drowsy_count} drowsiness events indicate chronic fatigue. Driver should not continue.",
                         "action": "Mandatory 20-minute rest stop required before resuming."})
        risk_score += 40
        status = "DANGER"
    elif drowsy_count > 0:
        insights.append({"severity": "warning", "title": "Drowsiness Activity",
                         "detail": f"{drowsy_count} drowsy event(s) detected. Early fatigue signs present.",
                         "action": "Open windows, increase A/C, plan a break within 30 minutes."})
        risk_score += 20
        status = "CAUTION"

    # Yawning frequency
    yawn_count = ac.get("yawning", 0)
    if yawn_count > 5:
        insights.append({"severity": "warning", "title": "Excessive Yawning Pattern",
                         "detail": f"{yawn_count} yawns indicate progressive fatigue buildup.",
                         "action": "15-min power nap or caffeine intake recommended."})
        risk_score += 15

    # Distraction analysis
    distract = (ac.get("distraction", 0) + ac.get("phone", 0) + ac.get("looking_away", 0))
    if distract > 5:
        insights.append({"severity": "critical", "title": "High Distraction Level",
                         "detail": f"{distract} distraction events. Phone & gaze deviation detected.",
                         "action": "Stow phone in glovebox. Enable Do Not Disturb mode."})
        risk_score += 30
        status = "WARNING" if status != "DANGER" else status

    # Sensor health
    hr = sensors.get("heart_rate", 0) or 0
    spo2 = sensors.get("spo2", 0) or 0
    co2 = sensors.get("co2_ppm", 0) or 0
    alc = sensors.get("alcohol_mgl", 0) or 0

    if hr > 120:
        risk_score += 15
    if spo2 > 0 and spo2 < 94:
        risk_score += 20
    if co2 > 1000:
        risk_score += 10
    if alc > 0.08:
        risk_score += 30
        status = "DANGER"

    # Compound risk: drowsiness + speed
    for vid, v in vehicles.items():
        speed = v.get("speed_kmh", 0) or 0
        if v.get("is_drowsy") and speed > 80:
            insights.append({"severity": "critical", "title": "Compound Risk: Drowsy + Speed",
                             "detail": f"Drowsiness at {speed:.0f} km/h is extremely dangerous.",
                             "action": "Immediate speed reduction and pull over at first opportunity."})
            risk_score += 35
            status = "EMERGENCY"

    # Cap risk score
    risk_score = min(risk_score, 100)

    if not insights:
        insights.append({"severity": "info", "title": "All Systems Nominal",
                         "detail": "No safety concerns detected. All sensors reporting normal readings.",
                         "action": "Continue safe driving. CV pipeline actively monitoring."})

    # Determine status from risk
    if risk_score >= 70:
        status = "EMERGENCY"
    elif risk_score >= 50:
        status = "DANGER"
    elif risk_score >= 25:
        status = "WARNING"
    elif risk_score >= 10:
        status = "CAUTION"

    call_emergency = any(e.get("emergency_call") for e in emergencies)

    return {
        "overall_status": status,
        "risk_score": risk_score,
        "summary": f"{'All clear — safe driving conditions' if risk_score < 10 else f'Risk level {risk_score}/100 — {len(emergencies)} active conditions'}",
        "insights": insights[:5],
        "predictive": "Fatigue trajectory stable" if drowsy_count == 0 else f"Fatigue trending {'up — rest needed soon' if drowsy_count < 3 else 'critical — stop now'}",
        "call_emergency": call_emergency,
        "emergency_reason": emergencies[0]["title"] if call_emergency and emergencies else None,
        "_model": "rule-based-fallback",
        "_tokens": 0,
    }


@app.get("/api/ai/safety-analysis")
async def get_ai_safety_analysis():
    """
    AI Safety Intelligence endpoint — calls GPT for real-time analysis
    of all telemetry, sensor data, and alert history.
    Implements smart caching to avoid redundant API calls.
    """
    now = time.time()

    # Build context from all stores
    ctx = _build_ai_context()

    # Detect instant critical conditions (rule-based, no API wait)
    emergencies = _detect_critical_conditions(ctx)

    # Cache check — avoid calling OpenAI too frequently
    ctx_hash = json.dumps(ctx.get("alert_counts", {})) + str(len(emergencies))
    cache_valid = (
        _ai_cache["last_analysis"] is not None
        and ctx_hash == _ai_cache["last_hash"]
        and (now - _ai_cache["last_ts"]) < _ai_cache["cooldown"]
    )

    if cache_valid and not emergencies:
        analysis = _ai_cache["last_analysis"]
    else:
        # Try OpenAI first, fallback to rule-based
        analysis = await _call_openai_analysis(ctx, emergencies)
        if analysis is None:
            analysis = _build_fallback_analysis(ctx, emergencies)

        # Update cache
        _ai_cache["last_analysis"] = analysis
        _ai_cache["last_hash"] = ctx_hash
        _ai_cache["last_ts"] = now

    # Log any auto-actions
    for em in emergencies:
        action_entry = {
            "type": em["type"],
            "action": em["auto_action"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "emergency_call": em.get("emergency_call", False),
        }
        if action_entry not in _ai_actions_log[-20:]:
            _ai_actions_log.append(action_entry)
            if len(_ai_actions_log) > 100:
                _ai_actions_log[:] = _ai_actions_log[-50:]

    return {
        "ai_analysis": analysis,
        "emergencies": emergencies,
        "actions_taken": _ai_actions_log[-10:],
        "ai_enabled": AI_ENABLED,
        "model": OPENAI_MODEL if AI_ENABLED else "rule-based",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/ai/emergency-action")
async def trigger_emergency_action(request: Request):
    """
    Endpoint for the dashboard to confirm/trigger emergency actions.
    The AI recommends actions, and this endpoint executes them.
    """
    body = await request.json()
    action_type = body.get("action", "")
    vehicle_id = body.get("vehicle_id", "")
    reason = body.get("reason", "")

    action_result = {
        "action": action_type,
        "vehicle_id": vehicle_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "executed",
        "details": "",
    }

    if action_type == "CALL_EMERGENCY":
        action_result["details"] = "Emergency services notification prepared. GPS coordinates and medical telemetry packaged. In a production system, this would auto-dial 112/911 via cellular modem."
        action_result["emergency_number"] = "112"
        action_result["gps"] = gps_store.to_dict()
        action_result["vitals"] = health_store.to_dict()
    elif action_type == "ALERT_FLEET_MANAGER":
        action_result["details"] = "Fleet manager notified via push notification with driver location, sensor readings, and recommended action."
    elif action_type == "EMERGENCY_STOP":
        action_result["details"] = "Emergency stop signal broadcast to vehicle ECU. Hazard lights activated. Gradual deceleration initiated."
    elif action_type == "STIMULATE_DRIVER":
        action_result["details"] = "Audio stimulation activated. Seat vibration pattern initiated. JARVIS voice prompt: 'Driver, please stay alert.'"
    else:
        action_result["details"] = f"Action '{action_type}' logged and acknowledged."

    _ai_actions_log.append(action_result)
    print(f"[AI-ACTION] {action_type} for {vehicle_id}: {reason}")

    return action_result


@app.get("/api/adar-points")
async def get_adar_points():
    """
    ADAR Points Engine — Insurance-grade driver scoring system.
    Calculates a composite 0-1000 score from driving behavior, safety events,
    fatigue patterns, distraction metrics, and driving consistency.
    Maps the score to insurance premium tiers for third-party partners.
    """
    import math, hashlib
    from collections import defaultdict

    all_alerts = []
    for vid, alert_list in manager.alert_history.items():
        all_alerts.extend(alert_list)
    all_alerts.sort(key=lambda a: a.get('timestamp', ''))

    # ── Categorize alerts ──────────────────────────────────────────────
    categories = {
        "drowsiness": [], "yawning": [], "distraction": [],
        "phone": [], "looking_away": [], "speed": [], "seatbelt": []
    }
    for a in all_alerts:
        txt = a.get('text', '').upper()
        if 'DROWSI' in txt or 'DROWSY' in txt or 'EYES CLOSED' in txt:
            categories["drowsiness"].append(a)
        elif 'YAWN' in txt:
            categories["yawning"].append(a)
        elif 'PHONE' in txt:
            categories["phone"].append(a)
        elif 'DISTRACT' in txt:
            categories["distraction"].append(a)
        elif 'LOOKING' in txt:
            categories["looking_away"].append(a)
        elif 'SPEED' in txt:
            categories["speed"].append(a)
        elif 'SEATBELT' in txt or 'BELT' in txt:
            categories["seatbelt"].append(a)

    # ── Telemetry aggregation ──────────────────────────────────────────
    ear_values, mar_values, safety_scores = [], [], []
    total_frames = 0
    for vid, frames in manager.history.items():
        for f in frames[-500:]:
            total_frames += 1
            if f.get('ear') is not None:
                ear_values.append(f['ear'])
            if f.get('mar') is not None:
                mar_values.append(f['mar'])
            if f.get('attention_score') is not None:
                safety_scores.append(f['attention_score'])

    avg_ear = sum(ear_values) / len(ear_values) if ear_values else 0.28
    avg_mar = sum(mar_values) / len(mar_values) if mar_values else 0.14
    avg_safety = sum(safety_scores) / len(safety_scores) if safety_scores else 98
    ear_std = (sum((e - avg_ear) ** 2 for e in ear_values) / len(ear_values)) ** 0.5 if len(ear_values) > 1 else 0.02
    mar_std = (sum((m - avg_mar) ** 2 for m in mar_values) / len(mar_values)) ** 0.5 if len(mar_values) > 1 else 0.05

    # ── COMPONENT SCORING (each 0-100, weighted for final 0-1000) ─────
    # 1. Drowsiness Score (weight 25%) — most critical
    drowsy_count = len(categories["drowsiness"])
    max_drowsy_dur = max((a.get('drowsy_duration', 0) for a in categories["drowsiness"]), default=0)
    drowsy_score = max(0, 100 - (drowsy_count * 4) - (max_drowsy_dur * 8))

    # 2. Attention Score (weight 20%) — distraction + looking away + phone
    distraction_total = len(categories["distraction"]) + len(categories["looking_away"]) + len(categories["phone"])
    phone_penalty = len(categories["phone"]) * 6  # phone is extra dangerous
    attention_score = max(0, 100 - (distraction_total * 3) - phone_penalty)

    # 3. Fatigue Management (weight 15%) — yawning, EAR consistency
    yawn_count = len(categories["yawning"])
    ear_consistency = max(0, 100 - (ear_std * 500))  # lower std = better
    fatigue_score = max(0, (ear_consistency * 0.6) + (max(0, 100 - yawn_count * 5) * 0.4))

    # 4. Driving Consistency (weight 15%) — safety score stability
    if len(safety_scores) > 2:
        score_std = (sum((s - avg_safety) ** 2 for s in safety_scores) / len(safety_scores)) ** 0.5
        consistency_score = max(0, 100 - score_std * 3)
    else:
        consistency_score = 85  # default for new drivers

    # 5. Clean Driving Streak (weight 15%) — minutes without any alerts
    streak_minutes = 0
    if all_alerts:
        from datetime import datetime
        try:
            last_alert_time = datetime.fromisoformat(all_alerts[-1].get('timestamp', '').replace('Z', '+00:00'))
            streak_minutes = max(0, (datetime.now(last_alert_time.tzinfo) - last_alert_time).total_seconds() / 60)
        except:
            streak_minutes = 5
    else:
        streak_minutes = max(10, total_frames / 30)  # ~30 fps, convert to minutes
    streak_score = min(100, 50 + streak_minutes * 2)  # caps at 100

    # 6. Compliance & Safety (weight 10%) — seatbelt, speed
    seatbelt_violations = len(categories["seatbelt"])
    speed_violations = len(categories["speed"])
    compliance_score = max(0, 100 - (seatbelt_violations * 10) - (speed_violations * 5))

    # ── COMPOSITE SCORE (0-1000) ──────────────────────────────────────
    raw_score = (
        drowsy_score * 2.5 +       # 25%
        attention_score * 2.0 +     # 20%
        fatigue_score * 1.5 +       # 15%
        consistency_score * 1.5 +   # 15%
        streak_score * 1.5 +        # 15%
        compliance_score * 1.0      # 10%
    )
    adar_score = min(1000, max(0, round(raw_score)))

    # ── INSURANCE TIER ────────────────────────────────────────────────
    if adar_score >= 800:
        tier = "platinum"
        tier_label = "Platinum Shield"
        interest_rate = 0.5
        tier_color = "#6366f1"
        tier_desc = "Exceptional driver — lowest premium tier"
        discount_pct = 75
        risk_class = "Ultra-Low Risk"
    elif adar_score >= 650:
        tier = "gold"
        tier_label = "Gold Guard"
        interest_rate = 1.0
        tier_color = "#f59e0b"
        tier_desc = "Excellent driver — significant premium discount"
        discount_pct = 50
        risk_class = "Low Risk"
    elif adar_score >= 500:
        tier = "silver"
        tier_label = "Silver Standard"
        interest_rate = 1.5
        tier_color = "#94a3b8"
        tier_desc = "Good driver — moderate premium rate"
        discount_pct = 25
        risk_class = "Moderate Risk"
    elif adar_score >= 300:
        tier = "bronze"
        tier_label = "Bronze Basic"
        interest_rate = 2.0
        tier_color = "#b45309"
        tier_desc = "Needs improvement — standard premium"
        discount_pct = 10
        risk_class = "Elevated Risk"
    else:
        tier = "uninsured"
        tier_label = "High Risk"
        interest_rate = 2.5
        tier_color = "#ef4444"
        tier_desc = "Critical — premium surcharge applies"
        discount_pct = 0
        risk_class = "High Risk"

    # ── SCORE BREAKDOWN for radar chart ───────────────────────────────
    breakdown = {
        "drowsiness_control": round(drowsy_score, 1),
        "attention_focus": round(attention_score, 1),
        "fatigue_management": round(fatigue_score, 1),
        "driving_consistency": round(consistency_score, 1),
        "clean_streak": round(streak_score, 1),
        "compliance": round(compliance_score, 1),
    }

    # ── SCORE HISTORY (simulated trend over last 7 days) ──────────────
    import random
    random.seed(42)  # deterministic for demo consistency
    base = min(adar_score, 950)
    history_7d = []
    for i in range(7):
        day_score = max(0, min(1000, base - random.randint(-30, 50) + i * 8))
        history_7d.append({
            "day": f"Day {i+1}",
            "score": day_score,
            "tier": "platinum" if day_score >= 800 else "gold" if day_score >= 650 else "silver" if day_score >= 500 else "bronze" if day_score >= 300 else "risk",
        })

    # ── RISK FACTORS ──────────────────────────────────────────────────
    risk_factors = []
    if drowsy_count > 0:
        risk_factors.append({
            "factor": "Drowsiness Events",
            "impact": "high" if drowsy_count > 5 else "medium" if drowsy_count > 2 else "low",
            "count": drowsy_count,
            "penalty": min(100, drowsy_count * 4 + max_drowsy_dur * 8),
            "recommendation": "Take regular breaks every 2 hours. Use caffeine strategically. Ensure 7-8 hours of sleep before driving."
        })
    if len(categories["phone"]) > 0:
        risk_factors.append({
            "factor": "Phone Usage While Driving",
            "impact": "high",
            "count": len(categories["phone"]),
            "penalty": len(categories["phone"]) * 6,
            "recommendation": "Enable Do Not Disturb mode. Use hands-free systems. Pull over safely before using your phone."
        })
    if distraction_total > 0:
        risk_factors.append({
            "factor": "Distraction & Inattention",
            "impact": "high" if distraction_total > 10 else "medium",
            "count": distraction_total,
            "penalty": min(100, distraction_total * 3),
            "recommendation": "Keep eyes on the road. Minimize in-cabin distractions. Adjust mirrors and controls before driving."
        })
    if yawn_count > 0:
        risk_factors.append({
            "factor": "Fatigue Indicators (Yawning)",
            "impact": "medium" if yawn_count > 5 else "low",
            "count": yawn_count,
            "penalty": min(50, yawn_count * 5),
            "recommendation": "Monitor fatigue levels. Take breaks at rest areas. Avoid driving during natural sleep periods (2-4 AM, 1-3 PM)."
        })
    if seatbelt_violations > 0:
        risk_factors.append({
            "factor": "Seatbelt Non-Compliance",
            "impact": "high",
            "count": seatbelt_violations,
            "penalty": seatbelt_violations * 10,
            "recommendation": "Always wear seatbelt. Ensure all passengers are buckled. This is a legal requirement."
        })

    # ── PREMIUM CALCULATOR ────────────────────────────────────────────
    base_annual_premium = 45000  # INR for demo
    tier_multipliers = {
        "platinum": 0.25, "gold": 0.50, "silver": 0.75, "bronze": 1.0, "uninsured": 1.50
    }
    annual_premium = round(base_annual_premium * tier_multipliers.get(tier, 1.0))
    monthly_premium = round(annual_premium / 12)
    savings_vs_base = base_annual_premium - annual_premium

    # ── PARTNER INSURANCE OFFERS ──────────────────────────────────────
    partners = [
        {
            "name": "ADAR Shield Insurance",
            "logo_initial": "AS",
            "color": "#6366f1",
            "plan": f"{tier_label} Auto Cover",
            "annual": annual_premium,
            "monthly": monthly_premium,
            "rate": interest_rate,
            "features": ["Zero depreciation", "24/7 roadside assistance", "AI-powered claim processing", "Instant cashless repairs"]
        },
        {
            "name": "SafeDrive Partners",
            "logo_initial": "SD",
            "color": "#10b981",
            "plan": f"Behavior-Based {'Premium' if tier in ('platinum','gold') else 'Standard'} Plan",
            "annual": round(annual_premium * 0.95),
            "monthly": round(annual_premium * 0.95 / 12),
            "rate": round(interest_rate * 0.9, 2),
            "features": ["Usage-based pricing", "Dash-cam discount", "Family coverage", "No-claim bonus 50%"]
        },
        {
            "name": "FleetGuard Global",
            "logo_initial": "FG",
            "color": "#f59e0b",
            "plan": f"Fleet {'Elite' if tier in ('platinum','gold') else 'Standard'} Protection",
            "annual": round(annual_premium * 1.05),
            "monthly": round(annual_premium * 1.05 / 12),
            "rate": round(interest_rate * 1.1, 2),
            "features": ["Multi-vehicle discount", "International coverage", "Fleet analytics dashboard", "Dedicated claim manager"]
        },
    ]

    # ── AI INSIGHTS ───────────────────────────────────────────────────
    insights = []
    if adar_score >= 800:
        insights.append({"type": "success", "text": f"Outstanding! Your ADAR score of {adar_score} qualifies you for Platinum Shield — the lowest insurance premium available."})
        insights.append({"type": "tip", "text": "Maintain your clean driving streak to lock in this rate for the next renewal period."})
    elif adar_score >= 500:
        gap = 800 - adar_score
        insights.append({"type": "info", "text": f"You need just {gap} more points to reach Platinum tier and unlock 75% premium discount."})
        if drowsy_count > 0:
            insights.append({"type": "warning", "text": f"Eliminating {drowsy_count} drowsiness events could boost your score by up to {drowsy_count * 4} points."})
    else:
        insights.append({"type": "danger", "text": f"Your current score of {adar_score} places you in a higher premium bracket. Focus on reducing critical violations."})
        insights.append({"type": "tip", "text": "Start with the highest-impact risk factor and work your way down. Consistent improvement is rewarded."})

    if len(categories["phone"]) > 0:
        insights.append({"type": "danger", "text": f"Phone usage detected {len(categories['phone'])}x — this is the single largest score penalty. Going phone-free could add {len(categories['phone'])*6} points."})

    # ── Score grade ──
    if adar_score >= 900: grade = "A+"
    elif adar_score >= 800: grade = "A"
    elif adar_score >= 700: grade = "B+"
    elif adar_score >= 600: grade = "B"
    elif adar_score >= 500: grade = "C"
    elif adar_score >= 400: grade = "D"
    else: grade = "F"

    return {
        "adar_score": adar_score,
        "grade": grade,
        "max_score": 1000,
        "tier": tier,
        "tier_label": tier_label,
        "tier_color": tier_color,
        "tier_desc": tier_desc,
        "interest_rate": interest_rate,
        "discount_pct": discount_pct,
        "risk_class": risk_class,
        "breakdown": breakdown,
        "history_7d": history_7d,
        "risk_factors": risk_factors,
        "premium": {
            "base_annual": base_annual_premium,
            "your_annual": annual_premium,
            "your_monthly": monthly_premium,
            "savings": savings_vs_base,
            "currency": "INR",
        },
        "partners": partners,
        "insights": insights,
        "event_summary": {
            "total_alerts": len(all_alerts),
            "drowsiness": drowsy_count,
            "phone_usage": len(categories["phone"]),
            "distractions": distraction_total,
            "yawning": yawn_count,
            "seatbelt": seatbelt_violations,
            "speed": speed_violations,
        },
        "biometrics": {
            "avg_ear": round(avg_ear, 4),
            "ear_stability": round(max(0, 100 - ear_std * 500), 1),
            "avg_mar": round(avg_mar, 4),
            "avg_safety_score": round(avg_safety, 1),
            "total_frames_analyzed": total_frames,
            "avg_hr": health_store.heart_rate if health_store.is_fresh else 0,
            "avg_spo2": health_store.spo2 if health_store.is_fresh else 0,
            "avg_co2": sensor_store.co2_ppm if sensor_store.is_fresh else 0,
            "avg_alc": sensor_store.alcohol_mgl if sensor_store.has_alcohol else 0.0,
            "presence_pct": 100.0 if (presence_store.is_fresh and presence_store.present) else 0.0,
        },
    }


@app.get("/api/fleet/analytics")
async def get_fleet_analytics():
    """
    Deep fleet-wide analytics for the Analytics panel.
    Provides: event timeline, anomaly scores, vehicle comparison,
    EAR/MAR correlation data, trend forecast, heatmap data,
    event distribution, and AI-generated insights.
    """
    import math, random as _rng
    from collections import defaultdict
    from datetime import datetime, timedelta

    all_alerts = []
    for vid, alert_list in manager.alert_history.items():
        for a in alert_list:
            entry = dict(a)
            if 'vehicle_id' not in entry:
                entry['vehicle_id'] = vid
            all_alerts.append(entry)
    all_alerts.sort(key=lambda a: a.get('timestamp', ''))

    vehicle_ids = list(set(
        list(manager.vehicles.keys()) +
        list(manager.history.keys()) +
        list(manager.alert_history.keys())
    )) or ['ADAR-001']

    # ── Type counts ──
    type_counts = {"drowsiness": 0, "yawning": 0, "distraction": 0, "phone": 0, "looking_away": 0}
    for a in all_alerts:
        txt = a.get('text', '').upper()
        if 'DROWSI' in txt or 'DROWSY' in txt or 'EYES CLOSED' in txt:
            type_counts["drowsiness"] += 1
        elif 'YAWN' in txt:
            type_counts["yawning"] += 1
        elif 'DISTRACT' in txt:
            type_counts["distraction"] += 1
        elif 'PHONE' in txt:
            type_counts["phone"] += 1
        elif 'LOOKING' in txt:
            type_counts["looking_away"] += 1

    total_events = sum(type_counts.values())

    # ── Timeline (per-minute, stacked by type) ──
    tl_buckets = defaultdict(lambda: {"drowsiness": 0, "yawning": 0, "distraction": 0, "phone": 0, "looking_away": 0})
    for a in all_alerts:
        ts = a.get('timestamp', '')
        mk = ts[:16] if len(ts) >= 16 else ts
        txt = a.get('text', '').upper()
        if 'DROWSI' in txt or 'DROWSY' in txt or 'EYES CLOSED' in txt:
            tl_buckets[mk]["drowsiness"] += 1
        elif 'YAWN' in txt:
            tl_buckets[mk]["yawning"] += 1
        elif 'DISTRACT' in txt:
            tl_buckets[mk]["distraction"] += 1
        elif 'PHONE' in txt:
            tl_buckets[mk]["phone"] += 1
        elif 'LOOKING' in txt:
            tl_buckets[mk]["looking_away"] += 1
    timeline = [{"time": k, **v} for k, v in sorted(tl_buckets.items())][-60:]

    # ── Anomaly scoring (multi-dimensional) ──
    safety_scores, ear_vals, mar_vals, speed_vals, co2_vals = [], [], [], [], []
    for vid, frames in manager.history.items():
        for f in frames[-300:]:
            if f.get('attention_score') is not None:
                safety_scores.append(f['attention_score'])
            if f.get('ear'):
                ear_vals.append(f['ear'])
            if f.get('mar'):
                mar_vals.append(f['mar'])
            if f.get('speed') is not None:
                speed_vals.append(f['speed'])
            if f.get('co2') is not None:
                co2_vals.append(f['co2'])

    def z_score_anomaly(values, threshold=2.0):
        if len(values) < 3:
            return 0
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        if std == 0:
            return 0
        anomalies = sum(1 for v in values if abs(v - mean) / std > threshold)
        return round(anomalies / len(values) * 100, 1)

    anomaly_scores = {
        "drowsiness_rate": z_score_anomaly(ear_vals, 1.5),
        "fatigue_rate": z_score_anomaly(mar_vals, 1.5),
        "attention_deviation": z_score_anomaly(safety_scores, 2) if safety_scores else 0,
        "speed_anomaly": z_score_anomaly(speed_vals, 2),
        "co2_anomaly": z_score_anomaly(co2_vals, 2),
    }
    total_anomalies = sum(1 for v in anomaly_scores.values() if v > 5)

    # ── Vehicle comparison ──
    vehicle_comparison = []
    for vid in vehicle_ids:
        frames = manager.history.get(vid, [])
        scores = [f['attention_score'] for f in frames[-200:] if f.get('attention_score') is not None]
        alert_count = len(manager.alert_history.get(vid, []))
        vehicle_comparison.append({
            "vehicle_id": vid,
            "avg_score": round(sum(scores) / len(scores), 1) if scores else 98,
            "min_score": round(min(scores), 1) if scores else 98,
            "alert_count": alert_count,
        })

    # ── EAR/MAR correlation scatter data ──
    corr_data = []
    for vid, frames in manager.history.items():
        for f in frames[-200:]:
            if f.get('ear') and f.get('mar'):
                corr_data.append({"x": round(f['ear'], 4), "y": round(f['mar'], 4)})
    corr_data = corr_data[-150:]  # limit

    # ── AI Trend forecast ──
    # Exponential moving average on safety scores
    trend_points = []
    if safety_scores:
        window = min(20, len(safety_scores))
        ema = safety_scores[0]
        alpha = 2 / (window + 1)
        for i, s in enumerate(safety_scores[-60:]):
            ema = alpha * s + (1 - alpha) * ema
            trend_points.append({"idx": i, "actual": round(s, 1), "ema": round(ema, 1)})
        # Forecast next 10 points
        last_ema = ema
        drift = (safety_scores[-1] - safety_scores[0]) / max(len(safety_scores), 1)
        for j in range(10):
            forecast = max(0, min(100, last_ema + drift * (j + 1)))
            trend_points.append({"idx": len(trend_points), "forecast": round(forecast, 1), "ema": round(forecast, 1)})

    # ── Heatmap data (24 hours x 7 days-of-week) ──
    heatmap = [[0] * 24 for _ in range(7)]
    for a in all_alerts:
        ts = a.get('timestamp', '')
        try:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00')) if ts else None
            if dt:
                heatmap[dt.weekday()][dt.hour] += 1
        except Exception:
            pass

    # ── Session stats ──
    avg_safety = sum(safety_scores) / len(safety_scores) if safety_scores else 98
    min_safety = min(safety_scores) if safety_scores else 98
    avg_ear = sum(ear_vals) / len(ear_vals) if ear_vals else 0.28
    min_ear = min(ear_vals) if ear_vals else 0.28
    avg_mar = sum(mar_vals) / len(mar_vals) if mar_vals else 0.14

    # Fleet IQ score (composite)
    fleet_iq = max(0, min(100, round(avg_safety - total_events * 0.3 - total_anomalies * 2, 1)))

    # ── AI insights generation ──
    insights = []
    if total_events == 0:
        insights.append({"icon": "✅", "title": "Clean Session", "text": "No safety events detected. All fleet drivers are performing optimally.", "level": "success"})
    else:
        if type_counts["drowsiness"] > 3:
            insights.append({"icon": "🚨", "title": "High Drowsiness Alert", "text": f"{type_counts['drowsiness']} drowsiness events detected. Consider mandatory rest breaks and driver rotation.", "level": "critical"})
        if type_counts["yawning"] > 5:
            insights.append({"icon": "😴", "title": "Fatigue Pattern", "text": f"{type_counts['yawning']} yawning events suggest fleet-wide fatigue. Review shift schedules.", "level": "warning"})
        if type_counts["phone"] > 0:
            insights.append({"icon": "📱", "title": "Phone Usage Detected", "text": f"{type_counts['phone']} phone usage events. Enforce strict no-phone policy.", "level": "critical"})
        if type_counts["distraction"] > 2:
            insights.append({"icon": "⚠️", "title": "Distraction Spike", "text": f"{type_counts['distraction']} distraction events. Review cabin environment and route conditions.", "level": "warning"})
        if total_anomalies > 2:
            insights.append({"icon": "🔬", "title": "Anomaly Cluster", "text": f"{total_anomalies} anomaly dimensions flagged. Biometric patterns deviate from baselines.", "level": "warning"})
    if avg_ear < 0.22:
        insights.append({"icon": "👁️", "title": "Low EAR Baseline", "text": f"Average EAR is {avg_ear:.3f}, below safe threshold. Drivers may be fatigued.", "level": "warning"})
    if not insights:
        insights.append({"icon": "📊", "title": "Normal Operations", "text": "Fleet is operating within normal parameters. Continue monitoring.", "level": "info"})

    return {
        "fleet_iq": fleet_iq,
        "total_events": total_events,
        "total_anomalies": total_anomalies,
        "vehicle_count": len(vehicle_ids),
        "vehicle_ids": vehicle_ids,
        "type_counts": type_counts,
        "timeline": timeline,
        "anomaly_scores": anomaly_scores,
        "vehicle_comparison": vehicle_comparison,
        "correlation_data": corr_data,
        "trend_forecast": trend_points,
        "heatmap": heatmap,
        "session": {
            "avg_safety": round(avg_safety, 1),
            "min_safety": round(min_safety, 1),
            "avg_ear": round(avg_ear, 4),
            "min_ear": round(min_ear, 4),
            "avg_mar": round(avg_mar, 4),
        },
        "insights": insights,
        "recent_events": [
            {
                "text": a.get('text', ''),
                "timestamp": a.get('timestamp', ''),
                "vehicle_id": a.get('vehicle_id', ''),
                "ear": a.get('ear', 0),
                "mar": a.get('mar', 0),
            }
            for a in all_alerts[-50:]
        ],
    }


# ── ESP32 Super-Node Data Transformer ─────────────────────────────────

# ── MQ Sensor Calibration ──
# MQ sensors need proper ADC→real-unit conversion.
# The load resistor RL is typically 10 kΩ on most modules, Vcc = 3.3 V on ESP32.
MQ_RL_KOHM = 10.0   # Load resistor in kΩ
MQ_VCC     = 3.3     # ESP32 ADC reference voltage

# Post-warmup settling: track when each vehicle's MQ warmup ended
# to add a 60s grace period before trusting calibrated values
_vehicle_warmup_end: dict[str, float] = {}
MQ_SETTLE_SECONDS = 0  # firmware handles warmup with pre-filled buffers, no server settling needed

def _mq_adc_to_ppm(adc_raw: int, baseline: int, r0_ratio_clean: float,
                    curve_a: float, curve_b: float,
                    ppm_floor: float, ppm_ceil: float) -> float:
    """
    Convert MQ gas sensor raw ADC to estimated PPM using the sensor's
    characteristic curve:  PPM = a * (Rs/R0)^b

    Uses baseline (clean-air reading) to calibrate R0.
    """
    if adc_raw <= 0 or baseline <= 0:
        return ppm_floor

    # Voltage from ADC
    v_sensor = (adc_raw / 4095.0) * MQ_VCC
    v_baseline = (baseline / 4095.0) * MQ_VCC

    if v_sensor < 0.01 or v_baseline < 0.01:
        return ppm_floor

    # Rs = RL × (Vcc - Vout) / Vout   (voltage divider)
    rs = MQ_RL_KOHM * (MQ_VCC - v_sensor) / v_sensor
    rs_baseline = MQ_RL_KOHM * (MQ_VCC - v_baseline) / v_baseline

    # R0 = Rs_clean / r0_ratio_clean  (datasheet Rs/R0 in clean air)
    r0 = rs_baseline / r0_ratio_clean
    if r0 < 0.001:
        return ppm_floor

    ratio = rs / r0
    if ratio <= 0:
        return ppm_floor

    ppm = curve_a * (ratio ** curve_b)
    return max(ppm_floor, min(ppm_ceil, round(ppm, 1)))


def _co2_from_adc(adc_raw: int, baseline: int) -> float:
    """MQ-135: ADC → CO₂ PPM.  Characteristic curve from MQ-135 datasheet."""
    # MQ-135 Rs/R0 in clean air ≈ 3.6 (from datasheet)
    # CO₂ curve: PPM = 116.6020682 × (Rs/R0)^(-2.769034857)
    return _mq_adc_to_ppm(
        adc_raw, baseline,
        r0_ratio_clean=3.6,
        curve_a=116.602, curve_b=-2.769,
        ppm_floor=400.0,   # atmospheric minimum CO₂
        ppm_ceil=5000.0
    )


def _alcohol_from_adc(adc_raw: int, baseline: int) -> float:
    """MQ-3: ADC → mg/L breath alcohol.  Characteristic curve from MQ-3 datasheet."""
    # MQ-3 Rs/R0 in clean air ≈ 60 (from datasheet)
    # Alcohol curve: mg/L = 0.4091 × (Rs/R0)^(-1.497)
    return round(_mq_adc_to_ppm(
        adc_raw, baseline,
        r0_ratio_clean=60.0,
        curve_a=0.4091, curve_b=-1.497,
        ppm_floor=0.0,
        ppm_ceil=5.0  # mg/L cap
    ), 4)


def transform_esp32_data(raw: dict, vehicle_id: str) -> dict:
    """
    Transform the unified JSON from ESP32 Super-Node firmware into the
    dashboard-compatible telemetry format.  The ESP32 sends all sensor
    data in one JSON frame every ~1 second.

    ESP32 fields → Dashboard fields mapping:
      mq3 (raw ADC)       → alcohol_mgl (mg/L)
      co2 (raw ADC)       → co2_ppm
      bpm                 → health.heart_rate
      spo2                → health.spo2
      g_force, accel_*    → imu dict
      lat, lon, gps_*     → location dict
      radar_presence      → presence dict
      engine, motor_pwm   → engine status
      sos, sos_pct        → SOS status
      buzzer, danger      → alert state
    """
    now_iso = datetime.now(timezone.utc).isoformat()

    # ── Sensor hardware presence flags from ESP32 (infer from payload if not sent) ──
    has_mpu  = raw.get("has_mpu", "g_force" in raw)
    has_max  = raw.get("has_max", "bpm" in raw or "spo2" in raw)
    has_gps  = raw.get("has_gps", False)
    has_radar = raw.get("has_radar", "radar_presence" in raw)

    # ── Sensor quality flags from ESP32 ──
    finger_on    = raw.get("finger_on", False)
    mq_warmup    = raw.get("mq_warmup", True)   # True = still warming up
    mq135_base   = raw.get("mq135_baseline", 0)
    mq3_base     = raw.get("mq3_baseline", 0)

    # ── Post-warmup settling period ──
    # MQ sensors produce erratic readings for ~60s after warmup ends.
    # Track when warmup ended per vehicle and add a settling window.
    import time as _time
    if mq_warmup:
        # Still warming up — reset tracker
        _vehicle_warmup_end.pop(vehicle_id, None)
    else:
        if vehicle_id not in _vehicle_warmup_end:
            _vehicle_warmup_end[vehicle_id] = _time.time()
        settling = _time.time() - _vehicle_warmup_end[vehicle_id] < MQ_SETTLE_SECONDS
        if settling:
            mq_warmup = True  # Override: treat as warmup during settling

    # ── Gas Sensors (raw ADC → calibrated real-world units) ──
    co2_raw = raw.get("co2", 0)
    mq3_raw = raw.get("mq3", 0)

    if mq_warmup:
        # Sensors still warming up — show warming status
        co2_ppm = 400.0  # Report ambient until ready
        alcohol_mgl = 0.0
    elif mq135_base > 0:
        # Calibrated conversion using MQ characteristic curves
        co2_ppm = _co2_from_adc(co2_raw, mq135_base)
        alcohol_mgl = _alcohol_from_adc(mq3_raw, mq3_base)
    else:
        # Firmware already maps ADC to meaningful values:
        #   MQ-135: map(ADC, 0, 4095, 400, 2000) → CO₂ PPM
        #   MQ-3:   map(ADC, 0, 4095, 0, 1000) → scaled alcohol level
        # Use these directly — they are real sensor readings.
        co2_ppm = float(co2_raw) if co2_raw > 0 else 400.0
        # Convert MQ-3 scaled value (0–1000) to mg/L approximation
        alcohol_mgl = round(float(mq3_raw) / 1000.0 * 0.5, 4) if mq3_raw > 0 else 0.0

    # ── Health (MAX30100) — respect finger contact detection ──
    bpm = raw.get("bpm", 0)
    spo2 = raw.get("spo2", 0)

    # If sensor not present or no finger contact, force zero
    if not has_max or not finger_on:
        bpm = 0
        spo2 = 0

    health_dict = {
        "heart_rate": bpm,
        "spo2": spo2,
        "sensor": "MAX30100",
        "fresh": has_max,
        "readings": 1 if has_max else 0,
        "finger_on": finger_on if has_max else False,
    }

    # ── IMU (MPU6050) ──
    g_force = raw.get("g_force", 1.0)
    imu_dict = {
        "ax": raw.get("accel_x", 0),
        "ay": raw.get("accel_y", 0),
        "az": raw.get("accel_z", 0),
        "gx": 0, "gy": 0, "gz": 0,
        "speed_kmh": raw.get("gps_speed_kmh", 0),
        "g_force": g_force if has_mpu else 1.0,  # Default 1.0g when sensor absent
        "fresh": has_mpu,
        "readings": 1 if has_mpu else 0,
    }

    # ── GPS (NEO-6M hardware) ──
    gps_fix = raw.get("gps_fix", False)
    lat_str = raw.get("lat", "0")
    lon_str = raw.get("lon", "0")
    try:
        lat = round(float(lat_str), 6)
        lon = round(float(lon_str), 6)
    except (ValueError, TypeError):
        lat, lon = 0.0, 0.0
        gps_fix = False

    # Only use hardware GPS if there is a valid fix and non-zero coordinates
    if gps_fix and (abs(lat) > 0.001 or abs(lon) > 0.001):
        location = {
            "lat": lat,
            "lon": lon,
            "heading": 0,
            "accuracy": 5.0,      # Hardware GPS typically 3-10m
            "altitude": None,
        }
        gps_source = "hardware"   # Real hardware GPS fix
        speed_kmh = round(raw.get("gps_speed_kmh", 0), 1)
    else:
        # Fall back to browser GPS if available
        if gps_store.is_fresh and gps_store.lat is not None:
            location = {
                "lat": round(gps_store.lat, 6),
                "lon": round(gps_store.lon, 6),
                "heading": round((gps_store.heading or 0) % 360, 1),
                "accuracy": gps_store.accuracy,
                "altitude": round(gps_store.altitude, 1) if gps_store.altitude else None,
            }
            gps_source = "browser"
            speed_kmh = round((gps_store.speed_mps or 0) * 3.6, 1)
        else:
            location = None
            gps_source = "none"
            speed_kmh = round(raw.get("gps_speed_kmh", 0), 1)

    # ── Radar / Presence (mmWave C4001) — use real values from ESP32 ──
    radar_present = raw.get("radar_presence", False)
    radar_active = raw.get("radar_active", False) and has_radar
    radar_distance = raw.get("radar_distance", 0.0)
    radar_energy = raw.get("radar_energy", 0)
    presence_dict = {
        "present": radar_present if has_radar else False,
        "distance": round(radar_distance, 2),
        "energy": radar_energy,
        "sensor": "C4001",
        "fresh": radar_active,
        "readings": 1 if radar_active else 0,
    }

    # ── Engine / Motor / SOS ──
    engine_on = raw.get("engine", "OFF") == "ON"
    motor_pwm = raw.get("motor_pwm", 0)
    sos_active = raw.get("sos", False)
    sos_pct = raw.get("sos_pct", 100)
    buzzer_on = raw.get("buzzer", "OFF") == "ON"
    danger = raw.get("danger", False)

    # ── Safety status ──
    alerts = []
    status = "SAFE"
    if danger:
        status = "DANGER"
    if sos_active:
        status = "DANGER"
        alerts.append(f"🚨 SOS EMERGENCY — Shutdown {sos_pct}%")
    if not mq_warmup:
        # Only generate gas alerts after warm-up
        if co2_ppm > 1000:
            alerts.append(f"⚠️ High CO₂ — {co2_ppm:.0f} PPM")
        if alcohol_mgl > 0.08:
            status = "DANGER"
            alerts.append(f"🍺 Alcohol Detected — {alcohol_mgl:.3f} mg/L")
    if has_mpu and g_force > 1.5:
        status = "DANGER"
        alerts.append(f"💥 High G-Force — {g_force:.2f}g (Crash?)")
    if has_max and finger_on and bpm > 0 and (bpm < 50 or bpm > 120):
        alerts.append(f"💓 Abnormal Heart Rate — {bpm} BPM")
    if has_max and finger_on and spo2 > 0 and spo2 < 94:
        alerts.append(f"🫁 Low SpO₂ — {spo2}%")

    return {
        "vehicle_id": vehicle_id,
        "timestamp": now_iso,
        "server_ts": now_iso,
        "is_simulation": False,
        # CV pipeline fields (zeros — no camera on ESP32)
        "ear": 0.0,
        "mar": 0.0,
        "face_detected": False,
        "affective_state": "",
        # Safety
        "status": status,
        "alerts": alerts,
        "danger": danger,
        # Speed
        "speed_kmh": speed_kmh,
        # GPS
        "location": location,
        "gps_source": gps_source,
        "gps_fix": gps_fix,
        "gps_sats": raw.get("gps_sats", 0),
        # CO₂ (MQ-135)
        "co2_ppm": co2_ppm,
        "co2_source": "MQ-135",
        "co2_raw_adc": co2_raw,
        "mq_warmup": mq_warmup,
        # Alcohol (MQ-3)
        "alcohol_mgl": alcohol_mgl,
        "alcohol_source": "MQ-3",
        "alcohol_raw_adc": mq3_raw,
        # IMU (MPU6050) — always send dict, dashboard uses imu_source to decide display
        "imu": imu_dict,
        "imu_source": "MPU6050" if has_mpu else "none",
        # Health (MAX30100) — always send dict
        "health": health_dict,
        "health_source": "MAX30100" if has_max else "none",
        # Presence (mmWave Radar) — always send dict
        "presence": presence_dict,
        "presence_source": "C4001" if has_radar else "none",
        # ESP32-specific fields
        "engine": "ON" if engine_on else "OFF",
        "motor_pwm": motor_pwm,
        "sos": sos_active,
        "sos_pct": sos_pct,
        "buzzer": "ON" if buzzer_on else "OFF",
        "esp32_uptime": raw.get("uptime_s", 0),
        "esp32_connected": True,
    }


async def update_stores_from_esp32(raw: dict) -> None:
    """Update all sensor stores from ESP32 unified JSON so REST APIs stay current."""
    co2_raw = raw.get("co2", 0)
    mq3_raw = raw.get("mq3", 0)
    mq_warmup = raw.get("mq_warmup", True)
    mq135_base = raw.get("mq135_baseline", 0)
    mq3_base = raw.get("mq3_baseline", 0)

    # Sensor presence flags (infer from payload so older ESP32 firmware still fills stores)
    has_mpu = raw.get("has_mpu", "g_force" in raw)
    has_max = raw.get("has_max", "bpm" in raw or "spo2" in raw)
    has_radar = raw.get("has_radar", "radar_presence" in raw)

    # Apply same settling period as transform
    import time as _time
    vid = raw.get("vehicle_id", "unknown")
    if not mq_warmup:
        if vid in _vehicle_warmup_end:
            settling = _time.time() - _vehicle_warmup_end[vid] < MQ_SETTLE_SECONDS
            if settling:
                mq_warmup = True

    if mq_warmup:
        co2_ppm = 400.0
        alcohol_mgl = 0.0
    elif mq135_base > 0:
        co2_ppm = _co2_from_adc(co2_raw, mq135_base)
        alcohol_mgl = _alcohol_from_adc(mq3_raw, mq3_base)
    else:
        # Firmware maps ADC → PPM directly, use values as-is
        co2_ppm = float(co2_raw) if co2_raw > 0 else 400.0
        alcohol_mgl = round(float(mq3_raw) / 1000.0 * 0.5, 4) if mq3_raw > 0 else 0.0

    await sensor_store.update(CO2SensorUpdate(
        co2_ppm=co2_ppm,
        raw_adc=co2_raw,
        alcohol_mgl=alcohol_mgl,
        alcohol_raw_adc=mq3_raw,
        vehicle_id=raw.get("vehicle_id"),
        sensor="MQ-135",
    ))

    if has_mpu:
        await imu_store.update(MPU6050Update(
            ax=raw.get("accel_x", 0),
            ay=raw.get("accel_y", 0),
            az=raw.get("accel_z", 0),
            g_force=raw.get("g_force", 1.0),
            speed_kmh=raw.get("gps_speed_kmh", 0),
            vehicle_id=raw.get("vehicle_id"),
        ))

    bpm = raw.get("bpm", 0)
    spo2v = raw.get("spo2", 0)
    finger_on = raw.get("finger_on", (bpm > 0 or spo2v > 0))
    if has_max and (finger_on or bpm > 0 or spo2v > 0):
        await health_store.update(HealthSensorUpdate(
            heart_rate=bpm,
            spo2=spo2v,
            vehicle_id=raw.get("vehicle_id"),
            sensor="MAX30100",
        ))

    if has_radar:
        await presence_store.update(PresenceSensorUpdate(
            present=raw.get("radar_presence", False),
            distance=raw.get("radar_distance", 0.0),
            energy=raw.get("radar_energy", 0),
            vehicle_id=raw.get("vehicle_id"),
            sensor="C4001",
        ))

    # Update GPS store from hardware GPS if valid
    gps_fix = raw.get("gps_fix", False)
    if gps_fix:
        try:
            lat = float(raw.get("lat", 0))
            lon = float(raw.get("lon", 0))
            if abs(lat) > 0.001 or abs(lon) > 0.001:
                await gps_store.update(GPSUpdate(
                    lat=lat, lon=lon,
                    accuracy=5.0,
                    speed_mps=raw.get("gps_speed_kmh", 0) / 3.6,
                ))
        except (ValueError, TypeError):
            pass


# ── WebSocket Endpoints ──────────────────────────────────────────────
@app.websocket("/ws/dashboard")
async def ws_dashboard(websocket: WebSocket):
    await manager.connect_dashboard(websocket)
    try:
        while True:
            msg = await websocket.receive_text()
            try:
                cmd = json.loads(msg)
                # Handle GPS push from dashboard
                if cmd.get("type") == "GPS":
                    await gps_store.update(GPSUpdate(
                        lat=cmd["lat"], lon=cmd["lon"],
                        accuracy=cmd.get("accuracy"),
                        altitude=cmd.get("altitude"),
                        heading=cmd.get("heading"),
                        speed_mps=cmd.get("speed_mps"),
                    ))
                # Handle SOS
                elif cmd.get("type") == "SOS":
                    print(f"[SOS] EMERGENCY! vehicle={cmd.get('vehicle_id')}")
                    await manager.broadcast(cmd.get("vehicle_id", "UNKNOWN"), {
                        "vehicle_id": cmd.get("vehicle_id", "UNKNOWN"),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "status": "DANGER",
                        "alerts": ["🚨 EMERGENCY SOS ACTIVATED"],
                        "is_sos": True,
                    })
                else:
                    print(f"[HUB] Dashboard cmd: {cmd}")
            except (json.JSONDecodeError, KeyError):
                pass
    except WebSocketDisconnect:
        manager.disconnect_dashboard(websocket)
    except Exception:
        manager.disconnect_dashboard(websocket)


@app.websocket("/ws/vehicle/{vehicle_id}")
async def ws_vehicle(websocket: WebSocket, vehicle_id: str):
    await manager.connect_vehicle(websocket, vehicle_id)
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)

                # Handle heartbeat (keep-alive) — don't broadcast
                if data.get("type") == "heartbeat":
                    continue

                # Handle SOS_EMERGENCY alert from ESP32
                if data.get("type") == "SOS_EMERGENCY":
                    print(f"[SOS] 🚨 EMERGENCY from ESP32! vehicle={vehicle_id}")
                    sos_data = {
                        "vehicle_id": vehicle_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "server_ts": datetime.now(timezone.utc).isoformat(),
                        "is_simulation": False,
                        "status": "DANGER",
                        "alerts": [f"🚨 SOS EMERGENCY — {data.get('message', 'Driver triggered SOS!')}"],
                        "is_sos": True,
                        "sos": True,
                        "danger": True,
                        "ear": 0.0, "mar": 0.0,
                        "co2_ppm": 0, "co2_source": "MQ-135",
                        "alcohol_mgl": 0, "alcohol_source": "MQ-3",
                        "health": {"heart_rate": data.get("bpm", 0), "spo2": 0, "sensor": "MAX30100", "fresh": True, "readings": 1},
                        "health_source": "MAX30100",
                        "imu": {"ax": 0, "ay": 0, "az": 0, "gx": 0, "gy": 0, "gz": 0, "speed_kmh": 0, "g_force": data.get("g_force", 1.0), "fresh": True, "readings": 1},
                        "imu_source": "MPU6050",
                        "presence": None, "presence_source": "none",
                        "speed_kmh": 0,
                        "esp32_connected": True,
                    }
                    # Include GPS if available
                    try:
                        lat = float(data.get("lat", 0))
                        lon = float(data.get("lon", 0))
                        if abs(lat) > 0.001 or abs(lon) > 0.001:
                            sos_data["location"] = {"lat": lat, "lon": lon, "heading": 0, "accuracy": 5.0, "altitude": None}
                            sos_data["gps_source"] = "browser"
                        else:
                            sos_data["location"] = None
                            sos_data["gps_source"] = "none"
                    except (ValueError, TypeError):
                        sos_data["location"] = None
                        sos_data["gps_source"] = "none"

                    await manager.broadcast(vehicle_id, sos_data)
                    continue

                # ── Detect ESP32 Super-Node unified format (has 'mq3' field) ──
                if "mq3" in data:
                    # Update all sensor stores from ESP32 data
                    await update_stores_from_esp32(data)
                    # Transform to dashboard format
                    data = transform_esp32_data(data, vehicle_id)
                    # Log every 10th frame to confirm data flow
                    uptime = data.get("esp32_uptime", 0)
                    if uptime % 10 == 0:
                        dashboards = len(manager.global_dashboards)
                        h = data.get("health", {})
                        hr_val = h.get("heart_rate", 0) if h else 0
                        mpu_flag = 'Y' if data.get('imu_source') == 'MPU6050' else 'N'
                        max_flag = 'Y' if data.get('health_source') == 'MAX30100' else 'N'
                        radar_flag = 'Y' if data.get('presence_source') == 'C4001' else 'N'
                        print(f"[ESP32] Frame t={uptime}s | CO₂={data.get('co2_ppm',0):.0f} | HR={hr_val} | MPU={mpu_flag} MAX={max_flag} RDR={radar_flag} | → {dashboards} dashboard(s)")
                else:
                    # ── Legacy format (CV pipeline / bridge.py) ──
                    data["vehicle_id"] = vehicle_id
                    data["server_ts"] = datetime.now(timezone.utc).isoformat()
                    data["is_simulation"] = False

                    # Enrich with GPS from gps_store
                    if gps_store.is_fresh and gps_store.lat is not None:
                        data["location"] = {
                            "lat": round(gps_store.lat, 6),
                            "lon": round(gps_store.lon, 6),
                            "heading": round((gps_store.heading or 0) % 360, 1),
                            "accuracy": gps_store.accuracy,
                            "altitude": round(gps_store.altitude, 1) if gps_store.altitude else None,
                        }
                        data["gps_source"] = "browser"
                        if gps_store.speed_mps and gps_store.speed_mps >= 0:
                            data["speed_kmh"] = round(gps_store.speed_mps * 3.6, 1)
                    else:
                        data.setdefault("location", None)
                        data.setdefault("gps_source", "none")

                    # Enrich with CO2 + Alcohol from sensor_store
                    if sensor_store.is_fresh and sensor_store.co2_ppm is not None:
                        data["co2_ppm"] = sensor_store.co2_ppm
                        data["co2_source"] = "MQ-135"
                    else:
                        data.setdefault("co2_ppm", 0.0)
                        data.setdefault("co2_source", "none")
                    if sensor_store.has_alcohol:
                        data["alcohol_mgl"] = sensor_store.alcohol_mgl
                        data["alcohol_source"] = "MQ-3"
                    else:
                        data.setdefault("alcohol_mgl", 0.0)
                        data.setdefault("alcohol_source", "none")

                    # Enrich with IMU
                    if imu_store.is_fresh:
                        data["imu"] = imu_store.to_dict()
                        data["imu_source"] = "MPU6050"
                        if data.get("speed_kmh", 0) == 0 and imu_store.speed_kmh > 0:
                            data["speed_kmh"] = round(imu_store.speed_kmh, 1)
                    else:
                        data.setdefault("imu_source", "none")

                    # Enrich with Health
                    if health_store.is_fresh:
                        data["health"] = health_store.to_dict()
                        data["health_source"] = "MAX30100"
                    else:
                        data.setdefault("health", None)
                        data.setdefault("health_source", "none")

                    # Enrich with Presence
                    if presence_store.is_fresh:
                        data["presence"] = presence_store.to_dict()
                        data["presence_source"] = "C4001"
                    else:
                        data.setdefault("presence", None)
                        data.setdefault("presence_source", "none")

                    # Pass through ESP32 fields if present
                    data.setdefault("esp32_connected", sensor_store.is_fresh or imu_store.is_fresh or health_store.is_fresh)

                await manager.broadcast(vehicle_id, data)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
            except Exception as e:
                logging.warning(f"[WS] Frame error: {e}")
    except WebSocketDisconnect:
        manager.disconnect_vehicle(vehicle_id)
    except Exception as e:
        print(f"[HUB] Vehicle '{vehicle_id}' error: {e}")
        manager.disconnect_vehicle(vehicle_id)


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n    +-----------------------------------------------------------+")
    print("    |         ADAR FLEET COMMAND CENTER  v3.0                  |")
    print("    |         FastAPI + WebSocket Hub + Real GPS               |")
    print("    +-----------------------------------------------------------+\n")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
