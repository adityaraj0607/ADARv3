# ADAR V3.0 — Fleet Command Center

A **separate** FastAPI WebSocket hub that receives telemetry from one or more
ADAR driver-monitoring units and displays them on a real-time fleet dashboard.

> **This is independent from the ADAR Driver Monitor (Flask, port 5000).**
> They communicate via WebSocket when both are running.

---

## Quick Start

```bash
cd fleet_command
pip install -r requirements.txt
python server.py            # → http://localhost:8000
```

Or with uvicorn directly:

```bash
uvicorn server:app --reload --port 8000
```

## Architecture

```
┌──────────────────────┐      WebSocket       ┌──────────────────────────┐
│  ADAR Driver Monitor │  ──────────────────►  │  Fleet Command Center    │
│  (Flask :5000)       │   local_bridge.py     │  (FastAPI :8000)         │
│  OpenCV + AI + HUD   │                       │  Dashboard + GPS + Map   │
└──────────────────────┘                       └──────────────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `server.py` | FastAPI app, WebSocket hub, GPS/CO2 stores, simulation engine |
| `local_bridge.py` | WebSocket client bridge (used by ADAR to send telemetry here) |
| `templates/dashboard.html` | Fleet dashboard UI (Tailwind + Chart.js + Leaflet map) |
| `static/index.html` | Fleet command center UI (Iron Man theme) |
| `static/js/dashboard.js` | Dashboard JavaScript |
| `static/css/style.css` | Command center stylesheet |
| `Procfile` | Render.com deployment config |
| `requirements.txt` | Python dependencies (FastAPI stack only) |

## Connecting ADAR → Fleet

When both systems are running, the ADAR driver monitor can optionally send
telemetry to the fleet server using `local_bridge.py`:

```python
from fleet_command.local_bridge import ADARBridge

bridge = ADARBridge(vehicle_id="ADAR-001")
bridge.connect("ws://localhost:8000/ws/vehicle/ADAR-001")
bridge.send_data(ear=0.25, mar=0.12, status="SAFE", lat=28.61, lon=77.20)
bridge.disconnect()
```

## Deployment

The fleet server is designed for cloud deployment (e.g., Render.com):

```
web: uvicorn server:app --host 0.0.0.0 --port $PORT
```
