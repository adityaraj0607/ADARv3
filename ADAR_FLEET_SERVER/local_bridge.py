"""
============================================================
 ADAR V3.0 — Local Bridge (The Connector)
 Import this into your main OpenCV script to send live
 telemetry to the Fleet Command Center cloud server.

 Usage:
     from local_bridge import ADARBridge

     bridge = ADARBridge()
     bridge.connect("ws://your-server.onrender.com/ws/vehicle/ADAR-001")

     # Inside your camera loop:
     bridge.send_data(ear=0.25, mar=0.12, co2=450, status="SAFE",
                      lat=28.6139, lon=77.2090, speed=55.0)

     # When done:
     bridge.disconnect()
============================================================
"""

import json
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import websocket  # pip install websocket-client


class ADARBridge:
    """
    Non-blocking WebSocket bridge that sends telemetry from the
    local OpenCV/ADAR pipeline to the Fleet Command Center server.
    Uses a background thread so it NEVER blocks the camera loop.
    """

    def __init__(self, vehicle_id: str = "ADAR-001", verbose: bool = True) -> None:
        self.vehicle_id = vehicle_id
        self.verbose = verbose

        self._ws: Optional[websocket.WebSocketApp] = None
        self._ws_raw: Optional[websocket.WebSocket] = None
        self._thread: Optional[threading.Thread] = None
        self._connected = threading.Event()
        self._stop_event = threading.Event()
        self._url: str = ""

        # Queue for outgoing messages (thread-safe)
        self._send_lock = threading.Lock()
        self._frame_count = 0
        self._last_send_time = 0.0

    # ── Public API ────────────────────────────────────────────────────

    def connect(self, url: str, timeout: float = 10.0) -> bool:
        """
        Connect to the Fleet Command Center WebSocket server.

        Args:
            url: Full WebSocket URL, e.g.
                 "ws://localhost:8000/ws/vehicle/ADAR-001"
                 "wss://adar-fleet.onrender.com/ws/vehicle/ADAR-001"
            timeout: Max seconds to wait for connection.

        Returns:
            True if connected successfully, False otherwise.
        """
        self._url = url
        self._stop_event.clear()
        self._connected.clear()

        self._thread = threading.Thread(target=self._run_forever, daemon=True)
        self._thread.start()

        # Wait for connection with timeout
        connected = self._connected.wait(timeout=timeout)
        if connected:
            self._log(f"Connected to {url}")
        else:
            self._log(f"Connection timeout after {timeout}s — will keep retrying in background")
        return connected

    def send_data(
        self,
        ear: float = 0.0,
        mar: float = 0.0,
        co2: float = 0.0,
        status: str = "SAFE",
        lat: float = 0.0,
        lon: float = 0.0,
        speed: float = 0.0,
        heading: float = 0.0,
        accuracy: Optional[float] = None,
        altitude: Optional[float] = None,
        alerts: Optional[list] = None,
        extra: Optional[dict] = None,
    ) -> bool:
        """
        Send a telemetry frame to the server. Non-blocking.

        Args:
            ear: Eye Aspect Ratio (0.0 - 0.5)
            mar: Mouth Aspect Ratio (0.0 - 1.0)
            co2: CO2 concentration in PPM
            status: "SAFE" or "DANGER"
            lat: GPS Latitude
            lon: GPS Longitude
            speed: Speed in km/h
            heading: Compass heading in degrees (0–360)
            accuracy: GPS accuracy in meters
            altitude: Altitude in meters
            alerts: List of active alert strings
            extra: Additional key-value pairs to include

        Returns:
            True if sent, False if not connected.
        """
        if not self._connected.is_set():
            return False

        self._frame_count += 1
        location = {
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "heading": round(heading % 360, 1),
        }
        if accuracy is not None:
            location["accuracy"] = round(accuracy, 1)
        if altitude is not None:
            location["altitude"] = round(altitude, 1)

        payload = {
            "vehicle_id": self.vehicle_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "frame": self._frame_count,
            "ear": round(ear, 4),
            "mar": round(mar, 4),
            "co2_ppm": round(co2, 1),
            "speed_kmh": round(speed, 1),
            "status": status,
            "alerts": alerts or [],
            "location": location,
        }

        if extra:
            payload.update(extra)

        try:
            with self._send_lock:
                if self._ws_raw and self._ws_raw.connected:
                    self._ws_raw.send(json.dumps(payload))
                    self._last_send_time = time.time()
                    return True
        except Exception as e:
            self._log(f"Send error: {e}")
        return False

    def disconnect(self) -> None:
        """Gracefully close the WebSocket connection."""
        self._stop_event.set()
        if self._ws_raw:
            try:
                self._ws_raw.close()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._connected.clear()
        self._log("Disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    @property
    def frames_sent(self) -> int:
        return self._frame_count

    # ── Internal ──────────────────────────────────────────────────────

    def _run_forever(self) -> None:
        """Background thread: connect and maintain the WebSocket."""
        while not self._stop_event.is_set():
            try:
                self._ws_raw = websocket.WebSocket()
                self._ws_raw.connect(self._url, timeout=10)
                self._connected.set()
                self._log("WebSocket link established")

                # Keep-alive loop — just hold the connection
                while not self._stop_event.is_set():
                    # Check if still connected
                    try:
                        self._ws_raw.ping()
                    except Exception:
                        break
                    self._stop_event.wait(timeout=5.0)

            except Exception as e:
                self._connected.clear()
                self._log(f"Connection failed: {e} — retrying in 3s")
                self._stop_event.wait(timeout=3.0)

        self._connected.clear()

    def _log(self, msg: str) -> None:
        if self.verbose:
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[ADARBridge {ts}] {msg}")


# ── Convenience: Quick test ───────────────────────────────────────────
if __name__ == "__main__":
    """
    Quick standalone test — connects to the server and sends
    simulated data for 30 seconds.
    """
    import math
    import argparse

    parser = argparse.ArgumentParser(description="ADAR Bridge Test Sender")
    parser.add_argument("--url", default="ws://localhost:8000/ws/vehicle/TEST-001",
                        help="WebSocket URL of the Fleet Command Center")
    parser.add_argument("--duration", type=int, default=30,
                        help="How many seconds to send test data")
    args = parser.parse_args()

    bridge = ADARBridge(vehicle_id="TEST-001")
    print(f"\n{'='*50}")
    print("  ADAR Bridge — Test Mode")
    print(f"  Target: {args.url}")
    print(f"  Duration: {args.duration}s")
    print(f"{'='*50}\n")

    if not bridge.connect(args.url):
        print("Could not connect. Is the server running?")
        exit(1)

    start = time.time()
    lat, lon = 28.6139, 77.2090        # New Delhi
    ear, co2 = 0.28, 420.0

    try:
        while time.time() - start < args.duration:
            t = time.time() - start

            # Simulate movement
            lat += math.sin(t * 0.1) * 0.00005
            lon += math.cos(t * 0.1) * 0.00005

            # Simulate drowsiness event at t=15s
            if 12 < t < 20:
                ear = max(0.12, ear - 0.003)
            else:
                ear = min(0.30, ear + 0.005)

            # CO2 creep
            co2 += 2.5 if t > 10 else -1
            co2 = max(350, min(1500, co2))

            status = "DANGER" if (ear < 0.20 or co2 > 1000) else "SAFE"
            alerts = []
            if ear < 0.20:
                alerts.append("DROWSINESS")
            if co2 > 1000:
                alerts.append("TOXIC AIR")

            bridge.send_data(
                ear=ear, mar=0.12, co2=co2, status=status,
                lat=lat, lon=lon, speed=45 + t * 0.5, alerts=alerts
            )
            time.sleep(0.5)

    except KeyboardInterrupt:
        pass
    finally:
        bridge.disconnect()
        print(f"\nSent {bridge.frames_sent} frames in {time.time()-start:.1f}s")
