"""
============================================================
 ADAR V3.0 — Fleet Bridge (The Connector)
 MODULE C: WebSocket client that sends live telemetry
 from the Driver Client to the Fleet Server.

 ARCHITECTURE: Queue-based single-thread WebSocket ownership
   - send_telemetry() puts messages into a thread-safe Queue
   - One background thread EXCLUSIVELY owns the WebSocket
   - No two threads ever touch the SSL socket => no BAD_LENGTH

 Usage:
     from bridge import FleetConnector

     fleet = FleetConnector(vehicle_id="ADAR-01")
     fleet.connect()   # ws://localhost:8000 by default

     # Inside your camera loop:
     fleet.send_telemetry(ear=0.25, mar=0.12, status="SAFE")

     # When done:
     fleet.disconnect()
============================================================
"""

import json
import queue
import ssl
import threading
import time
from datetime import datetime, timezone
from typing import Optional

try:
    import websocket  # pip install websocket-client
except ImportError:
    websocket = None
    print("[BRIDGE] ⚠ websocket-client not installed — fleet bridge disabled")


class FleetConnector:
    """
    Non-blocking WebSocket bridge that sends telemetry from the
    ADAR Driver Client to the Fleet Command Server.

    THREAD SAFETY:
      - The camera/processing thread calls send_telemetry() which
        only enqueues a JSON string into self._outbox (thread-safe Queue).
      - A single daemon thread (_run_forever) owns the WebSocket exclusively:
        it connects, drains the queue, sends heartbeats, and reconnects.
      - This eliminates the [SSL: BAD_LENGTH] error that occurred when
        two threads both wrote to the same SSL socket.
    """

    DEFAULT_URL = "ws://localhost:8000/ws/vehicle/{vehicle_id}"

    # Throttle: minimum interval between accepting frames into the queue
    CLOUD_SEND_INTERVAL = 0.5   # ~2 fps to cloud  (stable on free tier)
    LOCAL_SEND_INTERVAL = 0.08  # ~12 fps to localhost

    # Queue capacity — old frames are silently dropped when full
    QUEUE_SIZE = 50

    def __init__(
        self,
        vehicle_id: str = "ADAR-01",
        server_url: str | None = None,
        verbose: bool = True,
    ) -> None:
        self.vehicle_id = vehicle_id
        self.verbose = verbose
        self._url = (server_url or self.DEFAULT_URL).format(vehicle_id=vehicle_id)
        self._is_cloud = self._url.startswith("wss://")

        self._ws: Optional[object] = None
        self._thread: Optional[threading.Thread] = None
        self._connected = threading.Event()
        self._stop_event = threading.Event()

        # ── Queue-based send (thread-safe) ──
        self._outbox: queue.Queue = queue.Queue(maxsize=self.QUEUE_SIZE)
        self._frame_count = 0
        self._last_accept_time = 0.0
        self._send_interval = (
            self.CLOUD_SEND_INTERVAL if self._is_cloud else self.LOCAL_SEND_INTERVAL
        )

        # Reconnection settings
        self._reconnect_delay = 3.0
        self._max_reconnect_delay = 30.0

    # ── Public API ────────────────────────────────────────────────────

    def connect(self, url: str | None = None, timeout: float = 8.0) -> bool:
        """
        Connect to the Fleet Server in the background.

        Args:
            url: Override the WebSocket URL (optional).
            timeout: Max seconds to wait for initial connection.

        Returns:
            True if connected, False if timed out (will keep retrying).
        """
        if websocket is None:
            self._log("websocket-client not installed — cannot connect")
            return False

        if url:
            self._url = url
            self._is_cloud = self._url.startswith("wss://")
            self._send_interval = (
                self.CLOUD_SEND_INTERVAL if self._is_cloud else self.LOCAL_SEND_INTERVAL
            )

        self._stop_event.clear()
        self._connected.clear()

        self._thread = threading.Thread(target=self._run_forever, daemon=True)
        self._thread.start()

        connected = self._connected.wait(timeout=timeout)
        if connected:
            self._log(f"✅ Connected to Fleet Server at {self._url}")
        else:
            self._log(f"⏳ Server not available — will keep retrying in background")
        return connected

    def send_telemetry(
        self,
        ear: float = 0.0,
        mar: float = 0.0,
        co2: float = 0.0,
        status: str = "SAFE",
        lat: float = 0.0,
        lon: float = 0.0,
        speed: float = 0.0,
        heading: float = 0.0,
        accuracy: float | None = None,
        altitude: float | None = None,
        attention_score: float = 100.0,
        alerts: list | None = None,
        extra: dict | None = None,
    ) -> bool:
        """
        Enqueue a telemetry frame for the Fleet Server. Non-blocking.
        Returns True if enqueued, False if not connected or throttled.
        """
        if not self._connected.is_set():
            return False

        # Throttle — only accept frames at the configured rate
        now = time.time()
        if (now - self._last_accept_time) < self._send_interval:
            return False  # Skip this frame (not an error)

        self._frame_count += 1
        self._last_accept_time = now

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
            "attention_score": round(attention_score, 1),
            "alerts": alerts or [],
            "location": location,
        }

        if extra:
            payload.update(extra)

        # Put into queue — never touches the WebSocket directly
        try:
            self._outbox.put_nowait(json.dumps(payload))
            return True
        except queue.Full:
            # Queue is full — drop oldest frame and enqueue new one
            try:
                self._outbox.get_nowait()
            except queue.Empty:
                pass
            try:
                self._outbox.put_nowait(json.dumps(payload))
                return True
            except queue.Full:
                return False

    def disconnect(self) -> None:
        """Gracefully close the WebSocket connection."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._connected.clear()
        self._log("Disconnected from Fleet Server")

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    @property
    def frames_sent(self) -> int:
        return self._frame_count

    # ── Internal — SINGLE THREAD owns the WebSocket ───────────────────

    def _run_forever(self) -> None:
        """
        Background thread: connect → drain queue & send → heartbeat → reconnect.
        This is the ONLY thread that touches self._ws.
        """
        delay = self._reconnect_delay

        # SSL context for wss:// connections
        sslopt = None
        if self._is_cloud:
            ctx = ssl.create_default_context()
            ctx.check_hostname = True
            ctx.verify_mode = ssl.CERT_REQUIRED
            sslopt = {"ssl_context": ctx}

        while not self._stop_event.is_set():
            try:
                # ── Connect ──
                ws = websocket.WebSocket()
                connect_kwargs = {"timeout": 15}
                if sslopt:
                    connect_kwargs["sslopt"] = sslopt
                ws.connect(self._url, **connect_kwargs)
                self._ws = ws
                self._connected.set()
                delay = self._reconnect_delay  # Reset backoff
                self._log(
                    f"WebSocket link established "
                    f"({'cloud/SSL' if self._is_cloud else 'local'})"
                )

                # ── Main loop: drain queue + heartbeat + recv ──
                heartbeat_interval = 10.0 if self._is_cloud else 5.0
                last_heartbeat = time.time()
                drain_interval = 0.1  # Check queue every 100ms

                # Set short recv timeout so we can process incoming
                # ping/pong/close frames without blocking the send loop
                ws.settimeout(0.05)

                while not self._stop_event.is_set():
                    now = time.time()

                    # 1) Drain up to 5 messages per cycle from the queue
                    for _ in range(5):
                        try:
                            msg = self._outbox.get_nowait()
                            ws.send(msg)
                        except queue.Empty:
                            break
                        except Exception as e:
                            self._log(f"Send error: {e}")
                            raise  # Break out to reconnect

                    # 2) Heartbeat — use protocol-level ping + JSON fallback
                    if (now - last_heartbeat) >= heartbeat_interval:
                        try:
                            ws.ping()  # Protocol-level PING (proxy recognises this)
                            ws.send(json.dumps({
                                "type": "heartbeat",
                                "vehicle_id": self.vehicle_id,
                            }))
                            last_heartbeat = now
                        except Exception as e:
                            self._log(f"Heartbeat error: {e}")
                            raise  # Break out to reconnect

                    # 3) Read incoming frames — lets the library process
                    #    PING (auto-PONG), close frames, etc.
                    try:
                        ws.recv()
                    except websocket.WebSocketTimeoutException:
                        pass  # No data available — normal, not an error
                    except websocket.WebSocketConnectionClosedException:
                        self._log("Server closed the connection")
                        raise  # Trigger reconnect
                    except Exception:
                        pass  # Other recv errors, ignore

                    # Sleep briefly before next drain cycle
                    self._stop_event.wait(timeout=drain_interval)

            except Exception as e:
                self._connected.clear()
                self._log(f"Connection lost: {e} — retrying in {delay:.0f}s")
                try:
                    if self._ws:
                        self._ws.close()
                except Exception:
                    pass
                self._ws = None
                self._stop_event.wait(timeout=delay)
                delay = min(delay * 1.5, self._max_reconnect_delay)

        # Cleanup on stop
        self._connected.clear()
        try:
            if self._ws:
                self._ws.close()
        except Exception:
            pass
        self._ws = None

    def _log(self, msg: str) -> None:
        if self.verbose:
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[FleetBridge {ts}] {msg}")


# ── Quick standalone test ─────────────────────────────────────────────
if __name__ == "__main__":
    import math
    import argparse

    parser = argparse.ArgumentParser(description="ADAR Fleet Bridge — Test Mode")
    parser.add_argument("--url", default=None, help="Fleet Server WebSocket URL")
    parser.add_argument("--duration", type=int, default=30, help="Test duration (seconds)")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print("  ADAR Fleet Bridge — Standalone Test")
    print(f"{'='*55}\n")

    bridge = FleetConnector(vehicle_id="ADAR-TEST")
    if args.url:
        bridge.connect(url=args.url)
    else:
        bridge.connect()

    start = time.time()
    lat, lon = 28.6139, 77.2090

    try:
        while time.time() - start < args.duration:
            t = time.time() - start
            lat += math.sin(t * 0.1) * 0.00005
            lon += math.cos(t * 0.1) * 0.00005
            ear = max(0.12, 0.28 - 0.003 * t) if 12 < t < 20 else 0.28
            status = "DANGER" if ear < 0.20 else "SAFE"

            bridge.send_telemetry(
                ear=ear, mar=0.12, co2=420, status=status,
                lat=lat, lon=lon, speed=45 + t * 0.5,
                alerts=["DROWSINESS"] if ear < 0.20 else [],
            )
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.disconnect()
        print(f"\nSent {bridge.frames_sent} frames in {time.time()-start:.1f}s")
