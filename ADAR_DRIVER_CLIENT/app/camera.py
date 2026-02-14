"""
============================================================
 ADAR V3.0 — Lock-Free Threaded Camera (Production-Grade)
 Uses atomic reference swaps instead of threading.Lock.
============================================================
"""

import cv2
import time
import threading
import config


class Camera:
    """High-performance camera capture with atomic frame access."""

    def __init__(self, src=None):
        self.src = src if src is not None else config.CAMERA_INDEX
        self.cap = None
        self._current_frame = None    # atomic reference
        self._current_jpeg = None     # pre-encoded JPEG bytes
        self.running = False
        self._thread = None
        self.fps = 0.0
        self._frame_count = 0
        self._fps_timer = time.time()
        self._first_frame = False

    def start(self):
        """Open the camera and start the capture thread."""
        self._connect()
        if self.cap is None or not self.cap.isOpened():
            print(f"[CAMERA] ✗ Failed to open source {self.src}")
            return self

        self.running = True
        self._thread = threading.Thread(target=self._update, daemon=True, name="Camera")
        self._thread.start()
        print(f"[CAMERA] Capture thread started (source={self.src})")
        return self

    def _connect(self):
        """Open the camera device."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        # Try DirectShow first (lowest latency on Windows)
        try:
            cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
        except Exception:
            cap = cv2.VideoCapture(self.src)

        if not cap.isOpened():
            print(f"[CAMERA] ⚠ cv2.VideoCapture failed for source {self.src}")
            return

        # Configure resolution and FPS BEFORE any other settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for lowest latency

        # Try MJPG codec for faster decoding
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # type: ignore[attr-defined]
        except Exception:
            pass

        # Performance optimizations
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual mode
        try:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus for better performance
        except Exception:
            pass

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[CAMERA] Opened source {self.src} ({actual_w}x{actual_h} @ {actual_fps:.0f}fps)")

        self.cap = cap

    def _update(self):
        """Continuously read frames in a background thread."""
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                print(f"[CAMERA] Lost connection. Reconnecting...")
                time.sleep(config.CAMERA_RECONNECT_DELAY)
                self._connect()
                continue

            # Aggressively clear buffer first to get latest frame
            self.cap.grab()
            
            ret, frame = self.cap.read()
            if not ret or frame is None:
                continue

            # Resize if camera didn't respect our resolution settings
            h, w = frame.shape[:2]
            if w != config.CAMERA_WIDTH or h != config.CAMERA_HEIGHT:
                frame = cv2.resize(frame, (config.CAMERA_WIDTH, config.CAMERA_HEIGHT), interpolation=cv2.INTER_LINEAR)

            # Flip horizontally if enabled (for natural mirror view)
            if config.CAMERA_FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)

            # Atomic reference swap - always use latest frame
            self._current_frame = frame

            # Pre-encode JPEG for MJPEG streaming at full camera speed
            # This decouples the video stream from AI processing latency
            ret_j, jpeg = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65]
            )
            if ret_j:
                self._current_jpeg = jpeg.tobytes()

            if not self._first_frame:
                self._first_frame = True
                h, w = frame.shape[:2]
                print(f"[CAMERA] ✓ First frame captured ({w}x{h})")

            # FPS counter
            self._frame_count += 1
            elapsed = time.time() - self._fps_timer
            if elapsed >= 1.0:
                self.fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_timer = time.time()

    def read(self):
        """Return the latest frame (lock-free). Returns (ok, frame_copy)."""
        frame = self._current_frame
        if frame is None:
            return False, None
        # Return direct reference for speed (caller will copy if needed)
        return True, frame

    def stop(self):
        """Stop the capture thread and release the camera."""
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=3)
        if self.cap is not None:
            self.cap.release()
        print("[CAMERA] Stopped and released.")

    @property
    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()
