"""
============================================================
 ADAR V3.0 — J.A.R.V.I.S. Module
 The Assistant: Sends danger frames to GPT-4o Vision,
 receives a spoken warning, and plays audio via pygame.
 Runs entirely in a background thread with cooldown.
============================================================
"""

import io
import os
import cv2
import json
import re
import time
import base64
import tempfile
import threading
import pygame
import openai
import pyttsx3
import config
from app.database import log_incident


class Jarvis:
    """
    Background AI assistant.
    When safety_status == DANGER for enough consecutive frames,
    captures a frame, sends it to GPT-4o for analysis,
    generates a TTS audio warning, and plays it.
    """

    def __init__(self, socketio=None):
        # SocketIO ref for emitting alerts to dashboard
        self.socketio = socketio

        # OpenAI client
        self.client = None
        if config.OPENAI_API_KEY and config.OPENAI_API_KEY != "sk-your-openai-api-key-here":
            self.client = openai.OpenAI(
                api_key=config.OPENAI_API_KEY,
                max_retries=0,  # No SDK retries — we handle fallback ourselves
            )
            print("[JARVIS] OpenAI client initialized ✓")
        else:
            print("[JARVIS] ⚠ No valid API key — running in OFFLINE mode")

        # Audio engine — pygame for OpenAI TTS mp3
        pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=2048)
        print("[JARVIS] Audio engine (pygame) initialized ✓")

        # Local TTS engine — Windows SAPI5 via pyttsx3 (offline fallback)
        self._local_tts_lock = threading.Lock()
        try:
            self._local_engine = pyttsx3.init()
            self._local_engine.setProperty('rate', 180)
            self._local_engine.setProperty('volume', 1.0)
            voices = self._local_engine.getProperty('voices')
            # Prefer a male voice if available
            for v in voices:
                if 'male' in v.name.lower() or 'david' in v.name.lower():
                    self._local_engine.setProperty('voice', v.id)
                    break
            print("[JARVIS] Local TTS engine (SAPI5) initialized ✓")
        except Exception as e:
            self._local_engine = None
            print(f"[JARVIS] ⚠ Local TTS init failed: {e}")

        # State
        self.is_speaking = False
        self.last_alert_time = 0
        self.last_response = ""
        self.alert_count = 0
        self._lock = threading.Lock()

        # Rate limit backoff
        self._backoff_until = 0
        self._consecutive_429s = 0

    @property
    def is_ready(self):
        """Check if Jarvis can fire a new alert (cooldown + backoff respected)."""
        now = time.time()
        elapsed = now - self.last_alert_time
        return (
            not self.is_speaking
            and elapsed > config.JARVIS_COOLDOWN
            and now >= self._backoff_until
        )

    def trigger_alert(self, frame, telemetry, force=False, alert_reason=None):
        """
        Fire-and-forget: Spawn a background thread to handle
        the full GPT-5.2 Vision → TTS → Playback pipeline.
        force=True bypasses cooldown for critical drowsiness (8s+).
        alert_reason: specific trigger (YAWNING, LOOKING_AWAY, PHONE_USE, etc.)
        """
        if force:
            # Critical drowsy override — only check if not currently speaking
            now = time.time()
            if self.is_speaking or now < self._backoff_until:
                return
        elif not self.is_ready:
            return

        with self._lock:
            if self.is_speaking:
                return
            self.is_speaking = True

        thread = threading.Thread(
            target=self._process_alert,
            args=(frame.copy(), telemetry.copy(), alert_reason),
            daemon=True,
        )
        thread.start()

    def trigger_drowsy_alert(self, telemetry, force=False):
        """
        Instant local drowsy alert — bypasses GPT-5.2 entirely.
        Called when the drowsy timer hits 4s+ so the alert is
        guaranteed to fire regardless of API availability.
        """
        if force:
            now = time.time()
            if self.is_speaking or now < self._backoff_until:
                return
        elif not self.is_ready:
            return

        with self._lock:
            if self.is_speaking:
                return
            self.is_speaking = True

        thread = threading.Thread(
            target=self._process_drowsy_alert,
            args=(telemetry.copy(),),
            daemon=True,
        )
        thread.start()

    def _process_drowsy_alert(self, telemetry):
        """Direct local alert for sustained drowsiness — no GPT-5.2 call for speed."""
        try:
            self.last_alert_time = time.time()
            print("[JARVIS] 💤 Drowsy timer triggered — OFFLINE instant alert (bypassing GPT-5.2 for speed)")
            self._local_fallback_alert(telemetry, source="OFFLINE")
        finally:
            with self._lock:
                self.is_speaking = False

    def _process_alert(self, frame, telemetry, alert_reason=None):
        """Full alert pipeline: GPT-5.2-instant with timeout fallback. All alerts get TTS."""
        try:
            self.last_alert_time = time.time()
            reason_tag = alert_reason or "DANGER"
            print(f"[JARVIS] 🚨 {reason_tag} detected — sending to GPT-5.2-instant...")

            # 1. Encode frame to base64
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

            # 2. Build context prompt with specific alert reason
            context = self._build_context(telemetry, alert_reason=alert_reason)

            # 3. Call GPT-5.2-instant (1000ms timeout enforced)
            t0 = time.time()
            response_text = self._call_gpt(frame_b64, context)
            latency_ms = (time.time() - t0) * 1000
            self._consecutive_429s = 0
            print(f"[JARVIS] GPT-5.2 responded in {latency_ms:.0f}ms: {response_text}")

            # 4. Parse JSON response
            result = self._parse_gpt_response(response_text)
            status = result.get("status", "DANGER")
            reason = result.get("reason", "Attention required")
            confidence = result.get("confidence", 0.5)

            self.last_response = f"[{status}] {reason} ({confidence:.0%})"

            # 5. If GPT-5.2 confirms SAFE with high confidence, suppress alert
            if status == "SAFE" and confidence >= 0.8:
                print(f"[JARVIS] ✅ GPT-5.2 override: SAFE ({confidence:.0%}) — alert suppressed")
                self.alert_count += 1
                return

            # 6. Generate spoken warning from GPT reason
            spoken = f"Warning: {reason}."
            audio_data = self._generate_tts(spoken)

            # 7. Play audio — OpenAI TTS first, local SAPI5 fallback
            if audio_data:
                self._play_audio(audio_data)
            else:
                print("[JARVIS] OpenAI TTS unavailable — using local SAPI5 voice")
                self._speak_local(spoken)

            # 8. Log to database
            alert_type = self._determine_alert_type(telemetry)
            log_incident(
                alert_type=alert_type,
                severity=config.STATUS_DANGER,
                ear_value=telemetry.get("ear"),
                mar_value=telemetry.get("mar"),
                yaw_angle=telemetry.get("yaw"),
                pitch_angle=telemetry.get("pitch"),
                detected_objects=", ".join(telemetry.get("detected_objects", [])),
                jarvis_response=f"GPT-5.2: {response_text}",
                attention_score=telemetry.get("attention_score"),
                blink_rate=telemetry.get("blink_rate"),
            )

            # 9. Emit to dashboard with GPT-5.2 metadata
            if self.socketio:
                try:
                    self.socketio.emit("jarvis_alert", {
                        "message": spoken,
                        "alert_type": alert_type,
                        "gpt_status": status,
                        "gpt_confidence": confidence,
                        "latency_ms": round(latency_ms),
                        "source": "GPT-5.2",
                    })
                except Exception:
                    pass

            self.alert_count += 1

        except openai.APITimeoutError:
            # GPT-5.2 exceeded 1000ms — fallback to local rule-based alert
            print("[JARVIS] ⏱ GPT-5.2 timeout (>1000ms) — local fallback activated")
            self._local_fallback_alert(telemetry)

        except openai.RateLimitError:
            self._consecutive_429s += 1
            backoff = min(
                config.JARVIS_BACKOFF_BASE * (2 ** self._consecutive_429s),
                config.JARVIS_BACKOFF_MAX,
            )
            self._backoff_until = time.time() + backoff
            print(f"[JARVIS] ⚡ Rate limited — backing off {backoff}s")
            self._local_fallback_alert(telemetry)

        except Exception as e:
            print(f"[JARVIS] ❌ Error: {e}")
            self._local_fallback_alert(telemetry)

        finally:
            with self._lock:
                self.is_speaking = False

    def _build_context(self, telemetry, alert_reason=None):
        """Build the GPT-5.2-instant prompt with specific alert context."""
        ear = telemetry.get('ear', 0)
        mar = telemetry.get('mar', 0)
        yaw = telemetry.get('yaw', 0)
        pitch = telemetry.get('pitch', 0)
        attention = telemetry.get('attention_score', 0)
        blink_rate = telemetry.get('blink_rate', 0)
        objects = telemetry.get('detected_objects', [])
        
        # Specific alert context for better GPT responses
        alert_hints = {
            "YAWNING": "The driver is YAWNING (mouth wide open, MAR is very high). Warn about fatigue.",
            "LOOKING_AWAY": "The driver is NOT looking at the road (head turned significantly). Warn them to focus.",
            "LOOKING_DOWN": "The driver is LOOKING DOWN (possibly at phone/lap). Tell them to look up at the road.",
            "PHONE_USE": "The driver is USING A PHONE near their ear while driving. This is very dangerous.",
            "DISTRACTION": f"The driver is DISTRACTED. Objects detected: {', '.join(objects) if objects else 'unknown'}. Warn them.",
            "DRINKING": "The driver is DRINKING a beverage while driving. Tell them to focus on the road.",
            "OBJECT_DETECTED": f"Dangerous objects detected near driver: {', '.join(objects)}. Assess the risk.",
            "DANGER": "Multiple danger signals active. Assess the driver's state from the image.",
            "WARNING_SUSTAINED": "Sustained warning state — driver attention is degrading. Give a firm warning.",
        }
        hint = alert_hints.get(alert_reason, "Assess the driver's alertness from the image.")
        
        return (
            "You are JARVIS, an advanced real-time driver safety AI assistant. "
            "Analyze the driver's face in this camera frame and give a SHORT spoken warning.\n\n"
            f"ALERT TRIGGER: {alert_reason or 'GENERAL'}\n"
            f"Context: {hint}\n\n"
            f"Sensor readings: EAR={ear:.3f}, MAR={mar:.3f}, "
            f"Yaw={yaw:.1f}°, Pitch={pitch:.1f}°, "
            f"Attention={attention:.0f}/100, BlinkRate={blink_rate:.0f}/min\n"
            f"Objects in frame: {', '.join(objects) if objects else 'none'}\n\n"
            "Respond in EXACTLY this JSON format:\n"
            '{\n'
            '  "status": "DANGER" or "WARNING" or "SAFE",\n'
            '  "reason": "A short, varied, natural spoken warning (max 20 words)",\n'
            '  "confidence": 0.0 to 1.0\n'
            '}\n\n'
            "Rules: Each alert MUST be different wording. Be firm but caring. "
            "Never repeat the same phrase twice."
        )

    def _call_gpt(self, frame_b64: str, context: str) -> str:
        """Send frame + context to GPT-5.2-instant with 1s timeout."""
        assert self.client is not None, "OpenAI client not initialized"
        response = self.client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are JARVIS, a real-time safety AI. Be concise.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": context},
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
            max_completion_tokens=100,
            temperature=0.3,
            timeout=config.GPT_TIMEOUT,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""

    def _parse_gpt_response(self, text: str) -> dict:
        """Parse GPT-5.2 JSON response with robust fallback."""
        # Try direct JSON parse
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass
        # Try extracting JSON from markdown or mixed text
        match = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except (json.JSONDecodeError, TypeError):
                pass
        # Fallback: treat entire response as a danger reason
        return {"status": "DANGER", "reason": text[:120], "confidence": 0.5}

    def _local_fallback_alert(self, telemetry, source="LOCAL"):
        """Local rule-based alert — used for drowsy alerts (OFFLINE) or GPT failures."""
        alert_type = self._determine_alert_type(telemetry)
        fallback_msg = config.JARVIS_OFFLINE_MESSAGES.get(
            alert_type, config.JARVIS_OFFLINE_MESSAGES["GENERAL"]
        )
        self.last_response = f"[{source}] {fallback_msg}"
        print(f"[JARVIS] {source} alert: {fallback_msg}")

        # Generate TTS for the fallback message — try OpenAI, fall back to local SAPI5
        try:
            audio_data = self._generate_tts(fallback_msg)
            if audio_data:
                self._play_audio(audio_data)
            else:
                print("[JARVIS] OpenAI TTS unavailable — using local SAPI5 voice")
                self._speak_local(fallback_msg)
        except Exception as e:
            print(f"[JARVIS] All TTS failed in fallback: {e}")
            self._speak_local(fallback_msg)

        # Log incident
        log_incident(
            alert_type=alert_type,
            severity=config.STATUS_DANGER,
            ear_value=telemetry.get("ear"),
            mar_value=telemetry.get("mar"),
            yaw_angle=telemetry.get("yaw"),
            pitch_angle=telemetry.get("pitch"),
            detected_objects=", ".join(telemetry.get("detected_objects", [])),
            jarvis_response=f"[{source}] {fallback_msg}",
            attention_score=telemetry.get("attention_score"),
            blink_rate=telemetry.get("blink_rate"),
        )

        # Emit to dashboard with source tag
        if self.socketio:
            try:
                self.socketio.emit("jarvis_alert", {
                    "message": fallback_msg,
                    "alert_type": alert_type,
                    "source": source,
                    "fallback": True,
                })
            except Exception:
                pass

        self.alert_count += 1

    def _generate_tts(self, text: str) -> bytes | None:
        """Generate speech audio from text using OpenAI TTS."""
        try:
            assert self.client is not None
            response = self.client.audio.speech.create(
                model=config.OPENAI_TTS_MODEL,
                voice=config.OPENAI_TTS_VOICE,
                input=text,
                response_format="mp3",
            )
            print(f"[JARVIS] 🔊 OpenAI TTS generated ({len(response.content)} bytes)")
            return response.content
        except Exception as e:
            print(f"[JARVIS] OpenAI TTS failed: {e}")
            return None

    def _play_audio(self, audio_data):
        """Play mp3 audio bytes through pygame mixer using a temp file for reliability."""
        tmp_path = None
        try:
            # Write to temp file — BytesIO mp3 loading is unreliable on Windows
            fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
            os.write(fd, audio_data)
            os.close(fd)
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            print("[JARVIS] 🔊 Playing audio via pygame...")
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            print("[JARVIS] 🔊 Audio playback finished")
        except Exception as e:
            print(f"[JARVIS] Audio playback error: {e}")
        finally:
            # Clean up temp file
            try:
                pygame.mixer.music.unload()
            except Exception:
                pass
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _speak_local(self, text: str):
        """Offline TTS fallback using Windows SAPI5 (pyttsx3). No internet needed."""
        with self._local_tts_lock:
            try:
                if self._local_engine is None:
                    # Re-try init in case it failed at startup
                    self._local_engine = pyttsx3.init()
                    self._local_engine.setProperty('rate', 180)
                    self._local_engine.setProperty('volume', 1.0)
                self._local_engine.say(text)
                self._local_engine.runAndWait()
                print(f"[JARVIS] 🔊 Local SAPI5 spoke: {text[:60]}...")
            except Exception as e:
                print(f"[JARVIS] Local TTS error: {e}")
                # Engine may be in bad state, reset it
                try:
                    self._local_engine.stop()
                except Exception:
                    pass
                self._local_engine = None

    @staticmethod
    def _determine_alert_type(telemetry):
        """Pick the primary alert type for logging."""
        if telemetry.get("is_drowsy"):
            return "DROWSINESS"
        if telemetry.get("is_phone_near_ear"):
            return "PHONE_USE"
        if telemetry.get("is_distracted"):
            return "DISTRACTION"
        if telemetry.get("is_yawning"):
            return "YAWNING"
        if telemetry.get("is_looking_away"):
            return "LOOKING_AWAY"
        if telemetry.get("is_looking_down"):
            return "LOOKING_DOWN"
        if telemetry.get("is_drinking"):
            return "DRINKING"
        return "GENERAL"

    def trigger_offline_alert(self, telemetry):
        """
        Fallback when no API key: log the incident and print to console.
        """
        alert_type = self._determine_alert_type(telemetry)
        print(f"[JARVIS OFFLINE] ⚠ {alert_type} alert logged (no API key)")
        log_incident(
            alert_type=alert_type,
            severity=config.STATUS_DANGER,
            ear_value=telemetry.get("ear"),
            mar_value=telemetry.get("mar"),
            yaw_angle=telemetry.get("yaw"),
            pitch_angle=telemetry.get("pitch"),
            detected_objects=", ".join(telemetry.get("detected_objects", [])),
        )

    def shutdown(self):
        """Clean up audio resources."""
        pygame.mixer.quit()
        print("[JARVIS] Shut down.")
