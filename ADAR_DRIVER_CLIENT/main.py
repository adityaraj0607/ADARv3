"""
============================================================
 ADAR V3.0 — Main Entry Point
 Run: python main.py
============================================================
"""

import sys
import signal
import types
import socket
import time
from app import create_app, socketio
import config


def check_internet(timeout: float = 3.0) -> bool:
    """Quick internet check — tries DNS resolution + TCP connect."""
    targets = [("8.8.8.8", 53), ("1.1.1.1", 53), ("208.67.222.222", 53)]
    for host, port in targets:
        try:
            s = socket.create_connection((host, port), timeout=timeout)
            s.close()
            return True
        except OSError:
            continue
    return False

def signal_handler(sig: int, frame: types.FrameType | None) -> None: 
    """Handle Ctrl+C gracefully."""
    print("\n\n[ADAR] Shutting down gracefully...")
    try:
        from app.routes import stop_engine
        stop_engine()
    except (ImportError, Exception):
        pass
    sys.exit(0)


def main():
    print(r"""
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║     █████╗ ██████╗  █████╗ ██████╗   ██╗   ██╗██████╗║
    ║    ██╔══██╗██╔══██╗██╔══██╗██╔══██╗  ██║   ██║╚════██║
    ║    ███████║██║  ██║███████║██████╔╝  ██║   ██║ █████╔╝║
    ║    ██╔══██║██║  ██║██╔══██║██╔══██╗  ╚██╗ ██╔╝ ╚═══██║║
    ║    ██║  ██║██████╔╝██║  ██║██║  ██║   ╚████╔╝ ██████╔╝║
    ║    ╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝    ╚═══╝  ╚═════╝║
    ║                                                       ║
    ║   Advanced Driver Attention & Response System         ║
    ║   Version 3.0  —  Command Center                     ║
    ╚═══════════════════════════════════════════════════════╝
    """)

    print(f"  🌐 Dashboard:  http://localhost:{config.FLASK_PORT}")
    print(f"  📷 Camera:     Source {config.CAMERA_INDEX}")
    print(f"  🧠 AI Model:   YOLOv8 + MediaPipe Face Mesh")

    # ── Internet connectivity check ──
    print("  🔎 Network:    Checking connectivity...", end="", flush=True)
    internet_ok = check_internet()
    if internet_ok:
        print("\r  🌍 Network:    ✅ ONLINE — Internet connected         ")
    else:
        print("\r  🌍 Network:    ⚠️  OFFLINE — No internet connection    ")
        print("                 (GPT-5.2 alerts & spatial scan disabled)")

    jarvis_status = "✅ ONLINE" if (internet_ok and config.OPENAI_API_KEY and config.OPENAI_API_KEY != "sk-your-openai-api-key-here") else ("⚠️  OFFLINE (no internet)" if not internet_ok else "⚠️  OFFLINE (no API key)")
    print(f"  🤖 Jarvis:     {jarvis_status}")
    print(f"  💾 Database:   {config.DATABASE_URI}")
    print()

    app = create_app()

    # Register signal handler AFTER app is created to avoid interference during import/init
    signal.signal(signal.SIGINT, signal_handler)

    socketio.run(
        app,
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
        use_reloader=False,
        log_output=False,
    )


if __name__ == "__main__":
    main()


































