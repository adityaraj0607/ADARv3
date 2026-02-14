"""
============================================================
 ADAR V3.0 — Flask Application Factory
============================================================
"""

from flask import Flask
from flask_socketio import SocketIO

socketio = SocketIO()


def create_app():
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )

    # Load config
    import config
    app.config["SECRET_KEY"] = config.FLASK_SECRET_KEY
    app.config["SQLALCHEMY_DATABASE_URI"] = config.DATABASE_URI

    # Initialize extensions
    socketio.init_app(app, async_mode="threading", cors_allowed_origins="*")

    # Initialize database
    from app.database import init_db
    init_db()

    # Register routes
    from app.routes import register_routes
    register_routes(app, socketio)

    return app
