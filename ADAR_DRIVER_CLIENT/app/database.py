"""
============================================================
 ADAR V3.0 — Database Models & Session Management
 Logs every incident to adar_logs.db via SQLAlchemy.
============================================================
"""

from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
import config

Base = declarative_base()
engine = create_engine(config.DATABASE_URI, echo=False)
session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)


class Incident(Base):
    """Represents a single safety incident / alert."""
    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    alert_type = Column(String(50), nullable=False)        # DROWSINESS, YAWNING, DISTRACTION, HEAD_POSE
    severity = Column(String(20), nullable=False)           # WARNING, DANGER
    ear_value = Column(Float, nullable=True)
    mar_value = Column(Float, nullable=True)
    yaw_angle = Column(Float, nullable=True)
    pitch_angle = Column(Float, nullable=True)
    detected_objects = Column(String(200), nullable=True)   # e.g. "cell phone"
    jarvis_response = Column(Text, nullable=True)           # GPT-4o response text
    duration_seconds = Column(Float, nullable=True)
    attention_score = Column(Float, nullable=True)           # 0-100 composite score
    blink_rate = Column(Float, nullable=True)                # blinks per minute

    def __repr__(self):
        return f"<Incident #{self.id} [{self.alert_type}] {self.severity} @ {self.timestamp}>"

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,  # type: ignore[union-attr]
            "alert_type": self.alert_type,
            "severity": self.severity,
            "ear_value": self.ear_value,
            "mar_value": self.mar_value,
            "yaw_angle": self.yaw_angle,
            "pitch_angle": self.pitch_angle,
            "detected_objects": self.detected_objects,
            "jarvis_response": self.jarvis_response,
            "duration_seconds": self.duration_seconds,
            "attention_score": self.attention_score,
            "blink_rate": self.blink_rate,
        }


class SessionLog(Base):
    """Tracks driving sessions."""
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    start_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    end_time = Column(DateTime, nullable=True)
    total_incidents = Column(Integer, default=0)
    max_severity = Column(String(20), default="SAFE")

    def __repr__(self):
        return f"<Session #{self.id} from {self.start_time}>"


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(engine)
    _migrate_db()


def _migrate_db():
    """Add new columns to existing tables (safe for existing data)."""
    import sqlite3
    db_path = config.DATABASE_URI.replace("sqlite:///", "")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(incidents)")
        existing = {row[1] for row in cursor.fetchall()}
        migrations = [
            ("attention_score", "REAL"),
            ("blink_rate", "REAL"),
        ]
        for col_name, col_type in migrations:
            if col_name not in existing:
                cursor.execute(f"ALTER TABLE incidents ADD COLUMN {col_name} {col_type}")
                print(f"[DB] ✓ Migrated: added column '{col_name}'")
        conn.commit()
        conn.close()
    except Exception:
        pass  # Table may not exist yet — create_all handles it


def log_incident(alert_type: str, severity: str, **kwargs) -> Incident | None:
    """Log an incident to the database. Thread-safe."""
    session = Session()
    try:
        incident = Incident(
            alert_type=alert_type,
            severity=severity,
            ear_value=kwargs.get("ear_value"),
            mar_value=kwargs.get("mar_value"),
            yaw_angle=kwargs.get("yaw_angle"),
            pitch_angle=kwargs.get("pitch_angle"),
            detected_objects=kwargs.get("detected_objects"),
            jarvis_response=kwargs.get("jarvis_response"),
            duration_seconds=kwargs.get("duration_seconds"),
            attention_score=kwargs.get("attention_score"),
            blink_rate=kwargs.get("blink_rate"),
        )
        session.add(incident)
        session.commit()
        return incident
    except Exception as e:
        session.rollback()
        print(f"[DB ERROR] Failed to log incident: {e}")
        return None
    finally:
        Session.remove()


def get_recent_incidents(limit: int = 50):
    """Fetch the most recent incidents."""
    session = Session()
    try:
        incidents = (
            session.query(Incident)
            .order_by(Incident.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [i.to_dict() for i in incidents]
    finally:
        Session.remove()


def get_incident_stats(since=None):
    """Get aggregate statistics for the dashboard (scoped to current session)."""
    session = Session()
    try:
        from sqlalchemy import func
        tf = [Incident.timestamp >= since] if since else []
        total = session.query(func.count(Incident.id)).filter(
            *tf
        ).scalar() or 0
        drowsy = session.query(func.count(Incident.id)).filter(
            Incident.alert_type == "DROWSINESS", *tf
        ).scalar() or 0
        yawning = session.query(func.count(Incident.id)).filter(
            Incident.alert_type == "YAWNING", *tf
        ).scalar() or 0
        distraction = session.query(func.count(Incident.id)).filter(
            Incident.alert_type == "DISTRACTION", *tf
        ).scalar() or 0
        avg_attention = session.query(func.avg(Incident.attention_score)).filter(
            *tf
        ).scalar()
        return {
            "total": total,
            "drowsiness": drowsy,
            "yawning": yawning,
            "distraction": distraction,
            "avg_attention": round(avg_attention, 1) if avg_attention else None,
        }
    finally:
        Session.remove()
