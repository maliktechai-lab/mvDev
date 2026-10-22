import uuid
from datetime import datetime
from enum import Enum as PyEnum
from sqlalchemy import (
    Column, String, Boolean, DateTime, Float,
    ForeignKey, Text, Integer, Enum, JSON
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from db.database import Base


# ── Enums ──────────────────────────────────────────────────────────────────

class MediaType(str, PyEnum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class JobStatus(str, PyEnum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"


class Verdict(str, PyEnum):
    REAL    = "real"
    FAKE    = "fake"
    UNKNOWN = "unknown"


# ── User ───────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id         = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email      = Column(String(255), unique=True, nullable=False, index=True)
    username   = Column(String(50), unique=True, nullable=False, index=True)
    full_name  = Column(String(100), nullable=True)
    hashed_password = Column(String(255), nullable=False)
    is_active  = Column(Boolean, default=True, nullable=False)
    is_admin   = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    jobs = relationship("DetectionJob", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email}>"


# ── Detection Job ──────────────────────────────────────────────────────────

class DetectionJob(Base):
    __tablename__ = "detection_jobs"

    id           = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id      = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    celery_task_id = Column(String(255), nullable=True, index=True)  # None for sync image jobs

    # File metadata
    original_filename = Column(String(500), nullable=False)
    stored_filename   = Column(String(500), nullable=False)
    media_type        = Column(Enum(MediaType), nullable=False)
    file_size_bytes   = Column(Integer, nullable=False)
    file_path         = Column(String(1000), nullable=False)

    # Job state
    status     = Column(Enum(JobStatus), default=JobStatus.PENDING, nullable=False, index=True)
    error_msg  = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    ended_at   = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user   = relationship("User", back_populates="jobs")
    result = relationship("DetectionResult", back_populates="job", uselist=False, cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<DetectionJob id={self.id} status={self.status} media={self.media_type}>"


# ── Detection Result ───────────────────────────────────────────────────────

class DetectionResult(Base):
    __tablename__ = "detection_results"

    id     = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("detection_jobs.id"), unique=True, nullable=False)

    # Final verdict
    verdict    = Column(Enum(Verdict), nullable=False)
    confidence = Column(Float, nullable=False)   # 0.0 – 1.0

    # Per-model branch scores (stored as JSON)
    # e.g. {"spatial": 0.82, "frequency": 0.76, "attention": 0.79}
    branch_scores  = Column(JSON, nullable=True)

    # For video: frame-by-frame scores
    frame_scores   = Column(JSON, nullable=True)   # [{frame: 0, score: 0.9, timestamp: 0.0}, ...]

    # For audio: chunk-by-chunk scores
    chunk_scores   = Column(JSON, nullable=True)   # [{chunk: 0, score: 0.8, start: 0.0, end: 3.0}, ...]

    # Explainability
    heatmap_path   = Column(String(1000), nullable=True)  # Path to GradCAM overlay image
    heatmap_url    = Column(String(1000), nullable=True)  # Public URL

    # Media metadata
    duration_sec   = Column(Float, nullable=True)         # For video/audio
    frames_analyzed = Column(Integer, nullable=True)      # For video
    faces_detected  = Column(Integer, nullable=True)      # For image/video

    # Raw model output
    raw_output     = Column(JSON, nullable=True)

    created_at     = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    job = relationship("DetectionJob", back_populates="result")

    def __repr__(self) -> str:
        return f"<DetectionResult id={self.id} verdict={self.verdict} confidence={self.confidence:.2f}>"


# ── Refresh Token ──────────────────────────────────────────────────────────

class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id         = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id    = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    token_hash = Column(String(255), nullable=False, unique=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    revoked    = Column(Boolean, default=False)

    def __repr__(self) -> str:
        return f"<RefreshToken user={self.user_id} revoked={self.revoked}>"
