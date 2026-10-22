"""
Audio Detection Endpoint
=========================
POST /api/v1/detect/audio

Accepts an uploaded audio file, dispatches to Celery queue,
returns job_id for polling. Processing is async (10-30 seconds).
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from core.config import settings
from core.security import get_current_user
from db.database import get_db
from db.models import DetectionJob, JobStatus, MediaType, User

router = APIRouter(tags=["Detection"])

ALLOWED_AUDIO_TYPES = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav",
    "audio/ogg", "audio/flac", "audio/mp4", "audio/x-m4a",
    "audio/aac",
}


class AudioSubmitResponse(BaseModel):
    job_id: str
    status: str
    message: str
    poll_url: str


@router.post("/audio", response_model=AudioSubmitResponse, status_code=status.HTTP_202_ACCEPTED)
async def detect_audio(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Submit an audio file for synthetic voice detection.

    Processing is asynchronous.
    - Returns job_id immediately (HTTP 202 Accepted)
    - Poll GET /api/v1/jobs/{job_id} for status and result
    """
    # ── Validate type ──────────────────────────────────────────────────────
    content_type = file.content_type or ""
    # Fallback: accept by extension if MIME isn't recognized
    ext = Path(file.filename or "").suffix.lower()
    ext_allowed = ext in {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}

    if content_type not in ALLOWED_AUDIO_TYPES and not ext_allowed:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported audio format: {content_type}. Allowed: MP3, WAV, FLAC, OGG, M4A",
        )

    # ── Read & validate size ───────────────────────────────────────────────
    file_bytes = await file.read()
    file_size  = len(file_bytes)

    if file_size > settings.max_audio_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large: {file_size / 1e6:.1f} MB (max {settings.MAX_AUDIO_SIZE_MB} MB)",
        )
    if file_size < 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File too small to be valid audio",
        )

    # ── Save to disk ───────────────────────────────────────────────────────
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    stored_ext      = ext or ".mp3"
    stored_filename = f"{uuid.uuid4().hex}{stored_ext}"
    file_path       = upload_dir / stored_filename

    with open(file_path, "wb") as f:
        f.write(file_bytes)

    logger.info(f"Audio saved: {file_path} ({file_size / 1e6:.2f} MB)")

    # ── Create job record ──────────────────────────────────────────────────
    job = DetectionJob(
        user_id=current_user.id,
        original_filename=file.filename or stored_filename,
        stored_filename=stored_filename,
        media_type=MediaType.AUDIO,
        file_size_bytes=file_size,
        file_path=str(file_path),
        status=JobStatus.PENDING,
    )
    db.add(job)
    await db.flush()
    await db.refresh(job)
    job_id = str(job.id)

    # ── Dispatch Celery task ───────────────────────────────────────────────
    from pipelines.audio_pipeline import run_audio_detection
    task = run_audio_detection.apply_async(
        args=[job_id, str(file_path)],
        queue="audio",
    )
    job.celery_task_id = task.id
    logger.info(f"Audio job {job_id} queued → Celery task {task.id}")

    return AudioSubmitResponse(
        job_id=job_id,
        status="pending",
        message="Audio submitted for analysis. Poll the job status URL for results.",
        poll_url=f"/api/v1/jobs/{job_id}",
    )
