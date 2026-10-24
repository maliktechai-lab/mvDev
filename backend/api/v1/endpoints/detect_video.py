"""
Video Detection Endpoint
=========================
POST /api/v1/detect/video

Accepts an uploaded video, dispatches to Celery queue,
returns job_id immediately for polling via GET /api/v1/jobs/{id}.

Video processing is async because it can take 10-60 seconds.
The frontend polls GET /api/v1/jobs/{job_id} every 3s until status=completed.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from core.config import settings
from core.security import get_current_user
from db.database import get_db
from db.models import DetectionJob, JobStatus, MediaType, User

router = APIRouter(tags=["Detection"])

ALLOWED_VIDEO_TYPES = {
    "video/mp4", "video/webm", "video/avi",
    "video/quicktime", "video/x-matroska",
    "video/x-msvideo",
}


class VideoSubmitResponse(BaseModel):
    job_id: str
    status: str
    message: str
    poll_url: str


@router.post("/video", response_model=VideoSubmitResponse, status_code=status.HTTP_202_ACCEPTED)
async def detect_video(
    request: Request,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Submit a video for deepfake detection.

    Processing is asynchronous.
    - Returns job_id immediately (HTTP 202 Accepted)
    - Poll GET /api/v1/jobs/{job_id} for status and result
    - Status transitions: pending → running → completed | failed
    """
    # ── Validate type ──────────────────────────────────────────────────────
    content_type = file.content_type or ""
    if content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported video format: {content_type}. Allowed: MP4, WebM, AVI, MOV, MKV",
        )

    # ── Read file & validate size ──────────────────────────────────────────
    # SECURITY: Check Content-Length header before reading the body.
    # The header is client-supplied and therefore unreliable, but it lets us
    # reject obviously-oversized uploads (e.g. a 2 GB file) without buffering
    # the entire payload into memory first.  A hard limit is still enforced
    # after the actual read so a lying client cannot bypass this check.
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            cl_bytes = int(content_length)
            if cl_bytes > settings.max_video_size_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Content-Length {cl_bytes / 1e6:.0f} MB exceeds max {settings.MAX_VIDEO_SIZE_MB} MB",
                )
        except ValueError:
            pass  # Unparseable header — let the actual read enforce the limit

    file_bytes = await file.read()
    file_size  = len(file_bytes)

    if file_size > settings.max_video_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large: {file_size / 1e6:.0f} MB (max {settings.MAX_VIDEO_SIZE_MB} MB)",
        )
    if file_size < 10_000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File too small to be a valid video",
        )

    # ── Save to disk ───────────────────────────────────────────────────────
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # SECURITY: Sanitize extension against a strict whitelist to prevent path
    # traversal via crafted filenames. uuid4().hex ensures no user-controlled
    # bytes appear in the stored path.
    _SAFE_VIDEO_EXTS = {".mp4", ".webm", ".avi", ".mov", ".mkv"}
    raw_ext = Path(file.filename or "").suffix.lower()
    ext             = raw_ext if raw_ext in _SAFE_VIDEO_EXTS else ".mp4"
    stored_filename = f"{uuid.uuid4().hex}{ext}"
    file_path       = upload_dir / stored_filename

    with open(file_path, "wb") as f:
        f.write(file_bytes)

    logger.info(f"Video saved: {file_path} ({file_size / 1e6:.1f} MB)")

    # ── Create job record ──────────────────────────────────────────────────
    job = DetectionJob(
        user_id=current_user.id,
        original_filename=file.filename or stored_filename,
        stored_filename=stored_filename,
        media_type=MediaType.VIDEO,
        file_size_bytes=file_size,
        file_path=str(file_path),
        status=JobStatus.PENDING,
    )
    db.add(job)
    await db.flush()
    await db.refresh(job)
    job_id = str(job.id)

    # ── Dispatch Celery task ───────────────────────────────────────────────
    from pipelines.video_pipeline import run_video_detection
    task = run_video_detection.apply_async(
        args=[job_id, str(file_path)],
        queue="video",
    )

    # Save Celery task ID for monitoring
    job.celery_task_id = task.id
    logger.info(f"Video job {job_id} queued → Celery task {task.id}")

    return VideoSubmitResponse(
        job_id=job_id,
        status="pending",
        message="Video submitted for analysis. Poll the job status URL for results.",
        poll_url=f"/api/v1/jobs/{job_id}",
    )
