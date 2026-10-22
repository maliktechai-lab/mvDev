"""
Image Detection Endpoint
=========================
POST /api/v1/detect/image

Accepts an uploaded image, runs synchronous deepfake detection,
saves the job + result to DB, returns the full result.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from core.config import settings
from core.security import get_current_user
from db.database import get_db
from db.models import DetectionJob, DetectionResult, JobStatus, MediaType, Verdict, User

router = APIRouter(prefix="/detect", tags=["Detection"])

# Allowed MIME types for image uploads
ALLOWED_IMAGE_TYPES = {
    "image/jpeg", "image/jpg", "image/png",
    "image/webp", "image/bmp",
}


# ── Response Schema ────────────────────────────────────────────────────────

class BranchScoresOut(BaseModel):
    spatial:   float | None = None
    frequency: float | None = None
    attention: float | None = None


class ImageDetectionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    job_id:          str
    verdict:         str
    confidence:      float
    fake_probability: float
    real_probability: float
    faces_detected:  int
    branch_scores:   dict
    heatmap_url:     str | None
    processing_ms:   int


# ── Endpoint ───────────────────────────────────────────────────────────────

@router.post("/image", response_model=ImageDetectionResponse, status_code=status.HTTP_200_OK)
async def detect_image(
    file: UploadFile = File(...),
    generate_heatmap: bool = True,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Detect deepfake in an uploaded image.

    - Validates file type and size
    - Runs face detection → DeepGuard-Image (3 branches) → GradCAM
    - Saves job and result to database
    - Returns verdict, confidence, per-branch scores, heatmap URL
    """
    import time
    start_ms = int(time.time() * 1000)

    # ── Validate file type ─────────────────────────────────────────────────
    content_type = file.content_type or ""
    if content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {content_type}. Allowed: JPEG, PNG, WebP, BMP",
        )

    # ── Read and validate file size ────────────────────────────────────────
    file_bytes = await file.read()
    file_size  = len(file_bytes)

    if file_size > settings.max_image_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large: {file_size / 1e6:.1f} MB (max {settings.MAX_IMAGE_SIZE_MB} MB)",
        )
    if file_size < 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File too small (minimum 1 KB)",
        )

    # ── Save file to disk ──────────────────────────────────────────────────
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename or "upload.jpg").suffix.lower() or ".jpg"
    stored_filename = f"{uuid.uuid4().hex}{ext}"
    file_path       = upload_dir / stored_filename

    with open(file_path, "wb") as f:
        f.write(file_bytes)

    logger.info(f"Image saved: {file_path} ({file_size} bytes)")

    # ── Create job record ──────────────────────────────────────────────────
    job = DetectionJob(
        user_id=current_user.id,
        original_filename=file.filename or stored_filename,
        stored_filename=stored_filename,
        media_type=MediaType.IMAGE,
        file_size_bytes=file_size,
        file_path=str(file_path),
        status=JobStatus.RUNNING,
        started_at=datetime.now(timezone.utc),
    )
    db.add(job)
    await db.flush()
    await db.refresh(job)

    # ── Run inference pipeline ─────────────────────────────────────────────
    try:
        from pipelines.image_pipeline import get_image_pipeline
        pipeline = get_image_pipeline()
        result   = pipeline.run(str(file_path), generate_heatmap=generate_heatmap)

        verdict_enum = Verdict[result.verdict.upper()] if result.verdict in ("real", "fake") else Verdict.UNKNOWN

        # Save result to DB
        db_result = DetectionResult(
            job_id=job.id,
            verdict=verdict_enum,
            confidence=result.confidence,
            branch_scores=result.branch_scores,
            heatmap_path=result.heatmap_path,
            heatmap_url=result.heatmap_url,
            faces_detected=result.faces_detected,
            raw_output={
                "fake_probability": result.fake_probability,
                "real_probability": result.real_probability,
                "face_results": [
                    {
                        "face_index": fr.face_index,
                        "verdict":    fr.verdict,
                        "fake_prob":  fr.fake_probability,
                        "confidence": fr.confidence,
                    }
                    for fr in result.face_results
                ],
            },
        )
        db.add(db_result)

        job.status   = JobStatus.COMPLETED
        job.ended_at = datetime.now(timezone.utc)

        elapsed_ms = int(time.time() * 1000) - start_ms
        logger.info(
            f"Image detection complete: job={job.id} verdict={result.verdict} "
            f"confidence={result.confidence:.3f} ms={elapsed_ms}"
        )

        return ImageDetectionResponse(
            job_id=str(job.id),
            verdict=result.verdict,
            confidence=result.confidence,
            fake_probability=result.fake_probability,
            real_probability=result.real_probability,
            faces_detected=result.faces_detected,
            branch_scores=result.branch_scores,
            heatmap_url=result.heatmap_url,
            processing_ms=elapsed_ms,
        )

    except Exception as e:
        logger.exception(f"Detection failed for job {job.id}: {e}")

        job.status    = JobStatus.FAILED
        job.error_msg = str(e)
        job.ended_at  = datetime.now(timezone.utc)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {str(e)}",
        )
