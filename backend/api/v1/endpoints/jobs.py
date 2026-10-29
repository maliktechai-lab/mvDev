from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from sqlalchemy.orm import selectinload
from pydantic import BaseModel, ConfigDict
from datetime import datetime

from db.database import get_db
from db.models import DetectionJob, DetectionResult, User, JobStatus, MediaType, Verdict
from core.security import get_current_user

router = APIRouter(prefix="/jobs", tags=["Jobs"])


# ── Response Schemas ───────────────────────────────────────────────────────

class BranchScores(BaseModel):
    spatial: Optional[float] = None
    frequency: Optional[float] = None
    attention: Optional[float] = None
    temporal: Optional[float] = None
    biological: Optional[float] = None
    raw_waveform: Optional[float] = None
    spectral: Optional[float] = None
    prosody: Optional[float] = None


class FrameScoreResponse(BaseModel):
    """Per-frame score entry for video results.

    ``score`` is ``None`` when the frame could not be scored because no face
    was detected (``face_status="not_found"``) or the face detector raised an
    exception (``face_status="failed"``).  A missing score must never be
    treated as 0, 0.5, or any other fabricated value.
    """
    frame: int
    score: Optional[float]           # null ↔ unscored frame
    timestamp: float
    face_status: Optional[str] = None  # "detected" | "not_found" | "failed"
    window_count: Optional[int] = None


class VideoRawOutput(BaseModel):
    """Typed raw output fields for video detection results.

    Exposed in the API so the frontend does not need to cast to unknown.
    All fields are optional — older records may predate some fields.
    """
    fake_probability: Optional[float] = None
    real_probability: Optional[float] = None
    suspicious_timestamps: Optional[List[float]] = None
    faces_not_found: Optional[int] = None
    faces_failed: Optional[int] = None
    scored_frames_count: Optional[int] = None


class ResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    verdict: Verdict
    confidence: float
    branch_scores: Optional[dict]
    frame_scores: Optional[List[FrameScoreResponse]]
    chunk_scores: Optional[list]
    heatmap_url: Optional[str]
    duration_sec: Optional[float]
    frames_analyzed: Optional[int]
    faces_detected: Optional[int]
    raw_output: Optional[dict]  # Stored as freeform JSON; VideoRawOutput for video results
    created_at: datetime


class JobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    original_filename: str
    media_type: MediaType
    file_size_bytes: int
    status: JobStatus
    error_msg: Optional[str]
    started_at: Optional[datetime]
    ended_at: Optional[datetime]
    created_at: datetime
    result: Optional[ResultResponse]


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a specific detection job (with result if completed)."""
    result = await db.execute(
        select(DetectionJob)
        .options(selectinload(DetectionJob.result))
        .where(DetectionJob.id == job_id, DetectionJob.user_id == current_user.id)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return job


async def _list_jobs_impl(
    db: AsyncSession,
    current_user: User,
    page: int,
    page_size: int,
    media_type: Optional[MediaType],
    status_filter: Optional[JobStatus],
) -> List[JobResponse]:
    """Shared implementation for list-jobs routes."""
    query = (
        select(DetectionJob)
        .options(selectinload(DetectionJob.result))
        .where(DetectionJob.user_id == current_user.id)
        .order_by(desc(DetectionJob.created_at))
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    if media_type:
        query = query.where(DetectionJob.media_type == media_type)
    if status_filter:
        query = query.where(DetectionJob.status == status_filter)

    result = await db.execute(query)
    return result.scalars().all()


# Both /jobs and /jobs/ — prevents the 307 redirect when frontend omits the slash
@router.get("", response_model=List[JobResponse])
@router.get("/", response_model=List[JobResponse], include_in_schema=False)
async def list_jobs(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    media_type: Optional[MediaType] = Query(default=None),
    status_filter: Optional[JobStatus] = Query(default=None, alias="status"),
):
    """List all jobs for the current user with pagination."""
    return await _list_jobs_impl(db, current_user, page, page_size, media_type, status_filter)


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_job(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a job and its result."""
    result = await db.execute(
        select(DetectionJob)
        .where(DetectionJob.id == job_id, DetectionJob.user_id == current_user.id)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    await db.delete(job)
