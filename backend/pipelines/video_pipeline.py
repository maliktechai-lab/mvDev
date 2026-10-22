"""
Video Inference Pipeline
=========================
Orchestrates the full detection flow for a video file:

  1. Extract frames at configurable FPS (ffmpeg)
  2. Run face detection per frame (MTCNN)
  3. Batch frames → DeepGuard-Video model (4 branches)
  4. Generate frame-level scores and suspicious timestamps
  5. Generate GradCAM heatmaps for most suspicious frames
  6. Store result in DB, update job status

This runs as a Celery async task (videos can take 10-60 seconds).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torchvision.transforms as T
from loguru import logger

from core.celery_app import celery_app
from core.config import settings
from preprocessing.face_detector import get_face_detector
from preprocessing.frame_extractor import FrameExtractor, ExtractedFrame, VideoMetadata
from preprocessing.image_preprocessor import IMAGENET_MEAN, IMAGENET_STD


# ── Transform for individual frames ───────────────────────────────────────

frame_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Result types ───────────────────────────────────────────────────────────

@dataclass
class FrameDetectionResult:
    frame_index: int
    timestamp_sec: float
    fake_probability: float
    has_face: bool


@dataclass
class VideoDetectionResult:
    verdict: str
    confidence: float
    fake_probability: float
    real_probability: float
    frames_analyzed: int
    duration_sec: float
    faces_detected: int
    frame_scores: List[Dict]
    branch_scores: Dict
    heatmap_path: Optional[str]
    heatmap_url: Optional[str]
    suspicious_timestamps: List[float]   # Timestamps where fake_prob > 0.7
    error: Optional[str] = None


# ── Core pipeline logic ────────────────────────────────────────────────────

class VideoInferencePipeline:
    """Stateless video inference pipeline."""

    BATCH_SIZE    = 8    # Number of frames per model forward pass
    FRAME_FPS     = 1.0  # Extract 1 frame/sec by default
    MAX_FRAMES    = 300  # Safety cap

    def __init__(self):
        self.device    = settings.DEVICE
        self.extractor = FrameExtractor(
            target_fps=settings.VIDEO_FRAMES_PER_SEC,
            max_frames=self.MAX_FRAMES,
            target_size=(224, 224),
        )

    def run(
        self,
        video_path: str,
        generate_heatmap: bool = True,
    ) -> VideoDetectionResult:
        try:
            return self._run_pipeline(video_path, generate_heatmap)
        except Exception as e:
            logger.exception(f"Video pipeline error: {e}")
            return VideoDetectionResult(
                verdict="unknown", confidence=0.0,
                fake_probability=0.0, real_probability=0.0,
                frames_analyzed=0, duration_sec=0.0,
                faces_detected=0, frame_scores=[],
                branch_scores={}, heatmap_path=None,
                heatmap_url=None, suspicious_timestamps=[],
                error=str(e),
            )

    def _run_pipeline(self, video_path: str, generate_heatmap: bool) -> VideoDetectionResult:
        from models.registry import get_video_model, get_gradcam
        from models.image_model import DeepGuardImageModel

        model = get_video_model()

        # ── 1. Extract frames ──────────────────────────────────────────────
        frames, meta = self.extractor.extract(video_path)
        if not frames:
            raise ValueError("No frames could be extracted from video")

        logger.info(f"Video: {meta.duration_sec:.1f}s, {len(frames)} frames extracted")

        # ── 2. Face detection per frame ────────────────────────────────────
        detector    = get_face_detector(self.device)
        face_frames = []    # PIL images of detected faces (or full frame fallback)
        face_count  = 0

        for frame in frames:
            faces = detector.detect(frame.image)
            if faces:
                face_frames.append(faces[0].crop)  # Use primary face
                face_count += 1
            else:
                face_frames.append(frame.image)    # Fallback: full frame

        logger.info(f"Faces detected in {face_count}/{len(frames)} frames")

        # ── 3. Convert frames to tensors ───────────────────────────────────
        frame_tensors = [frame_transform(img) for img in face_frames]  # list of (3,224,224)

        # ── 4. Run model in batches of BATCH_SIZE sequences ───────────────
        all_frame_scores: List[float] = []
        all_branch_scores: List[Dict] = []
        overall_fake_probs: List[float] = []

        # Slide a window of BATCH_SIZE frames across the video
        window = min(self.BATCH_SIZE, len(frame_tensors))
        stride = max(1, window // 2)

        i = 0
        while i + window <= len(frame_tensors):
            batch_frames = frame_tensors[i:i + window]
            seq = torch.stack(batch_frames).unsqueeze(0).to(self.device)  # (1, T, 3, 224, 224)

            with torch.no_grad():
                out = model(seq, return_branch_scores=True)

            fake_prob   = float(out["probs"][0, 1])
            frame_sc    = out["frame_scores"][0].tolist()  # T scores
            branch_sc   = out.get("branch_scores", {})

            overall_fake_probs.append(fake_prob)
            all_frame_scores.extend(frame_sc)
            all_branch_scores.append(branch_sc)
            i += stride

        # Handle remaining frames if any
        if not overall_fake_probs:
            seq = torch.stack(frame_tensors[:window]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = model(seq, return_branch_scores=True)
            overall_fake_probs = [float(out["probs"][0, 1])]
            all_frame_scores   = out["frame_scores"][0].tolist()
            all_branch_scores  = [out.get("branch_scores", {})]

        # ── 5. Aggregate scores ────────────────────────────────────────────
        # Trim/pad frame scores to match extracted frames
        all_frame_scores = all_frame_scores[:len(frames)]
        while len(all_frame_scores) < len(frames):
            all_frame_scores.append(all_frame_scores[-1] if all_frame_scores else 0.5)

        # Average branch scores across windows
        avg_branch = {}
        if all_branch_scores:
            keys = all_branch_scores[0].keys()
            for k in keys:
                vals = [d[k] for d in all_branch_scores if k in d]
                avg_branch[k] = float(np.mean(vals)) if vals else 0.0

        # Final fake probability: max of sliding-window scores (conservative)
        final_fake_prob = float(np.mean(overall_fake_probs))
        verdict         = "fake" if final_fake_prob >= settings.IMAGE_CONFIDENCE_THRESHOLD else "real"
        confidence      = max(final_fake_prob, 1.0 - final_fake_prob)

        # Build per-frame score list for DB storage
        frame_score_list = [
            {
                "frame":     i,
                "score":     round(score, 4),
                "timestamp": round(frames[i].timestamp_sec, 2) if i < len(frames) else 0.0,
            }
            for i, score in enumerate(all_frame_scores)
        ]

        # Suspicious timestamps (fake_prob > 0.7)
        suspicious = [
            fs["timestamp"]
            for fs in frame_score_list
            if fs["score"] > 0.7
        ]

        # ── 6. GradCAM on most suspicious frame ───────────────────────────
        heatmap_path = None
        heatmap_url  = None

        if generate_heatmap and frame_score_list:
            try:
                most_suspicious_idx = int(np.argmax(all_frame_scores))
                most_suspicious_img = face_frames[most_suspicious_idx]
                most_suspicious_tensor = frame_tensors[most_suspicious_idx].unsqueeze(0).to(self.device)

                # Use image model's GradCAM (spatial branch target)
                gradcam = get_gradcam()
                _, heatmap_path = gradcam.generate(most_suspicious_tensor, most_suspicious_img)
                fname       = Path(heatmap_path).name
                heatmap_url = f"/heatmaps/{fname}"
            except Exception as e:
                logger.warning(f"Video GradCAM failed: {e}")

        logger.info(
            f"Video detection done: verdict={verdict} "
            f"fake_prob={final_fake_prob:.3f} "
            f"frames={len(frames)} suspicious={len(suspicious)}"
        )

        return VideoDetectionResult(
            verdict=verdict,
            confidence=confidence,
            fake_probability=final_fake_prob,
            real_probability=1.0 - final_fake_prob,
            frames_analyzed=len(frames),
            duration_sec=meta.duration_sec,
            faces_detected=face_count,
            frame_scores=frame_score_list,
            branch_scores=avg_branch,
            heatmap_path=heatmap_path,
            heatmap_url=heatmap_url,
            suspicious_timestamps=suspicious,
        )


# ── Celery task ────────────────────────────────────────────────────────────

@celery_app.task(
    name="pipelines.video_pipeline.run_video_detection",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
)
def run_video_detection(self, job_id: str, video_path: str) -> dict:
    """
    Celery task: run video detection and update job status in DB.

    Args:
        job_id:     UUID string of the DetectionJob record
        video_path: Absolute path to the uploaded video file

    Returns:
        dict with result summary (stored in Celery result backend)
    """
    import asyncio
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    from sqlalchemy import select
    from db.models import DetectionJob, DetectionResult, JobStatus, Verdict

    logger.info(f"[Celery] Starting video detection for job={job_id}")

    # Update job status to RUNNING
    async def update_db(status, **result_kwargs):
        from core.config import settings as cfg
        engine  = create_async_engine(cfg.DATABASE_URL, pool_pre_ping=True)
        Session = async_sessionmaker(engine, expire_on_commit=False)

        async with Session() as db:
            res = await db.execute(select(DetectionJob).where(DetectionJob.id == job_id))
            job = res.scalar_one_or_none()
            if not job:
                return

            job.status = status
            if status == JobStatus.RUNNING:
                job.started_at = datetime.now(timezone.utc)
            else:
                job.ended_at = datetime.now(timezone.utc)

            if result_kwargs:
                verdict_str = result_kwargs.get("verdict", "unknown")
                verdict_enum = Verdict[verdict_str.upper()] if verdict_str in ("real", "fake") else Verdict.UNKNOWN

                db_result = DetectionResult(
                    job_id=job.id,
                    verdict=verdict_enum,
                    confidence=result_kwargs.get("confidence", 0.0),
                    branch_scores=result_kwargs.get("branch_scores"),
                    frame_scores=result_kwargs.get("frame_scores"),
                    heatmap_path=result_kwargs.get("heatmap_path"),
                    heatmap_url=result_kwargs.get("heatmap_url"),
                    duration_sec=result_kwargs.get("duration_sec"),
                    frames_analyzed=result_kwargs.get("frames_analyzed"),
                    faces_detected=result_kwargs.get("faces_detected"),
                    raw_output={
                        "fake_probability": result_kwargs.get("fake_probability"),
                        "real_probability": result_kwargs.get("real_probability"),
                        "suspicious_timestamps": result_kwargs.get("suspicious_timestamps", []),
                    },
                )
                db.add(db_result)

            await db.commit()
        await engine.dispose()

    try:
        asyncio.run(update_db(JobStatus.RUNNING))

        pipeline = VideoInferencePipeline()
        result   = pipeline.run(video_path, generate_heatmap=True)

        if result.error:
            raise RuntimeError(result.error)

        asyncio.run(update_db(
            JobStatus.COMPLETED,
            verdict=result.verdict,
            confidence=result.confidence,
            fake_probability=result.fake_probability,
            real_probability=result.real_probability,
            branch_scores=result.branch_scores,
            frame_scores=result.frame_scores,
            heatmap_path=result.heatmap_path,
            heatmap_url=result.heatmap_url,
            duration_sec=result.duration_sec,
            frames_analyzed=result.frames_analyzed,
            faces_detected=result.faces_detected,
            suspicious_timestamps=result.suspicious_timestamps,
        ))

        logger.info(f"[Celery] Video detection complete for job={job_id}")
        return {"job_id": job_id, "verdict": result.verdict, "confidence": result.confidence}

    except Exception as exc:
        logger.exception(f"[Celery] Video detection failed for job={job_id}: {exc}")
        asyncio.run(update_db(JobStatus.FAILED))
        raise self.retry(exc=exc)


# ── Singleton ──────────────────────────────────────────────────────────────

_pipeline: Optional[VideoInferencePipeline] = None


def get_video_pipeline() -> VideoInferencePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = VideoInferencePipeline()
    return _pipeline
