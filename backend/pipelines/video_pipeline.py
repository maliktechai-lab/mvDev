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

class FaceDetectionStatus:
    """Explicit status codes for face detection on a single frame."""
    DETECTED  = "detected"   # face crop was used
    NOT_FOUND = "not_found"  # no face found; frame skipped (not scored)
    FAILED    = "failed"     # detector raised an exception


@dataclass
class FrameDetectionResult:
    frame_index:      int
    timestamp_sec:    float
    fake_probability: float
    face_status:      str   # one of FaceDetectionStatus constants


@dataclass
class VideoDetectionResult:
    verdict: str
    confidence: float
    fake_probability: float
    real_probability: float
    frames_analyzed: int
    duration_sec: float
    faces_detected: int
    faces_not_found: int
    faces_failed: int
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
                faces_detected=0, faces_not_found=0, faces_failed=0,
                frame_scores=[],
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

        # ── 2. Face detection per frame — explicit per-frame status ────────
        # For each frame we record the face status and, when a face was
        # detected, the corresponding PIL crop for scoring.
        #
        # Frames with no face (NOT_FOUND or FAILED) are excluded from model
        # inference and from the scored-frame list.  Their absence is
        # represented explicitly in the output — no fake score is invented
        # for a missing face.
        face_statuses: List[str] = []   # parallel to `frames`
        scoreable_indices: List[int] = []   # indices into `frames` that have a face crop
        face_crops: List = []               # PIL crops for scoreable frames only

        detector = get_face_detector(self.device)

        for frame_idx, frame in enumerate(frames):
            try:
                faces = detector.detect(frame.image)
            except Exception as exc:
                logger.warning(
                    f"Face detector raised for frame {frame_idx}: {exc}"
                )
                face_statuses.append(FaceDetectionStatus.FAILED)
                continue

            if faces:
                face_statuses.append(FaceDetectionStatus.DETECTED)
                scoreable_indices.append(frame_idx)
                face_crops.append(faces[0].crop)
            else:
                face_statuses.append(FaceDetectionStatus.NOT_FOUND)

        face_count     = face_statuses.count(FaceDetectionStatus.DETECTED)
        not_found_count = face_statuses.count(FaceDetectionStatus.NOT_FOUND)
        failed_count   = face_statuses.count(FaceDetectionStatus.FAILED)

        logger.info(
            f"Face detection: {face_count} detected, "
            f"{not_found_count} not found, {failed_count} failed "
            f"(out of {len(frames)} frames)"
        )

        # ── 3. Convert scoreable frames to tensors ─────────────────────────
        frame_tensors = [frame_transform(crop) for crop in face_crops]

        # ── 4. Indexed accumulation — one bucket per scoreable frame ───────
        #
        # We slide a window of BATCH_SIZE frames over the scoreable frames.
        # The model returns one frame-level score per position in the window.
        # Each score is accumulated into the bucket for the *original frame
        # index* it came from, so every contributing window is averaged out.
        #
        # score_accum[scoreable_indices[k]] += model_frame_score_at_position_k
        # score_count[scoreable_indices[k]] += 1

        # Keys are original frame indices; values accumulate window scores.
        score_accum: Dict[int, float] = {}
        score_count: Dict[int, int]   = {}
        all_branch_scores: List[Dict] = []
        overall_fake_probs: List[float] = []

        if frame_tensors:
            window = min(self.BATCH_SIZE, len(frame_tensors))
            stride = max(1, window // 2)

            i = 0
            while i + window <= len(frame_tensors):
                batch_frames = frame_tensors[i : i + window]
                seq = torch.stack(batch_frames).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    out = model(seq, return_branch_scores=True)

                fake_prob  = float(out["probs"][0, 1])
                window_frame_scores = out["frame_scores"][0].tolist()  # length == window
                branch_sc  = out.get("branch_scores", {})

                overall_fake_probs.append(fake_prob)
                all_branch_scores.append(branch_sc)

                # Accumulate each position's score back to its original frame index
                for pos, frame_score in enumerate(window_frame_scores):
                    original_idx = scoreable_indices[i + pos]
                    score_accum[original_idx] = score_accum.get(original_idx, 0.0) + frame_score
                    score_count[original_idx] = score_count.get(original_idx, 0) + 1

                i += stride

            # Handle remaining frames that didn't fill a full window
            remainder_start = i
            if remainder_start < len(frame_tensors):
                batch_frames = frame_tensors[remainder_start:]
                actual_window = len(batch_frames)
                seq = torch.stack(batch_frames).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    out = model(seq, return_branch_scores=True)

                fake_prob  = float(out["probs"][0, 1])
                window_frame_scores = out["frame_scores"][0].tolist()[:actual_window]
                branch_sc  = out.get("branch_scores", {})

                overall_fake_probs.append(fake_prob)
                all_branch_scores.append(branch_sc)

                for pos, frame_score in enumerate(window_frame_scores):
                    original_idx = scoreable_indices[remainder_start + pos]
                    score_accum[original_idx] = score_accum.get(original_idx, 0.0) + frame_score
                    score_count[original_idx] = score_count.get(original_idx, 0) + 1

        # ── 5. Compute per-original-frame averaged scores ──────────────────
        # Frames without a face are explicitly marked as unscored (null).
        frame_score_list = []
        for orig_idx, frame in enumerate(frames):
            status = face_statuses[orig_idx]
            if orig_idx in score_accum and score_count[orig_idx] > 0:
                avg_score = score_accum[orig_idx] / score_count[orig_idx]
                frame_score_list.append({
                    "frame":       orig_idx,
                    "score":       round(avg_score, 4),
                    "timestamp":   round(frame.timestamp_sec, 2),
                    "face_status": status,
                    "window_count": score_count[orig_idx],
                })
            else:
                # Frame was not scored — face absent or detector failed
                frame_score_list.append({
                    "frame":       orig_idx,
                    "score":       None,
                    "timestamp":   round(frame.timestamp_sec, 2),
                    "face_status": status,
                    "window_count": 0,
                })

        # ── 6. Aggregate branch scores and final verdict ───────────────────
        avg_branch: Dict = {}
        if all_branch_scores:
            keys = all_branch_scores[0].keys()
            for k in keys:
                vals = [d[k] for d in all_branch_scores if k in d]
                avg_branch[k] = float(np.mean(vals)) if vals else 0.0

        if overall_fake_probs:
            final_fake_prob = float(np.mean(overall_fake_probs))
            verdict    = "fake" if final_fake_prob >= settings.IMAGE_CONFIDENCE_THRESHOLD else "real"
            confidence = max(final_fake_prob, 1.0 - final_fake_prob)
        else:
            # No frames had detectable faces — cannot determine a verdict.
            # Setting verdict="real" here would be a false claim.
            logger.warning(
                "No frames were scored (no faces detected or all failed). "
                "Verdict is unknown."
            )
            final_fake_prob = 0.0
            verdict    = "unknown"
            confidence = 0.0

        # Suspicious timestamps: only include frames that were actually scored
        suspicious = [
            fs["timestamp"]
            for fs in frame_score_list
            if fs["score"] is not None and fs["score"] > 0.7
        ]

        # ── 7. GradCAM on most suspicious scored frame ─────────────────────
        heatmap_path = None
        heatmap_url  = None

        scored_entries = [fs for fs in frame_score_list if fs["score"] is not None]
        if generate_heatmap and scored_entries:
            try:
                most_suspicious_entry = max(scored_entries, key=lambda x: x["score"])
                most_suspicious_orig_idx = most_suspicious_entry["frame"]
                # Resolve back to the position in the scoreable list
                scoreable_pos = scoreable_indices.index(most_suspicious_orig_idx)
                most_suspicious_img    = face_crops[scoreable_pos]
                most_suspicious_tensor = frame_tensors[scoreable_pos].unsqueeze(0).to(self.device)

                gradcam = get_gradcam()
                _, heatmap_path = gradcam.generate(most_suspicious_tensor, most_suspicious_img)
                fname       = Path(heatmap_path).name
                heatmap_url = f"/heatmaps/{fname}"
            except Exception as e:
                logger.warning(f"Video GradCAM failed: {e}")

        logger.info(
            f"Video detection done: verdict={verdict} "
            f"fake_prob={final_fake_prob:.3f} "
            f"frames={len(frames)} scored={len(scored_entries)} "
            f"suspicious={len(suspicious)}"
        )

        return VideoDetectionResult(
            verdict=verdict,
            confidence=confidence,
            fake_probability=final_fake_prob,
            # When verdict is unknown (no scored frames), neither probability
            # has any evidential basis.  Both are 0.0 rather than 1.0 - 0.0 = 1.0,
            # which would wrongly imply the video is real.
            real_probability=0.0 if verdict == "unknown" else (1.0 - final_fake_prob),
            frames_analyzed=len(frames),
            duration_sec=meta.duration_sec,
            faces_detected=face_count,
            faces_not_found=not_found_count,
            faces_failed=failed_count,
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
    return asyncio.run(_run_video_detection_async(self, job_id, video_path))


async def _run_video_detection_async(task, job_id: str, video_path: str) -> dict:
    """Async implementation of the video detection task."""
    from sqlalchemy import select
    from db.database import AsyncSessionLocal
    from db.models import DetectionJob, DetectionResult, JobStatus, Verdict
    from core.model_readiness import check_video_model, require_model_ready

    logger.info(f"[Celery] Starting video detection for job={job_id}")

    # ── Model readiness gate (fail fast before any heavy work) ─────────────
    try:
        readiness = check_video_model()
        require_model_ready(readiness)
    except RuntimeError as exc:
        logger.error(f"[Celery] Model not ready for job={job_id}: {exc}")
        async with AsyncSessionLocal() as db:
            res = await db.execute(select(DetectionJob).where(DetectionJob.id == job_id))
            job = res.scalar_one_or_none()
            if job:
                job.status    = JobStatus.FAILED
                job.error_msg = str(exc)
                job.ended_at  = datetime.now(timezone.utc)
                await db.commit()
        raise task.retry(exc=exc)

    try:
        # ── Mark job as RUNNING ────────────────────────────────────────────
        async with AsyncSessionLocal() as db:
            res = await db.execute(select(DetectionJob).where(DetectionJob.id == job_id))
            job = res.scalar_one_or_none()
            if not job:
                logger.error(f"[Celery] Job {job_id} not found in DB")
                return {"job_id": job_id, "error": "job not found"}
            job.status     = JobStatus.RUNNING
            job.started_at = datetime.now(timezone.utc)
            await db.commit()

        # ── Run inference ──────────────────────────────────────────────────
        pipeline = VideoInferencePipeline()
        result   = pipeline.run(video_path, generate_heatmap=True)

        if result.error:
            raise RuntimeError(result.error)

        # ── Persist result ─────────────────────────────────────────────────
        async with AsyncSessionLocal() as db:
            res = await db.execute(select(DetectionJob).where(DetectionJob.id == job_id))
            job = res.scalar_one_or_none()
            if not job:
                logger.error(f"[Celery] Job {job_id} disappeared before result save")
                return {"job_id": job_id, "error": "job not found after inference"}

            verdict_str  = result.verdict
            verdict_enum = Verdict[verdict_str.upper()] if verdict_str in ("real", "fake") else Verdict.UNKNOWN

            db_result = DetectionResult(
                job_id=job.id,
                verdict=verdict_enum,
                confidence=result.confidence,
                branch_scores=result.branch_scores,
                frame_scores=result.frame_scores,
                heatmap_path=result.heatmap_path,
                heatmap_url=result.heatmap_url,
                duration_sec=result.duration_sec,
                frames_analyzed=result.frames_analyzed,
                faces_detected=result.faces_detected,
                raw_output={
                    "fake_probability":      result.fake_probability,
                    "real_probability":      result.real_probability,
                    "suspicious_timestamps": result.suspicious_timestamps,
                    "faces_not_found":       result.faces_not_found,
                    "faces_failed":          result.faces_failed,
                    "scored_frames_count":   sum(
                        1 for fs in result.frame_scores if fs.get("score") is not None
                    ),
                },
            )
            db.add(db_result)

            job.status   = JobStatus.COMPLETED
            job.ended_at = datetime.now(timezone.utc)
            await db.commit()

        logger.info(f"[Celery] Video detection complete for job={job_id}")
        return {"job_id": job_id, "verdict": result.verdict, "confidence": result.confidence}

    except Exception as exc:
        logger.exception(f"[Celery] Video detection failed for job={job_id}: {exc}")
        # Persist FAILED status so the gap is not lost on rollback
        try:
            async with AsyncSessionLocal() as db:
                res = await db.execute(select(DetectionJob).where(DetectionJob.id == job_id))
                job = res.scalar_one_or_none()
                if job:
                    job.status    = JobStatus.FAILED
                    job.error_msg = str(exc)
                    job.ended_at  = datetime.now(timezone.utc)
                    await db.commit()
        except Exception as db_exc:
            logger.error(f"[Celery] Failed to persist FAILED status for job={job_id}: {db_exc}")
        raise task.retry(exc=exc)

    finally:
        # ── Temp file cleanup ──────────────────────────────────────────────
        try:
            p = Path(video_path)
            if p.exists():
                p.unlink()
                logger.debug(f"[Celery] Cleaned up temp file: {video_path}")
        except Exception as cleanup_exc:
            logger.warning(f"[Celery] Could not remove temp file {video_path}: {cleanup_exc}")


# ── Singleton ──────────────────────────────────────────────────────────────

_pipeline: Optional[VideoInferencePipeline] = None


def get_video_pipeline() -> VideoInferencePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = VideoInferencePipeline()
    return _pipeline
