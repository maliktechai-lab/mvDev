"""
Audio Inference Pipeline
=========================
Full detection flow for an audio file:

  1. Load & validate audio
  2. Resample to 16kHz mono
  3. Segment into 3-second overlapping chunks
  4. Extract: waveform, mel-spec, MFCC, prosody per chunk
  5. Run DeepGuard-Audio model on each chunk
  6. Aggregate chunk scores → overall verdict
  7. Flag suspicious segment timestamps

Runs as Celery async task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from loguru import logger

from core.celery_app import celery_app
from core.config import settings
from preprocessing.audio_preprocessor import get_audio_preprocessor, AudioChunk


# ── Result types ───────────────────────────────────────────────────────────

@dataclass
class ChunkResult:
    chunk_index: int
    start_sec: float
    end_sec: float
    fake_probability: float
    verdict: str


@dataclass
class AudioDetectionResult:
    verdict: str
    confidence: float
    fake_probability: float
    real_probability: float
    duration_sec: float
    chunks_analyzed: int
    chunk_scores: List[Dict]
    branch_scores: Dict
    suspicious_timestamps: List[float]
    error: Optional[str] = None


# ── Pipeline ───────────────────────────────────────────────────────────────

class AudioInferencePipeline:
    """Stateless audio inference pipeline."""

    def __init__(self):
        self.device = settings.DEVICE

    def run(self, audio_path: str) -> AudioDetectionResult:
        try:
            return self._run_pipeline(audio_path)
        except Exception as e:
            logger.exception(f"Audio pipeline error: {e}")
            return AudioDetectionResult(
                verdict="unknown", confidence=0.0,
                fake_probability=0.0, real_probability=0.0,
                duration_sec=0.0, chunks_analyzed=0,
                chunk_scores=[], branch_scores={},
                suspicious_timestamps=[], error=str(e),
            )

    def _run_pipeline(self, audio_path: str) -> AudioDetectionResult:
        from models.registry import get_audio_model

        model = get_audio_model()

        # ── 1. Preprocess ──────────────────────────────────────────────────
        preprocessor = get_audio_preprocessor(self.device)
        chunks, duration_sec = preprocessor.preprocess(audio_path)

        if not chunks:
            raise ValueError("No audio chunks extracted")

        logger.info(f"Audio: {duration_sec:.1f}s → {len(chunks)} chunks")

        # ── 2. Run model per chunk ─────────────────────────────────────────
        chunk_results: List[ChunkResult] = []
        all_branch_scores: List[Dict] = []

        for chunk in chunks:
            waveform = chunk.waveform.unsqueeze(0).to(self.device)       # (1, 1, T)
            mel_spec = chunk.mel_spec.unsqueeze(0).to(self.device)       # (1, n_mels, T)
            mfcc     = chunk.mfcc.unsqueeze(0).to(self.device)           # (1, n_mfcc, T)
            prosody  = chunk.prosody.unsqueeze(0).to(self.device)        # (1, 16)

            with torch.no_grad():
                out = model(waveform, mel_spec, mfcc, prosody, return_branch_scores=True)

            fake_prob   = float(out["probs"][0, 1])
            branch_sc   = out.get("branch_scores", {})

            chunk_results.append(ChunkResult(
                chunk_index=chunk.chunk_index,
                start_sec=chunk.start_sec,
                end_sec=chunk.end_sec,
                fake_probability=fake_prob,
                verdict="fake" if fake_prob >= 0.5 else "real",
            ))
            all_branch_scores.append(branch_sc)

        # ── 3. Aggregate ───────────────────────────────────────────────────
        fake_probs      = [r.fake_probability for r in chunk_results]
        final_fake_prob = float(np.mean(fake_probs))
        verdict         = "fake" if final_fake_prob >= settings.IMAGE_CONFIDENCE_THRESHOLD else "real"
        confidence      = max(final_fake_prob, 1.0 - final_fake_prob)

        # Average branch scores
        avg_branch = {}
        if all_branch_scores:
            for k in all_branch_scores[0].keys():
                vals = [d[k] for d in all_branch_scores if k in d]
                avg_branch[k] = float(np.mean(vals)) if vals else 0.0

        # Chunk score list for DB
        chunk_score_list = [
            {
                "chunk": r.chunk_index,
                "score": round(r.fake_probability, 4),
                "start": round(r.start_sec, 2),
                "end":   round(r.end_sec, 2),
            }
            for r in chunk_results
        ]

        # Suspicious timestamps
        suspicious = [r.start_sec for r in chunk_results if r.fake_probability > 0.7]

        logger.info(
            f"Audio detection: verdict={verdict} "
            f"fake_prob={final_fake_prob:.3f} chunks={len(chunks)} "
            f"suspicious={len(suspicious)}"
        )

        return AudioDetectionResult(
            verdict=verdict,
            confidence=confidence,
            fake_probability=final_fake_prob,
            real_probability=1.0 - final_fake_prob,
            duration_sec=duration_sec,
            chunks_analyzed=len(chunks),
            chunk_scores=chunk_score_list,
            branch_scores=avg_branch,
            suspicious_timestamps=suspicious,
        )


# ── Celery task ────────────────────────────────────────────────────────────

@celery_app.task(
    name="pipelines.audio_pipeline.run_audio_detection",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
)
def run_audio_detection(self, job_id: str, audio_path: str) -> dict:
    """
    Celery task: run audio detection and update job status in DB.

    Uses asyncio.run + shared AsyncSessionLocal (not create_async_engine per call).
    Enforces model readiness gate before any heavy work.
    Cleans up the uploaded audio file in a finally block.
    """
    import asyncio
    return asyncio.run(_run_audio_detection_async(self, job_id, audio_path))


async def _run_audio_detection_async(task, job_id: str, audio_path: str) -> dict:
    """Async implementation of the audio Celery task."""
    from sqlalchemy import select
    from db.database import AsyncSessionLocal
    from db.models import DetectionJob, DetectionResult, JobStatus, Verdict
    from core.model_readiness import check_audio_model, require_model_ready

    logger.info(f"[Celery/audio] Starting detection for job={job_id}")

    # ── Model readiness gate — fail fast ───────────────────────────────────
    try:
        readiness = check_audio_model()
        require_model_ready(readiness)
    except RuntimeError as exc:
        logger.error(f"[Celery/audio] Model not ready for job={job_id}: {exc}")
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
        # ── Mark RUNNING ───────────────────────────────────────────────────
        async with AsyncSessionLocal() as db:
            res = await db.execute(select(DetectionJob).where(DetectionJob.id == job_id))
            job = res.scalar_one_or_none()
            if not job:
                logger.error(f"[Celery/audio] Job {job_id} not found")
                return {"job_id": job_id, "error": "job not found"}
            job.status     = JobStatus.RUNNING
            job.started_at = datetime.now(timezone.utc)
            await db.commit()

        # ── Run inference ──────────────────────────────────────────────────
        pipeline = AudioInferencePipeline()
        result   = pipeline.run(audio_path)

        if result.error:
            raise RuntimeError(result.error)

        # ── Persist result ─────────────────────────────────────────────────
        async with AsyncSessionLocal() as db:
            res = await db.execute(select(DetectionJob).where(DetectionJob.id == job_id))
            job = res.scalar_one_or_none()
            if not job:
                logger.error(f"[Celery/audio] Job {job_id} disappeared before result save")
                return {"job_id": job_id, "error": "job not found after inference"}

            verdict_str  = result.verdict
            verdict_enum = Verdict[verdict_str.upper()] if verdict_str in ("real", "fake") else Verdict.UNKNOWN

            db_result = DetectionResult(
                job_id=job.id,
                verdict=verdict_enum,
                confidence=result.confidence,
                branch_scores=result.branch_scores,
                chunk_scores=result.chunk_scores,
                duration_sec=result.duration_sec,
                raw_output={
                    "fake_probability":     result.fake_probability,
                    "real_probability":     result.real_probability,
                    "suspicious_timestamps": result.suspicious_timestamps,
                },
            )
            db.add(db_result)
            job.status   = JobStatus.COMPLETED
            job.ended_at = datetime.now(timezone.utc)
            await db.commit()

        logger.info(f"[Celery/audio] Detection complete for job={job_id}")
        return {"job_id": job_id, "verdict": result.verdict, "confidence": result.confidence}

    except Exception as exc:
        logger.exception(f"[Celery/audio] Detection failed for job={job_id}: {exc}")
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
            logger.error(f"[Celery/audio] Failed to persist FAILED status: {db_exc}")
        raise task.retry(exc=exc)

    finally:
        # ── Clean up temp audio file ───────────────────────────────────────
        try:
            p = Path(audio_path)
            if p.exists():
                p.unlink()
                logger.debug(f"[Celery/audio] Cleaned up: {audio_path}")
        except Exception as cleanup_exc:
            logger.warning(f"[Celery/audio] Could not remove temp file {audio_path}: {cleanup_exc}")


# ── Singleton ──────────────────────────────────────────────────────────────

_pipeline: Optional[AudioInferencePipeline] = None


def get_audio_pipeline() -> AudioInferencePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = AudioInferencePipeline()
    return _pipeline
