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
    """Celery task: run audio detection and update job in DB."""
    import asyncio
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from sqlalchemy import select
    from db.models import DetectionJob, DetectionResult, JobStatus, Verdict

    logger.info(f"[Celery] Starting audio detection for job={job_id}")

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
                verdict_str  = result_kwargs.get("verdict", "unknown")
                verdict_enum = Verdict[verdict_str.upper()] if verdict_str in ("real", "fake") else Verdict.UNKNOWN
                db_result = DetectionResult(
                    job_id=job.id,
                    verdict=verdict_enum,
                    confidence=result_kwargs.get("confidence", 0.0),
                    branch_scores=result_kwargs.get("branch_scores"),
                    chunk_scores=result_kwargs.get("chunk_scores"),
                    duration_sec=result_kwargs.get("duration_sec"),
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

        pipeline = AudioInferencePipeline()
        result   = pipeline.run(audio_path)

        if result.error:
            raise RuntimeError(result.error)

        asyncio.run(update_db(
            JobStatus.COMPLETED,
            verdict=result.verdict,
            confidence=result.confidence,
            fake_probability=result.fake_probability,
            real_probability=result.real_probability,
            branch_scores=result.branch_scores,
            chunk_scores=result.chunk_scores,
            duration_sec=result.duration_sec,
            suspicious_timestamps=result.suspicious_timestamps,
        ))

        return {"job_id": job_id, "verdict": result.verdict}

    except Exception as exc:
        logger.exception(f"[Celery] Audio detection failed for job={job_id}: {exc}")
        asyncio.run(update_db(JobStatus.FAILED))
        raise self.retry(exc=exc)


# ── Singleton ──────────────────────────────────────────────────────────────

_pipeline: Optional[AudioInferencePipeline] = None


def get_audio_pipeline() -> AudioInferencePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = AudioInferencePipeline()
    return _pipeline
