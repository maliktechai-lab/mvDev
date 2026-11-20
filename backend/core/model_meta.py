"""
Model Metadata Helpers
======================
Provides checkpoint fingerprinting and a consistent ``model_status`` block
that is attached to every inference response so callers can always determine
whether a result came from a trained model or random weights.

The checkpoint SHA-256 hash lets operators correlate API results to a specific
model version without exposing weights.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from core.config import settings
from core.model_readiness import ModelStatus, check_image_model, check_video_model, check_audio_model


def _checkpoint_sha256(path: str) -> Optional[str]:
    """Return the first 16 hex chars of the SHA-256 hash of a checkpoint file.

    Returns None if the file does not exist or cannot be read.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except OSError:
        return None


def image_model_status_block() -> dict:
    """
    Return a ``model_status`` dict for inclusion in every image inference response.

    Example output (trained model)::

        {
            "status": "ready",
            "is_trained": true,
            "checkpoint_hash": "a3f9c1d2e4b78f56",
            "environment": "production",
            "warning": null
        }

    Example output (random weights)::

        {
            "status": "dummy",
            "is_trained": false,
            "checkpoint_hash": null,
            "environment": "development",
            "warning": "Model is running on random weights. Results are MEANINGLESS."
        }
    """
    readiness = check_image_model()
    is_trained = readiness.status == ModelStatus.READY and not readiness.is_dummy

    warning: Optional[str] = None
    if readiness.status in (ModelStatus.UNAVAILABLE, ModelStatus.DUMMY):
        warning = (
            "This result was produced by a model running on RANDOM (untrained) weights. "
            "The verdict and scores are MEANINGLESS and must not be used for real detection."
        )
    elif readiness.status == ModelStatus.CORRUPT:
        warning = "Model checkpoint is corrupt. This result may be unreliable."

    return {
        "status": readiness.status.value,
        "is_trained": is_trained,
        "checkpoint_hash": _checkpoint_sha256(settings.IMAGE_MODEL_PATH) if is_trained else None,
        "environment": settings.APP_ENV,
        "warning": warning,
    }


def video_model_status_block() -> dict:
    """Return a ``model_status`` dict for video inference responses."""
    readiness = check_video_model()
    is_trained = readiness.status == ModelStatus.READY and not readiness.is_dummy

    warning: Optional[str] = None
    if readiness.status in (ModelStatus.UNAVAILABLE, ModelStatus.DUMMY):
        warning = (
            "This result was produced by a model running on RANDOM (untrained) weights. "
            "The verdict and scores are MEANINGLESS and must not be used for real detection."
        )

    return {
        "status": readiness.status.value,
        "is_trained": is_trained,
        "checkpoint_hash": _checkpoint_sha256(settings.VIDEO_MODEL_PATH) if is_trained else None,
        "environment": settings.APP_ENV,
        "warning": warning,
    }


def audio_model_status_block() -> dict:
    """Return a ``model_status`` dict for audio inference responses."""
    readiness = check_audio_model()
    is_trained = readiness.status == ModelStatus.READY and not readiness.is_dummy

    warning: Optional[str] = None
    if readiness.status in (ModelStatus.UNAVAILABLE, ModelStatus.DUMMY):
        warning = (
            "This result was produced by a model running on RANDOM (untrained) weights. "
            "The verdict and scores are MEANINGLESS and must not be used for real detection."
        )

    return {
        "status": readiness.status.value,
        "is_trained": is_trained,
        "checkpoint_hash": _checkpoint_sha256(settings.AUDIO_MODEL_PATH) if is_trained else None,
        "environment": settings.APP_ENV,
        "warning": warning,
    }
