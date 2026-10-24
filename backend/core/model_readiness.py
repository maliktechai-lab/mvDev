"""
Model Readiness
===============
Tracks whether each model has a valid, loadable checkpoint.

Rules:
- UNAVAILABLE  → no checkpoint file exists at the configured path.
- CORRUPT      → file exists but torch.load fails or keys are wrong.
- READY        → file loaded successfully and architecture matches.

In production (APP_ENV != "development"), inference endpoints MUST
refuse requests when the required model is not READY.  They must never
silently serve predictions from randomly-initialised weights.

In development mode a UNAVAILABLE/CORRUPT model is allowed to run
on random weights, but every response is tagged with a clear warning.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

from core.config import settings


class ModelStatus(str, enum.Enum):
    UNAVAILABLE = "unavailable"   # checkpoint file missing
    CORRUPT     = "corrupt"       # file exists but cannot be loaded / wrong keys
    READY       = "ready"         # checkpoint loaded and verified


@dataclass
class ModelReadiness:
    name:    str
    status:  ModelStatus
    path:    str
    message: str
    params:  Optional[int] = None
    error:   Optional[str] = None


def _verify_checkpoint(path: str, model_builder, expected_key_prefix: str) -> ModelReadiness:
    """
    Try to load a checkpoint and verify it is compatible with the model
    architecture. Returns a ModelReadiness record.
    """
    p = Path(path)
    name = p.stem

    if not p.exists():
        return ModelReadiness(
            name=name,
            status=ModelStatus.UNAVAILABLE,
            path=path,
            message=f"Checkpoint not found: {path}. Train the model first.",
        )

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)

        # Accept both raw state_dict and trainer-style {"model_state_dict": ...}
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt):
            # Looks like a raw state_dict
            state_dict = ckpt
        else:
            return ModelReadiness(
                name=name,
                status=ModelStatus.CORRUPT,
                path=path,
                message="Checkpoint format unrecognised — expected state_dict or {model_state_dict: ...}",
                error="Unexpected checkpoint structure",
            )

        # Spot-check: at least one key must start with expected_key_prefix
        keys = list(state_dict.keys())
        if not any(k.startswith(expected_key_prefix) for k in keys):
            return ModelReadiness(
                name=name,
                status=ModelStatus.CORRUPT,
                path=path,
                message=(
                    f"Checkpoint keys do not match expected architecture "
                    f"(expected prefix '{expected_key_prefix}', got: {keys[:3]})"
                ),
                error="Architecture mismatch",
            )

        # Try actually loading the weights into the model
        model = model_builder(device="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if len(missing) > len(state_dict) * 0.1:
            # More than 10 % of keys missing — likely wrong model version
            return ModelReadiness(
                name=name,
                status=ModelStatus.CORRUPT,
                path=path,
                message=f"Too many missing keys ({len(missing)}) — checkpoint may be from a different model version.",
                error=f"Missing keys: {missing[:5]}",
            )

        n_params = sum(p.numel() for p in model.parameters())
        return ModelReadiness(
            name=name,
            status=ModelStatus.READY,
            path=path,
            message=f"Loaded successfully ({n_params:,} parameters).",
            params=n_params,
        )

    except Exception as exc:
        return ModelReadiness(
            name=name,
            status=ModelStatus.CORRUPT,
            path=path,
            message=f"Failed to load checkpoint: {exc}",
            error=str(exc),
        )


# ── Public API ─────────────────────────────────────────────────────────────

def check_image_model() -> ModelReadiness:
    from models.image_model import build_image_model
    return _verify_checkpoint(
        settings.IMAGE_MODEL_PATH,
        build_image_model,
        expected_key_prefix="spatial_branch",
    )


def check_video_model() -> ModelReadiness:
    from models.video_model import build_video_model
    return _verify_checkpoint(
        settings.VIDEO_MODEL_PATH,
        build_video_model,
        expected_key_prefix="spatial_branch",
    )


def check_audio_model() -> ModelReadiness:
    from models.audio_model import build_audio_model
    return _verify_checkpoint(
        settings.AUDIO_MODEL_PATH,
        build_audio_model,
        expected_key_prefix="raw_branch",
    )


def get_all_model_readiness() -> dict[str, ModelReadiness]:
    return {
        "image": check_image_model(),
        "video": check_video_model(),
        "audio": check_audio_model(),
    }


def require_model_ready(readiness: ModelReadiness) -> None:
    """
    Called before inference. Raises RuntimeError if:
    - status is UNAVAILABLE or CORRUPT in production
    - status is CORRUPT in any environment (corrupt weights are never safe to use)

    In development + UNAVAILABLE, logs a loud warning but allows random-weight
    inference so the pipeline can be tested without trained weights.
    """
    if readiness.status == ModelStatus.READY:
        return

    if readiness.status == ModelStatus.CORRUPT:
        raise RuntimeError(
            f"Model '{readiness.name}' checkpoint is corrupt and cannot be used. "
            f"Detail: {readiness.message}"
        )

    # UNAVAILABLE
    if settings.APP_ENV != "development":
        raise RuntimeError(
            f"Model '{readiness.name}' is not available in production. "
            f"No trained checkpoint found at '{readiness.path}'. "
            "Train the model or provide a checkpoint before serving requests."
        )

    # Development + UNAVAILABLE: warn loudly but allow random-weight inference
    logger.warning(
        f"[DEV MODE] Model '{readiness.name}' has no trained checkpoint. "
        "Running with RANDOM WEIGHTS — predictions are meaningless. "
        "Do not interpret these results as real deepfake detection."
    )
