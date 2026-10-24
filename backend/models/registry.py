"""
Model Registry
==============
Manages model loading and caching for all media types.

Safety contract:
- In PRODUCTION  (APP_ENV != 'development'):
    Missing or corrupt checkpoint → RuntimeError raised immediately.
    Random-weight inference is NEVER served.

- In DEVELOPMENT (APP_ENV == 'development'):
    Missing checkpoint → model loaded with random weights; every call-site
    that uses the model receives a `random_weights=True` flag on the returned
    model instance so it can tag responses accordingly.
    Corrupt checkpoint → RuntimeError (corrupt weights are never safe, even in dev).

Use `require_model_ready()` from core.model_readiness before any inference call
to enforce these rules at the API boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from loguru import logger

from core.config import settings


# ── Lazy-loaded model singletons ───────────────────────────────────────────
_image_model = None
_video_model = None
_audio_model = None
_gradcam     = None


# ── Helper ─────────────────────────────────────────────────────────────────

def _load_or_raise(
    ckpt_path: str,
    build_fn,
    load_fn,
    model_name: str,
    device: str,
):
    """
    Load a model from checkpoint.

    - READY:       Returns model loaded from checkpoint.
    - UNAVAILABLE (dev): Returns random-weight model with .random_weights=True attribute.
    - UNAVAILABLE (prod): Raises RuntimeError.
    - CORRUPT (any env): Raises RuntimeError.
    """
    p = Path(ckpt_path)

    if not p.exists():
        msg = (
            f"Model '{model_name}': no checkpoint found at '{ckpt_path}'. "
            "Train the model or provide a checkpoint before serving requests. "
            "See README for training instructions."
        )
        if settings.APP_ENV != "development":
            raise RuntimeError(
                f"[PRODUCTION] {msg} "
                "Production inference with untrained models is not permitted."
            )
        logger.warning(
            f"\n{'='*70}\n"
            f"[DEV MODE] {msg}\n"
            f"Initialising '{model_name}' with RANDOM WEIGHTS.\n"
            f"Predictions are MEANINGLESS. Do not use results as real detections.\n"
            f"{'='*70}"
        )
        model = build_fn(device=device)
        model.eval()
        model.random_weights = True  # type: ignore[attr-defined]
        return model

    # File exists — try to load it
    try:
        model = load_fn(ckpt_path, device=device)
        model.eval()
        model.random_weights = False  # type: ignore[attr-defined]
        logger.success(f"Model '{model_name}' loaded from checkpoint: {ckpt_path}")
        return model
    except Exception as exc:
        raise RuntimeError(
            f"Model '{model_name}': checkpoint at '{ckpt_path}' exists but "
            f"could not be loaded. It may be corrupt or from an incompatible "
            f"model version. Error: {exc}"
        ) from exc


# ── Image model ────────────────────────────────────────────────────────────

def get_image_model():
    global _image_model
    if _image_model is not None:
        return _image_model

    from models.image_model import build_image_model, load_image_model

    _image_model = _load_or_raise(
        ckpt_path=settings.IMAGE_MODEL_PATH,
        build_fn=build_image_model,
        load_fn=load_image_model,
        model_name="deepguard-image",
        device=settings.DEVICE,
    )
    return _image_model


# ── Video model ────────────────────────────────────────────────────────────

def get_video_model():
    global _video_model
    if _video_model is not None:
        return _video_model

    from models.video_model import build_video_model, load_video_model

    _video_model = _load_or_raise(
        ckpt_path=settings.VIDEO_MODEL_PATH,
        build_fn=build_video_model,
        load_fn=load_video_model,
        model_name="deepguard-video",
        device=settings.DEVICE,
    )
    return _video_model


# ── Audio model ────────────────────────────────────────────────────────────

def get_audio_model():
    global _audio_model
    if _audio_model is not None:
        return _audio_model

    from models.audio_model import build_audio_model, load_audio_model

    _audio_model = _load_or_raise(
        ckpt_path=settings.AUDIO_MODEL_PATH,
        build_fn=build_audio_model,
        load_fn=load_audio_model,
        model_name="deepguard-audio",
        device=settings.DEVICE,
    )
    return _audio_model


# ── GradCAM ────────────────────────────────────────────────────────────────

def get_gradcam():
    global _gradcam
    if _gradcam is not None:
        return _gradcam

    from utils.gradcam import GradCAMGenerator
    model    = get_image_model()
    _gradcam = GradCAMGenerator(model=model, device=settings.DEVICE)
    logger.info("GradCAM generator initialized")
    return _gradcam


# ── Warmup ─────────────────────────────────────────────────────────────────

def warmup_models():
    """
    Pre-load all models at startup to avoid cold-start latency on first request.

    In development with no checkpoints this will load random-weight models and
    run a dummy forward pass to confirm the architecture is functional.

    In production this will FAIL FAST if any model checkpoint is missing or
    corrupt — the application should not start serving traffic in that state.
    """
    logger.info("Warming up ML models…")
    errors: list[str] = []

    # ── Image model ────────────────────────────────────────────────────────
    try:
        model = get_image_model()
        dummy = torch.zeros(1, 3, 224, 224).to(settings.DEVICE)
        with torch.no_grad():
            _ = model(dummy)
        n_params = sum(p.numel() for p in model.parameters())
        suffix = " [RANDOM WEIGHTS — DEV ONLY]" if getattr(model, "random_weights", False) else ""
        logger.success(f"Image model ready — {n_params:,} params{suffix}")
    except Exception as e:
        msg = f"Image model warmup FAILED: {e}"
        logger.error(msg)
        errors.append(msg)

    # ── Video model ────────────────────────────────────────────────────────
    try:
        vmodel = get_video_model()
        dummy_video = torch.zeros(1, 8, 3, 224, 224).to(settings.DEVICE)
        with torch.no_grad():
            _ = vmodel(dummy_video)
        n_params = sum(p.numel() for p in vmodel.parameters())
        suffix = " [RANDOM WEIGHTS — DEV ONLY]" if getattr(vmodel, "random_weights", False) else ""
        logger.success(f"Video model ready — {n_params:,} params{suffix}")
    except Exception as e:
        msg = f"Video model warmup FAILED: {e}"
        logger.error(msg)
        errors.append(msg)

    # ── Audio model ────────────────────────────────────────────────────────
    try:
        amodel = get_audio_model()
        T         = 16000 * 3
        T_frames  = 188
        dummy_wave    = torch.zeros(1, 1, T).to(settings.DEVICE)
        dummy_mel     = torch.zeros(1, 80, T_frames).to(settings.DEVICE)
        dummy_mfcc    = torch.zeros(1, 40, T_frames).to(settings.DEVICE)
        dummy_prosody = torch.zeros(1, 16).to(settings.DEVICE)
        with torch.no_grad():
            _ = amodel(dummy_wave, dummy_mel, dummy_mfcc, dummy_prosody)
        n_params = sum(p.numel() for p in amodel.parameters())
        suffix = " [RANDOM WEIGHTS — DEV ONLY]" if getattr(amodel, "random_weights", False) else ""
        logger.success(f"Audio model ready — {n_params:,} params{suffix}")
    except Exception as e:
        msg = f"Audio model warmup FAILED: {e}"
        logger.error(msg)
        errors.append(msg)

    if errors:
        if settings.APP_ENV != "development":
            raise RuntimeError(
                f"Production startup aborted — {len(errors)} model(s) failed to load:\n"
                + "\n".join(f"  • {e}" for e in errors)
            )
        else:
            logger.warning(
                f"[DEV] {len(errors)} model(s) unavailable (no checkpoints). "
                "Inference will run on random weights. "
                "This is expected in a fresh development environment."
            )
    else:
        logger.success("All models warmed up and ready.")
