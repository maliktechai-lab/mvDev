"""
Model Registry
==============
Manages model loading and caching for all media types.
All models are built entirely from scratch — no pretrained weights.
Models load from checkpoint if available, otherwise start with random weights.
"""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger

from core.config import settings

# ── Lazy-loaded model singletons ───────────────────────────────────────────
_image_model = None
_video_model = None
_audio_model = None
_gradcam     = None


# ── Image model ────────────────────────────────────────────────────────────

def get_image_model():
    global _image_model
    if _image_model is not None:
        return _image_model

    from models.image_model import build_image_model, load_image_model

    device = settings.DEVICE
    ckpt   = settings.IMAGE_MODEL_PATH

    if Path(ckpt).exists():
        logger.info(f"Loading image model from checkpoint: {ckpt}")
        try:
            _image_model = load_image_model(ckpt, device=device)
            logger.success(f"Image model loaded from checkpoint (device={device})")
        except Exception as e:
            logger.warning(f"Checkpoint load failed ({e}), initializing from scratch")
            _image_model = build_image_model(device=device)
    else:
        logger.warning(
            f"No checkpoint at {ckpt}. "
            "Initializing image model from scratch with random weights. "
            "Run backend/training/train_image.py to train."
        )
        _image_model = build_image_model(device=device)

    _image_model.eval()
    return _image_model


# ── Video model ────────────────────────────────────────────────────────────

def get_video_model():
    global _video_model
    if _video_model is not None:
        return _video_model

    from models.video_model import build_video_model, load_video_model

    device = settings.DEVICE
    ckpt   = settings.VIDEO_MODEL_PATH

    if Path(ckpt).exists():
        logger.info(f"Loading video model from checkpoint: {ckpt}")
        try:
            _video_model = load_video_model(ckpt, device=device)
            logger.success(f"Video model loaded from checkpoint (device={device})")
        except Exception as e:
            logger.warning(f"Checkpoint load failed ({e}), initializing from scratch")
            _video_model = build_video_model(device=device)
    else:
        logger.warning(
            f"No checkpoint at {ckpt}. "
            "Initializing video model from scratch with random weights. "
            "Run backend/training/train_video.py to train."
        )
        _video_model = build_video_model(device=device)

    _video_model.eval()
    return _video_model


# ── Audio model ────────────────────────────────────────────────────────────

def get_audio_model():
    global _audio_model
    if _audio_model is not None:
        return _audio_model

    from models.audio_model import build_audio_model, load_audio_model

    device = settings.DEVICE
    ckpt   = settings.AUDIO_MODEL_PATH

    if Path(ckpt).exists():
        logger.info(f"Loading audio model from checkpoint: {ckpt}")
        try:
            _audio_model = load_audio_model(ckpt, device=device)
            logger.success(f"Audio model loaded from checkpoint (device={device})")
        except Exception as e:
            logger.warning(f"Checkpoint load failed ({e}), initializing from scratch")
            _audio_model = build_audio_model(device=device)
    else:
        logger.warning(
            f"No checkpoint at {ckpt}. "
            "Initializing audio model from scratch with random weights. "
            "Run backend/training/train_audio.py to train."
        )
        _audio_model = build_audio_model(device=device)

    _audio_model.eval()
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
    All models are from scratch — no network downloads.
    """
    logger.info("Warming up ML models (all from scratch, no downloads)...")

    # Image model
    try:
        model = get_image_model()
        dummy = torch.zeros(1, 3, 224, 224).to(settings.DEVICE)
        with torch.no_grad():
            _ = model(dummy)
        n_params = sum(p.numel() for p in model.parameters())
        logger.success(f"Image model warmed up — {n_params:,} parameters")
    except Exception as e:
        logger.warning(f"Image model warmup failed (non-fatal): {e}")

    # Video model
    try:
        vmodel = get_video_model()
        # (B=1, T=8, C=3, H=224, W=224)
        dummy_video = torch.zeros(1, 8, 3, 224, 224).to(settings.DEVICE)
        with torch.no_grad():
            _ = vmodel(dummy_video)
        n_params = sum(p.numel() for p in vmodel.parameters())
        logger.success(f"Video model warmed up — {n_params:,} parameters")
    except Exception as e:
        logger.warning(f"Video model warmup failed (non-fatal): {e}")

    # Audio model
    try:
        amodel = get_audio_model()
        T         = 16000 * 3   # 3 seconds at 16kHz
        T_frames  = 188         # mel/mfcc time frames at hop=256
        dummy_wave    = torch.zeros(1, 1, T).to(settings.DEVICE)
        dummy_mel     = torch.zeros(1, 80, T_frames).to(settings.DEVICE)
        dummy_mfcc    = torch.zeros(1, 40, T_frames).to(settings.DEVICE)
        dummy_prosody = torch.zeros(1, 16).to(settings.DEVICE)
        with torch.no_grad():
            _ = amodel(dummy_wave, dummy_mel, dummy_mfcc, dummy_prosody)
        n_params = sum(p.numel() for p in amodel.parameters())
        logger.success(f"Audio model warmed up — {n_params:,} parameters")
    except Exception as e:
        logger.warning(f"Audio model warmup failed (non-fatal): {e}")
