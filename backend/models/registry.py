"""
Model Registry
==============
Manages model loading and caching for all media types.
Models are loaded once at startup and reused across requests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from loguru import logger

from core.config import settings

# ── Lazy-loaded model references ───────────────────────────────────────────
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
            logger.warning(f"Checkpoint load failed ({e}), using pretrained backbone")
            _image_model = build_image_model(pretrained=True, device=device)
    else:
        logger.warning(
            f"No image checkpoint at {ckpt}. "
            "Using pretrained EfficientNet-B4. Run training/train_image.py to fine-tune."
        )
        _image_model = build_image_model(pretrained=True, device=device)

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
            logger.success(f"Video model loaded (device={device})")
        except Exception as e:
            logger.warning(f"Video checkpoint load failed ({e}), using pretrained backbone")
            _video_model = build_video_model(pretrained=True, device=device)
    else:
        logger.warning(
            f"No video checkpoint at {ckpt}. "
            "Using pretrained EfficientNet backbone. Run training/train_video.py to fine-tune."
        )
        _video_model = build_video_model(pretrained=True, device=device)

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
            logger.success(f"Audio model loaded (device={device})")
        except Exception as e:
            logger.warning(f"Audio checkpoint load failed ({e}), using untrained model")
            _audio_model = build_audio_model(device=device)
    else:
        logger.warning(
            f"No audio checkpoint at {ckpt}. "
            "Using untrained RawNet2. Run training/train_audio.py to fine-tune."
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
    """Pre-load all models at startup to avoid cold-start latency."""
    logger.info("Warming up ML models...")

    # Image model
    try:
        model = get_image_model()
        dummy = torch.zeros(1, 3, 224, 224).to(settings.DEVICE)
        with torch.no_grad():
            _ = model(dummy)
        logger.success("Image model warmed up")
    except Exception as e:
        logger.warning(f"Image model warmup failed (non-fatal): {e}")

    # Video model
    try:
        vmodel = get_video_model()
        # (B=1, T=8, C=3, H=224, W=224)
        dummy_video = torch.zeros(1, 8, 3, 224, 224).to(settings.DEVICE)
        with torch.no_grad():
            _ = vmodel(dummy_video)
        logger.success("Video model warmed up")
    except Exception as e:
        logger.warning(f"Video model warmup failed (non-fatal): {e}")

    # Audio model
    try:
        amodel = get_audio_model()
        T = 16000 * 3   # 3 seconds at 16kHz
        T_frames = 188  # ~3s at hop=256
        dummy_wave    = torch.zeros(1, 1, T).to(settings.DEVICE)
        dummy_mel     = torch.zeros(1, 80, T_frames).to(settings.DEVICE)
        dummy_mfcc    = torch.zeros(1, 40, T_frames).to(settings.DEVICE)
        dummy_prosody = torch.zeros(1, 16).to(settings.DEVICE)
        with torch.no_grad():
            _ = amodel(dummy_wave, dummy_mel, dummy_mfcc, dummy_prosody)
        logger.success("Audio model warmed up")
    except Exception as e:
        logger.warning(f"Audio model warmup failed (non-fatal): {e}")
