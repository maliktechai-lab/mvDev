"""
Tests for models.registry
==========================
Validates get_image_model() and warmup_models() behaviour in dev / prod.
"""

from __future__ import annotations

import pytest

from core.config import settings


def _reset_registry():
    """Clear the registry's module-level model singletons between tests."""
    import models.registry as reg
    reg._image_model = None
    reg._video_model = None
    reg._audio_model = None
    reg._gradcam     = None


# ── 1. get_image_model raises in production when no checkpoint ─────────────

def test_get_image_model_raises_in_production_when_no_checkpoint(
    tmp_path, monkeypatch
):
    """Production + missing checkpoint → RuntimeError."""
    _reset_registry()
    nonexistent = str(tmp_path / "no_model.pth")
    monkeypatch.setattr(settings, "APP_ENV", "production")
    monkeypatch.setattr(settings, "IMAGE_MODEL_PATH", nonexistent)

    from models.registry import get_image_model
    with pytest.raises(RuntimeError, match="PRODUCTION"):
        get_image_model()


# ── 2. get_image_model returns random-weight model in dev ─────────────────

def test_get_image_model_returns_random_weights_in_dev(tmp_path, monkeypatch):
    """Development + missing checkpoint → model with random_weights=True."""
    _reset_registry()
    nonexistent = str(tmp_path / "no_model_dev.pth")
    monkeypatch.setattr(settings, "APP_ENV", "development")
    monkeypatch.setattr(settings, "IMAGE_MODEL_PATH", nonexistent)

    from models.registry import get_image_model
    model = get_image_model()
    assert getattr(model, "random_weights", False) is True, (
        "Expected model.random_weights == True when no checkpoint in dev mode"
    )


# ── 3. warmup_models logs error but doesn't raise in dev ──────────────────

def test_warmup_models_logs_error_not_raises_in_dev(tmp_path, monkeypatch, caplog):
    """Dev mode + no checkpoints: warmup_models() must not raise."""
    _reset_registry()
    nonexistent = str(tmp_path / "none.pth")
    monkeypatch.setattr(settings, "APP_ENV", "development")
    monkeypatch.setattr(settings, "IMAGE_MODEL_PATH", nonexistent)
    monkeypatch.setattr(settings, "VIDEO_MODEL_PATH", nonexistent)
    monkeypatch.setattr(settings, "AUDIO_MODEL_PATH", nonexistent)

    from models.registry import warmup_models

    # Should complete without raising even though no checkpoints exist
    warmup_models()


# ── 4. warmup_models raises in production ─────────────────────────────────

def test_warmup_models_raises_in_production(tmp_path, monkeypatch):
    """Production + no checkpoints: warmup_models() raises RuntimeError."""
    _reset_registry()
    nonexistent = str(tmp_path / "none.pth")
    monkeypatch.setattr(settings, "APP_ENV", "production")
    monkeypatch.setattr(settings, "IMAGE_MODEL_PATH", nonexistent)
    monkeypatch.setattr(settings, "VIDEO_MODEL_PATH", nonexistent)
    monkeypatch.setattr(settings, "AUDIO_MODEL_PATH", nonexistent)

    from models.registry import warmup_models

    with pytest.raises(RuntimeError, match="Production startup aborted"):
        warmup_models()
