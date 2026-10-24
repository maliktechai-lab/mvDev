"""
Tests for core.model_readiness
==============================
Validates the three checkpoint states (UNAVAILABLE, CORRUPT, READY)
and the require_model_ready() gating logic.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from core.model_readiness import (
    ModelStatus,
    ModelReadiness,
    check_image_model,
    require_model_ready,
    _verify_checkpoint,
)
from core.config import settings


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_readiness(status: ModelStatus, name: str = "test_model") -> ModelReadiness:
    return ModelReadiness(
        name=name,
        status=status,
        path="/fake/path/model.pth",
        message="test",
        error=None if status == ModelStatus.READY else "some error",
    )


# ── 1. UNAVAILABLE when no checkpoint ─────────────────────────────────────

def test_unavailable_when_no_checkpoint(tmp_path):
    """check_image_model() with a nonexistent path → UNAVAILABLE."""
    nonexistent = str(tmp_path / "does_not_exist.pth")

    # Patch the settings path so _verify_checkpoint uses our nonexistent path
    original = settings.IMAGE_MODEL_PATH
    settings.IMAGE_MODEL_PATH = nonexistent
    try:
        result = check_image_model()
        assert result.status == ModelStatus.UNAVAILABLE, (
            f"Expected UNAVAILABLE but got {result.status}: {result.message}"
        )
    finally:
        settings.IMAGE_MODEL_PATH = original


# ── 2. CORRUPT when garbage bytes ─────────────────────────────────────────

def test_corrupt_when_garbage_bytes(corrupt_checkpoint_file, monkeypatch):
    """check_image_model() with a garbage file → CORRUPT."""
    monkeypatch.setattr(settings, "IMAGE_MODEL_PATH", str(corrupt_checkpoint_file))
    result = check_image_model()
    assert result.status == ModelStatus.CORRUPT, (
        f"Expected CORRUPT but got {result.status}: {result.message}"
    )


# ── 3. CORRUPT when wrong keys ────────────────────────────────────────────

def test_corrupt_when_wrong_keys(tmp_path, monkeypatch):
    """Valid torch file but wrong key prefix → CORRUPT."""
    # Create a valid torch file whose keys do NOT start with 'spatial_branch'
    wrong_state_dict = {
        "encoder.layer0.weight": torch.zeros(4, 4),
        "encoder.layer1.bias": torch.zeros(4),
    }
    wrong_ckpt = tmp_path / "wrong_keys.pth"
    torch.save(wrong_state_dict, str(wrong_ckpt))

    monkeypatch.setattr(settings, "IMAGE_MODEL_PATH", str(wrong_ckpt))
    result = check_image_model()
    assert result.status == ModelStatus.CORRUPT, (
        f"Expected CORRUPT (wrong key prefix) but got {result.status}: {result.message}"
    )
    assert "Architecture mismatch" in (result.error or "")


# ── 4. READY when valid checkpoint ────────────────────────────────────────

def test_ready_when_valid_checkpoint(tmp_checkpoint_file, monkeypatch):
    """
    A checkpoint built from the real model's zeroed state_dict → READY.

    The fixture uses build_image_model() to get the exact key set, zeros
    all weights, and saves that.  load_state_dict(strict=False) then has
    0 missing keys, well below the 10 % threshold.
    """
    monkeypatch.setattr(settings, "IMAGE_MODEL_PATH", str(tmp_checkpoint_file))
    result = check_image_model()
    assert result.status == ModelStatus.READY, (
        f"Expected READY but got {result.status}: {result.message}"
    )


# ── 5. require_model_ready raises in prod when UNAVAILABLE ────────────────

def test_require_model_ready_raises_in_production_when_unavailable(mock_settings_prod):
    """UNAVAILABLE + production → RuntimeError."""
    r = _make_readiness(ModelStatus.UNAVAILABLE)
    with pytest.raises(RuntimeError, match="not available in production"):
        require_model_ready(r)


# ── 6. require_model_ready allows dev when UNAVAILABLE ───────────────────

def test_require_model_ready_allows_dev_when_unavailable(mock_settings_dev):
    """UNAVAILABLE + development → no exception raised."""
    r = _make_readiness(ModelStatus.UNAVAILABLE)
    # Should NOT raise — logs a warning instead
    require_model_ready(r)  # passes silently


# ── 7. require_model_ready always raises on CORRUPT ───────────────────────

@pytest.mark.parametrize("env", ["development", "production"])
def test_require_model_ready_always_raises_on_corrupt(env, monkeypatch):
    """CORRUPT checkpoint → RuntimeError in any environment."""
    monkeypatch.setattr(settings, "APP_ENV", env)
    r = _make_readiness(ModelStatus.CORRUPT)
    with pytest.raises(RuntimeError, match="corrupt"):
        require_model_ready(r)
