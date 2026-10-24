"""
conftest.py — Shared pytest fixtures for DeepGuard test suite.

Sets up sys.path so `backend/` modules are importable, and provides
reusable fixtures for DB mocking, settings patching, image bytes,
and checkpoint files.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# ── sys.path setup ─────────────────────────────────────────────────────────
# pytest.ini already adds `backend` via pythonpath, but add it here too
# so editors and standalone test runs work without pytest.ini in scope.
BACKEND_DIR = str(Path(__file__).parent.parent / "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# ── Environment defaults needed before settings is imported ────────────────
# Pydantic-settings reads from .env — we override key required fields
# so tests can import `core.config` without a real .env present.
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-pytest-only")
os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-secret-key-for-pytest-only")
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://test:test@localhost:5432/testdb",
)
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_BROKER_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
os.environ.setdefault("APP_ENV", "development")


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def mock_db_session():
    """AsyncMock that mimics SQLAlchemy AsyncSession."""
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    session.execute = AsyncMock()
    return session


@pytest.fixture
def mock_settings_dev(monkeypatch):
    """Patch APP_ENV to 'development' in core.config.settings."""
    from core.config import settings
    monkeypatch.setattr(settings, "APP_ENV", "development")
    return settings


@pytest.fixture
def mock_settings_prod(monkeypatch):
    """Patch APP_ENV to 'production' in core.config.settings."""
    from core.config import settings
    monkeypatch.setattr(settings, "APP_ENV", "production")
    return settings


@pytest.fixture
def test_image_bytes() -> bytes:
    """Return raw bytes of a valid 50×50 JPEG image created in-memory."""
    from PIL import Image as PILImage

    img = PILImage.new("RGB", (50, 50), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def test_corrupt_image_bytes() -> bytes:
    """Return obviously non-image bytes."""
    return b"NOT_AN_IMAGE" * 100


@pytest.fixture
def tmp_checkpoint_file(tmp_path) -> Path:
    """
    Write a structurally valid .pth file that passes model_readiness READY checks.

    Strategy: build the real DeepGuardImageModel, zero out all its weights,
    and save that as the checkpoint.  This guarantees:
    - At least one key starts with 'spatial_branch' ✓
    - load_state_dict(strict=False) has 0 missing keys (well under 10 %) ✓
    """
    import torch
    from models.image_model import build_image_model

    model = build_image_model(device="cpu")
    # Zero all parameters so the file is small and deterministic
    state_dict = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}

    checkpoint_path = tmp_path / "deepguard_image.pth"
    torch.save(state_dict, str(checkpoint_path))
    return checkpoint_path


@pytest.fixture
def corrupt_checkpoint_file(tmp_path) -> Path:
    """Write a file containing garbage bytes (not a valid torch checkpoint)."""
    p = tmp_path / "corrupt_model.pth"
    p.write_bytes(b"\x00\x01\x02\x03\xff\xfe\xfd" * 50)
    return p
