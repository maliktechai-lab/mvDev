"""
Tests for api.v1.endpoints.health
===================================
Uses FastAPI TestClient with DB dependency overridden.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from core.config import settings


# ── App factory ────────────────────────────────────────────────────────────

def _make_test_app():
    """
    Build a minimal FastAPI app that includes only the health router,
    with the DB dependency overridden so no real Postgres is needed.
    """
    from fastapi import FastAPI
    from api.v1.endpoints.health import router as health_router
    from db.database import get_db

    app = FastAPI()
    app.include_router(health_router)

    # Override DB to return an AsyncMock session
    async def fake_db():
        session = AsyncMock()
        yield session

    app.dependency_overrides[get_db] = fake_db
    return app


# ── Tests ──────────────────────────────────────────────────────────────────

def test_ping_returns_pong():
    """GET /health/ping → {"ping": "pong"}"""
    app = _make_test_app()
    with TestClient(app) as client:
        resp = client.get("/health/ping")
    assert resp.status_code == 200
    assert resp.json() == {"ping": "pong"}


def test_health_models_returns_correct_schema():
    """GET /health/models → has 'overall', 'models' keys with image/video/audio entries."""
    app = _make_test_app()
    with TestClient(app) as client:
        resp = client.get("/health/models")

    assert resp.status_code == 200
    body = resp.json()
    assert "overall" in body, "Response must have 'overall' key"
    assert "models" in body, "Response must have 'models' key"

    models = body["models"]
    assert "image" in models, "models dict must include 'image'"
    assert "video" in models, "models dict must include 'video'"
    assert "audio" in models, "models dict must include 'audio'"

    # Each model entry should have at least 'status' and 'path'
    for name, entry in models.items():
        assert "status" in entry, f"Model '{name}' missing 'status' field"
        assert "path" in entry, f"Model '{name}' missing 'path' field"


def test_health_models_all_unavailable_without_checkpoints(tmp_path, monkeypatch):
    """With no checkpoints present, overall should be 'unavailable'."""
    nonexistent = str(tmp_path / "nonexistent.pth")
    monkeypatch.setattr(settings, "IMAGE_MODEL_PATH", nonexistent)
    monkeypatch.setattr(settings, "VIDEO_MODEL_PATH", nonexistent)
    monkeypatch.setattr(settings, "AUDIO_MODEL_PATH", nonexistent)

    app = _make_test_app()
    with TestClient(app) as client:
        resp = client.get("/health/models")

    assert resp.status_code == 200
    body = resp.json()
    assert body["overall"] == "unavailable", (
        f"Expected overall='unavailable', got '{body['overall']}'"
    )
    for name, entry in body["models"].items():
        assert entry["status"] == "unavailable", (
            f"Model '{name}' expected 'unavailable', got '{entry['status']}'"
        )
