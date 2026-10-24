"""
Tests for pipelines.image_pipeline and api.v1.endpoints.detect
================================================================
Uses mocked model registry and FastAPI TestClient.
DB and auth dependencies are fully overridden.
"""

from __future__ import annotations

import io
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient

from core.config import settings
from core.model_readiness import ModelReadiness, ModelStatus


# ── Minimal mock model ─────────────────────────────────────────────────────

class _FakeImageModel(torch.nn.Module):
    """
    A tiny stand-in for DeepGuardImageModel that returns the expected
    output dict shape: {"probs": Tensor[B, 2], "branch_scores": {...}}.
    """
    random_weights = False

    def forward(self, x, return_branch_scores=False):
        batch = x.shape[0]
        # Simulate 70 % real, 30 % fake probability
        probs = torch.tensor([[0.7, 0.3]] * batch)
        out = {"probs": probs}
        if return_branch_scores:
            out["branch_scores"] = {
                "spatial": 0.3,
                "frequency": 0.28,
                "attention": 0.31,
            }
        return out


def _fake_model():
    m = _FakeImageModel()
    m.eval()
    return m


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_detect_app():
    """
    Build a FastAPI app with only the detect router mounted.
    Overrides:
    - get_db → AsyncMock session
    - get_current_user → fake User object
    """
    from fastapi import FastAPI
    from api.v1.endpoints.detect import router as detect_router
    from db.database import get_db
    from core.security import get_current_user

    app = FastAPI()
    app.include_router(detect_router)

    # ── Fake DB session ────────────────────────────────────────────────────
    async def fake_db():
        session = AsyncMock()
        # job.id must be accessible after flush+refresh
        fake_job = MagicMock()
        fake_job.id = uuid.uuid4()
        session.flush = AsyncMock()
        session.refresh = AsyncMock(side_effect=lambda obj: setattr(obj, "id", uuid.uuid4()))
        session.add = MagicMock()
        session.commit = AsyncMock()
        yield session

    # ── Fake user ──────────────────────────────────────────────────────────
    fake_user = MagicMock()
    fake_user.id = uuid.uuid4()
    fake_user.email = "test@example.com"
    fake_user.is_active = True

    async def fake_current_user():
        return fake_user

    app.dependency_overrides[get_db] = fake_db
    app.dependency_overrides[get_current_user] = fake_current_user

    return app


def _jpeg_upload_bytes() -> bytes:
    """Create a minimal valid JPEG in memory."""
    from PIL import Image as PILImage

    img = PILImage.new("RGB", (50, 50), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ── 1. Pipeline returns result with expected fields ─────────────────────

def test_pipeline_returns_result_shape(monkeypatch):
    """
    Mock get_image_model() → _FakeImageModel; mock face detection to
    return a simple face; run pipeline.run() with a valid JPEG; verify
    the result has verdict, confidence, face_results fields.
    """
    import pipelines.image_pipeline as pip_mod
    from preprocessing.face_detector import DetectedFace
    from preprocessing.image_preprocessor import ImagePreprocessor

    # Mock the model
    monkeypatch.setattr(pip_mod, "get_image_model", _fake_model)

    # Mock the preprocessor to return one face without real MTCNN
    tensor = torch.zeros(1, 3, 224, 224)
    fake_face = MagicMock(spec=DetectedFace)
    fake_face.crop = MagicMock()
    fake_face.box = (0, 0, 50, 50)

    fake_preprocessor = MagicMock()
    fake_preprocessor.preprocess.return_value = (
        [tensor],      # tensors list
        [fake_face],   # faces list
        MagicMock(),   # original_image
    )

    monkeypatch.setattr(pip_mod, "get_image_preprocessor", lambda device: fake_preprocessor)
    # Disable GradCAM to avoid needing a real model
    monkeypatch.setattr(pip_mod, "get_gradcam", MagicMock(side_effect=Exception("no gradcam")))

    pipeline = pip_mod.ImageInferencePipeline()
    result = pipeline.run(_jpeg_upload_bytes(), generate_heatmap=False)

    assert hasattr(result, "verdict"), "Result missing 'verdict'"
    assert hasattr(result, "confidence"), "Result missing 'confidence'"
    assert hasattr(result, "face_results"), "Result missing 'face_results'"
    assert result.error is None, f"Unexpected error: {result.error}"
    assert result.verdict in ("real", "fake", "unknown")
    assert 0.0 <= result.confidence <= 1.0


# ── 2. Pipeline handles corrupt image ────────────────────────────────────

def test_pipeline_handles_corrupt_image(monkeypatch):
    """
    Running the pipeline on b'garbage' should return a result with
    result.error set (not raise).
    """
    import pipelines.image_pipeline as pip_mod

    # Don't mock model — the pipeline should fail at preprocess stage
    # and catch the exception internally.
    # Reset singleton to avoid cached mocked model from other tests
    pip_mod._pipeline = None

    # We need to ensure get_image_model doesn't load a real model (no checkpoint).
    # In dev mode it returns random weights, which is fine for this test.
    monkeypatch.setattr(settings, "APP_ENV", "development")
    monkeypatch.setattr(settings, "IMAGE_MODEL_PATH", "/nonexistent/model.pth")

    # Clear registry singleton
    import models.registry as reg
    reg._image_model = None

    pipeline = pip_mod.ImageInferencePipeline()
    result = pipeline.run(b"THIS IS NOT A VALID IMAGE")

    # Pipeline wraps exceptions → error field is populated
    assert result.error is not None, "Expected result.error to be set for corrupt input"


# ── 3. Detect endpoint returns 503 when model UNAVAILABLE ────────────────

def test_detect_endpoint_returns_503_when_model_unavailable(monkeypatch, test_image_bytes):
    """POST /detect/image with UNAVAILABLE model in production → 503."""
    # Force production so require_model_ready raises
    monkeypatch.setattr(settings, "APP_ENV", "production")

    unavailable_readiness = ModelReadiness(
        name="deepguard_image",
        status=ModelStatus.UNAVAILABLE,
        path="/nonexistent.pth",
        message="No checkpoint",
    )

    with patch("api.v1.endpoints.detect.check_image_model", return_value=unavailable_readiness):
        app = _make_detect_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/detect/image",
                files={"file": ("test.jpg", test_image_bytes, "image/jpeg")},
            )

    assert resp.status_code == 503, (
        f"Expected 503 for unavailable model in production, got {resp.status_code}"
    )


# ── 4. Detect endpoint rejects invalid MIME type ──────────────────────────

def test_detect_endpoint_rejects_invalid_mime(monkeypatch, test_image_bytes):
    """POST with content_type='text/html' → 415."""
    # Ensure model gate doesn't block first (use dev mode + mock ready model)
    monkeypatch.setattr(settings, "APP_ENV", "development")

    ready_readiness = ModelReadiness(
        name="deepguard_image",
        status=ModelStatus.READY,
        path="/fake.pth",
        message="Ready",
        params=1000,
    )

    with patch("api.v1.endpoints.detect.check_image_model", return_value=ready_readiness):
        app = _make_detect_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/detect/image",
                files={"file": ("test.html", test_image_bytes, "text/html")},
            )

    assert resp.status_code == 415, (
        f"Expected 415 for text/html MIME, got {resp.status_code}"
    )


# ── 5. Detect endpoint rejects magic bytes mismatch ──────────────────────

def test_detect_endpoint_rejects_magic_bytes_mismatch(monkeypatch):
    """POST with MIME=image/jpeg but fake content → 415."""
    monkeypatch.setattr(settings, "APP_ENV", "development")

    ready_readiness = ModelReadiness(
        name="deepguard_image",
        status=ModelStatus.READY,
        path="/fake.pth",
        message="Ready",
        params=1000,
    )

    fake_content = b"NOT_REAL_IMAGE_CONTENT_AT_ALL" * 100

    with patch("api.v1.endpoints.detect.check_image_model", return_value=ready_readiness):
        app = _make_detect_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/detect/image",
                files={"file": ("fake.jpg", fake_content, "image/jpeg")},
            )

    assert resp.status_code == 415, (
        f"Expected 415 for magic byte mismatch, got {resp.status_code}: {resp.text}"
    )


# ── 6. Detect endpoint rejects oversized file ─────────────────────────────

def test_detect_endpoint_rejects_oversized_file(monkeypatch, test_image_bytes):
    """A file exceeding MAX_IMAGE_SIZE_MB → 413."""
    monkeypatch.setattr(settings, "APP_ENV", "development")
    # Override max size to 1 byte so our test image triggers the limit
    monkeypatch.setattr(settings, "MAX_IMAGE_SIZE_MB", 0)

    ready_readiness = ModelReadiness(
        name="deepguard_image",
        status=ModelStatus.READY,
        path="/fake.pth",
        message="Ready",
        params=1000,
    )

    with patch("api.v1.endpoints.detect.check_image_model", return_value=ready_readiness):
        app = _make_detect_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/detect/image",
                files={"file": ("test.jpg", test_image_bytes, "image/jpeg")},
            )

    assert resp.status_code == 413, (
        f"Expected 413 for oversized file, got {resp.status_code}: {resp.text}"
    )
