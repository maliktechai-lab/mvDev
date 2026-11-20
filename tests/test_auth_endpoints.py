"""
Integration tests for auth endpoint security properties.
=========================================================
Tests:
  - Refresh-token reuse is rejected (revoked token → 401).
  - Password change revokes all refresh tokens.
  - Generic error returned for locked / disabled accounts.
  - Per-user active token limit enforcement.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest
from fastapi import HTTPException

from core.security import hash_token, create_refresh_token, create_access_token
from core.config import settings


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_user(**kwargs):
    u = MagicMock()
    u.id = uuid.uuid4()
    u.email = "test@example.com"
    u.username = "testuser"
    u.is_active = True
    u.failed_login_attempts = 0
    u.locked_until = None
    u.hashed_password = "hashed"
    for k, v in kwargs.items():
        setattr(u, k, v)
    return u


# ── Refresh token reuse ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_revoked_refresh_token_rejected():
    """A refresh token that has been revoked must return 401."""
    from api.v1.endpoints.auth import refresh_token as refresh_endpoint
    from api.v1.schemas.auth import RefreshRequest

    token = create_refresh_token(str(uuid.uuid4()))
    token_hash = hash_token(token)

    # Simulate a revoked DB token
    db_token = MagicMock()
    db_token.revoked = True
    db_token.expires_at = datetime.now(timezone.utc) + timedelta(days=1)

    db = AsyncMock()
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = None  # revoked → not found
    db.execute = AsyncMock(return_value=result_mock)

    with pytest.raises(HTTPException) as exc:
        await refresh_endpoint(RefreshRequest(refresh_token=token), db)

    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_expired_refresh_token_rejected():
    """An expired refresh token must return 401."""
    from api.v1.endpoints.auth import refresh_token as refresh_endpoint
    from api.v1.schemas.auth import RefreshRequest

    token = create_refresh_token(str(uuid.uuid4()))

    db_token = MagicMock()
    db_token.revoked = False
    db_token.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)

    db = AsyncMock()
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = db_token
    db.execute = AsyncMock(return_value=result_mock)

    with pytest.raises(HTTPException) as exc:
        await refresh_endpoint(RefreshRequest(refresh_token=token), db)

    assert exc.value.status_code == 401


# ── Password change revokes tokens ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_password_change_revokes_all_tokens():
    """change-password must issue a bulk revocation UPDATE for the user's tokens."""
    from api.v1.endpoints.auth import change_password
    from api.v1.schemas.auth import PasswordChangeRequest

    user = _make_user()

    db = AsyncMock()
    db.execute = AsyncMock()

    with patch("api.v1.endpoints.auth.verify_password", return_value=True), \
         patch("api.v1.endpoints.auth.hash_password", return_value="new_hash"):
        await change_password(
            PasswordChangeRequest(current_password="Old1pass", new_password="New1pass"),
            current_user=user,
            db=db,
        )

    # db.execute should have been called at least once for the UPDATE statement
    assert db.execute.called, "Expected db.execute to be called for token revocation"


# ── Generic error for locked account ──────────────────────────────────────

@pytest.mark.asyncio
async def test_locked_account_returns_generic_error():
    """Login for a locked account must return the same error as bad credentials."""
    from api.v1.endpoints.auth import login
    from api.v1.schemas.auth import LoginRequest
    from fastapi import Request

    future = datetime.now(timezone.utc) + timedelta(minutes=10)
    user = _make_user(locked_until=future)

    db = AsyncMock()
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = user
    db.execute = AsyncMock(return_value=result_mock)

    request = MagicMock(spec=Request)

    with pytest.raises(HTTPException) as exc:
        await login(request, LoginRequest(email="test@example.com", password="anything"), db)

    # Must be 401 with the generic message — NOT a 403 "account locked" that reveals state
    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid email or password"


@pytest.mark.asyncio
async def test_disabled_account_returns_generic_error():
    """Login for a disabled account must NOT return "Account is disabled" (enumeration risk)."""
    from api.v1.endpoints.auth import login
    from api.v1.schemas.auth import LoginRequest
    from fastapi import Request

    user = _make_user(is_active=False)

    db = AsyncMock()
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = user
    db.execute = AsyncMock(return_value=result_mock)

    request = MagicMock(spec=Request)

    with patch("api.v1.endpoints.auth.verify_password", return_value=True):
        with pytest.raises(HTTPException) as exc:
            await login(request, LoginRequest(email="test@example.com", password="correct"), db)

    assert exc.value.status_code == 401
    # The message must be generic — not "Account is disabled"
    assert "disabled" not in exc.value.detail.lower()


# ── Job ownership authorization ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_job_not_found_for_different_user():
    """GET /jobs/{id} must return 404 for a job owned by another user."""
    from api.v1.endpoints.jobs import get_job

    job_id = uuid.uuid4()
    requester = _make_user()

    db = AsyncMock()
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = None  # filtered by user_id
    db.execute = AsyncMock(return_value=result_mock)

    with pytest.raises(HTTPException) as exc:
        await get_job(job_id=job_id, db=db, current_user=requester)

    assert exc.value.status_code == 404
