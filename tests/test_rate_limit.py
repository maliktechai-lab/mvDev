"""
Tests for core.rate_limit — account lockout helpers.
======================================================
Validates the per-account brute-force protection layer independently of the
HTTP layer.  The slowapi per-IP layer is tested via integration in
test_login_lockout.py.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.rate_limit import (
    LOGIN_LOCKOUT_MINUTES,
    LOGIN_MAX_FAILURES,
    check_account_lockout,
    clear_login_failures,
    record_login_failure,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_user(failed: int = 0, locked_until=None):
    u = MagicMock()
    u.id = "test-user-id"
    u.failed_login_attempts = failed
    u.locked_until = locked_until
    return u


def _make_db():
    db = AsyncMock()
    db.commit = AsyncMock()
    return db


# ── check_account_lockout ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_check_lockout_passes_when_no_lockout():
    """Account with no lockout should not raise."""
    user = _make_user()
    await check_account_lockout(user)  # must not raise


@pytest.mark.asyncio
async def test_check_lockout_passes_when_lockout_expired():
    """Account with an expired lockout should not raise."""
    past = datetime.now(timezone.utc) - timedelta(minutes=1)
    user = _make_user(locked_until=past)
    await check_account_lockout(user)  # must not raise


@pytest.mark.asyncio
async def test_check_lockout_raises_when_locked():
    """Account with an active lockout should raise HTTP 401."""
    from fastapi import HTTPException
    future = datetime.now(timezone.utc) + timedelta(minutes=10)
    user = _make_user(locked_until=future)
    with pytest.raises(HTTPException) as exc_info:
        await check_account_lockout(user)
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_check_lockout_noop_for_none_user():
    """None user (email not found) should not raise."""
    await check_account_lockout(None)  # must not raise


# ── record_login_failure ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_record_failure_increments_counter():
    """Each failure increments failed_login_attempts by 1."""
    user = _make_user(failed=2)
    db = _make_db()
    await record_login_failure(db, user)
    assert user.failed_login_attempts == 3


@pytest.mark.asyncio
async def test_record_failure_locks_at_threshold():
    """Reaching LOGIN_MAX_FAILURES sets locked_until."""
    user = _make_user(failed=LOGIN_MAX_FAILURES - 1)
    db = _make_db()
    await record_login_failure(db, user)
    assert user.failed_login_attempts == LOGIN_MAX_FAILURES
    assert user.locked_until is not None
    assert user.locked_until > datetime.now(timezone.utc)


@pytest.mark.asyncio
async def test_record_failure_noop_for_none_user():
    """None user should not raise or write to DB."""
    db = _make_db()
    await record_login_failure(db, None)
    db.commit.assert_not_called()


# ── clear_login_failures ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_clear_failures_resets_counter_and_lockout():
    """Successful login clears counter and locked_until."""
    future = datetime.now(timezone.utc) + timedelta(minutes=5)
    user = _make_user(failed=3, locked_until=future)
    db = _make_db()
    await clear_login_failures(db, user)
    assert user.failed_login_attempts == 0
    assert user.locked_until is None
    db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_clear_failures_noop_when_no_failures():
    """No DB write when counter is already 0."""
    user = _make_user(failed=0, locked_until=None)
    db = _make_db()
    await clear_login_failures(db, user)
    db.commit.assert_not_called()
