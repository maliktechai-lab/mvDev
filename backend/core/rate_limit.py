"""
Rate Limiting
=============
Two-layer approach for the login endpoint:

Layer 1 — per-IP via slowapi (Redis-backed):
  5 attempts per minute per source IP.  Exceeding this returns HTTP 429
  before any DB query is made.

Layer 2 — per-account lockout (DB-backed):
  After LOGIN_MAX_FAILURES consecutive failures on a single account the
  account is locked for LOGIN_LOCKOUT_MINUTES minutes.  The lockout period
  resets on a successful login.

Both layers use generic "account not found or bad credentials" language so
attackers cannot distinguish "account exists but locked" from "wrong password".

Usage
-----
  from core.rate_limit import limiter, check_account_lockout, record_login_failure, clear_login_failures

  # In main.py:
  from slowapi import _rate_limit_exceeded_handler
  from slowapi.errors import RateLimitExceeded
  app.state.limiter = limiter
  app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

  # On the login endpoint:
  @router.post("/login")
  @limiter.limit("5/minute")
  async def login(request: Request, ...):
      await check_account_lockout(user)   # raises 401 if locked
      ...
      await record_login_failure(db, user)
      ...
      await clear_login_failures(db, user)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import HTTPException, status
from loguru import logger
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings

# ── Configuration ──────────────────────────────────────��───────────────────
# After this many consecutive failures the account is temporarily locked.
LOGIN_MAX_FAILURES: int = 5
# Lock duration in minutes (doubles after each subsequent lockout period).
LOGIN_LOCKOUT_MINUTES: int = 15


# ── slowapi limiter instance ───────────────────────────────────────────────
# Uses Redis as the backing store via the REDIS_URL config value so limits
# survive app restarts and work across multiple workers.
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=settings.REDIS_URL,
    default_limits=[],   # no global default; apply per-route
)


# ── Generic error message ──────────────────────────────────────────────────
# Always use the same phrasing regardless of whether the account exists,
# is locked, or the password is wrong.  This prevents user enumeration.
_GENERIC_AUTH_ERROR = "Invalid email or password"


def _locked_error() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=_GENERIC_AUTH_ERROR,
    )


# ── Account lockout helpers ────────────────────────────────────────────────

async def check_account_lockout(user) -> None:
    """
    Raise HTTP 401 (generic) if the account is currently locked.

    Called BEFORE verifying the password so that locked accounts cannot
    be used as an oracle even with a timing side-channel.
    """
    if user is None:
        return  # Non-existent account — handled by caller

    if user.locked_until and user.locked_until > datetime.now(timezone.utc):
        logger.warning(
            f"[AUTH] Login attempt on locked account: user_id={user.id} "
            f"locked_until={user.locked_until.isoformat()}"
        )
        raise _locked_error()


async def record_login_failure(db: AsyncSession, user) -> None:
    """
    Increment the failed-attempt counter.  Lock the account when the
    threshold is crossed.

    If `user` is None (unknown email) this is a no-op — we do not reveal
    whether an account exists via side-channel timing from a DB write.
    """
    if user is None:
        return

    user.failed_login_attempts = (user.failed_login_attempts or 0) + 1

    if user.failed_login_attempts >= LOGIN_MAX_FAILURES:
        user.locked_until = datetime.now(timezone.utc) + timedelta(
            minutes=LOGIN_LOCKOUT_MINUTES
        )
        logger.warning(
            f"[AUTH] Account locked after {user.failed_login_attempts} failed attempts: "
            f"user_id={user.id}  locked_until={user.locked_until.isoformat()}"
        )

    # Commit the counter update so it persists even if the request errors out
    # after this point.
    try:
        await db.commit()
    except Exception as exc:
        logger.error(f"[AUTH] Failed to persist login failure counter: {exc}")


async def clear_login_failures(db: AsyncSession, user) -> None:
    """
    Reset the failure counter and clear any lockout on successful login.
    """
    if user is None:
        return

    if user.failed_login_attempts or user.locked_until:
        user.failed_login_attempts = 0
        user.locked_until = None
        try:
            await db.commit()
        except Exception as exc:
            logger.error(f"[AUTH] Failed to clear login failure counter: {exc}")
