"""
Tests for core.config — production safety validation.
======================================================
Validates that validate_production() correctly blocks startup when
dangerous configuration is present and allows safe configuration.
"""

from __future__ import annotations

import pytest


# ── Helpers ────────────────────────────────────────────────────────────────

def _patched_settings(**kwargs):
    """Return a Settings-like object with selected fields overridden."""
    from core.config import Settings
    from unittest.mock import MagicMock, PropertyMock

    # Build a real settings object then override specific attrs
    s = MagicMock(spec=Settings)
    # Safe defaults
    s.APP_ENV = kwargs.get("APP_ENV", "production")
    s.SECRET_KEY = kwargs.get("SECRET_KEY", "a" * 64)
    s.JWT_SECRET_KEY = kwargs.get("JWT_SECRET_KEY", "b" * 64)
    s.DEBUG = kwargs.get("DEBUG", False)
    s.POSTGRES_PASSWORD = kwargs.get("POSTGRES_PASSWORD", "strongpassword123!")
    s.FLOWER_BASIC_AUTH = kwargs.get("FLOWER_BASIC_AUTH", "admin:strongpassword!")
    allowed = kwargs.get("ALLOWED_ORIGINS", "https://app.example.com")
    s.allowed_origins_list = [o.strip() for o in allowed.split(",")]
    # Bind validate_production from the real class
    from core.config import Settings as RealSettings
    s.validate_production = lambda: RealSettings.validate_production(s)
    return s


# ── Development mode — no restrictions ────────────────────────────────────

def test_validate_production_noop_in_development():
    """Development mode skips all checks."""
    s = _patched_settings(
        APP_ENV="development",
        SECRET_KEY="change-this-to-a-very-long-random-secret-key-in-production",
        JWT_SECRET_KEY="change-this-jwt-secret-key-too",
        DEBUG=True,
        POSTGRES_PASSWORD="deepguard_password",
        ALLOWED_ORIGINS="http://localhost:3000",
        FLOWER_BASIC_AUTH="admin:changeme",
    )
    s.validate_production()  # must not raise


# ── Production mode — safe config ─────────────────────────────────────────

def test_validate_production_passes_with_safe_config():
    """All-safe production config should not raise."""
    s = _patched_settings()
    s.validate_production()  # must not raise


# ── Production mode — individual failure cases ────────────────────────────

@pytest.mark.parametrize("secret", [
    "change-this-to-a-very-long-random-secret-key-in-production",
    "changeme",
    "",
])
def test_validate_production_blocks_weak_secret_key(secret):
    """Known placeholder SECRET_KEY in production must raise RuntimeError."""
    s = _patched_settings(SECRET_KEY=secret)
    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        s.validate_production()


@pytest.mark.parametrize("secret", [
    "change-this-jwt-secret-key-too",
    "changeme",
    "",
])
def test_validate_production_blocks_weak_jwt_key(secret):
    """Known placeholder JWT_SECRET_KEY in production must raise RuntimeError."""
    s = _patched_settings(JWT_SECRET_KEY=secret)
    with pytest.raises(RuntimeError, match="JWT_SECRET_KEY"):
        s.validate_production()


def test_validate_production_blocks_debug_true():
    """DEBUG=true in production must raise RuntimeError."""
    s = _patched_settings(DEBUG=True)
    with pytest.raises(RuntimeError, match="DEBUG"):
        s.validate_production()


@pytest.mark.parametrize("pw", ["deepguard_password", "password", "postgres", ""])
def test_validate_production_blocks_weak_db_password(pw):
    """Known weak POSTGRES_PASSWORD in production must raise RuntimeError."""
    s = _patched_settings(POSTGRES_PASSWORD=pw)
    with pytest.raises(RuntimeError, match="POSTGRES_PASSWORD"):
        s.validate_production()


@pytest.mark.parametrize("origins", [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://app.example.com,http://localhost:8000",
])
def test_validate_production_blocks_localhost_origins(origins):
    """localhost/127.0.0.1 in ALLOWED_ORIGINS in production must raise RuntimeError."""
    s = _patched_settings(ALLOWED_ORIGINS=origins)
    with pytest.raises(RuntimeError, match="ALLOWED_ORIGINS"):
        s.validate_production()


def test_validate_production_blocks_default_flower_auth():
    """Default FLOWER_BASIC_AUTH in production must raise RuntimeError."""
    s = _patched_settings(FLOWER_BASIC_AUTH="admin:changeme")
    with pytest.raises(RuntimeError, match="FLOWER_BASIC_AUTH"):
        s.validate_production()


def test_validate_production_reports_all_errors_at_once():
    """All failures should be reported in one RuntimeError, not one-by-one."""
    s = _patched_settings(
        SECRET_KEY="change-this-to-a-very-long-random-secret-key-in-production",
        JWT_SECRET_KEY="change-this-jwt-secret-key-too",
        DEBUG=True,
        POSTGRES_PASSWORD="deepguard_password",
        ALLOWED_ORIGINS="http://localhost:3000",
        FLOWER_BASIC_AUTH="admin:changeme",
    )
    with pytest.raises(RuntimeError) as exc_info:
        s.validate_production()
    msg = str(exc_info.value)
    # All five issues must appear in the single error message
    assert "SECRET_KEY" in msg
    assert "JWT_SECRET_KEY" in msg
    assert "DEBUG" in msg
    assert "POSTGRES_PASSWORD" in msg
    assert "ALLOWED_ORIGINS" in msg
