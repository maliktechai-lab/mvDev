"""
Tests for core.security (auth helpers)
========================================
Validates password hashing, JWT creation and decoding.
No DB or HTTP client required.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from core.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
)


# ── 1. Password hashing round-trip ────────────────────────────────────────

def test_password_hashing_roundtrip():
    """hash_password then verify_password should return True."""
    plain = "SuperSecret123!"
    hashed = hash_password(plain)

    assert hashed != plain, "Hash must differ from plaintext"
    assert verify_password(plain, hashed) is True
    assert verify_password("WrongPassword", hashed) is False


# ── 2. Create and decode access token ────────────────────────────────────

def test_create_and_decode_access_token():
    """create_access_token → decode_token should return correct sub and type."""
    subject = "user-uuid-1234"
    token = create_access_token(subject)

    payload = decode_token(token)
    assert payload["sub"] == subject
    assert payload["type"] == "access"
    assert "exp" in payload


# ── 3. Create and decode refresh token ───────────────────────────────────

def test_create_and_decode_refresh_token():
    """create_refresh_token → decode_token should return type='refresh'."""
    subject = "user-uuid-5678"
    token = create_refresh_token(subject)

    payload = decode_token(token)
    assert payload["sub"] == subject
    assert payload["type"] == "refresh"
    assert "exp" in payload


# ── 4. Wrong token type rejected ─────────────────────────────────────────

def test_wrong_token_type_rejected():
    """
    A refresh token must not be accepted as an access token.

    The get_current_user dependency rejects tokens where type != 'access'.
    decode_token itself doesn't check type — that's the dependency's job.
    We verify the type field is present and wrong so the endpoint would reject it.
    """
    subject = "user-uuid-9999"
    refresh_token = create_refresh_token(subject)

    # decode_token succeeds (signature is valid)
    payload = decode_token(refresh_token)

    # But the type is 'refresh', not 'access' — endpoint would 401
    assert payload["type"] == "refresh"
    assert payload["type"] != "access", (
        "Refresh token should NOT be accepted as access token (type mismatch)"
    )


# ── 5. Expired / tampered token raises HTTPException ─────────────────────

def test_tampered_token_raises():
    """A token with modified payload should fail signature verification."""
    token = create_access_token("user-abc")
    # Tamper with the token payload section (middle part)
    parts = token.split(".")
    assert len(parts) == 3
    bad_token = parts[0] + ".AAAA_tampered_BBBB." + parts[2]

    with pytest.raises(HTTPException) as exc_info:
        decode_token(bad_token)
    assert exc_info.value.status_code == 401
