from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from db.database import get_db
from db.models import User, RefreshToken
from core.security import (
    hash_password, verify_password,
    create_access_token, create_refresh_token,
    decode_token, hash_token, get_current_user,
)
from core.config import settings
from core.rate_limit import limiter, check_account_lockout, record_login_failure, clear_login_failures
from api.v1.schemas.auth import (
    UserRegister, UserResponse, LoginRequest,
    TokenResponse, RefreshRequest, PasswordChangeRequest,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


# ── Register ───────────────────────────────────────────────────────────────

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(payload: UserRegister, db: AsyncSession = Depends(get_db)):
    # Check email uniqueness
    result = await db.execute(select(User).where(User.email == payload.email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    # Check username uniqueness
    result = await db.execute(select(User).where(User.username == payload.username))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already taken")

    user = User(
        email=payload.email,
        username=payload.username,
        full_name=payload.full_name,
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    await db.flush()
    await db.refresh(user)
    return user


# ── Login ──────────────────────────────────────────────────────────────────

@router.post("/login", response_model=TokenResponse)
@limiter.limit("5/minute")
async def login(request: Request, payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    """
    Authenticate and issue a JWT access + refresh token pair.

    Rate limiting:
      - Layer 1 (IP): 5 requests/minute per source IP via slowapi/Redis.
        Exceeding this returns HTTP 429 before any DB query is made.
      - Layer 2 (account): after 5 consecutive failures the account is locked
        for 15 minutes.  Counter resets on successful login.

    Both layers return the same generic error message so attackers cannot
    distinguish "account exists but locked" from "wrong password".
    """
    _GENERIC_ERROR = "Invalid email or password"

    result = await db.execute(select(User).where(User.email == payload.email))
    user = result.scalar_one_or_none()

    # Check lockout before password verification to prevent timing oracle.
    await check_account_lockout(user)

    if not user or not verify_password(payload.password, user.hashed_password):
        # Record the failure (no-op if user is None).
        await record_login_failure(db, user)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_GENERIC_ERROR,
        )

    if not user.is_active:
        # Do not reveal the distinction between disabled and bad credentials.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_GENERIC_ERROR,
        )

    # Successful login — clear any accumulated failure counters.
    await clear_login_failures(db, user)

    access_token  = create_access_token(str(user.id), {"email": user.email, "username": user.username})
    refresh_token = create_refresh_token(str(user.id))

    # Store hashed refresh token in DB
    expires_at = datetime.now(timezone.utc) + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)

    # Enforce a per-user limit on active refresh tokens (max 10 concurrent sessions).
    # Revoke the oldest tokens if the limit is exceeded.
    _MAX_ACTIVE_TOKENS = 10
    active_tokens_result = await db.execute(
        select(RefreshToken)
        .where(RefreshToken.user_id == user.id, RefreshToken.revoked == False)
        .order_by(RefreshToken.created_at)
    )
    active_tokens = active_tokens_result.scalars().all()
    if len(active_tokens) >= _MAX_ACTIVE_TOKENS:
        # Revoke the oldest ones to stay within the limit
        for old_token in active_tokens[:len(active_tokens) - _MAX_ACTIVE_TOKENS + 1]:
            old_token.revoked = True

    db_token = RefreshToken(
        user_id=user.id,
        token_hash=hash_token(refresh_token),
        expires_at=expires_at,
    )
    db.add(db_token)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


# ── Refresh Token ──────────────────────────────────────────────────────────

@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(payload: RefreshRequest, db: AsyncSession = Depends(get_db)):
    token_data = decode_token(payload.refresh_token)

    if token_data.get("type") != "refresh":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Wrong token type")

    token_hash = hash_token(payload.refresh_token)
    result = await db.execute(
        select(RefreshToken).where(
            RefreshToken.token_hash == token_hash,
            RefreshToken.revoked == False,
        )
    )
    db_token = result.scalar_one_or_none()

    if not db_token or db_token.expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token expired or revoked")

    # Revoke old token (rotation)
    db_token.revoked = True

    # Issue new pair
    user_id = token_data["sub"]
    new_access  = create_access_token(user_id)
    new_refresh = create_refresh_token(user_id)

    new_expires = datetime.now(timezone.utc) + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    db.add(RefreshToken(
        user_id=user_id,
        token_hash=hash_token(new_refresh),
        expires_at=new_expires,
    ))

    return TokenResponse(
        access_token=new_access,
        refresh_token=new_refresh,
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


# ── Logout ─────────────────────────────────────────────────────────────────

@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(payload: RefreshRequest, db: AsyncSession = Depends(get_db)):
    token_hash = hash_token(payload.refresh_token)
    result = await db.execute(select(RefreshToken).where(RefreshToken.token_hash == token_hash))
    db_token = result.scalar_one_or_none()
    if db_token:
        db_token.revoked = True


# ── Current User ───────────────────────────────────────────────────────────

@router.get("/me", response_model=UserResponse)
async def me(current_user: User = Depends(get_current_user)):
    return current_user


# ── Change Password ────────────────────────────────────────────────────────

@router.post("/change-password", status_code=status.HTTP_204_NO_CONTENT)
async def change_password(
    payload: PasswordChangeRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not verify_password(payload.current_password, current_user.hashed_password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Current password is incorrect")

    current_user.hashed_password = hash_password(payload.new_password)

    # Revoke ALL active refresh tokens for this user so any stolen token
    # cannot be used after a password change.
    from sqlalchemy import update as sa_update
    await db.execute(
        sa_update(RefreshToken)
        .where(RefreshToken.user_id == current_user.id, RefreshToken.revoked == False)
        .values(revoked=True)
    )
