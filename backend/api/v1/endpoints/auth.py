from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException, status
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
async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    # TODO(security): No rate limiting is applied to this endpoint.
    # Brute-force attacks (credential stuffing, password spraying) are possible.
    # Mitigate by adding one of:
    #   - slowapi (pip install slowapi) with a Redis backend for per-IP limits
    #   - A reverse-proxy-level rate limit in Nginx (limit_req_zone)
    #   - Account lockout after N consecutive failures (requires a failed_attempts
    #     counter column on the User model and a cooldown period)
    # See DESIGN.md § "Security — Brute-force protection" for full discussion.
    result = await db.execute(select(User).where(User.email == payload.email))
    user = result.scalar_one_or_none()

    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is disabled")

    access_token  = create_access_token(str(user.id), {"email": user.email, "username": user.username})
    refresh_token = create_refresh_token(str(user.id))

    # Store hashed refresh token in DB
    expires_at = datetime.now(timezone.utc) + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
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
