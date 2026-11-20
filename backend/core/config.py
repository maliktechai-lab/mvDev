from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import List


# ── Weak / example secrets that must never be used in production ───────────
_PLACEHOLDER_SECRETS = {
    "change-this-to-a-very-long-random-secret-key-in-production",
    "change-this-jwt-secret-key-too",
    "secret",
    "changeme",
    "",
}

_WEAK_DB_PASSWORDS = {
    "deepguard_password",
    "password",
    "postgres",
    "changeme",
    "secret",
    "",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    APP_NAME: str = "DeepGuard"
    APP_ENV: str = "development"
    DEBUG: bool = False
    SECRET_KEY: str
    ALLOWED_ORIGINS: str = "http://localhost:3000"

    # Database
    DATABASE_URL: str
    POSTGRES_DB: str = "deepguard"
    POSTGRES_USER: str = "deepguard_user"
    POSTGRES_PASSWORD: str = "deepguard_password"

    # Redis / Celery
    REDIS_URL: str = "redis://redis:6379/0"
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/1"

    # JWT
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # File Storage
    UPLOAD_DIR: str = "./uploads"
    HEATMAP_DIR: str = "./heatmaps"
    MAX_IMAGE_SIZE_MB: int = 10
    MAX_VIDEO_SIZE_MB: int = 500
    MAX_AUDIO_SIZE_MB: int = 50

    # ML Models
    # Docker:     /app/checkpoints/   (volume-mounted from ./ml/checkpoints)
    # Local dev:  ./checkpoints/      (symlink or copy from ml/checkpoints)
    # Override via .env — see .env.example for both scenarios.
    MODEL_DIR: str = "./checkpoints"
    IMAGE_MODEL_PATH: str = "./checkpoints/deepguard_image.pth"
    VIDEO_MODEL_PATH: str = "./checkpoints/deepguard_video.pth"
    AUDIO_MODEL_PATH: str = "./checkpoints/deepguard_audio.pth"
    DEVICE: str = "cpu"

    # Inference
    IMAGE_CONFIDENCE_THRESHOLD: float = 0.5
    VIDEO_FRAMES_PER_SEC: int = 1
    AUDIO_CHUNK_DURATION_SEC: int = 3

    # Celery Flower basic-auth (user:password).  Set in .env.
    # Required in non-development environments.
    FLOWER_BASIC_AUTH: str = "admin:changeme"

    @property
    def allowed_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]

    @property
    def max_image_size_bytes(self) -> int:
        return self.MAX_IMAGE_SIZE_MB * 1024 * 1024

    @property
    def max_video_size_bytes(self) -> int:
        return self.MAX_VIDEO_SIZE_MB * 1024 * 1024

    @property
    def max_audio_size_bytes(self) -> int:
        return self.MAX_AUDIO_SIZE_MB * 1024 * 1024

    # ── Production-only validators ─────────────────────────────────────────
    # These run at startup and raise ValueError (which pydantic wraps as a
    # ValidationError) if any dangerous configuration is detected in production.

    @field_validator("SECRET_KEY")
    @classmethod
    def secret_key_not_placeholder(cls, v: str, info) -> str:
        # We cannot read APP_ENV from `info` at this stage reliably in all
        # pydantic-settings versions; validation is enforced in validate_production()
        # below instead.  Here we just reject the empty string.
        if not v or not v.strip():
            raise ValueError("SECRET_KEY must not be empty")
        return v

    @field_validator("JWT_SECRET_KEY")
    @classmethod
    def jwt_key_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("JWT_SECRET_KEY must not be empty")
        return v

    def validate_production(self) -> None:
        """
        Raise RuntimeError if production-unsafe configuration is detected.

        Called explicitly from main.py on startup so that the error message
        is clear and actionable rather than buried in a Pydantic traceback.

        Rules enforced when APP_ENV == 'production':
          - SECRET_KEY must not be a known placeholder value.
          - JWT_SECRET_KEY must not be a known placeholder value.
          - DEBUG must be False.
          - POSTGRES_PASSWORD must not be a known weak value.
          - ALLOWED_ORIGINS must not contain localhost or 127.0.0.1.
        """
        if self.APP_ENV == "development":
            return  # No restrictions in development

        errors: list[str] = []

        if self.SECRET_KEY in _PLACEHOLDER_SECRETS:
            errors.append(
                "SECRET_KEY is set to a known placeholder value. "
                "Generate a strong secret: python -c \"import secrets; print(secrets.token_hex(64))\""
            )

        if self.JWT_SECRET_KEY in _PLACEHOLDER_SECRETS:
            errors.append(
                "JWT_SECRET_KEY is set to a known placeholder value. "
                "Generate a strong secret: python -c \"import secrets; print(secrets.token_hex(64))\""
            )

        if self.DEBUG:
            errors.append(
                "DEBUG=true is not permitted in production. Set DEBUG=false in .env."
            )

        if self.POSTGRES_PASSWORD in _WEAK_DB_PASSWORDS:
            errors.append(
                "POSTGRES_PASSWORD is set to a known weak value. "
                "Use a strong random password in production."
            )

        for origin in self.allowed_origins_list:
            if "localhost" in origin or "127.0.0.1" in origin:
                errors.append(
                    f"ALLOWED_ORIGINS contains a localhost/loopback entry ('{origin}') "
                    "in production mode. Set ALLOWED_ORIGINS to your real frontend domain."
                )
                break  # one message is enough

        if self.FLOWER_BASIC_AUTH in ("admin:changeme", ""):
            errors.append(
                "FLOWER_BASIC_AUTH is set to the default placeholder. "
                "Set a strong user:password in .env (FLOWER_BASIC_AUTH=user:strongpassword)."
            )

        if errors:
            bullet_list = "\n".join(f"  • {e}" for e in errors)
            raise RuntimeError(
                f"[PRODUCTION] Startup aborted — unsafe configuration detected:\n{bullet_list}\n\n"
                "Fix all items above before running in production."
            )


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
