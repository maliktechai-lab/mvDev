from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AnyHttpUrl, field_validator
from typing import List


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


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
