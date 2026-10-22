from celery import Celery
from core.config import settings

celery_app = Celery(
    "deepguard",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "pipelines.video_pipeline",
        "pipelines.audio_pipeline",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_routes={
        "pipelines.video_pipeline.*": {"queue": "video"},
        "pipelines.audio_pipeline.*": {"queue": "audio"},
    },
    task_time_limit=600,          # 10 min hard limit
    task_soft_time_limit=540,     # 9 min soft limit (sends warning)
    result_expires=86400,         # Results expire after 24h
)
