"""
Health Endpoints
================
GET /api/v1/health        — liveness + DB + Redis connectivity
GET /api/v1/health/ping   — minimal liveness probe (no DB)
GET /api/v1/health/models — model checkpoint readiness (separate from app health)
"""

import time
from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from db.database import get_db
from core.config import settings
from core.model_readiness import ModelStatus, get_all_model_readiness

router = APIRouter(tags=["Health"])


@router.get("/health/ping")
async def ping():
    """Minimal liveness probe for load balancers and Docker HEALTHCHECK."""
    return {"ping": "pong"}


@router.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    """
    Liveness + readiness probe.

    Checks:
    - API process is alive
    - PostgreSQL is reachable and accepting queries
    - Redis connectivity (via Celery broker URL)

    Model readiness is NOT included here — see GET /health/models.
    A missing model checkpoint does NOT make the application unhealthy;
    it just means certain inference endpoints will refuse requests.
    """
    start = time.time()

    # ── Database check ────────────────────────────────────────────────────
    db_ok = False
    db_latency_ms: Optional[float] = None
    db_error: Optional[str] = None
    try:
        t0 = time.time()
        await db.execute(text("SELECT 1"))
        db_latency_ms = round((time.time() - t0) * 1000, 2)
        db_ok = True
    except Exception as exc:
        db_error = str(exc)

    # ── Redis check ───────────────────────────────────────────────────────
    redis_ok = False
    redis_latency_ms: Optional[float] = None
    redis_error: Optional[str] = None
    try:
        import redis.asyncio as aioredis
        t0 = time.time()
        r = aioredis.from_url(settings.REDIS_URL, socket_connect_timeout=2)
        await r.ping()
        await r.aclose()
        redis_latency_ms = round((time.time() - t0) * 1000, 2)
        redis_ok = True
    except Exception as exc:
        redis_error = str(exc)

    total_ms = round((time.time() - start) * 1000, 2)
    overall = "healthy" if (db_ok and redis_ok) else "degraded"

    return {
        "status": overall,
        "app": settings.APP_NAME,
        "env": settings.APP_ENV,
        "version": "1.0.0",
        "checks": {
            "database": {
                "ok": db_ok,
                "latency_ms": db_latency_ms,
                **({"error": db_error} if db_error else {}),
            },
            "redis": {
                "ok": redis_ok,
                "latency_ms": redis_latency_ms,
                **({"error": redis_error} if redis_error else {}),
            },
        },
        "response_time_ms": total_ms,
    }


@router.get("/health/models")
async def model_readiness():
    """
    Report the readiness of each ML model checkpoint.

    This is intentionally separate from GET /health so that:
    - A missing model does not fail load balancer health checks.
    - Clients can query model availability before submitting inference jobs.

    Statuses:
    - ready       → checkpoint exists and loaded successfully.
    - unavailable → no checkpoint file found; model needs training.
    - corrupt     → file exists but cannot be loaded or has wrong architecture.

    In production (APP_ENV=production), inference endpoints will refuse
    requests when the required model is not 'ready'.

    In development mode (APP_ENV=development), inference endpoints will run
    with random weights and tag responses with a warning.
    """
    results = get_all_model_readiness()

    models_out = {}
    for name, r in results.items():
        models_out[name] = {
            "status": r.status,
            "path": r.path,
            "message": r.message,
            **({"params": r.params} if r.params else {}),
            **({"error": r.error} if r.error else {}),
        }

    any_ready    = any(r.status == ModelStatus.READY    for r in results.values())
    any_corrupt  = any(r.status == ModelStatus.CORRUPT  for r in results.values())
    all_ready    = all(r.status == ModelStatus.READY    for r in results.values())

    if all_ready:
        overall = "all_ready"
    elif any_corrupt:
        overall = "degraded"
    elif any_ready:
        overall = "partial"
    else:
        overall = "unavailable"

    return {
        "overall": overall,
        "env": settings.APP_ENV,
        "inference_mode": (
            "production"  if settings.APP_ENV != "development" else "development_random_weights"
        ),
        "models": models_out,
    }
