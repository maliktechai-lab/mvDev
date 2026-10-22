from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from redis.asyncio import Redis
import time

from db.database import get_db
from core.config import settings

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    """
    Liveness + readiness probe.
    Checks: API up, DB connected, returns app metadata.
    """
    start = time.time()

    # Check DB
    db_ok = False
    db_latency = None
    try:
        t0 = time.time()
        await db.execute(text("SELECT 1"))
        db_latency = round((time.time() - t0) * 1000, 2)
        db_ok = True
    except Exception as e:
        db_error = str(e)

    total_ms = round((time.time() - start) * 1000, 2)

    return {
        "status": "healthy" if db_ok else "degraded",
        "app": settings.APP_NAME,
        "env": settings.APP_ENV,
        "version": "1.0.0",
        "checks": {
            "database": {"ok": db_ok, "latency_ms": db_latency},
        },
        "response_time_ms": total_ms,
    }


@router.get("/health/ping")
async def ping():
    """Minimal liveness probe for load balancers."""
    return {"ping": "pong"}
