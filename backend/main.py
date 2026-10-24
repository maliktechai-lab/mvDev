import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from core.config import settings
from core.logging import setup_logging
from db.database import create_tables
from api.v1.router import api_router


# ── Lifespan ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    setup_logging()
    logger.info(f"Starting {settings.APP_NAME} — env={settings.APP_ENV}")

    # Ensure storage directories exist
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.HEATMAP_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Create DB tables (dev only — in prod Alembic handles this)
    if settings.APP_ENV == "development":
        await create_tables()
        logger.info("Database tables verified")

    # Warm up ML models (loads weights, runs dummy forward pass)
    from models.registry import warmup_models
    warmup_models()

    logger.info(f"{settings.APP_NAME} is ready")
    yield

    # Shutdown
    logger.info(f"{settings.APP_NAME} shutting down")


# ── App Instance ───────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    description="Deepfake & Synthetic Media Detection Platform",
    version="1.0.0",
    docs_url="/api/docs" if settings.DEBUG else None,
    redoc_url="/api/redoc" if settings.DEBUG else None,
    openapi_url="/api/openapi.json" if settings.DEBUG else None,
    lifespan=lifespan,
)


# ── Middleware ─────────────────────────────────────────────────────────────

# CORS
# SECURITY NOTE: allow_headers=["*"] is intentionally permissive for development.
# In production, restrict this to the specific headers your frontend sends
# (e.g. ["Authorization", "Content-Type", "Accept", "X-Request-ID"]).
# The allowed origins are controlled via the ALLOWED_ORIGINS env var, which
# should be set to your exact frontend domain(s) in production (never "*").
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Gzip responses
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ── Request logging middleware ─────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"→ {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"← {request.method} {request.url.path} [{response.status_code}]")
    return response


# ── Global exception handler ───────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception on {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error. Our team has been notified."},
    )


# ── Static files ───────────────────────────────────────────────────────────

app.mount("/heatmaps", StaticFiles(directory=settings.HEATMAP_DIR), name="heatmaps")


# ── Routes ─────────────────────────────────────────────────────────────────

app.include_router(api_router)


# ── Root ───────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {"message": f"Welcome to {settings.APP_NAME} API", "docs": "/api/docs"}
