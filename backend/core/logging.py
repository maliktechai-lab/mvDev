import sys
import logging
from loguru import logger
from core.config import settings


class InterceptHandler(logging.Handler):
    """Intercept standard logging and route through loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging() -> None:
    """Configure loguru as the single logging backend."""
    logger.remove()

    log_level = "DEBUG" if settings.DEBUG else "INFO"
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # Console output
    logger.add(
        sys.stdout,
        format=log_format,
        level=log_level,
        colorize=True,
        enqueue=True,
    )

    # File output with rotation
    logger.add(
        "logs/deepguard_{time:YYYY-MM-DD}.log",
        format=log_format,
        level="INFO",
        rotation="00:00",       # New file at midnight
        retention="30 days",    # Keep 30 days of logs
        compression="zip",
        enqueue=True,
    )

    # Intercept uvicorn, sqlalchemy, celery logs
    for log_name in ["uvicorn", "uvicorn.error", "uvicorn.access",
                     "sqlalchemy.engine", "celery", "fastapi"]:
        logging.getLogger(log_name).handlers = [InterceptHandler()]
        logging.getLogger(log_name).propagate = False

    logger.info(f"Logging configured — level={log_level}, env={settings.APP_ENV}")
