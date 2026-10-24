"""
Tests for Celery task imports
==============================
Verifies that video and audio pipeline tasks are importable as Celery tasks,
and that the celery app config is correct.
No live Redis or broker connection needed.
"""

from __future__ import annotations

import pytest
from celery import Task


# ── 1. Video task importable ─────────────────────────────────────────────

def test_video_task_importable():
    """run_video_detection should be importable and be a Celery Task."""
    from pipelines.video_pipeline import run_video_detection

    assert callable(run_video_detection), "run_video_detection must be callable"
    # Celery tasks are instances of celery.app.task.Task (bound to the app)
    assert isinstance(run_video_detection, Task), (
        f"run_video_detection should be a Celery Task, got {type(run_video_detection)}"
    )


# ── 2. Audio task importable ─────────────────────────────────────────────

def test_audio_task_importable():
    """run_audio_detection should be importable and be a Celery Task."""
    from pipelines.audio_pipeline import run_audio_detection

    assert callable(run_audio_detection), "run_audio_detection must be callable"
    assert isinstance(run_audio_detection, Task), (
        f"run_audio_detection should be a Celery Task, got {type(run_audio_detection)}"
    )


# ── 3. Celery app config ──────────────────────────────────────────────────

def test_celery_app_config():
    """celery_app.conf.task_serializer should be 'json'."""
    from core.celery_app import celery_app

    assert celery_app.conf.task_serializer == "json", (
        f"Expected task_serializer='json', got '{celery_app.conf.task_serializer}'"
    )
    assert celery_app.conf.accept_content == ["json"], (
        f"Expected accept_content=['json'], got {celery_app.conf.accept_content}"
    )
    assert celery_app.conf.result_serializer == "json", (
        f"Expected result_serializer='json', got '{celery_app.conf.result_serializer}'"
    )
