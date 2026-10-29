"""
Regression tests for backend/training/metrics.py.

Covers:
  Task 2 — scientific evaluation path must raise on degenerate input,
            never silently return AUC=0.5 or EER=0.5.

No real dataset is used.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = str(Path(__file__).parent.parent / "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


# ---------------------------------------------------------------------------
# compute_metrics — scientific path
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    """compute_metrics must raise ValueError for any degenerate input."""

    def test_empty_labels_raises(self):
        from training.metrics import compute_metrics
        with pytest.raises(ValueError, match="empty"):
            compute_metrics([], [])

    def test_real_only_labels_raise(self):
        """All-real labels → AUC and EER undefined → ValueError."""
        from training.metrics import compute_metrics
        labels = [0, 0, 0, 0]
        scores = [0.1, 0.2, 0.3, 0.15]
        with pytest.raises(ValueError):
            compute_metrics(labels, scores)

    def test_fake_only_labels_raise(self):
        """All-fake labels → AUC and EER undefined → ValueError."""
        from training.metrics import compute_metrics
        labels = [1, 1, 1, 1]
        scores = [0.8, 0.9, 0.7, 0.85]
        with pytest.raises(ValueError):
            compute_metrics(labels, scores)

    def test_mismatched_lengths_raise(self):
        """Different-length labels and scores → ValueError."""
        from training.metrics import compute_metrics
        with pytest.raises(ValueError, match="same length"):
            compute_metrics([0, 1, 0], [0.1, 0.9])

    def test_valid_two_class_returns_metrics(self):
        """Two-class input → returns EpochMetrics with sane AUC and EER."""
        from training.metrics import compute_metrics
        # Perfect separation: real=0→low scores, fake=1→high scores
        labels = [0, 0, 0, 1, 1, 1]
        scores = [0.1, 0.2, 0.15, 0.85, 0.9, 0.8]
        m = compute_metrics(labels, scores)
        assert m.auc > 0.9, f"Expected near-perfect AUC, got {m.auc}"
        assert 0.0 <= m.eer <= 1.0
        assert 0.0 <= m.accuracy <= 1.0

    def test_auc_not_silently_returned_as_half(self):
        """Single-class input must NOT silently return AUC=0.5."""
        from training.metrics import compute_metrics
        labels = [0, 0, 0, 0]
        scores = [0.1, 0.2, 0.3, 0.4]
        with pytest.raises(ValueError):
            result = compute_metrics(labels, scores)
            # If the bug is still present, this assertion catches it
            assert result.auc != 0.5, (
                "AUC silently returned 0.5 for single-class input — defect not fixed"
            )

    def test_eer_not_silently_returned_as_half(self):
        """Single-class input must NOT silently return EER=0.5."""
        from training.metrics import compute_metrics
        labels = [1, 1, 1, 1]
        scores = [0.7, 0.8, 0.9, 0.6]
        with pytest.raises(ValueError):
            result = compute_metrics(labels, scores)
            assert result.eer != 0.5, (
                "EER silently returned 0.5 for single-class input — defect not fixed"
            )


# ---------------------------------------------------------------------------
# compute_eer — standalone
# ---------------------------------------------------------------------------

class TestComputeEer:
    """compute_eer must raise when the labels are not two-class."""

    def test_eer_single_class_raises(self):
        import numpy as np
        from training.metrics import compute_eer
        labels = np.array([0, 0, 0])
        scores = np.array([0.1, 0.3, 0.2])
        with pytest.raises(ValueError, match="undefined"):
            compute_eer(labels, scores)

    def test_eer_valid_two_class_computes(self):
        import numpy as np
        from training.metrics import compute_eer
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        eer = compute_eer(labels, scores)
        assert 0.0 <= eer <= 1.0


# ---------------------------------------------------------------------------
# compute_metrics_safe — dashboard fallback path
# ---------------------------------------------------------------------------

class TestComputeMetricsSafe:
    """compute_metrics_safe must swallow errors and return a zeroed EpochMetrics."""

    def test_safe_empty_input_returns_zero(self):
        from training.metrics import compute_metrics_safe, EpochMetrics
        result = compute_metrics_safe([], [])
        assert result.auc == 0.0
        assert result.eer == 0.0

    def test_safe_single_class_returns_zero(self):
        from training.metrics import compute_metrics_safe
        result = compute_metrics_safe([0, 0, 0], [0.1, 0.2, 0.3])
        assert result.auc == 0.0

    def test_safe_valid_two_class_returns_metrics(self):
        from training.metrics import compute_metrics_safe
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.2, 0.8, 0.9]
        result = compute_metrics_safe(labels, scores)
        assert result.auc > 0.5
