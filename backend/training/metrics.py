"""
Training Metrics
================
AUC-ROC, EER, Accuracy, F1, and confusion matrix for deepfake detection.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, roc_curve,
)


@dataclass
class EpochMetrics:
    loss:     float = 0.0
    auc:      float = 0.0
    accuracy: float = 0.0
    f1:       float = 0.0
    eer:      float = 0.0   # Equal Error Rate
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    def __str__(self) -> str:
        return (
            f"loss={self.loss:.4f}  "
            f"auc={self.auc:.4f}  "
            f"acc={self.accuracy:.4f}  "
            f"f1={self.f1:.4f}  "
            f"eer={self.eer:.4f}"
        )


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute Equal Error Rate — the point where FAR == FRR.
    Lower is better (0 = perfect, 0.5 = random).

    Raises ValueError if the input does not contain both classes, which
    makes EER undefined.  Callers on the scientific evaluation path must
    not suppress this error.
    """
    unique = np.unique(labels)
    if len(unique) < 2:
        raise ValueError(
            f"EER is undefined: labels contain only class(es) {unique.tolist()}. "
            "Both REAL (0) and FAKE (1) must be present."
        )
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    # Find the threshold where |FPR - FNR| is minimized
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    return eer


def compute_metrics(
    labels: List[int],
    scores: List[float],    # fake probability
    threshold: float = 0.5,
) -> EpochMetrics:
    """
    Compute all metrics from predictions.

    This is the *scientific evaluation path*.  It raises ValueError when
    the input is degenerate (empty, single-class, or mismatched lengths)
    rather than returning a silent default.  Do not catch these errors in
    training loops — they indicate a data pipeline problem that must be
    fixed.

    For a dashboard-friendly path that tolerates missing data, use
    ``compute_metrics_safe`` below.

    Args:
        labels: Ground truth labels (0=real, 1=fake)
        scores: Predicted fake probabilities [0, 1]
        threshold: Decision threshold
    """
    if len(labels) == 0:
        raise ValueError(
            "compute_metrics received an empty labels list. "
            "The split produced no samples."
        )
    if len(labels) != len(scores):
        raise ValueError(
            f"compute_metrics: labels ({len(labels)}) and scores ({len(scores)}) "
            "must have the same length."
        )

    labels_arr  = np.array(labels)
    scores_arr  = np.array(scores)
    preds_arr   = (scores_arr >= threshold).astype(int)

    unique_classes = np.unique(labels_arr)
    if len(unique_classes) < 2:
        raise ValueError(
            f"AUC-ROC and EER are undefined: labels contain only "
            f"class(es) {unique_classes.tolist()}. "
            "Both REAL (0) and FAKE (1) must be present in every evaluated split."
        )

    auc = float(roc_auc_score(labels_arr, scores_arr))
    eer = compute_eer(labels_arr, scores_arr)

    acc = float(accuracy_score(labels_arr, preds_arr))
    f1  = float(f1_score(labels_arr, preds_arr, zero_division=0))

    cm = confusion_matrix(labels_arr, preds_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    return EpochMetrics(
        auc=auc, accuracy=acc, f1=f1, eer=eer,
        tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
    )


def compute_metrics_safe(
    labels: List[int],
    scores: List[float],
    threshold: float = 0.5,
) -> EpochMetrics:
    """
    Dashboard-friendly wrapper around ``compute_metrics``.

    Returns a zeroed ``EpochMetrics`` (with loss=0, auc=0, eer=0, …)
    when the input is degenerate, instead of raising.  Never use this
    on the scientific evaluation path — it hides data pipeline errors.
    """
    try:
        return compute_metrics(labels, scores, threshold)
    except (ValueError, Exception):
        return EpochMetrics()


class MetricsTracker:
    """Accumulates predictions and labels across batches, computes epoch metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._labels: List[int]   = []
        self._scores: List[float] = []
        self._losses: List[float] = []

    def update(self, labels: List[int], scores: List[float], loss: float):
        self._labels.extend(labels)
        self._scores.extend(scores)
        self._losses.append(loss)

    def compute(self) -> EpochMetrics:
        metrics = compute_metrics(self._labels, self._scores)
        metrics.loss = float(np.mean(self._losses)) if self._losses else 0.0
        return metrics
