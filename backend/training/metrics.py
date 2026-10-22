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
    """
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

    Args:
        labels: Ground truth labels (0=real, 1=fake)
        scores: Predicted fake probabilities [0, 1]
        threshold: Decision threshold
    """
    labels_arr  = np.array(labels)
    scores_arr  = np.array(scores)
    preds_arr   = (scores_arr >= threshold).astype(int)

    try:
        auc = float(roc_auc_score(labels_arr, scores_arr))
    except ValueError:
        auc = 0.5

    try:
        eer = compute_eer(labels_arr, scores_arr)
    except Exception:
        eer = 0.5

    acc = float(accuracy_score(labels_arr, preds_arr))
    f1  = float(f1_score(labels_arr, preds_arr, zero_division=0))

    cm = confusion_matrix(labels_arr, preds_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    return EpochMetrics(
        auc=auc, accuracy=acc, f1=f1, eer=eer,
        tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
    )


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
