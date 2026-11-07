"""
DeepGuard-Image — Full Evaluation & Evidence Generator
=======================================================
Loads the trained checkpoint and produces a complete set of evaluation
artifacts as hard evidence that the model works:

Outputs (all written to ml/evaluation/<run_timestamp>/):
  report.txt                — Human-readable report with all metrics
  metrics.json              — Machine-readable metrics (CI-friendly)
  roc_curve.png             — AUC-ROC curve plot
  confusion_matrix.png      — Confusion matrix (absolute + normalised)
  score_distribution.png    — Real vs fake score histograms
  threshold_sweep.png       — Accuracy / F1 / TPR / FPR vs threshold
  calibration_curve.png     — Reliability diagram (confidence vs accuracy)
  gradcam_samples/          — GradCAM overlays for 20 sample images
    real_correct_N.png      — Correctly classified real
    fake_correct_N.png      — Correctly classified fake
    real_wrong_N.png        — Real misclassified as fake (false positive)
    fake_wrong_N.png        — Fake misclassified as real (false negative)

Usage
-----
  python backend/training/evaluate_image.py \\
      --checkpoint ml/checkpoints/deepguard_image.pth \\
      --dataset_root data/processed/cifake \\
      --output_dir ml/evaluation

  # Limit samples for a quick run
  python backend/training/evaluate_image.py \\
      --checkpoint ml/checkpoints/deepguard_image.pth \\
      --dataset_root data/processed/cifake \\
      --max_samples 2000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Subset

# Allow running from repo root or backend/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.image_model import build_image_model
from training.dataset import DeepfakeImageDataset, LABEL_REAL, LABEL_FAKE
from training.metrics import compute_metrics, EpochMetrics


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")   # headless — no display needed
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        logger.error("matplotlib is not installed. Run: pip install matplotlib")
        sys.exit(1)


def _require_cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        logger.error("opencv-python-headless is not installed.")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────
# Inference — collect all predictions on the test set
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model on all batches.

    Returns:
        labels  — (N,) int   ground-truth (0=real, 1=fake)
        scores  — (N,) float fake probability
        preds   — (N,) int   hard prediction at threshold 0.5
    """
    model.eval()
    all_labels: List[int]   = []
    all_scores: List[float] = []

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        out    = model(images)
        probs  = F.softmax(out["logits"], dim=1)
        scores = probs[:, 1].cpu().tolist()
        all_labels.extend(labels.tolist())
        all_scores.extend(scores)
        if (batch_idx + 1) % 20 == 0:
            logger.info(f"  Inference: {batch_idx+1}/{len(loader)} batches")

    labels_arr = np.array(all_labels, dtype=int)
    scores_arr = np.array(all_scores, dtype=float)
    preds_arr  = (scores_arr >= 0.5).astype(int)
    return labels_arr, scores_arr, preds_arr


# ──────────────────────────────────────────────────────────────────────────
# Plot: ROC Curve
# ──────────────────────────────────────────────────────────────────────────

def plot_roc_curve(labels, scores, auc: float, out_path: Path) -> None:
    from sklearn.metrics import roc_curve
    plt = _require_matplotlib()

    fpr, tpr, _ = roc_curve(labels, scores)
    fig, ax     = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2563EB", lw=2,
            label=f"DeepGuard-Image  AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color="#94A3B8", lw=1, linestyle="--", label="Random (AUC = 0.50)")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#2563EB")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — DeepGuard-Image (Test Set)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.01])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────
# Plot: Confusion Matrix
# ──────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    labels, preds, tp, tn, fp, fn, out_path: Path,
) -> None:
    plt = _require_matplotlib()
    import matplotlib.colors as mcolors

    cm_abs  = np.array([[tn, fp], [fn, tp]])
    cm_norm = cm_abs.astype(float) / cm_abs.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data, fmt, title in [
        (axes[0], cm_abs,  "d",    "Absolute Counts"),
        (axes[1], cm_norm, ".2%",  "Row-Normalised"),
    ]:
        im = ax.imshow(data, cmap="Blues", vmin=0,
                       vmax=data.max())
        for i in range(2):
            for j in range(2):
                ax.text(j, i, format(data[i, j], fmt),
                        ha="center", va="center",
                        color="white" if data[i, j] > data.max() * 0.5 else "black",
                        fontsize=14, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: Real", "Pred: Fake"], fontsize=11)
        ax.set_yticklabels(["True: Real", "True: Fake"], fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax)

    fig.suptitle("Confusion Matrix — DeepGuard-Image (Test Set)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────
# Plot: Score distribution
# ──────────────────────────────────────────────────────────────────────────

def plot_score_distribution(labels, scores, out_path: Path) -> None:
    plt = _require_matplotlib()

    real_scores = scores[labels == LABEL_REAL]
    fake_scores = scores[labels == LABEL_FAKE]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 51)
    ax.hist(real_scores, bins=bins, alpha=0.65, color="#16A34A",
            label=f"Real  (n={len(real_scores):,})", density=True)
    ax.hist(fake_scores, bins=bins, alpha=0.65, color="#DC2626",
            label=f"Fake  (n={len(fake_scores):,})", density=True)
    ax.axvline(0.5, color="black", lw=1.5, linestyle="--", label="Threshold = 0.50")
    ax.set_xlabel("Predicted Fake Probability", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Score Distribution — Real vs Fake (Test Set)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────
# Plot: Threshold sweep
# ──────────────────────────────────────────────────────────────────────────

def plot_threshold_sweep(labels, scores, out_path: Path) -> None:
    plt = _require_matplotlib()

    thresholds = np.linspace(0.01, 0.99, 99)
    accs, f1s, tprs, fprs = [], [], [], []

    for t in thresholds:
        preds = (scores >= t).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        accs.append((tp + tn) / len(labels))
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0)
        tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, accs, label="Accuracy",   color="#2563EB", lw=2)
    ax.plot(thresholds, f1s,  label="F1 Score",   color="#D97706", lw=2)
    ax.plot(thresholds, tprs, label="TPR (Recall)", color="#16A34A", lw=1.5, linestyle="--")
    ax.plot(thresholds, fprs, label="FPR",         color="#DC2626", lw=1.5, linestyle="--")
    ax.axvline(0.5, color="black", lw=1, linestyle=":", label="Default threshold")
    ax.set_xlabel("Decision Threshold", fontsize=12)
    ax.set_ylabel("Metric Value", fontsize=12)
    ax.set_title("Threshold Sweep — Test Set", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.01])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────
# Plot: Calibration curve
# ──────────────────────────────────────────────────────────────────────────

def plot_calibration_curve(labels, scores, out_path: Path) -> None:
    plt = _require_matplotlib()

    n_bins = 10
    bin_edges  = np.linspace(0, 1, n_bins + 1)
    bin_mids   = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_accs   = []
    bin_confs  = []
    bin_counts = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (scores >= lo) & (scores < hi)
        if mask.sum() == 0:
            continue
        bin_accs.append(labels[mask].mean())
        bin_confs.append(scores[mask].mean())
        bin_counts.append(mask.sum())

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94A3B8", lw=1.5, label="Perfect calibration")
    ax.scatter(bin_confs, bin_accs, s=[c * 0.5 for c in bin_counts],
               color="#2563EB", alpha=0.8, zorder=5)
    ax.plot(bin_confs, bin_accs, color="#2563EB", lw=2, label="DeepGuard-Image")
    ax.set_xlabel("Mean Predicted Probability (confidence)", fontsize=12)
    ax.set_ylabel("Fraction of Positives (actual accuracy)", fontsize=12)
    ax.set_title("Calibration Curve — DeepGuard-Image", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────
# GradCAM samples
# ──────────────────────────────────────────────────────────────────────────

def generate_gradcam_samples(
    model:      torch.nn.Module,
    dataset:    DeepfakeImageDataset,
    labels_arr: np.ndarray,
    scores_arr: np.ndarray,
    preds_arr:  np.ndarray,
    out_dir:    Path,
    n_each:     int = 5,
    device:     str = "cpu",
) -> None:
    """
    Save GradCAM heatmap overlays for sample images from each outcome bucket:
      real_correct, fake_correct, real_wrong (FP), fake_wrong (FN)
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        import cv2
    except ImportError:
        logger.warning("pytorch-grad-cam or cv2 not available — skipping GradCAM samples")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Target layer: last conv in spatial_branch.stage5
    target_layer = model.spatial_branch.stage5[-1].conv2.conv

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return self.m(x)["logits"]

    cam_gen = GradCAM(model=Wrapper(model), target_layers=[target_layer])

    # Build index lists for each bucket
    real_correct = list(np.where((labels_arr == 0) & (preds_arr == 0))[0])
    fake_correct = list(np.where((labels_arr == 1) & (preds_arr == 1))[0])
    real_wrong   = list(np.where((labels_arr == 0) & (preds_arr == 1))[0])   # FP
    fake_wrong   = list(np.where((labels_arr == 1) & (preds_arr == 0))[0])   # FN

    buckets = [
        ("real_correct", real_correct, 0),
        ("fake_correct", fake_correct, 1),
        ("real_wrong",   real_wrong,   1),   # target class = fake to see what fooled it
        ("fake_wrong",   fake_wrong,   1),
    ]

    import torchvision.transforms as T
    from training.dataset import IMAGENET_MEAN, IMAGENET_STD

    # Inverse-normalise tensor → [0,1] numpy for overlay
    inv_mean = [-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]
    inv_std  = [1.0 / s for s in IMAGENET_STD]
    unnorm   = T.Normalize(mean=inv_mean, std=inv_std)

    saved_total = 0
    for bucket_name, indices, target_cls in buckets:
        np.random.shuffle(indices)
        for i, idx in enumerate(indices[:n_each]):
            tensor_chw, true_label = dataset[idx]
            input_tensor = tensor_chw.unsqueeze(0).to(device)

            # GradCAM
            try:
                targets      = [ClassifierOutputTarget(target_cls)]
                grayscale    = cam_gen(input_tensor=input_tensor, targets=targets)[0]
            except Exception as e:
                logger.warning(f"GradCAM failed for idx={idx}: {e}")
                continue

            # Reconstruct original image for overlay
            orig_np = unnorm(tensor_chw).permute(1, 2, 0).numpy()
            orig_np = np.clip(orig_np, 0, 1).astype(np.float32)

            overlay = show_cam_on_image(
                img=orig_np, mask=grayscale,
                use_rgb=True, colormap=cv2.COLORMAP_JET, image_weight=0.5,
            )

            score = float(scores_arr[idx])
            fname = (
                f"{bucket_name}_{i+1}"
                f"__score{score:.2f}"
                f"__true{'real' if true_label == 0 else 'fake'}.png"
            )
            Image.fromarray(overlay).save(out_dir / fname)
            saved_total += 1

    logger.info(f"  GradCAM: {saved_total} overlays saved → {out_dir}")


# ──────────────────────────────────────────────────────────────────────────
# Text report
# ──────────────────────────────────────────────────────────────────────────

def write_report(
    out_path: Path,
    checkpoint_path: str,
    dataset_root: str,
    n_test: int,
    metrics: EpochMetrics,
    elapsed: float,
    timestamp: str,
) -> None:
    precision = metrics.tp / (metrics.tp + metrics.fp) if (metrics.tp + metrics.fp) > 0 else 0.0
    recall    = metrics.tp / (metrics.tp + metrics.fn) if (metrics.tp + metrics.fn) > 0 else 0.0
    specificity = metrics.tn / (metrics.tn + metrics.fp) if (metrics.tn + metrics.fp) > 0 else 0.0

    lines = [
        "=" * 65,
        "  DeepGuard-Image — Evaluation Report",
        "=" * 65,
        f"  Timestamp       : {timestamp}",
        f"  Checkpoint      : {checkpoint_path}",
        f"  Dataset         : {dataset_root}",
        f"  Test samples    : {n_test:,}",
        f"  Inference time  : {elapsed:.1f}s",
        "",
        "── Primary Metrics ────────────────────────────────────────",
        f"  AUC-ROC         : {metrics.auc:.4f}",
        f"  Accuracy        : {metrics.accuracy:.4f}  ({metrics.accuracy*100:.2f}%)",
        f"  F1 Score        : {metrics.f1:.4f}",
        f"  EER             : {metrics.eer:.4f}  (Equal Error Rate — lower is better)",
        "",
        "── Per-Class Metrics (threshold = 0.50) ───────────────────",
        f"  Precision       : {precision:.4f}  (of predicted fakes, how many were fake)",
        f"  Recall / TPR    : {recall:.4f}  (of real fakes, how many caught)",
        f"  Specificity     : {specificity:.4f}  (of real images, how many correctly classified)",
        "",
        "── Confusion Matrix ────────────────────────────────────────",
        f"  TP (fake→fake)  : {metrics.tp:,}",
        f"  TN (real→real)  : {metrics.tn:,}",
        f"  FP (real→fake)  : {metrics.fp:,}  ← false alarm",
        f"  FN (fake→real)  : {metrics.fn:,}  ← missed fake",
        "",
        "── Interpretation ──────────────────────────────────────────",
    ]

    # Auto-grade
    if metrics.auc >= 0.97:
        grade = "Excellent  (AUC ≥ 0.97)"
    elif metrics.auc >= 0.93:
        grade = "Good  (AUC ≥ 0.93)"
    elif metrics.auc >= 0.85:
        grade = "Fair  (AUC ≥ 0.85)"
    else:
        grade = "Needs improvement  (AUC < 0.85)"

    lines += [
        f"  Overall grade   : {grade}",
        f"  EER assessment  : {'Good (< 10%)' if metrics.eer < 0.10 else 'Fair (< 20%)' if metrics.eer < 0.20 else 'Needs improvement'}",
        "",
        "── Evidence Artifacts ──────────────────────────────────────",
        "  roc_curve.png            — ROC curve with shaded area",
        "  confusion_matrix.png     — Absolute + normalised CM",
        "  score_distribution.png   — Real vs fake score histograms",
        "  threshold_sweep.png      — Metrics vs decision threshold",
        "  calibration_curve.png    — Model confidence calibration",
        "  gradcam_samples/         — GradCAM heatmaps (20 samples)",
        "  metrics.json             — Machine-readable metrics",
        "=" * 65,
    ]

    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")
    logger.success(f"\n{text}")
    logger.info(f"\n  Report saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DeepGuard-Image and generate evidence artifacts"
    )
    parser.add_argument("--checkpoint",    required=True,
                        help="Path to trained .pth checkpoint")
    parser.add_argument("--dataset_root",  default="data/processed/cifake")
    parser.add_argument("--output_dir",    default="ml/evaluation")
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument("--workers",       type=int, default=2)
    parser.add_argument("--max_samples",   type=int, default=None,
                        help="Limit test samples (for quick runs)")
    parser.add_argument("--device",        default=None)
    parser.add_argument("--no_gradcam",    action="store_true",
                        help="Skip GradCAM sample generation")
    parser.add_argument("--gradcam_n",     type=int, default=5,
                        help="GradCAM samples per bucket (5 buckets × N = total images)")
    args = parser.parse_args()

    # ── Setup ──────────────────────────────────────────────────────────────
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir    = Path(args.output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {run_dir}")

    # ── Load model ─────────────────────────────────────────────────────────
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if "dummy" in ckpt and ckpt["dummy"]:
        logger.error(
            "This is a DUMMY checkpoint (random weights). "
            "Evaluation on a dummy model is meaningless. "
            "Train the model first."
        )
        sys.exit(1)

    model = build_image_model(device=device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}")
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {n_params:,} parameters")
    if "epoch" in ckpt:
        logger.info(f"Checkpoint from epoch {ckpt['epoch']}")
    if "metrics" in ckpt:
        saved = ckpt["metrics"]
        logger.info(
            f"Saved training metrics — "
            f"val_auc={saved.get('auc', '?'):.4f}  "
            f"val_acc={saved.get('accuracy', '?'):.4f}"
        )

    # ── Load test dataset ──────────────────────────────────────────────────
    logger.info(f"Loading test set from {args.dataset_root}")
    test_ds = DeepfakeImageDataset(
        root_dir=args.dataset_root,
        split="test",
        augment=False,
    )

    if args.max_samples and args.max_samples < len(test_ds):
        indices = np.random.choice(len(test_ds), args.max_samples, replace=False)
        test_ds_eval = Subset(test_ds, indices.tolist())
        logger.info(f"Using {args.max_samples} random test samples")
    else:
        test_ds_eval = test_ds

    test_loader = DataLoader(
        test_ds_eval,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # ── Inference ──────────────────────────────────────────────────────────
    logger.info(f"Running inference on {len(test_ds_eval)} samples…")
    t_start = time.time()
    labels_arr, scores_arr, preds_arr = collect_predictions(model, test_loader, device)
    elapsed = time.time() - t_start
    logger.info(f"Inference complete: {elapsed:.1f}s  ({len(labels_arr)/elapsed:.0f} samples/s)")

    # ── Metrics ────────────────────────────────────────────────────────────
    metrics = compute_metrics(labels_arr.tolist(), scores_arr.tolist())
    logger.success(
        f"AUC={metrics.auc:.4f}  "
        f"Acc={metrics.accuracy:.4f}  "
        f"F1={metrics.f1:.4f}  "
        f"EER={metrics.eer:.4f}"
    )

    # ── Save metrics.json ──────────────────────────────────────────────────
    precision = metrics.tp / (metrics.tp + metrics.fp) if (metrics.tp + metrics.fp) > 0 else 0.0
    recall    = metrics.tp / (metrics.tp + metrics.fn) if (metrics.tp + metrics.fn) > 0 else 0.0
    specificity = metrics.tn / (metrics.tn + metrics.fp) if (metrics.tn + metrics.fp) > 0 else 0.0

    metrics_dict = {
        "model":       "DeepGuard-Image",
        "checkpoint":  args.checkpoint,
        "dataset":     args.dataset_root,
        "split":       "test",
        "n_samples":   int(len(labels_arr)),
        "timestamp":   timestamp,
        "auc_roc":     round(float(metrics.auc), 6),
        "accuracy":    round(float(metrics.accuracy), 6),
        "f1_score":    round(float(metrics.f1), 6),
        "eer":         round(float(metrics.eer), 6),
        "precision":   round(precision, 6),
        "recall":      round(recall, 6),
        "specificity": round(specificity, 6),
        "tp": int(metrics.tp),
        "tn": int(metrics.tn),
        "fp": int(metrics.fp),
        "fn": int(metrics.fn),
        "inference_seconds": round(elapsed, 2),
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_dict, indent=2), encoding="utf-8")
    logger.info(f"  Saved: {metrics_path}")

    # ── Plots ──────────────────────────────────────────────────────────────
    logger.info("Generating plots…")
    plot_roc_curve(
        labels_arr, scores_arr, metrics.auc,
        run_dir / "roc_curve.png",
    )
    plot_confusion_matrix(
        labels_arr, preds_arr,
        metrics.tp, metrics.tn, metrics.fp, metrics.fn,
        run_dir / "confusion_matrix.png",
    )
    plot_score_distribution(labels_arr, scores_arr, run_dir / "score_distribution.png")
    plot_threshold_sweep(labels_arr, scores_arr, run_dir / "threshold_sweep.png")
    plot_calibration_curve(labels_arr, scores_arr, run_dir / "calibration_curve.png")

    # ── GradCAM samples ────────────────────────────────────────────────────
    if not args.no_gradcam:
        logger.info("Generating GradCAM sample overlays…")
        generate_gradcam_samples(
            model       = model,
            dataset     = test_ds,
            labels_arr  = labels_arr,
            scores_arr  = scores_arr,
            preds_arr   = preds_arr,
            out_dir     = run_dir / "gradcam_samples",
            n_each      = args.gradcam_n,
            device      = device,
        )

    # ── Text report ────────────────────────────────────────────────────────
    write_report(
        out_path         = run_dir / "report.txt",
        checkpoint_path  = args.checkpoint,
        dataset_root     = args.dataset_root,
        n_test           = int(len(labels_arr)),
        metrics          = metrics,
        elapsed          = elapsed,
        timestamp        = timestamp,
    )

    logger.success(f"\nAll evaluation artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
