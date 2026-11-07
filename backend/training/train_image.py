"""
DeepGuard-Image — Training Script
===================================
Trains DeepGuardImageModel entirely from scratch on a labelled
real/fake image dataset (default: CIFAKE).

Features
--------
  - Deep supervision  (auxiliary branch losses at 0.3 weight)
  - Label smoothing   (reduces overconfidence on noisy labels)
  - AdamW + cosine LR decay with linear warm-up
  - Gradient clipping (max_norm 1.0)
  - Mixed-precision AMP (GPU only; CPU trains in full fp32)
  - Class-balanced WeightedRandomSampler
  - Early stopping on validation AUC-ROC
  - Best + periodic checkpoints
  - Resume from checkpoint (--resume)
  - Epoch-level + batch-level logging

Usage
-----
  # Full training (CPU, CIFAKE, sensible defaults)
  python backend/training/train_image.py \\
      --dataset_root data/processed/cifake \\
      --checkpoint_dir ml/checkpoints \\
      --epochs 30 \\
      --batch_size 32 \\
      --device cpu

  # Resume from a checkpoint
  python backend/training/train_image.py \\
      --dataset_root data/processed/cifake \\
      --checkpoint_dir ml/checkpoints \\
      --resume ml/checkpoints/deepguard_image_best.pth

  # Quick smoke test (no dataset needed)
  python backend/training/train_image.py --smoke-test

  # GPU training
  python backend/training/train_image.py \\
      --dataset_root data/processed/cifake \\
      --epochs 50 --batch_size 64 --device cuda
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

# Allow running from repo root or backend/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.image_model import DeepGuardImageModel, build_image_model
from training.dataset import build_dataloaders
from training.metrics import MetricsTracker, EpochMetrics


# ──────────────────────────────────────────────────────────────────────────
# Loss: main fusion head + per-branch auxiliary heads (deep supervision)
# ──────────────────────────────────────────────────────────────────────────

class DeepSupervisionLoss(nn.Module):
    """
    Weighted sum of four cross-entropy losses:
      main × 1.0  +  spatial × 0.3  +  frequency × 0.3  +  attention × 0.3

    Deep supervision forces each branch to independently learn
    discriminative features even before the fusion head is applied.

    Label smoothing (ε = 0.1 by default) reduces overconfidence and
    improves calibration on noisy/ambiguous face images.
    """

    def __init__(self, label_smoothing: float = 0.1, aux_weight: float = 0.3):
        super().__init__()
        self.main = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.aux  = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.w    = aux_weight

    def forward(
        self,
        logits:        torch.Tensor,   # (B, 2)
        aux_spatial:   torch.Tensor,   # (B, 2)
        aux_frequency: torch.Tensor,   # (B, 2)
        aux_attention: torch.Tensor,   # (B, 2)
        labels:        torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        loss  = self.main(logits,        labels)
        loss += self.w * self.aux(aux_spatial,   labels)
        loss += self.w * self.aux(aux_frequency, labels)
        loss += self.w * self.aux(aux_attention, labels)
        return loss


# ──────────────────────────────────────────────────────────────────────────
# LR schedule: linear warm-up → cosine decay
# ──────────────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """Linear warmup for the first `warmup_epochs`, then cosine decay."""
    import math

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ──────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────

class ImageModelTrainer:

    LOG_INTERVAL = 100   # log every N batches

    def __init__(
        self,
        model:           DeepGuardImageModel,
        device:          str,
        learning_rate:   float = 3e-4,
        weight_decay:    float = 1e-4,
        epochs:          int   = 30,
        warmup_epochs:   int   = 3,
        patience:        int   = 8,
        checkpoint_dir:  str   = "ml/checkpoints",
        canonical_name:  str   = "deepguard_image.pth",
        label_smoothing: float = 0.1,
        resume_path:     Optional[str] = None,
    ):
        self.model          = model.to(device)
        self.device         = device
        self.epochs         = epochs
        self.patience       = patience
        self.ckpt_dir       = Path(checkpoint_dir)
        self.canonical_name = canonical_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = DeepSupervisionLoss(
            label_smoothing=label_smoothing, aux_weight=0.3,
        )
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )
        self.scheduler = build_scheduler(self.optimizer, warmup_epochs, epochs)

        # AMP only on GPU — CPU training uses fp32
        self.use_amp = (device == "cuda")
        if self.use_amp:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
        else:
            self.scaler = None

        self.best_auc          = 0.0
        self.no_improve_epochs = 0
        self.start_epoch       = 1

        # Resume
        if resume_path:
            self._load_checkpoint(resume_path)

    # ── Checkpoint ─────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, metrics: EpochMetrics, tag: str):
        path = self.ckpt_dir / f"deepguard_image_{tag}.pth"
        torch.save(
            {
                "epoch":                epoch,
                "model_state_dict":     self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_auc":             self.best_auc,
                "metrics": {
                    "auc":      metrics.auc,
                    "accuracy": metrics.accuracy,
                    "f1":       metrics.f1,
                    "eer":      metrics.eer,
                    "loss":     metrics.loss,
                },
            },
            path,
        )
        if tag == "best":
            canonical = self.ckpt_dir / self.canonical_name
            shutil.copy2(path, canonical)
            logger.info(f"  ✓ Canonical checkpoint → {canonical}")
        logger.info(f"  ✓ Saved {path}")

    def _load_checkpoint(self, path: str):
        logger.info(f"Resuming from {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        if "model_state_dict" not in ckpt:
            raise ValueError(f"Checkpoint at {path} has no 'model_state_dict' key")
        missing, unexpected = self.model.load_state_dict(
            ckpt["model_state_dict"], strict=False
        )
        if missing:
            logger.warning(f"  Missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            logger.warning(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
        if "optimizer_state_dict" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                logger.warning(f"  Could not restore optimizer state: {e}")
        if "scheduler_state_dict" in ckpt:
            try:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception as e:
                logger.warning(f"  Could not restore scheduler state: {e}")
        self.best_auc    = ckpt.get("best_auc", 0.0)
        self.start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(
            f"  Resumed from epoch {self.start_epoch - 1}  "
            f"best_auc={self.best_auc:.4f}"
        )

    # ── Epoch pass ─────────────────────────────────────────────────────────

    def _run_epoch(self, loader, train: bool) -> EpochMetrics:
        self.model.train(train)
        tracker = MetricsTracker()
        phase   = "TRAIN" if train else "VAL  "
        n_batches = len(loader)

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if train and self.use_amp:
                from torch.cuda.amp import autocast
                with autocast():
                    out  = self.model(images)
                    loss = self.criterion(
                        out["logits"], out["aux_spatial"],
                        out["aux_frequency"], out["aux_attention"], labels,
                    )
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            elif train:
                out  = self.model(images)
                loss = self.criterion(
                    out["logits"], out["aux_spatial"],
                    out["aux_frequency"], out["aux_attention"], labels,
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            else:
                with torch.no_grad():
                    out  = self.model(images)
                    loss = self.criterion(
                        out["logits"], out["aux_spatial"],
                        out["aux_frequency"], out["aux_attention"], labels,
                    )

            probs  = F.softmax(out["logits"], dim=1)
            scores = probs[:, 1].detach().cpu().tolist()
            lbls   = labels.detach().cpu().tolist()
            tracker.update(lbls, scores, float(loss.item()))

            if (batch_idx + 1) % self.LOG_INTERVAL == 0 or batch_idx == 0:
                cur_lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"  [{phase}] batch {batch_idx+1:04d}/{n_batches}  "
                    f"loss={loss.item():.4f}  lr={cur_lr:.2e}"
                )

        return tracker.compute()

    # ── Main training loop ──────────────────────────────────────────────────

    def train(self, train_loader, val_loader) -> None:
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info("=" * 70)
        logger.info("DeepGuard-Image — Training from scratch")
        logger.info(f"  Parameters : {n_params:,}")
        logger.info(f"  Device     : {self.device}")
        logger.info(f"  Epochs     : {self.start_epoch} → {self.epochs}")
        logger.info(f"  AMP        : {self.use_amp}")
        logger.info(f"  Patience   : {self.patience}")
        logger.info(f"  Best AUC   : {self.best_auc:.4f} (from resume)")
        logger.info("=" * 70)

        for epoch in range(self.start_epoch, self.epochs + 1):
            t0 = time.time()
            logger.info(f"\n── Epoch {epoch}/{self.epochs} ──────────────────")

            train_m = self._run_epoch(train_loader, train=True)
            val_m   = self._run_epoch(val_loader,   train=False)
            self.scheduler.step()

            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch:03d}  "
                f"train_loss={train_m.loss:.4f}  train_auc={train_m.auc:.4f}  "
                f"val_loss={val_m.loss:.4f}  val_auc={val_m.auc:.4f}  "
                f"val_acc={val_m.accuracy:.4f}  val_f1={val_m.f1:.4f}  "
                f"val_eer={val_m.eer:.4f}  time={elapsed:.0f}s"
            )

            # ── Save best ──────────────────────────────────────────────────
            if val_m.auc > self.best_auc:
                self.best_auc          = val_m.auc
                self.no_improve_epochs = 0
                self._save_checkpoint(epoch, val_m, tag="best")
                logger.success(
                    f"  ↑ New best  val_auc={self.best_auc:.4f}  "
                    f"acc={val_m.accuracy:.4f}  f1={val_m.f1:.4f}  "
                    f"eer={val_m.eer:.4f}"
                )
            else:
                self.no_improve_epochs += 1
                logger.info(
                    f"  No improvement  "
                    f"({self.no_improve_epochs}/{self.patience})"
                )

            # ── Periodic checkpoint every 5 epochs ────────────────────────
            if epoch % 5 == 0:
                self._save_checkpoint(epoch, val_m, tag=f"epoch{epoch:03d}")

            # ── Early stopping ─────────────────────────────────────────────
            if self.no_improve_epochs >= self.patience:
                logger.warning(f"Early stopping triggered at epoch {epoch}.")
                break

        logger.success(
            f"\nTraining complete.  Best val AUC: {self.best_auc:.4f}"
        )


# ──────────────────────────────────────────────────────────────────────────
# Smoke test — no dataset needed
# ──────────────────────────────────────────────────────────────────────────

def run_smoke_test(device: str, checkpoint_dir: str, canonical_name: str) -> None:
    """
    5-step synthetic smoke test.
    Verifies the full forward→backward→save→reload code path.
    Does NOT produce a usable model.
    """
    logger.info("=" * 60)
    logger.info("SMOKE TEST — 5 gradient steps on synthetic data")
    logger.info("=" * 60)

    model     = build_image_model(device=device)
    criterion = DeepSupervisionLoss(label_smoothing=0.0)
    opt       = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses: list[float] = []
    for step in range(5):
        x = torch.randn(4, 3, 224, 224).to(device)
        y = torch.randint(0, 2, (4,)).to(device)
        out  = model(x)
        loss = criterion(
            out["logits"], out["aux_spatial"],
            out["aux_frequency"], out["aux_attention"], y,
        )
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss))
        logger.info(f"  step {step+1}/5  loss={loss:.4f}")

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    smoke_file = ckpt_dir / "_smoke_test.pth"

    torch.save({"model_state_dict": model.state_dict(), "epoch": 0}, smoke_file)

    # Reload strictly
    reload = build_image_model(device="cpu")
    state  = torch.load(smoke_file, map_location="cpu", weights_only=True)
    missing, unexpected = reload.load_state_dict(state["model_state_dict"], strict=True)
    assert not missing,    f"Missing keys: {missing[:5]}"
    assert not unexpected, f"Unexpected keys: {unexpected[:5]}"

    smoke_file.unlink()
    logger.success("SMOKE TEST PASSED — code paths verified.")


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train DeepGuard-Image model from scratch"
    )

    parser.add_argument("--dataset_root",  default="data/processed/cifake")
    parser.add_argument("--checkpoint_dir", default="ml/checkpoints")
    parser.add_argument("--canonical_name", default="deepguard_image.pth",
                        help="Filename the registry and API look for")
    parser.add_argument("--epochs",        type=int,   default=30)
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--lr",            type=float, default=3e-4)
    parser.add_argument("--weight_decay",  type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int,   default=3)
    parser.add_argument("--patience",      type=int,   default=8)
    parser.add_argument("--device",        default=None,
                        help="cuda / cpu (auto-detected if omitted)")
    parser.add_argument("--workers",       type=int,   default=2,
                        help="DataLoader worker processes (0 = main process)")
    parser.add_argument("--max_per_class", type=int,   default=None,
                        help="Cap per class — quick smoke / debug runs")
    parser.add_argument("--resume",        default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--smoke-test",    action="store_true", dest="smoke_test",
                        help="Quick 5-step code path verification (no dataset needed)")

    args = parser.parse_args()

    # ── Device ──────────────────────────────────────────────────────────────
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Device: {device}")

    # ── Reproducibility ──────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # ── Smoke test ──────────────────────────────────────────────────────────
    if args.smoke_test:
        run_smoke_test(device, args.checkpoint_dir, args.canonical_name)
        return

    # ── Data ────────────────────────────────────────────────────────────────
    logger.info(f"Loading dataset from {args.dataset_root}")
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.workers,
        max_per_class=args.max_per_class,
        seed=args.seed,
        augment=True,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_image_model(device=device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: DeepGuardImageModel  params={n_params:,}")

    # ── Train ────────────────────────────────────────────────────────────────
    trainer = ImageModelTrainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
        canonical_name=args.canonical_name,
        label_smoothing=args.label_smoothing,
        resume_path=args.resume,
    )
    trainer.train(train_loader, val_loader)

    # ── Test set evaluation ──────────────────────────────────────────────────
    logger.info("\nEvaluating on held-out test set (best checkpoint)…")

    best_ckpt = Path(args.checkpoint_dir) / "deepguard_image_best.pth"
    if best_ckpt.exists():
        ckpt  = torch.load(best_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"  Loaded best checkpoint from epoch {ckpt.get('epoch', '?')}")

    model.eval()
    tracker = MetricsTracker()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            out    = model(images)
            probs  = F.softmax(out["logits"], dim=1)
            tracker.update(labels.cpu().tolist(), probs[:, 1].cpu().tolist(), 0.0)

    test_m = tracker.compute()
    logger.success(
        f"\n{'='*60}\n"
        f"TEST SET RESULTS\n"
        f"  AUC-ROC  : {test_m.auc:.4f}\n"
        f"  Accuracy : {test_m.accuracy:.4f}\n"
        f"  F1 Score : {test_m.f1:.4f}\n"
        f"  EER      : {test_m.eer:.4f}\n"
        f"  TP={test_m.tp}  TN={test_m.tn}  FP={test_m.fp}  FN={test_m.fn}\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()
