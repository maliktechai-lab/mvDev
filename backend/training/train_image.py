"""
Image Model Trainer
====================
Full training pipeline for DeepGuard-Image (from scratch) with:
  - Deep supervision (auxiliary losses per branch)
  - Label smoothing
  - AdamW + CosineAnnealingLR with warm-up
  - Early stopping (on validation AUC)
  - Mixed precision (AMP) when GPU available
  - Checkpoint saving (best by AUC + periodic)
  - Epoch-level logging

Usage:
    python backend/training/train_image.py \
        --dataset_root ml/datasets/processed \
        --epochs 50 \
        --batch_size 32 \
        --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from loguru import logger

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.image_model import DeepGuardImageModel, build_image_model
from training.dataset import build_dataloaders
from training.metrics import MetricsTracker, EpochMetrics


# ── Loss with deep supervision ─────────────────────────────────────────────

class DeepSupervisionLoss(nn.Module):
    """
    Combines main fusion loss with per-branch auxiliary losses.
    Main loss weight = 1.0; each auxiliary branch = 0.3.
    Label smoothing reduces overconfident predictions on noisy data.
    """

    def __init__(self, label_smoothing: float = 0.1, aux_weight: float = 0.3):
        super().__init__()
        self.main_loss  = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.aux_loss   = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.aux_weight = aux_weight

    def forward(
        self,
        logits:        torch.Tensor,
        aux_spatial:   torch.Tensor,
        aux_frequency: torch.Tensor,
        aux_attention: torch.Tensor,
        labels:        torch.Tensor,
    ) -> torch.Tensor:
        loss  = self.main_loss(logits, labels)
        loss += self.aux_weight * self.aux_loss(aux_spatial,   labels)
        loss += self.aux_weight * self.aux_loss(aux_frequency, labels)
        loss += self.aux_weight * self.aux_loss(aux_attention, labels)
        return loss


# ── LR warmup + cosine decay ───────────────────────────────────────────────

def build_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """Linear warmup for the first `warmup_epochs`, then cosine decay."""
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Trainer ────────────────────────────────────────────────────────────────

class ImageModelTrainer:

    def __init__(
        self,
        model:           DeepGuardImageModel,
        device:          str,
        learning_rate:   float = 3e-4,
        weight_decay:    float = 1e-4,
        epochs:          int   = 50,
        warmup_epochs:   int   = 5,
        patience:        int   = 10,
        checkpoint_dir:  str   = "ml/checkpoints",
        label_smoothing: float = 0.1,
    ):
        self.model          = model.to(device)
        self.device         = device
        self.epochs         = epochs
        self.patience       = patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = DeepSupervisionLoss(
            label_smoothing=label_smoothing, aux_weight=0.3
        )

        # Single unified optimizer — no backbone/head split needed for scratch model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        self.scheduler = build_scheduler(self.optimizer, warmup_epochs, epochs)

        # Mixed precision (GPU only)
        self.use_amp = (device == "cuda")
        self.scaler  = GradScaler(enabled=self.use_amp)

        self.best_auc          = 0.0
        self.no_improve_epochs = 0

    # ── One epoch pass ─────────────────────────────────────────────────────

    def _run_epoch(self, loader, train: bool) -> EpochMetrics:
        self.model.train(train)
        tracker = MetricsTracker()

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                out  = self.model(images)
                loss = self.criterion(
                    out["logits"],
                    out["aux_spatial"],
                    out["aux_frequency"],
                    out["aux_attention"],
                    labels,
                )

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            probs        = F.softmax(out["logits"], dim=1)
            fake_scores  = probs[:, 1].detach().cpu().tolist()
            batch_labels = labels.detach().cpu().tolist()
            tracker.update(batch_labels, fake_scores, float(loss.item()))

            if batch_idx % 50 == 0:
                phase = "train" if train else "val  "
                cur_lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"[{phase}] batch {batch_idx:04d}/{len(loader)}  "
                    f"loss={loss.item():.4f}  lr={cur_lr:.2e}"
                )

        return tracker.compute()

    # ── Checkpoint ─────────────────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, metrics: EpochMetrics, tag: str = "best"):
        path = self.checkpoint_dir / f"deepguard_image_{tag}.pth"
        torch.save({
            "epoch":               epoch,
            "model_state_dict":    self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": {
                "auc":      metrics.auc,
                "accuracy": metrics.accuracy,
                "f1":       metrics.f1,
                "eer":      metrics.eer,
            },
        }, path)
        # Also save as the canonical name the registry looks for
        canonical = self.checkpoint_dir / "deepguard_image.pth"
        if tag == "best":
            import shutil
            shutil.copy2(path, canonical)
        logger.info(f"Checkpoint saved → {path}")

    # ── Full training loop ─────────────────────────────────────────────────

    def train(self, train_loader, val_loader):
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"Starting from-scratch training: {self.epochs} epochs on {self.device}  "
            f"model_params={n_params:,}"
        )

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_m = self._run_epoch(train_loader, train=True)
            val_m   = self._run_epoch(val_loader,   train=False)
            self.scheduler.step()

            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch:03d}/{self.epochs}  "
                f"train: {train_m}  val: {val_m}  "
                f"time={elapsed:.1f}s"
            )

            if val_m.auc > self.best_auc:
                self.best_auc          = val_m.auc
                self.no_improve_epochs = 0
                self.save_checkpoint(epoch, val_m, tag="best")
                logger.success(f"↑ New best val AUC: {self.best_auc:.4f}")
            else:
                self.no_improve_epochs += 1
                logger.info(
                    f"No improvement for {self.no_improve_epochs}/{self.patience} epochs"
                )

            if epoch % 5 == 0:
                self.save_checkpoint(epoch, val_m, tag=f"epoch_{epoch:03d}")

            if self.no_improve_epochs >= self.patience:
                logger.warning(f"Early stopping at epoch {epoch}")
                break

        logger.info(f"Training complete. Best val AUC: {self.best_auc:.4f}")


# ── CLI entry point ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train DeepGuard-Image model from scratch")
    parser.add_argument("--dataset_root",  default="ml/datasets/processed")
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--lr",            type=float, default=3e-4)
    parser.add_argument("--weight_decay",  type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int,   default=5)
    parser.add_argument("--patience",      type=int,   default=10)
    parser.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers",       type=int,   default=4)
    parser.add_argument("--max_per_class", type=int,   default=None,
                        help="Cap samples per class — useful for quick smoke tests")
    parser.add_argument("--checkpoint_dir", default="ml/checkpoints")
    args = parser.parse_args()

    logger.info(f"Config: {vars(args)}")

    # Data
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.workers,
        max_per_class=args.max_per_class,
    )

    # Model — always from scratch
    model = build_image_model(device=args.device)
    logger.info(f"Model built from scratch — {sum(p.numel() for p in model.parameters()):,} params")

    # Train
    trainer = ImageModelTrainer(
        model=model,
        device=args.device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
    )
    trainer.train(train_loader, val_loader)

    # Test set evaluation
    logger.info("Evaluating on test set...")
    tracker = MetricsTracker()
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            out    = model(images)
            probs  = F.softmax(out["logits"], dim=1)
            tracker.update(labels.cpu().tolist(), probs[:, 1].cpu().tolist(), 0.0)

    test_m = tracker.compute()
    logger.success(f"Test results: {test_m}")


if __name__ == "__main__":
    main()
