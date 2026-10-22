"""
Image Model Trainer
====================
Full training pipeline for DeepGuard-Image with:
  - Deep supervision (auxiliary losses per branch)
  - Label smoothing
  - AdamW + CosineAnnealingLR
  - Early stopping (on validation AUC)
  - Mixed precision (AMP) when GPU available
  - Checkpoint saving (best by AUC)
  - Epoch-level logging

Usage:
    python backend/training/train_image.py \
        --dataset_root ml/datasets/processed \
        --epochs 30 \
        --batch_size 32 \
        --device cuda
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from loguru import logger

from models.image_model import DeepGuardImageModel, build_image_model
from training.dataset import build_dataloaders
from training.metrics import MetricsTracker, EpochMetrics


# ── Loss with deep supervision ─────────────────────────────────────────────

class DeepSupervisionLoss(nn.Module):
    """
    Combines main loss with auxiliary branch losses.
    Main loss weight = 1.0; each auxiliary branch = 0.3
    Also applies label smoothing to reduce overconfidence.
    """

    def __init__(self, label_smoothing: float = 0.1):
        super().__init__()
        self.main_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.aux_loss  = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.aux_weight = 0.3

    def forward(
        self,
        logits: torch.Tensor,
        aux_spatial: torch.Tensor,
        aux_frequency: torch.Tensor,
        aux_attention: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        loss = self.main_loss(logits, labels)
        loss += self.aux_weight * self.aux_loss(aux_spatial,   labels)
        loss += self.aux_weight * self.aux_loss(aux_frequency, labels)
        loss += self.aux_weight * self.aux_loss(aux_attention, labels)
        return loss


# ── Trainer ────────────────────────────────────────────────────────────────

class ImageModelTrainer:

    def __init__(
        self,
        model: DeepGuardImageModel,
        device: str,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        epochs: int = 30,
        patience: int = 7,
        checkpoint_dir: str = "ml/checkpoints",
    ):
        self.model          = model.to(device)
        self.device         = device
        self.epochs         = epochs
        self.patience       = patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = DeepSupervisionLoss(label_smoothing=0.1)

        # Separate LR for backbone (lower) vs new heads (higher)
        backbone_params = list(self.model.spatial_branch.backbone.parameters())
        head_params = [
            p for n, p in self.model.named_parameters()
            if "backbone" not in n
        ]
        self.optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": learning_rate * 0.1},
            {"params": head_params,     "lr": learning_rate},
        ], weight_decay=weight_decay)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

        # Mixed precision (GPU only)
        self.use_amp = (device == "cuda")
        self.scaler  = GradScaler(enabled=self.use_amp)

        self.best_auc = 0.0
        self.no_improve_epochs = 0

    # ── Single epoch ──────────────────────────────────────────────────────

    def _run_epoch(self, loader, train: bool) -> EpochMetrics:
        self.model.train(train)
        tracker = MetricsTracker()

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                out = self.model(images)
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
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            probs = F.softmax(out["logits"], dim=1)
            fake_scores = probs[:, 1].detach().cpu().tolist()
            batch_labels = labels.detach().cpu().tolist()

            tracker.update(batch_labels, fake_scores, float(loss.item()))

            if batch_idx % 50 == 0:
                phase = "train" if train else "val"
                logger.info(f"[{phase}] batch {batch_idx}/{len(loader)}  loss={loss.item():.4f}")

        return tracker.compute()

    # ── Save checkpoint ────────────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, metrics: EpochMetrics, tag: str = "best"):
        path = self.checkpoint_dir / f"deepguard_image_{tag}.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": {
                "auc": metrics.auc,
                "accuracy": metrics.accuracy,
                "f1": metrics.f1,
                "eer": metrics.eer,
            },
        }, path)
        logger.info(f"Checkpoint saved → {path}")

    # ── Full training loop ─────────────────────────────────────────────────

    def train(self, train_loader, val_loader):
        logger.info(f"Starting training: {self.epochs} epochs on {self.device}")

        for epoch in range(1, self.epochs + 1):
            start = time.time()

            train_metrics = self._run_epoch(train_loader, train=True)
            val_metrics   = self._run_epoch(val_loader,   train=False)

            self.scheduler.step()
            elapsed = time.time() - start

            logger.info(
                f"Epoch {epoch:03d}/{self.epochs}  "
                f"train: {train_metrics}  "
                f"val: {val_metrics}  "
                f"time={elapsed:.1f}s"
            )

            # Save best checkpoint by validation AUC
            if val_metrics.auc > self.best_auc:
                self.best_auc = val_metrics.auc
                self.no_improve_epochs = 0
                self.save_checkpoint(epoch, val_metrics, tag="best")
                logger.success(f"↑ New best AUC: {self.best_auc:.4f}")
            else:
                self.no_improve_epochs += 1
                logger.info(f"No improvement for {self.no_improve_epochs}/{self.patience} epochs")

            # Save periodic checkpoint every 5 epochs
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, val_metrics, tag=f"epoch_{epoch:03d}")

            # Early stopping
            if self.no_improve_epochs >= self.patience:
                logger.warning(f"Early stopping triggered at epoch {epoch}")
                break

        logger.info(f"Training complete. Best val AUC: {self.best_auc:.4f}")


# ── CLI entry point ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train DeepGuard-Image model")
    parser.add_argument("--dataset_root", default="ml/datasets/processed")
    parser.add_argument("--epochs",       type=int, default=30)
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--lr",           type=float, default=2e-4)
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers",      type=int, default=4)
    parser.add_argument("--max_per_class", type=int, default=None,
                        help="Cap samples per class (for fast dev runs)")
    parser.add_argument("--pretrained",   action="store_true", default=True)
    parser.add_argument("--checkpoint_dir", default="ml/checkpoints")
    args = parser.parse_args()

    logger.info(f"Config: {vars(args)}")

    # Build data
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.workers,
        max_per_class=args.max_per_class,
    )

    # Build model
    model = build_image_model(pretrained=args.pretrained, device=args.device)
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    trainer = ImageModelTrainer(
        model=model,
        device=args.device,
        learning_rate=args.lr,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
    )
    trainer.train(train_loader, val_loader)

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    tracker = MetricsTracker()
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            out = model(images)
            probs = F.softmax(out["logits"], dim=1)
            tracker.update(
                labels.cpu().tolist(),
                probs[:, 1].cpu().tolist(),
                0.0,
            )
    test_metrics = tracker.compute()
    logger.success(f"Test set results: {test_metrics}")


if __name__ == "__main__":
    main()
