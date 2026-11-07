"""
DeepGuard-Image-Lite — CPU Training Script
==========================================
Trains the 1.8M-parameter lite model on CIFAKE.
Designed to run on a CPU-only machine in a reasonable time.

Expected wall-clock time on AMD Ryzen 3 (4 cores):
  Full CIFAKE (90k train): ~25 min/epoch × 20 epochs = ~8 h
  Subset 10k/class:        ~ 3 min/epoch × 20 epochs = ~1 h  ← default

Usage
-----
  # Fast proof-of-concept (10k/class, ~1 h total)
  python backend/training/train_image_lite.py \\
      --dataset_root data/processed/cifake \\
      --checkpoint_dir ml/checkpoints \\
      --max_per_class 10000 \\
      --epochs 20

  # Full dataset training (~8 h total)
  python backend/training/train_image_lite.py \\
      --dataset_root data/processed/cifake \\
      --checkpoint_dir ml/checkpoints \\
      --epochs 30

  # Resume after interruption
  python backend/training/train_image_lite.py \\
      --dataset_root data/processed/cifake \\
      --resume ml/checkpoints/deepguard_image_lite_best.pth

  # Smoke test (no dataset)
  python backend/training/train_image_lite.py --smoke-test

Output
------
  ml/checkpoints/deepguard_image_lite_best.pth   ← best by val AUC
  ml/checkpoints/deepguard_image_lite.pth        ← canonical (copy of best)
  ml/checkpoints/deepguard_image_lite_epochNNN.pth
"""

from __future__ import annotations

import argparse
import math
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.image_model_lite import DeepGuardImageModelLite, build_image_model_lite
from training.dataset import build_dataloaders
from training.metrics import MetricsTracker, EpochMetrics


# ──────────────────────────────────────────────────────────────────────────

class DeepSupervisionLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.1, aux_weight: float = 0.3):
        super().__init__()
        self.main = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.aux  = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.w    = aux_weight

    def forward(self, logits, aux_s, aux_f, aux_a, labels):
        return (
            self.main(logits, labels)
            + self.w * self.aux(aux_s, labels)
            + self.w * self.aux(aux_f, labels)
            + self.w * self.aux(aux_a, labels)
        )


def build_scheduler(optimizer, warmup: int, total: int):
    def lr_lambda(ep: int) -> float:
        if ep < warmup:
            return float(ep + 1) / max(1, warmup)
        p = (ep - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ──────────────────────────────────────────────────────────────────────────

class LiteTrainer:

    LOG_INTERVAL = 50

    def __init__(
        self,
        model:           DeepGuardImageModelLite,
        device:          str,
        lr:              float = 3e-4,
        weight_decay:    float = 1e-4,
        epochs:          int   = 20,
        warmup_epochs:   int   = 2,
        patience:        int   = 6,
        checkpoint_dir:  str   = "ml/checkpoints",
        canonical_name:  str   = "deepguard_image_lite.pth",
        label_smoothing: float = 0.1,
        resume_path:     Optional[str] = None,
    ):
        self.model         = model.to(device)
        self.device        = device
        self.epochs        = epochs
        self.patience      = patience
        self.ckpt_dir      = Path(checkpoint_dir)
        self.canonical     = canonical_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = DeepSupervisionLoss(label_smoothing, 0.3)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999),
        )
        self.scheduler = build_scheduler(self.optimizer, warmup_epochs, epochs)

        self.best_auc          = 0.0
        self.no_improve_epochs = 0
        self.start_epoch       = 1

        if resume_path:
            self._load(resume_path)

    # ── Checkpoint ──────────────────────────────────────────────────────────

    def _save(self, epoch: int, metrics: EpochMetrics, tag: str):
        path = self.ckpt_dir / f"deepguard_image_lite_{tag}.pth"
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
            canonical = self.ckpt_dir / self.canonical
            shutil.copy2(path, canonical)
            logger.info(f"  → Canonical: {canonical}")
        logger.info(f"  → Saved: {path}")

    def _load(self, path: str):
        logger.info(f"Resuming from {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        try:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception:
            pass
        self.best_auc    = ckpt.get("best_auc", 0.0)
        self.start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(f"  Resumed: epoch={self.start_epoch-1}  best_auc={self.best_auc:.4f}")

    # ── Epoch pass ──────────────────────────────────────────────────────────

    def _epoch(self, loader, train: bool) -> EpochMetrics:
        self.model.train(train)
        tracker  = MetricsTracker()
        phase    = "TRAIN" if train else "VAL  "
        n        = len(loader)

        for i, (imgs, lbls) in enumerate(loader):
            imgs = imgs.to(self.device, non_blocking=True)
            lbls = lbls.to(self.device, non_blocking=True)

            if train:
                out  = self.model(imgs)
                loss = self.criterion(
                    out["logits"], out["aux_spatial"],
                    out["aux_frequency"], out["aux_attention"], lbls,
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            else:
                with torch.no_grad():
                    out  = self.model(imgs)
                    loss = self.criterion(
                        out["logits"], out["aux_spatial"],
                        out["aux_frequency"], out["aux_attention"], lbls,
                    )

            scores = F.softmax(out["logits"], dim=1)[:, 1].detach().cpu().tolist()
            tracker.update(lbls.cpu().tolist(), scores, float(loss))

            if (i + 1) % self.LOG_INTERVAL == 0 or i == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"  [{phase}] {i+1:04d}/{n}  "
                    f"loss={loss:.4f}  lr={lr:.2e}"
                )

        return tracker.compute()

    # ── Train loop ───────────────────────────────────────────────────────────

    def train(self, train_loader, val_loader) -> None:
        n = sum(p.numel() for p in self.model.parameters())
        logger.info("=" * 65)
        logger.info("DeepGuard-Image-Lite — CPU Training")
        logger.info(f"  Parameters : {n:,}")
        logger.info(f"  Device     : {self.device}")
        logger.info(f"  Epochs     : {self.start_epoch} → {self.epochs}")
        logger.info(f"  Train size : {len(train_loader.dataset):,}  batches={len(train_loader)}")
        logger.info(f"  Val size   : {len(val_loader.dataset):,}   batches={len(val_loader)}")
        logger.info("=" * 65)

        for epoch in range(self.start_epoch, self.epochs + 1):
            t0 = time.time()
            logger.info(f"\n── Epoch {epoch}/{self.epochs} ──────────────────────────────")

            train_m = self._epoch(train_loader, train=True)
            val_m   = self._epoch(val_loader,   train=False)
            self.scheduler.step()
            elapsed = time.time() - t0

            logger.info(
                f"Epoch {epoch:03d}  "
                f"train_loss={train_m.loss:.4f} train_auc={train_m.auc:.4f}  "
                f"val_loss={val_m.loss:.4f} val_auc={val_m.auc:.4f}  "
                f"val_acc={val_m.accuracy:.4f} val_f1={val_m.f1:.4f}  "
                f"val_eer={val_m.eer:.4f}  {elapsed:.0f}s"
            )

            if val_m.auc > self.best_auc:
                self.best_auc          = val_m.auc
                self.no_improve_epochs = 0
                self._save(epoch, val_m, "best")
                logger.success(
                    f"  ↑ New best  auc={self.best_auc:.4f}  "
                    f"acc={val_m.accuracy:.4f}  f1={val_m.f1:.4f}"
                )
            else:
                self.no_improve_epochs += 1
                logger.info(
                    f"  No improvement ({self.no_improve_epochs}/{self.patience})"
                )

            if epoch % 5 == 0:
                self._save(epoch, val_m, f"epoch{epoch:03d}")

            if self.no_improve_epochs >= self.patience:
                logger.warning(f"Early stopping at epoch {epoch}.")
                break

        logger.success(f"Training done.  Best val AUC: {self.best_auc:.4f}")


# ──────────────────────────────────────────────────────────────────────────

def run_smoke_test(device: str, ckpt_dir: str) -> None:
    logger.info("=" * 55)
    logger.info("SMOKE TEST — DeepGuard-Image-Lite")
    logger.info("=" * 55)

    model = build_image_model_lite(device=device)
    crit  = DeepSupervisionLoss(0.0)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(5):
        x = torch.randn(4, 3, 224, 224).to(device)
        y = torch.randint(0, 2, (4,)).to(device)
        out  = model(x)
        loss = crit(
            out["logits"], out["aux_spatial"],
            out["aux_frequency"], out["aux_attention"], y,
        )
        opt.zero_grad(); loss.backward(); opt.step()
        logger.info(f"  step {step+1}/5  loss={loss:.4f}")

    p = Path(ckpt_dir) / "_smoke_lite.pth"
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "epoch": 0}, p)

    reload = build_image_model_lite(device="cpu")
    state  = torch.load(p, map_location="cpu", weights_only=True)
    missing, unexpected = reload.load_state_dict(state["model_state_dict"], strict=True)
    assert not missing and not unexpected

    p.unlink()
    logger.success("SMOKE TEST PASSED")


# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train DeepGuard-Image-Lite on CPU"
    )
    parser.add_argument("--dataset_root",   default="data/processed/cifake")
    parser.add_argument("--checkpoint_dir", default="ml/checkpoints")
    parser.add_argument("--canonical_name", default="deepguard_image_lite.pth")
    parser.add_argument("--epochs",         type=int,   default=20)
    parser.add_argument("--batch_size",     type=int,   default=32)
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--weight_decay",   type=float, default=1e-4)
    parser.add_argument("--label_smoothing",type=float, default=0.1)
    parser.add_argument("--warmup_epochs",  type=int,   default=2)
    parser.add_argument("--patience",       type=int,   default=6)
    parser.add_argument("--device",         default=None)
    parser.add_argument("--workers",        type=int,   default=2)
    parser.add_argument("--max_per_class",  type=int,   default=None,
                        help="Limit per class — None = full dataset (90k/45k/20k)")
    parser.add_argument("--resume",         default=None)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--smoke-test",     action="store_true", dest="smoke_test")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    if args.smoke_test:
        run_smoke_test(device, args.checkpoint_dir)
        return

    logger.info(f"Device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.workers,
        max_per_class=args.max_per_class,
        seed=args.seed,
        augment=True,
    )

    model = build_image_model_lite(device=device)

    trainer = LiteTrainer(
        model=model, device=device,
        lr=args.lr, weight_decay=args.weight_decay,
        epochs=args.epochs, warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
        canonical_name=args.canonical_name,
        label_smoothing=args.label_smoothing,
        resume_path=args.resume,
    )
    trainer.train(train_loader, val_loader)

    # ── Test set evaluation ─────────────────────────────────────────────────
    logger.info("\nEvaluating on test set…")
    best_path = Path(args.checkpoint_dir) / "deepguard_image_lite_best.pth"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"  Loaded best checkpoint (epoch {ckpt.get('epoch','?')})")

    model.eval()
    tracker = MetricsTracker()
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            out  = model(imgs)
            probs = F.softmax(out["logits"], dim=1)
            tracker.update(lbls.tolist(), probs[:, 1].tolist(), 0.0)

    m = tracker.compute()
    logger.success(
        f"\n{'='*60}\n"
        f"TEST SET RESULTS — DeepGuard-Image-Lite\n"
        f"  AUC-ROC  : {m.auc:.4f}\n"
        f"  Accuracy : {m.accuracy:.4f}\n"
        f"  F1 Score : {m.f1:.4f}\n"
        f"  EER      : {m.eer:.4f}\n"
        f"  TP={m.tp}  TN={m.tn}  FP={m.fp}  FN={m.fn}\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()
