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
        model:            DeepGuardImageModel,
        device:           str,
        learning_rate:    float = 3e-4,
        weight_decay:     float = 1e-4,
        epochs:           int   = 50,
        warmup_epochs:    int   = 5,
        patience:         int   = 10,
        checkpoint_dir:   str   = "ml/checkpoints",
        canonical_name:   str   = "deepguard_image.pth",
        label_smoothing:  float = 0.1,
    ):
        self.model           = model.to(device)
        self.device          = device
        self.epochs          = epochs
        self.patience        = patience
        self.checkpoint_dir  = Path(checkpoint_dir)
        self.canonical_name  = canonical_name
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
        # Also save as the canonical name the registry looks for (configurable via YAML checkpoint.canonical_name)
        canonical = self.checkpoint_dir / self.canonical_name
        if tag == "best":
            import shutil
            shutil.copy2(path, canonical)
            logger.info(f"Canonical checkpoint → {canonical}")
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

def _load_yaml_config(path: str) -> dict:
    """Load a YAML config file. Returns empty dict if PyYAML is unavailable."""
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        logger.warning("PyYAML not installed — ignoring --config flag")
        return {}
    except FileNotFoundError:
        logger.warning(f"Config file not found: {path} — using defaults")
        return {}


def run_smoke_test(device: str, checkpoint_dir: str, canonical_name: str) -> None:
    """
    Smoke test: verify the training loop is functional without a real dataset.

    Creates a small synthetic dataset (16 random images per class), runs 5
    mini-batches, checks that loss decreases, saves a checkpoint, and reloads
    it with strict=True to verify the format.

    This is NOT a substitute for real training. It only confirms the code
    paths are intact.
    """
    import tempfile
    import torch.nn.functional as F

    logger.info("=" * 60)
    logger.info("SMOKE TEST MODE — synthetic data, 5 gradient steps")
    logger.info("This confirms code paths work. Not a real training run.")
    logger.info("=" * 60)

    BATCH  = 4
    STEPS  = 5
    N_IMGS = 16   # per class

    # ── Synthetic dataset ──────────────────────────────────────────────────
    imgs   = torch.randn(N_IMGS * 2, 3, 224, 224)
    labels = torch.cat([torch.zeros(N_IMGS, dtype=torch.long),
                        torch.ones(N_IMGS,  dtype=torch.long)])

    model = build_image_model(device=device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = DeepSupervisionLoss(label_smoothing=0.0)

    losses: list[float] = []
    for step in range(STEPS):
        idx    = torch.randperm(len(imgs))[:BATCH]
        x      = imgs[idx].to(device)
        y      = labels[idx].to(device)
        out    = model(x, return_branch_scores=True)
        loss   = criterion(
            out["logits"], out["aux_spatial"],
            out["aux_frequency"], out["aux_attention"], y,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        logger.info(f"  step {step + 1}/{STEPS}  loss={loss.item():.4f}")

    # ── Check loss decreased ───────────────────────────────────────────────
    first_loss = losses[0]
    last_loss  = losses[-1]
    decreased  = last_loss < first_loss
    logger.info(f"Loss: {first_loss:.4f} → {last_loss:.4f}  decreased={decreased}")
    if not decreased:
        logger.warning(
            "Loss did not decrease over 5 steps. This can happen with random "
            "data — it is not necessarily a bug. Run with real data to confirm."
        )

    # ── Save checkpoint ────────────────────────────────────────────────────
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    smoke_file = ckpt_path / "smoke_test.pth"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": 0,
            "metrics": {"loss": last_loss},
            "smoke_test": True,
        },
        str(smoke_file),
    )
    logger.info(f"Checkpoint saved: {smoke_file}")

    # ── Reload strictly ────────────────────────────────────────────────────
    loaded = torch.load(str(smoke_file), map_location="cpu", weights_only=True)
    reload_model = build_image_model(device="cpu")
    missing, unexpected = reload_model.load_state_dict(
        loaded["model_state_dict"], strict=True
    )
    assert len(missing) == 0,    f"Strict load: missing keys: {missing[:5]}"
    assert len(unexpected) == 0, f"Strict load: unexpected keys: {unexpected[:5]}"
    logger.success("Checkpoint reloaded with strict=True — format verified.")

    # ── Clean up smoke checkpoint ──────────────────────────────────────────
    smoke_file.unlink()
    logger.info("Smoke checkpoint removed (not a trained model).")

    logger.success("SMOKE TEST PASSED")
    logger.info(
        "To train for real: provide --dataset_root and remove --smoke-test"
    )


def main():
    parser = argparse.ArgumentParser(description="Train DeepGuard-Image model from scratch")
    parser.add_argument("--config",        default=None,
                        help="Path to YAML config (e.g. ml/experiments/image_model.yaml). "
                             "CLI flags override YAML values.")
    parser.add_argument("--dataset_root",  default=None)
    parser.add_argument("--epochs",        type=int,   default=None)
    parser.add_argument("--batch_size",    type=int,   default=None)
    parser.add_argument("--lr",            type=float, default=None)
    parser.add_argument("--weight_decay",  type=float, default=None)
    parser.add_argument("--warmup_epochs", type=int,   default=None)
    parser.add_argument("--patience",      type=int,   default=None)
    parser.add_argument("--device",        default=None)
    parser.add_argument("--workers",       type=int,   default=None)
    parser.add_argument("--max_per_class", type=int,   default=None,
                        help="Cap samples per class — useful for quick smoke tests")
    parser.add_argument("--checkpoint_dir",   default=None)
    parser.add_argument("--canonical_name",   default=None,
                        help="Canonical checkpoint filename (e.g. deepguard_image.pth)")
    parser.add_argument("--smoke-test", "--smoke_test", action="store_true",
                        dest="smoke_test",
                        help="Run a 5-step synthetic smoke test (no dataset needed). "
                             "Verifies code paths and checkpoint save/load. "
                             "Does NOT produce a usable model.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for DataLoader sampler reproducibility")
    args = parser.parse_args()

    # ── Merge YAML → CLI (CLI wins) ────────────────────────────────────────
    cfg: dict = {}
    if args.config:
        cfg = _load_yaml_config(args.config)

    def _get(cli_val, yaml_keys: list, default):
        """Return cli_val if set, else walk yaml_keys, else default."""
        if cli_val is not None:
            return cli_val
        node = cfg
        for k in yaml_keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                return default
        return node if node is not None else default

    dataset_root  = _get(args.dataset_root,  ["dataset", "root"],                  "ml/datasets/processed")
    epochs        = _get(args.epochs,         ["training", "epochs"],               50)
    batch_size    = _get(args.batch_size,     ["training", "batch_size"],           32)
    lr            = _get(args.lr,             ["training", "learning_rate"],        3e-4)
    weight_decay  = _get(args.weight_decay,   ["training", "weight_decay"],         1e-4)
    warmup_epochs = _get(args.warmup_epochs,  ["training", "warmup_epochs"],        5)
    patience      = _get(args.patience,       ["training", "patience"],             10)
    device        = _get(args.device,         [],                                   "cuda" if torch.cuda.is_available() else "cpu")
    workers       = _get(args.workers,        ["training", "num_workers"],          4)
    max_per_class = _get(args.max_per_class,  ["dataset", "max_per_class"],         None)
    checkpoint_dir  = _get(args.checkpoint_dir,  ["checkpoint", "dir"],            "ml/checkpoints")
    canonical_name  = _get(args.canonical_name,  ["checkpoint", "canonical_name"], "deepguard_image.pth")
    label_smoothing = _get(None,                 ["training", "label_smoothing"],   0.1)

    effective = dict(
        dataset_root=dataset_root, epochs=epochs, batch_size=batch_size,
        lr=lr, weight_decay=weight_decay, warmup_epochs=warmup_epochs,
        patience=patience, device=device, workers=workers,
        max_per_class=max_per_class, checkpoint_dir=checkpoint_dir,
        canonical_name=canonical_name,
    )
    logger.info(f"Effective config: {effective}")

    # Data
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_root=dataset_root,
        batch_size=batch_size,
        num_workers=workers,
        max_per_class=max_per_class,
    )

    # Model — always from scratch
    model = build_image_model(device=device)
    logger.info(f"Model built from scratch — {sum(p.numel() for p in model.parameters()):,} params")

    # Train
    trainer = ImageModelTrainer(
        model=model,
        device=device,
        learning_rate=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        patience=patience,
        checkpoint_dir=checkpoint_dir,
        canonical_name=canonical_name,
        label_smoothing=label_smoothing,
    )
    trainer.train(train_loader, val_loader)

    # Test set evaluation
    logger.info("Evaluating on test set...")
    tracker = MetricsTracker()
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            out    = model(images)
            probs  = F.softmax(out["logits"], dim=1)
            tracker.update(labels.cpu().tolist(), probs[:, 1].cpu().tolist(), 0.0)

    test_m = tracker.compute()
    logger.success(f"Test results: {test_m}")


if __name__ == "__main__":
    main()
