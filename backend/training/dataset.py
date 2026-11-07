"""
Deepfake Image Dataset Loader
================================
Loads a processed dataset for training DeepGuard-Image.

Expected directory structure (produced by scripts/prepare_cifake.py):
  <root>/
    train/
      real/  *.jpg
      fake/  *.jpg
    val/
      real/  *.jpg
      fake/  *.jpg
    test/
      real/  *.jpg
      fake/  *.jpg

Supports CIFAKE and any similarly structured dataset (FaceForensics++ frames,
custom face-crop datasets, etc.).

Reproducibility
---------------
Samples are sorted deterministically.  Shuffling is handled by the DataLoader
(or WeightedRandomSampler) so the same seed always produces the same order
across runs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T


# ── Labels ─────────────────────────────────────────────────────────────────
LABEL_REAL = 0
LABEL_FAKE = 1

# ── ImageNet normalisation ──────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
INPUT_SIZE    = 224


# ── Transforms ─────────────────────────────────────────────────────────────

def get_inference_transform() -> T.Compose:
    """Deterministic transform used at val / test / inference time."""
    return T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=T.InterpolationMode.LANCZOS),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_training_transform() -> T.Compose:
    """
    Augmentation strategy for training.

    Goal: teach the model to detect manipulation artifacts robustly across
    compression levels, scales, and minor colour shifts — without destroying
    the high-frequency GAN/diffusion fingerprints the model must also detect.

    Kept deliberately *light* on geometric transforms so spatial forgery
    patterns (blending boundaries, asymmetric noise) are not erased.
    """
    return T.Compose([
        # Random crop from slightly larger image (preserves most content)
        T.Resize((INPUT_SIZE + 32, INPUT_SIZE + 32)),
        T.RandomCrop(INPUT_SIZE),

        # Horizontal flip — faces are bilaterally symmetric
        T.RandomHorizontalFlip(p=0.5),

        # Mild colour jitter — real cameras have colour variance
        T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.05),

        # Rare grayscale — forces model to not rely solely on colour
        T.RandomGrayscale(p=0.05),

        # JPEG-like blur: deepfakes are often re-compressed after generation
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.25),

        # Random rotation ±10° — slight head tilt should not fool the model
        T.RandomRotation(degrees=10, interpolation=T.InterpolationMode.BILINEAR),

        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

        # Random erasing simulates occlusion / watermarks
        T.RandomErasing(p=0.1, scale=(0.02, 0.08)),
    ])


# ── Dataset ─────────────────────────────────────────────────────────────────

class DeepfakeImageDataset(Dataset):
    """
    Binary classification dataset: real (0) vs fake (1).

    Supports any directory structured as:
      <root>/<split>/real/*.jpg
      <root>/<split>/fake/*.jpg

    Also accepts .png and .webp in addition to .jpg.
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(
        self,
        root_dir: str,
        split: str = "train",           # "train" | "val" | "test"
        augment: bool = True,
        max_per_class: Optional[int] = None,
    ):
        self.root  = Path(root_dir) / split
        self.split = split
        self.transform = (
            get_training_transform() if (split == "train" and augment)
            else get_inference_transform()
        )

        real_dir = self.root / "real"
        fake_dir = self.root / "fake"

        if not real_dir.exists() or not fake_dir.exists():
            raise FileNotFoundError(
                f"Dataset split '{split}' not found at {self.root}.\n"
                "Expected real/ and fake/ subdirectories.\n"
                "Run scripts/prepare_cifake.py first."
            )

        def _collect(directory: Path, label: int) -> List[Tuple[str, int]]:
            files = sorted(
                str(p) for p in directory.iterdir()
                if p.suffix.lower() in self.EXTENSIONS
            )
            return [(f, label) for f in files]

        real_files = _collect(real_dir, LABEL_REAL)
        fake_files = _collect(fake_dir, LABEL_FAKE)

        if not real_files:
            raise ValueError(f"No real images found in {real_dir}")
        if not fake_files:
            raise ValueError(f"No fake images found in {fake_dir}")

        if max_per_class is not None:
            real_files = real_files[:max_per_class]
            fake_files = fake_files[:max_per_class]

        self.samples: List[Tuple[str, int]] = real_files + fake_files

        n_real, n_fake = len(real_files), len(fake_files)
        print(
            f"[Dataset/{split}]  real={n_real:,}  fake={n_fake:,}  "
            f"total={len(self.samples):,}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(
                f"[Dataset/{self.split}] Failed to open image {path!r}: {e}"
            ) from e
        return self.transform(img), label

    def get_class_weights(self) -> torch.Tensor:
        """Per-sample weights for WeightedRandomSampler (handles class imbalance)."""
        labels = [lbl for _, lbl in self.samples]
        n_real = labels.count(LABEL_REAL)
        n_fake = labels.count(LABEL_FAKE)
        total  = len(labels)
        w_real = total / (2 * n_real) if n_real > 0 else 1.0
        w_fake = total / (2 * n_fake) if n_fake > 0 else 1.0
        return torch.FloatTensor(
            [w_real if lbl == LABEL_REAL else w_fake for lbl in labels]
        )

    @property
    def class_balance(self) -> dict:
        n_real = sum(1 for _, lbl in self.samples if lbl == LABEL_REAL)
        n_fake = sum(1 for _, lbl in self.samples if lbl == LABEL_FAKE)
        return {"real": n_real, "fake": n_fake, "total": len(self.samples)}


def build_dataloaders(
    dataset_root: str,
    batch_size: int = 32,
    num_workers: int = 2,
    max_per_class: Optional[int] = None,
    seed: int = 42,
    augment: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test dataloaders with reproducible balanced sampling.

    Args:
        dataset_root:  Path to processed dataset root (train/val/test inside).
        batch_size:    Samples per batch.
        num_workers:   DataLoader worker processes (0 = main process only).
        max_per_class: Cap per class — useful for quick smoke tests.
        seed:          Seed for reproducible WeightedRandomSampler.
        augment:       Apply training augmentation to train split.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds = DeepfakeImageDataset(dataset_root, "train", augment=augment,  max_per_class=max_per_class)
    val_ds   = DeepfakeImageDataset(dataset_root, "val",   augment=False, max_per_class=max_per_class)
    test_ds  = DeepfakeImageDataset(dataset_root, "test",  augment=False, max_per_class=max_per_class)

    # Reproducible balanced sampler for training
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=train_ds.get_class_weights(),
        num_samples=len(train_ds),
        replacement=True,
        generator=generator,
    )

    persistent = num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
    )

    return train_loader, val_loader, test_loader
