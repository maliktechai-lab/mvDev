"""
FaceForensics++ Dataset Loader
================================
Loads the FF++ dataset (and optionally Celeb-DF) for training.

Expected directory structure after preprocessing:
  ml/datasets/processed/
    train/
      real/  *.jpg
      fake/  *.jpg
    val/
      real/  *.jpg
      fake/  *.jpg
    test/
      real/  *.jpg
      fake/  *.jpg

Run ml/datasets/prepare_ff.py to generate this structure from raw FF++ videos.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from preprocessing.image_preprocessor import get_training_transform, get_inference_transform


# ── Labels ─────────────────────────────────────────────────────────────────
LABEL_REAL = 0
LABEL_FAKE = 1


class DeepfakeImageDataset(Dataset):
    """
    Dataset for image-based deepfake detection.
    Reads from a directory with real/ and fake/ subdirectories.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",       # train | val | test
        augment: bool = True,
        max_per_class: Optional[int] = None,
    ):
        self.root = Path(root_dir) / split
        self.transform = get_training_transform(augment=(split == "train" and augment))

        real_dir = self.root / "real"
        fake_dir = self.root / "fake"

        if not real_dir.exists() or not fake_dir.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.root}. "
                "Run ml/datasets/prepare_ff.py first."
            )

        real_files = sorted([
            (str(p), LABEL_REAL)
            for p in real_dir.glob("*.jpg")
        ])
        fake_files = sorted([
            (str(p), LABEL_FAKE)
            for p in fake_dir.glob("*.jpg")
        ])

        # Optional cap per class (for quick experiments)
        if max_per_class:
            real_files = real_files[:max_per_class]
            fake_files = fake_files[:max_per_class]

        self.samples: List[Tuple[str, int]] = real_files + fake_files
        random.shuffle(self.samples)

        n_real = sum(1 for _, l in self.samples if l == LABEL_REAL)
        n_fake = sum(1 for _, l in self.samples if l == LABEL_FAKE)
        print(f"[Dataset] {split}: {n_real} real, {n_fake} fake → {len(self.samples)} total")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            tensor = self.transform(img)
        except Exception as e:
            print(f"[Dataset] Warning: failed to load {path}: {e}")
            # Return a blank image rather than crashing
            tensor = torch.zeros(3, 224, 224)
            label = LABEL_REAL
        return tensor, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute per-sample weights for WeightedRandomSampler (handles class imbalance)."""
        labels = [l for _, l in self.samples]
        n_real = labels.count(LABEL_REAL)
        n_fake = labels.count(LABEL_FAKE)
        total  = len(labels)

        # Weight inversely proportional to class frequency
        w_real = total / (2 * n_real) if n_real > 0 else 1.0
        w_fake = total / (2 * n_fake) if n_fake > 0 else 1.0

        weights = [w_real if l == LABEL_REAL else w_fake for l in labels]
        return torch.FloatTensor(weights)


def build_dataloaders(
    dataset_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    max_per_class: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/val/test dataloaders.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds = DeepfakeImageDataset(dataset_root, "train", augment=True,  max_per_class=max_per_class)
    val_ds   = DeepfakeImageDataset(dataset_root, "val",   augment=False, max_per_class=max_per_class)
    test_ds  = DeepfakeImageDataset(dataset_root, "test",  augment=False, max_per_class=max_per_class)

    # Balanced sampling for training
    sampler = WeightedRandomSampler(
        weights=train_ds.get_class_weights(),
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, test_loader
