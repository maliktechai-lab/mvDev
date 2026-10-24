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

Reproducibility
---------------
The dataset does NOT shuffle samples internally — shuffling is handled by
the DataLoader (shuffle=True for train) so it is controlled by the
PyTorch global seed set at training startup. This prevents the dataset
object itself from being non-deterministic across runs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Optional

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

    Samples are sorted deterministically — DataLoader shuffle controls
    training randomness so the same seed always produces the same order.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",       # train | val | test
        augment: bool = True,
        max_per_class: Optional[int] = None,
    ):
        self.root = Path(root_dir) / split
        self.split = split
        self.transform = get_training_transform(augment=(split == "train" and augment))

        real_dir = self.root / "real"
        fake_dir = self.root / "fake"

        if not real_dir.exists() or not fake_dir.exists():
            raise FileNotFoundError(
                f"Dataset split '{split}' not found at {self.root}. "
                "Expected real/ and fake/ subdirectories. "
                "Run ml/datasets/prepare_ff.py first."
            )

        # Sorted for determinism — DataLoader shuffles for training
        real_files = sorted([
            (str(p), LABEL_REAL)
            for p in real_dir.glob("*.jpg")
        ])
        fake_files = sorted([
            (str(p), LABEL_FAKE)
            for p in fake_dir.glob("*.jpg")
        ])

        if len(real_files) == 0 and len(fake_files) == 0:
            raise ValueError(
                f"Dataset split '{split}' at {self.root} is empty. "
                "No .jpg files found in real/ or fake/. "
                "Check that prepare_ff.py completed successfully."
            )

        # Optional cap per class (for quick experiments / smoke tests)
        if max_per_class is not None:
            real_files = real_files[:max_per_class]
            fake_files = fake_files[:max_per_class]

        # Deterministic interleave: real first, then fake — no shuffle here
        self.samples: List[Tuple[str, int]] = real_files + fake_files

        n_real = len(real_files)
        n_fake = len(fake_files)
        print(
            f"[Dataset/{split}] {n_real} real + {n_fake} fake = {len(self.samples)} samples"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            tensor = self.transform(img)
        except Exception as e:
            print(f"[Dataset] Warning: failed to load {path}: {e}")
            # Return a blank tensor rather than crashing the DataLoader
            tensor = torch.zeros(3, 224, 224)
            label = LABEL_REAL
        return tensor, label

    def get_class_weights(self) -> torch.Tensor:
        """Per-sample weights for WeightedRandomSampler (handles class imbalance)."""
        labels = [lbl for _, lbl in self.samples]
        n_real = labels.count(LABEL_REAL)
        n_fake = labels.count(LABEL_FAKE)
        total  = len(labels)

        # Weight inversely proportional to class frequency
        w_real = total / (2 * n_real) if n_real > 0 else 1.0
        w_fake = total / (2 * n_fake) if n_fake > 0 else 1.0

        weights = [w_real if lbl == LABEL_REAL else w_fake for lbl in labels]
        return torch.FloatTensor(weights)

    @property
    def class_balance(self) -> dict:
        """Return class counts for verification."""
        n_real = sum(1 for _, lbl in self.samples if lbl == LABEL_REAL)
        n_fake = sum(1 for _, lbl in self.samples if lbl == LABEL_FAKE)
        return {"real": n_real, "fake": n_fake, "total": len(self.samples)}


def build_dataloaders(
    dataset_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    max_per_class: Optional[int] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/val/test dataloaders with reproducible sampling.

    Args:
        dataset_root:  Path to processed dataset (containing train/val/test/).
        batch_size:    Samples per batch.
        num_workers:   DataLoader worker processes.
        max_per_class: Cap per class (useful for smoke tests).
        seed:          Seed for WeightedRandomSampler generator.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds = DeepfakeImageDataset(dataset_root, "train", augment=True,  max_per_class=max_per_class)
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
