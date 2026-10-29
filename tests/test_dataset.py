"""
Regression tests for backend/training/dataset.py.

Covers:
  Task 1 — fail-fast on decode failure and transform failure
  Task 2 — split validation (empty, single-class, two-class)

No real dataset is used; all images are created in-memory.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image

# Ensure backend is on sys.path (pytest.ini sets pythonpath=backend, but also
# set it explicitly so the file can be run standalone with python -m pytest).
BACKEND_DIR = str(Path(__file__).parent.parent / "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_valid_jpg(path: Path) -> None:
    """Write a minimal valid JPEG to *path*."""
    img = Image.new("RGB", (32, 32), color=(100, 150, 200))
    img.save(str(path), "JPEG")


def _make_split(root: Path, split: str, n_real: int, n_fake: int) -> None:
    """Create <root>/<split>/real/ and fake/ with the given number of images."""
    real_dir = root / split / "real"
    fake_dir = root / split / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_real):
        _write_valid_jpg(real_dir / f"real_{i:04d}.jpg")
    for i in range(n_fake):
        _write_valid_jpg(fake_dir / f"fake_{i:04d}.jpg")


# ---------------------------------------------------------------------------
# Task 1 — Fail-fast on decode failure
# ---------------------------------------------------------------------------

class TestDecodeFailure:
    """__getitem__ must raise RuntimeError (not return zero tensor) on bad image."""

    def test_decode_failure_raises(self, tmp_path):
        """Corrupt JPEG content → RuntimeError with the file path in the message."""
        from training.dataset import DeepfakeImageDataset

        _make_split(tmp_path, "train", n_real=1, n_fake=1)

        # Overwrite one real image with garbage bytes
        corrupt = tmp_path / "train" / "real" / "real_0000.jpg"
        corrupt.write_bytes(b"\x00\x01\x02\x03GARBAGE_NOT_JPEG" * 20)

        ds = DeepfakeImageDataset(str(tmp_path), split="train", augment=False)

        # Find the index of the corrupt sample
        corrupt_idx = next(
            i for i, (p, _) in enumerate(ds.samples) if "real_0000" in p
        )

        with pytest.raises(RuntimeError) as exc_info:
            _ = ds[corrupt_idx]

        assert "real_0000.jpg" in str(exc_info.value), (
            "RuntimeError must include the offending file path"
        )

    def test_decode_failure_does_not_change_label(self, tmp_path):
        """On decode failure, the original label must not be silently changed to REAL."""
        from training.dataset import DeepfakeImageDataset, LABEL_FAKE

        _make_split(tmp_path, "train", n_real=1, n_fake=1)

        corrupt = tmp_path / "train" / "fake" / "fake_0000.jpg"
        corrupt.write_bytes(b"NOT_AN_IMAGE" * 30)

        ds = DeepfakeImageDataset(str(tmp_path), split="train", augment=False)
        corrupt_idx = next(
            i for i, (p, _) in enumerate(ds.samples) if "fake_0000" in p
        )

        # Confirm the stored label is still FAKE before the error fires
        assert ds.samples[corrupt_idx][1] == LABEL_FAKE

        with pytest.raises(RuntimeError):
            _ = ds[corrupt_idx]

    def test_decode_failure_does_not_return_zero_tensor(self, tmp_path):
        """Must not silently return a zero tensor on decode failure."""
        import torch
        from training.dataset import DeepfakeImageDataset

        _make_split(tmp_path, "train", n_real=1, n_fake=1)

        corrupt = tmp_path / "train" / "real" / "real_0000.jpg"
        corrupt.write_bytes(b"CORRUPT" * 50)

        ds = DeepfakeImageDataset(str(tmp_path), split="train", augment=False)
        corrupt_idx = next(
            i for i, (p, _) in enumerate(ds.samples) if "real_0000" in p
        )

        with pytest.raises(RuntimeError):
            tensor, _ = ds[corrupt_idx]
            # If we somehow reach here (which we should not), assert it's not zero
            assert not torch.all(tensor == 0), "Must not silently return zero tensor"


# ---------------------------------------------------------------------------
# Task 1 — Fail-fast on transform failure
# ---------------------------------------------------------------------------

class TestTransformFailure:
    """__getitem__ must raise RuntimeError when the transform raises."""

    def test_transform_failure_raises(self, tmp_path):
        """Mock transform that raises → RuntimeError with file path in message."""
        from training.dataset import DeepfakeImageDataset

        _make_split(tmp_path, "train", n_real=1, n_fake=1)
        ds = DeepfakeImageDataset(str(tmp_path), split="train", augment=False)

        # Replace the dataset's transform with one that always raises
        bad_transform = MagicMock(side_effect=RuntimeError("mock transform error"))
        ds.transform = bad_transform

        idx = 0
        path = ds.samples[idx][0]

        with pytest.raises(RuntimeError) as exc_info:
            _ = ds[idx]

        assert Path(path).name in str(exc_info.value), (
            "RuntimeError from a transform failure must include the offending file path"
        )


# ---------------------------------------------------------------------------
# Task 2 — Split validation: class presence requirements
# ---------------------------------------------------------------------------

class TestSplitValidation:
    """Both REAL and FAKE must be present; errors must be descriptive."""

    def test_empty_split_raises(self, tmp_path):
        """Split with zero real AND zero fake samples → ValueError."""
        from training.dataset import DeepfakeImageDataset

        _make_split(tmp_path, "train", n_real=0, n_fake=0)
        # We still need the directories to exist even with 0 files
        (tmp_path / "train" / "real").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "fake").mkdir(parents=True, exist_ok=True)

        with pytest.raises(ValueError, match="empty"):
            DeepfakeImageDataset(str(tmp_path), split="train", augment=False)

    def test_real_only_split_raises(self, tmp_path):
        """Split with only REAL samples → ValueError mentioning 'FAKE'."""
        from training.dataset import DeepfakeImageDataset

        _make_split(tmp_path, "train", n_real=3, n_fake=0)
        # Ensure fake dir exists but is empty
        (tmp_path / "train" / "fake").mkdir(parents=True, exist_ok=True)

        with pytest.raises(ValueError, match="(?i)fake"):
            DeepfakeImageDataset(str(tmp_path), split="train", augment=False)

    def test_fake_only_split_raises(self, tmp_path):
        """Split with only FAKE samples → ValueError mentioning 'REAL'."""
        from training.dataset import DeepfakeImageDataset

        _make_split(tmp_path, "train", n_real=0, n_fake=3)
        # Ensure real dir exists but is empty
        (tmp_path / "train" / "real").mkdir(parents=True, exist_ok=True)

        with pytest.raises(ValueError, match="(?i)real"):
            DeepfakeImageDataset(str(tmp_path), split="train", augment=False)

    def test_valid_two_class_split_succeeds(self, tmp_path):
        """Split with both classes → no error, correct sample count."""
        from training.dataset import DeepfakeImageDataset

        _make_split(tmp_path, "train", n_real=4, n_fake=4)
        ds = DeepfakeImageDataset(str(tmp_path), split="train", augment=False)

        balance = ds.class_balance
        assert balance["real"] == 4
        assert balance["fake"] == 4
        assert balance["total"] == 8

    def test_valid_split_returns_correct_labels(self, tmp_path):
        """Samples loaded from real/ have label 0, from fake/ have label 1."""
        from training.dataset import DeepfakeImageDataset, LABEL_REAL, LABEL_FAKE

        _make_split(tmp_path, "val", n_real=2, n_fake=2)
        ds = DeepfakeImageDataset(str(tmp_path), split="val", augment=False)

        real_labels = [lbl for path, lbl in ds.samples if "real" in path]
        fake_labels = [lbl for path, lbl in ds.samples if "fake" in path]
        assert all(l == LABEL_REAL for l in real_labels)
        assert all(l == LABEL_FAKE for l in fake_labels)
