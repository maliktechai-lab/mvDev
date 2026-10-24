"""
FaceForensics++ Dataset Preparation Script
===========================================
Extracts face crops from FF++ video frames and organises them into the
structure expected by the training pipeline.

  Input:  Raw FF++ videos (downloaded separately — see README)
  Output: ml/datasets/processed/{train,val,test}/{real,fake}/*.jpg
          ml/datasets/processed/manifest.json  ← reproducibility record

Reproducibility
---------------
Source videos are split into train/val/test BEFORE any frames are
extracted.  This prevents frames from the same source video from
appearing in different splits (data leakage).

A JSON manifest is written at output_dir/manifest.json recording:
  - The random seed used
  - The split ratios
  - Which source videos were assigned to each split

Usage:
    python ml/datasets/prepare_ff.py \\
        --raw_dir  ml/datasets/raw/FaceForensics \\
        --out_dir  ml/datasets/processed \\
        --max_frames 50 \\
        --seed 42

FF++ Download:
    Visit https://github.com/ondyari/FaceForensics
    Request access: http://kaldir.vc.in.tum.de/faceforensics/
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add backend to path for face detector import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "backend"))
from preprocessing.face_detector import FaceDetector


# FF++ manipulation types to include as fake samples
FAKE_METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]

# Source-video-level split ratios — must sum to 1.0
SPLIT_RATIOS: Dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}


# ── Frame extraction ────────────────────────────────────────────────────────

def extract_frames(
    video_path: str,
    max_frames: int = 50,
    sample_fps: float = 1.0,
) -> List[np.ndarray]:
    """Return up to max_frames RGB frames sampled at sample_fps."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(fps / sample_fps))
    frames: List[np.ndarray] = []
    idx = 0

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1

    cap.release()
    return frames


def process_video(
    video_path: str,
    output_dir: Path,
    detector: FaceDetector,
    max_frames: int = 50,
) -> int:
    """Extract one face crop per frame and save as JPG. Returns count saved."""
    stem   = Path(video_path).stem
    frames = extract_frames(video_path, max_frames=max_frames)
    saved  = 0

    for i, rgb in enumerate(frames):
        try:
            faces = detector.detect(Image.fromarray(rgb))
            if faces:
                out = output_dir / f"{stem}_f{i:04d}.jpg"
                faces[0].crop.save(str(out), "JPEG", quality=95)
                saved += 1
        except Exception as exc:
            print(f"  Warning: frame {i} of {stem}: {exc}")

    return saved


# ── Source-video-level split ────────────────────────────────────────────────

def split_videos(videos: List[Path], seed: int) -> Dict[str, List[Path]]:
    """
    Assign each source video to exactly one split using a reproducible seed.

    All frames from a given video will be in the SAME split, preventing
    data leakage between train/val/test.
    """
    rng    = random.Random(seed)
    paths  = sorted(videos)       # deterministic base order
    rng.shuffle(paths)

    n     = len(paths)
    n_tr  = max(1, int(n * SPLIT_RATIOS["train"]))
    n_val = max(1, int(n * SPLIT_RATIOS["val"]))
    return {
        "train": paths[:n_tr],
        "val":   paths[n_tr : n_tr + n_val],
        "test":  paths[n_tr + n_val:],       # remainder — no videos dropped
    }


# ── Main ────────────────────────────────────────────────────────────────────

def prepare_dataset(
    raw_dir: str,
    out_dir: str,
    max_frames: int = 50,
    device: str = "cpu",
    seed: int = 42,
) -> None:
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)

    print(f"Raw data : {raw_path}")
    print(f"Output   : {out_path}")
    print(f"Seed     : {seed}")

    for split in SPLIT_RATIOS:
        for label in ("real", "fake"):
            (out_path / split / label).mkdir(parents=True, exist_ok=True)

    detector = FaceDetector(device=device)
    manifest: Dict = {"seed": seed, "splits": SPLIT_RATIOS, "sources": {}}

    # ── Real videos ─────────────────────────────────────────────────────────
    real_dir = raw_path / "original_sequences" / "actors" / "c23" / "videos"
    if not real_dir.exists():
        real_dir = raw_path / "original_sequences" / "youtube" / "c23" / "videos"

    real_videos = sorted(real_dir.glob("*.mp4")) if real_dir.exists() else []
    print(f"\nReal source videos found: {len(real_videos)}")

    real_splits = split_videos(real_videos, seed=seed)
    manifest["sources"]["real"] = {
        s: [str(v) for v in vs] for s, vs in real_splits.items()
    }

    for split, vids in real_splits.items():
        n = sum(
            process_video(str(v), out_path / split / "real", detector, max_frames)
            for v in tqdm(vids, desc=f"real/{split}")
        )
        print(f"  real/{split}: {n} crops from {len(vids)} videos")

    # ── Fake videos ──────────────────────────────────────────────────────────
    for method in FAKE_METHODS:
        fake_dir = raw_path / "manipulated_sequences" / method / "c23" / "videos"
        if not fake_dir.exists():
            print(f"\nSkipping {method} — not found: {fake_dir}")
            continue

        fake_videos = sorted(fake_dir.glob("*.mp4"))
        print(f"\nFake source videos ({method}): {len(fake_videos)}")

        fake_splits = split_videos(fake_videos, seed=seed)
        manifest["sources"][f"fake_{method}"] = {
            s: [str(v) for v in vs] for s, vs in fake_splits.items()
        }

        for split, vids in fake_splits.items():
            n = sum(
                process_video(str(v), out_path / split / "fake", detector, max_frames)
                for v in tqdm(vids, desc=f"fake/{method}/{split}")
            )
            print(f"  fake/{method}/{split}: {n} crops from {len(vids)} videos")

    # ── Write manifest ───────────────────────────────────────────────────────
    manifest_path = out_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nManifest written → {manifest_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n── Dataset Summary ──")
    total = 0
    for split in SPLIT_RATIOS:
        for label in ("real", "fake"):
            count = len(list((out_path / split / label).glob("*.jpg")))
            print(f"  {split}/{label}: {count:,}")
            total += count
    print(f"  Total  : {total:,} face crops")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare FF++ dataset")
    parser.add_argument("--raw_dir",    default="ml/datasets/raw/FaceForensics",
                        help="Path to downloaded FF++ data")
    parser.add_argument("--out_dir",    default="ml/datasets/processed",
                        help="Output directory for processed crops")
    parser.add_argument("--max_frames", type=int, default=50)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Seed for reproducible source-video splits")
    args = parser.parse_args()

    prepare_dataset(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        max_frames=args.max_frames,
        device=args.device,
        seed=args.seed,
    )
