"""
FaceForensics++ Dataset Preparation Script
===========================================
Extracts face crops from FF++ video frames and organizes them
into the structure expected by the training pipeline.

Input: Raw FF++ videos (downloaded separately)
Output: ml/datasets/processed/{train,val,test}/{real,fake}/*.jpg

Usage:
    python ml/datasets/prepare_ff.py \
        --raw_dir ml/datasets/raw/FaceForensics \
        --out_dir ml/datasets/processed \
        --max_frames 50

FF++ Download:
    Visit https://github.com/ondyari/FaceForensics
    Request access at: http://kaldir.vc.in.tum.de/faceforensics/
    Or use the download script: python faceforensics_download_v4.py . -d all -c c23 -t videos
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
from PIL import Image
from tqdm import tqdm

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))
from preprocessing.face_detector import FaceDetector


# FF++ manipulation types we use for fake samples
FAKE_METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]

# Train/val/test split ratios
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}


def extract_frames_from_video(
    video_path: str,
    max_frames: int = 50,
    sample_fps: float = 1.0,
) -> List[np.ndarray]:
    """Extract up to max_frames frames from a video at sample_fps."""
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, int(fps / sample_fps))
    frames = []
    frame_idx = 0

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            # Convert BGR→RGB
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_idx += 1

    cap.release()
    return frames


def process_video(
    video_path: str,
    output_dir: Path,
    detector: FaceDetector,
    max_frames: int = 50,
    label: str = "real",
) -> int:
    """Extract faces from video and save as JPGs. Returns count saved."""
    import numpy as np

    video_name = Path(video_path).stem
    frames = extract_frames_from_video(video_path, max_frames=max_frames)
    saved = 0

    for i, frame_rgb in enumerate(frames):
        try:
            pil_img = Image.fromarray(frame_rgb)
            faces = detector.detect(pil_img)
            for j, face in enumerate(faces[:1]):  # Take only primary face per frame
                out_path = output_dir / f"{video_name}_f{i:04d}_face{j}.jpg"
                face.crop.save(str(out_path), "JPEG", quality=95)
                saved += 1
        except Exception as e:
            print(f"Warning: frame {i} of {video_name}: {e}")

    return saved


def split_files(files: List[Path]) -> dict:
    """Randomly split files into train/val/test."""
    random.shuffle(files)
    n = len(files)
    n_train = int(n * SPLITS["train"])
    n_val   = int(n * SPLITS["val"])
    return {
        "train": files[:n_train],
        "val":   files[n_train:n_train + n_val],
        "test":  files[n_train + n_val:],
    }


def prepare_dataset(
    raw_dir: str,
    out_dir: str,
    max_frames: int = 50,
    device: str = "cpu",
):
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)

    print(f"Raw data: {raw_path}")
    print(f"Output:   {out_path}")

    # Create output structure
    for split in ["train", "val", "test"]:
        for label in ["real", "fake"]:
            (out_path / split / label).mkdir(parents=True, exist_ok=True)

    detector = FaceDetector(device=device)

    # ── Process real videos ────────────────────────────────────────────────
    real_video_dir = raw_path / "original_sequences" / "actors" / "c23" / "videos"
    if not real_video_dir.exists():
        # Try alternative FF++ path
        real_video_dir = raw_path / "original_sequences" / "youtube" / "c23" / "videos"

    real_videos = sorted(real_video_dir.glob("*.mp4")) if real_video_dir.exists() else []
    print(f"\nFound {len(real_videos)} real videos")

    real_splits = split_files(real_videos)
    for split, vids in real_splits.items():
        for vid in tqdm(vids, desc=f"Real [{split}]"):
            process_video(
                str(vid),
                out_path / split / "real",
                detector,
                max_frames=max_frames,
                label="real",
            )

    # ── Process fake videos ────────────────────────────────────────────────
    for method in FAKE_METHODS:
        fake_video_dir = raw_path / "manipulated_sequences" / method / "c23" / "videos"
        if not fake_video_dir.exists():
            print(f"Skipping {method} — directory not found")
            continue

        fake_videos = sorted(fake_video_dir.glob("*.mp4"))
        print(f"\nFound {len(fake_videos)} fake videos ({method})")

        fake_splits = split_files(fake_videos)
        for split, vids in fake_splits.items():
            for vid in tqdm(vids, desc=f"Fake/{method} [{split}]"):
                process_video(
                    str(vid),
                    out_path / split / "fake",
                    detector,
                    max_frames=max_frames,
                    label="fake",
                )

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n── Dataset Summary ──")
    total = 0
    for split in ["train", "val", "test"]:
        for label in ["real", "fake"]:
            count = len(list((out_path / split / label).glob("*.jpg")))
            print(f"  {split}/{label}: {count}")
            total += count
    print(f"  Total: {total} face crops")


if __name__ == "__main__":
    import numpy as np

    parser = argparse.ArgumentParser(description="Prepare FF++ dataset")
    parser.add_argument("--raw_dir",    default="ml/datasets/raw/FaceForensics")
    parser.add_argument("--out_dir",    default="ml/datasets/processed")
    parser.add_argument("--max_frames", type=int, default=50)
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()

    prepare_dataset(args.raw_dir, args.out_dir, args.max_frames, args.device)
