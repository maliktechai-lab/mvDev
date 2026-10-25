#!/usr/bin/env python3
"""Prepare FaceForensics++ face crops using official video-ID/pair splits.

Input layout (after approved download):
  RAW/original_sequences/youtube/c23/videos/<id>.mp4
  RAW/manipulated_sequences/<METHOD>/c23/videos/<target>_<source>.mp4

Output layout (compatible with DeepfakeImageDataset):
  OUT/{train,val,test}/{real,fake}/*.jpg
  OUT/manifest.csv
  OUT/summary.json

This script does not download the gated dataset. Follow the official dataset terms.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

SPLITS = ("train", "val", "test")
DEFAULT_METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
VIDEO_SUFFIXES = {".mp4", ".MP4"}


def normalize_id(value: str) -> str:
    value = str(value).strip()
    if not value.isdigit():
        raise ValueError(f"Expected numeric video ID, got {value!r}")
    number = int(value)
    if not 0 <= number <= 999:
        raise ValueError(f"Video ID must be in 000..999: {value!r}")
    return f"{number:03d}"


def load_splits(split_dir: Path):
    ids_by_split: Dict[str, set[str]] = {}
    pairs_by_split: Dict[str, set[frozenset[str]]] = {}
    id_to_split: Dict[str, str] = {}
    pair_to_split: Dict[frozenset[str], str] = {}

    for split in SPLITS:
        path = split_dir / f"{split}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing split file: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not data:
            raise ValueError(f"Expected non-empty list of pairs in {path}")
        seen_pairs: set[frozenset[str]] = set()
        split_ids: set[str] = set()
        for pair in data:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(f"Invalid pair in {path}: {pair!r}")
            a, b = normalize_id(pair[0]), normalize_id(pair[1])
            if a == b:
                raise ValueError(f"Self-pair in {path}: {pair!r}")
            key = frozenset((a, b))
            if key in seen_pairs:
                raise ValueError(f"Duplicate unordered pair in {path}: {pair!r}")
            seen_pairs.add(key)
            split_ids.update((a, b))
        for video_id in split_ids:
            old = id_to_split.get(video_id)
            if old is not None and old != split:
                raise ValueError(f"Video ID {video_id} overlaps {old}/{split}")
            id_to_split[video_id] = split
        for pair_key in seen_pairs:
            old = pair_to_split.get(pair_key)
            if old is not None and old != split:
                raise ValueError(f"Pair {sorted(pair_key)} overlaps {old}/{split}")
            pair_to_split[pair_key] = split
        ids_by_split[split] = split_ids
        pairs_by_split[split] = seen_pairs
    return ids_by_split, pairs_by_split, id_to_split, pair_to_split


def validate_only(split_dir: Path) -> None:
    ids_by_split, pairs_by_split, id_to_split, pair_to_split = load_splits(split_dir)
    print("========== OFFICIAL FF++ SPLIT VALIDATION ==========")
    for split in SPLITS:
        print(f"{split}: {len(pairs_by_split[split])} pairs, {len(ids_by_split[split])} IDs")
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            print(f"{a}/{b}: {(ids_by_split[a] & ids_by_split[b]).__len__()} overlapping IDs, "
                  f"{len(pairs_by_split[a] & pairs_by_split[b])} overlapping pairs")
    print(f"Total distinct source IDs: {len(id_to_split)}")
    print(f"Total distinct unordered pairs: {len(pair_to_split)}")
    print("PASS: official split files are internally consistent.")


def list_videos(folder: Path) -> List[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".mp4")


def inventory(raw_dir: Path, methods: Tuple[str, ...], id_to_split, pair_to_split):
    originals_dir = raw_dir / "original_sequences" / "youtube" / "c23" / "videos"
    records = []
    skipped = Counter()
    method_counts = Counter()

    for path in list_videos(originals_dir):
        try:
            video_id = normalize_id(path.stem)
        except ValueError:
            skipped["original_bad_filename"] += 1
            continue
        split = id_to_split.get(video_id)
        if split is None:
            skipped["original_id_not_in_official_splits"] += 1
            continue
        records.append({"path": path, "split": split, "label": "real", "method": "original",
                        "video_id": video_id, "target_id": "", "source_id": ""})

    fake_re = re.compile(r"^(\d{1,3})_(\d{1,3})$")
    for method in methods:
        directory = raw_dir / "manipulated_sequences" / method / "c23" / "videos"
        if not directory.is_dir():
            skipped[f"missing_method_dir:{method}"] += 1
            continue
        for path in list_videos(directory):
            match = fake_re.fullmatch(path.stem)
            if not match:
                skipped[f"fake_bad_filename:{method}"] += 1
                continue
            target_id, source_id = normalize_id(match.group(1)), normalize_id(match.group(2))
            pair_key = frozenset((target_id, source_id))
            split = pair_to_split.get(pair_key)
            if split is None:
                skipped[f"fake_pair_not_in_official_splits:{method}"] += 1
                continue
            if id_to_split.get(target_id) != split or id_to_split.get(source_id) != split:
                skipped[f"fake_pair_crosses_split:{method}"] += 1
                continue
            records.append({"path": path, "split": split, "label": "fake", "method": method,
                            "video_id": path.stem, "target_id": target_id, "source_id": source_id})
            method_counts[(split, method)] += 1
    return records, skipped, method_counts, originals_dir


def extract_crops(video_path: Path, out_dir: Path, prefix: str, detector,
                  max_frames: int, sample_fps: float):
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], "video_open_failed"
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:
        fps = 25.0
    interval = max(1, int(round(fps / sample_fps)))
    crops = []
    frame_idx = sampled = 0
    error = ""
    try:
        while sampled < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            idx = frame_idx
            frame_idx += 1
            if idx % interval:
                continue
            sampled += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                faces = detector.detect(Image.fromarray(rgb))
                if not faces:
                    continue
                face = max(faces, key=lambda f: (max(0, f.box[2]-f.box[0]) * max(0, f.box[3]-f.box[1]), f.probability))
                dst = out_dir / f"{prefix}_f{idx:06d}.jpg"
                face.crop.save(dst, "JPEG", quality=95)
                crops.append({"frame_index": idx, "crop_path": str(dst), "face_probability": face.probability})
            except Exception as exc:
                error = f"frame_{idx}:{type(exc).__name__}:{exc}"
    finally:
        cap.release()
    return crops, error


def run_prepare(args):
    split_dir, raw_dir, out_dir = Path(args.split_dir), Path(args.raw_dir), Path(args.out_dir)
    _, _, id_to_split, pair_to_split = load_splits(split_dir)
    records, skipped, method_counts, originals_dir = inventory(raw_dir, tuple(args.methods), id_to_split, pair_to_split)

    print("========== RAW VIDEO INVENTORY ==========")
    print(f"Raw root: {raw_dir}")
    print(f"Original videos directory: {originals_dir}")
    print(f"Selected methods: {', '.join(args.methods)}")
    for split in SPLITS:
        for label in ("real", "fake"):
            n = sum(r["split"] == split and r["label"] == label for r in records)
            print(f"{split}/{label}: {n} videos")
    if skipped:
        print("Skipped file summary:")
        for reason, count in sorted(skipped.items()):
            print(f"  {reason}: {count}")

    if args.inventory_only:
        if not records:
            raise SystemExit("No recognized videos found. Check --raw_dir and the extracted layout.")
        print("PASS: inventory only; no frames extracted.")
        return
    if not records:
        raise SystemExit("No recognized videos found. Check --raw_dir and the extracted layout.")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"Output is not empty: {out_dir}. Choose a fresh --out_dir; nothing was overwritten.")
    if args.max_frames < 1 or args.sample_fps <= 0:
        raise SystemExit("--max_frames must be >= 1 and --sample_fps must be > 0")

    backend_dir = Path(__file__).resolve().parent.parent / "backend"
    sys.path.insert(0, str(backend_dir))
    from preprocessing.face_detector import FaceDetector

    out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        for label in ("real", "fake"):
            (out_dir / split / label).mkdir(parents=True, exist_ok=True)

    detector = FaceDetector(device=args.device)
    manifest_path = out_dir / "manifest.csv"
    summary = {
        "raw_dir": str(raw_dir), "split_dir": str(split_dir), "methods": list(args.methods),
        "max_frames_per_video": args.max_frames, "sample_fps": args.sample_fps,
        "device": args.device, "video_counts": {}, "crop_counts": {},
        "skipped": dict(skipped), "videos_without_crops": [],
    }
    video_counts = Counter()
    crop_counts = Counter()
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["split", "label", "method", "video_path", "video_id", "target_id", "source_id",
                  "frame_index", "crop_path", "face_probability", "processing_error"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            split, label, method = rec["split"], rec["label"], rec["method"]
            video_counts[(split, label)] += 1
            # Method in prefix prevents collisions between manipulation methods.
            prefix = f"{method}_{rec['video_id']}"
            target_dir = out_dir / split / label
            crops, err = extract_crops(rec["path"], target_dir, prefix, detector, args.max_frames, args.sample_fps)
            crop_counts[(split, label)] += len(crops)
            if not crops:
                summary["videos_without_crops"].append(str(rec["path"]))
            if crops:
                for crop in crops:
                    writer.writerow({"split": split, "label": label, "method": method,
                        "video_path": str(rec["path"]), "video_id": rec["video_id"],
                        "target_id": rec["target_id"], "source_id": rec["source_id"],
                        "frame_index": crop["frame_index"], "crop_path": crop["crop_path"],
                        "face_probability": crop["face_probability"], "processing_error": err})
            else:
                writer.writerow({"split": split, "label": label, "method": method,
                    "video_path": str(rec["path"]), "video_id": rec["video_id"],
                    "target_id": rec["target_id"], "source_id": rec["source_id"],
                    "frame_index": "", "crop_path": "", "face_probability": "", "processing_error": err})
            print(f"[{split}/{label}/{method}] {rec['path'].name}: {len(crops)} crops")

    for split in SPLITS:
        for label in ("real", "fake"):
            summary["video_counts"][f"{split}/{label}"] = video_counts[(split, label)]
            summary["crop_counts"][f"{split}/{label}"] = crop_counts[(split, label)]
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("========== PREPARATION SUMMARY ==========")
    for split in SPLITS:
        print(f"{split}: real crops={crop_counts[(split, 'real')]:,}; fake crops={crop_counts[(split, 'fake')]:,}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")
    print("Note: evaluate on the held-out test split only after model choices are finalized.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split_dir", default="ml/datasets/splits/ffpp_official")
    parser.add_argument("--raw_dir", default="ml/datasets/raw/FaceForensics")
    parser.add_argument("--out_dir", default="ml/datasets/processed_ffpp_official")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS), choices=DEFAULT_METHODS)
    parser.add_argument("--max_frames", type=int, default=50)
    parser.add_argument("--sample_fps", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--validate_splits_only", action="store_true")
    parser.add_argument("--inventory_only", action="store_true")
    args = parser.parse_args()
    if args.validate_splits_only:
        validate_only(Path(args.split_dir))
    else:
        run_prepare(args)


if __name__ == "__main__":
    main()
