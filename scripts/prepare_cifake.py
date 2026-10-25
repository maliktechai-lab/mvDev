from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import random
import shutil
import sys

SOURCE = Path("data/raw/cifake")
OUTPUT = Path("data/processed/cifake")
REPORTS = Path("data/reports")
SEED = 42
VAL_FRACTION = 0.10

LABELS = {
    "REAL": "real",
    "FAKE": "fake",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if not SOURCE.is_dir():
        sys.exit(f"Dataset not found: {SOURCE}")

    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        sys.exit(
            f"Output already contains files: {OUTPUT}\n"
            "Refusing to overwrite existing data."
        )

    for split in ("train", "val", "test"):
        for label in ("real", "fake"):
            source_split = "test" if split == "test" else "train"
            source_label = label.upper()

            source_dir = SOURCE / source_split / source_label

            if not source_dir.is_dir():
                sys.exit(f"Missing directory: {source_dir}")

            files = sorted(source_dir.glob("*.jpg"))

            if not files:
                sys.exit(f"No JPG images found in {source_dir}")

            if split == "test":
                selected = files
            else:
                rng = random.Random(
                    SEED + (0 if label == "real" else 1)
                )
                shuffled = files.copy()
                rng.shuffle(shuffled)

                val_count = int(len(shuffled) * VAL_FRACTION)

                if split == "val":
                    selected = shuffled[:val_count]
                else:
                    selected = shuffled[val_count:]

            dest_dir = OUTPUT / split / label
            dest_dir.mkdir(parents=True, exist_ok=True)

            print(
                f"Preparing {split}/{label}: "
                f"{len(selected):,} images"
            )

            for index, source_path in enumerate(selected, start=1):
                dest_path = dest_dir / source_path.name

                shutil.copyfile(source_path, dest_path)

                if index % 10000 == 0:
                    print(f"  Copied {index:,}/{len(selected):,}")

    print("\nPreparing manifests and checking duplicate hashes...")

    REPORTS.mkdir(parents=True, exist_ok=True)
    manifest_path = REPORTS / "cifake_manifest.csv"

    records = []
    hash_locations = defaultdict(set)
    counts = Counter()

    with manifest_path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split", "label", "path",
                "source_path", "sha256", "seed"
            ],
        )
        writer.writeheader()

        for split in ("train", "val", "test"):
            for label in ("real", "fake"):
                dest_dir = OUTPUT / split / label

                for dest_path in sorted(dest_dir.glob("*.jpg")):
                    source_path = (
                        SOURCE
                        / ("test" if split == "test" else "train")
                        / label.upper()
                        / dest_path.name
                    )

                    digest = sha256_file(dest_path)

                    record = {
                        "split": split,
                        "label": label,
                        "path": str(dest_path),
                        "source_path": str(source_path),
                        "sha256": digest,
                        "seed": SEED,
                    }

                    writer.writerow(record)
                    records.append(record)
                    hash_locations[digest].add(split)
                    counts[(split, label)] += 1

    duplicate_path = REPORTS / "cifake_cross_split_duplicates.csv"
    duplicate_hashes = {
        digest: splits
        for digest, splits in hash_locations.items()
        if len(splits) > 1
    }

    with duplicate_path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["sha256", "splits"])

        for digest, splits in sorted(duplicate_hashes.items()):
            writer.writerow([digest, ",".join(sorted(splits))])

    print("\n========== DATASET SUMMARY ==========")

    for split in ("train", "val", "test"):
        real_count = counts[(split, "real")]
        fake_count = counts[(split, "fake")]

        print(
            f"{split}: REAL={real_count:,}, "
            f"FAKE={fake_count:,}, "
            f"TOTAL={real_count + fake_count:,}"
        )

    print(f"\nSeed: {SEED}")
    print(f"Manifest: {manifest_path}")
    print(f"Duplicate report: {duplicate_path}")
    print(f"Exact hashes appearing across splits: {len(duplicate_hashes)}")

    if duplicate_hashes:
        print(
            "WARNING: Inspect the duplicate report before trusting "
            "evaluation results. No samples were automatically removed."
        )
    else:
        print("No exact image duplicates were detected across splits.")


if __name__ == "__main__":
    main()
