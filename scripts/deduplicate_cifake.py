from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import shutil
import sys

SOURCE = Path("data/processed/cifake")
OUTPUT = Path("data/processed/cifake_clean")
REPORTS = Path("data/reports")

# When duplicates occur across splits, preserve the higher-priority copy.
PRIORITY = {"test": 0, "val": 1, "train": 2}
LABELS = ("real", "fake")
SPLITS = ("train", "val", "test")


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    if not SOURCE.is_dir():
        sys.exit(f"Source dataset not found: {SOURCE}")

    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        sys.exit(
            f"Output directory is not empty: {OUTPUT}\n"
            "No files were modified. Choose a fresh output directory."
        )

    images = []

    print("Scanning dataset and calculating hashes...")

    for split in SPLITS:
        for label in LABELS:
            folder = SOURCE / split / label

            if not folder.is_dir():
                sys.exit(f"Missing directory: {folder}")

            files = sorted(folder.glob("*.jpg"))

            if not files:
                sys.exit(f"No JPG images found in {folder}")

            for path in files:
                images.append({
                    "path": path,
                    "split": split,
                    "label": label,
                    "sha256": sha256_file(path),
                })

    print(f"Images scanned: {len(images):,}")

    groups = defaultdict(list)

    for image in images:
        groups[image["sha256"]].append(image)

    # Do not make automatic decisions if one image has conflicting labels.
    label_conflicts = []

    for digest, copies in groups.items():
        labels = {item["label"] for item in copies}

        if len(labels) > 1:
            label_conflicts.append((digest, copies))

    if label_conflicts:
        print(
            f"ERROR: Found {len(label_conflicts)} hashes "
            "with conflicting labels."
        )
        print("No output dataset was created.")

        for digest, copies in label_conflicts[:10]:
            print(f"\nSHA256: {digest}")

            for item in copies:
                print(
                    f"  {item['split']}/{item['label']}: "
                    f"{item['path']}"
                )

        sys.exit(1)

    winners = []
    removed = []

    for digest, copies in groups.items():
        # Keep test first, then validation, then training.
        # Within each split, keep one deterministic path.
        ordered = sorted(
            copies,
            key=lambda item: (
                PRIORITY[item["split"]],
                str(item["path"]),
            ),
        )

        winner = ordered[0]
        winners.append(winner)

        for duplicate in ordered[1:]:
            removed.append({
                "sha256": digest,
                "removed_path": str(duplicate["path"]),
                "removed_split": duplicate["split"],
                "removed_label": duplicate["label"],
                "retained_path": str(winner["path"]),
                "retained_split": winner["split"],
                "retained_label": winner["label"],
            })

    print(f"Unique image hashes: {len(groups):,}")
    print(f"Duplicate copies to remove: {len(removed):,}")
    print("Label conflicts: 0")

    # Create the new output dataset; leave the source untouched.
    OUTPUT.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    clean_manifest = REPORTS / "cifake_clean_manifest.csv"
    removal_report = REPORTS / "cifake_duplicates_removed.csv"

    try:
        with clean_manifest.open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.writer(f)

            writer.writerow([
                "split", "label", "path",
                "source_path", "sha256"
            ])

            for item in sorted(
                winners,
                key=lambda x: (
                    PRIORITY[x["split"]],
                    x["label"],
                    str(x["path"]),
                ),
            ):
                split = item["split"]
                label = item["label"]

                destination_dir = OUTPUT / split / label
                destination_dir.mkdir(parents=True, exist_ok=True)

                destination = destination_dir / item["path"].name

                if destination.exists():
                    raise FileExistsError(
                        f"Unexpected destination collision: {destination}"
                    )

                shutil.copy2(item["path"], destination)

                writer.writerow([
                    split,
                    label,
                    str(destination),
                    str(item["path"]),
                    item["sha256"],
                ])

                counts[(split, label)] += 1

        with removal_report.open(
            "w", newline="", encoding="utf-8"
        ) as f:
            fields = [
                "sha256",
                "removed_path",
                "removed_split",
                "removed_label",
                "retained_path",
                "retained_split",
                "retained_label",
            ]

            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(removed)

    except Exception:
        print(
            "\nERROR while creating clean dataset. "
            "Some files may have been copied to the new output directory."
        )
        print("The original processed dataset remains untouched.")
        raise

    print("\n========== CLEAN DATASET SUMMARY ==========")

    for split in SPLITS:
        real_count = counts[(split, "real")]
        fake_count = counts[(split, "fake")]

        print(
            f"{split}: REAL={real_count:,}, "
            f"FAKE={fake_count:,}, "
            f"TOTAL={real_count + fake_count:,}"
        )

    print(f"\nClean dataset: {OUTPUT}")
    print(f"Clean manifest: {clean_manifest}")
    print(f"Removal report: {removal_report}")
    print(f"Removed duplicate copies: {len(removed):,}")
    print(f"Remaining unique images: {sum(counts.values()):,}")


if __name__ == "__main__":
    main()
