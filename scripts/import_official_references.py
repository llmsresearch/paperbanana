#!/usr/bin/env python3
"""Import reference examples from the official PaperBananaBench dataset.

This script is a thin CLI wrapper around DatasetManager for manual imports.
For most users, `paperbanana data download` is the recommended approach.

Usage:
    # Download and cache expanded references (recommended)
    paperbanana data download

    # Or use this script directly:
    python scripts/import_official_references.py

    # Import from a local PaperBananaBench directory
    python scripts/import_official_references.py --local /path/to/PaperBananaBench

    # Import both diagram and plot references
    python scripts/import_official_references.py --task both
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

from PIL import Image


def compute_aspect_ratio(image_path: Path) -> float | None:
    """Compute aspect ratio from an image file."""
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            return round(w / h, 2) if h > 0 else None
    except Exception:
        return None


def convert_ref_entry(
    entry: dict,
    field_map: dict,
    source_images_dir: Path,
    dest_images_dir: Path,
    task: str,
) -> dict | None:
    """Convert a single official ref.json entry to community index.json format."""
    result = {}

    for official_key, community_key in field_map.items():
        value = entry.get(official_key, "")
        if community_key == "source_context" and isinstance(value, (dict, list)):
            # Plot data is JSON — serialize it
            value = json.dumps(value, indent=2)
        result[community_key] = value

    # Resolve image path
    gt_image_rel = entry.get("path_to_gt_image", "")
    if gt_image_rel:
        source_image = source_images_dir / gt_image_rel
        if not source_image.exists():
            # Try with images/ prefix
            source_image = source_images_dir.parent / gt_image_rel
        if not source_image.exists():
            print(f"  Warning: image not found for {result['id']}: {source_image}")
            return None

        # Copy image to destination
        dest_filename = f"{result['id']}.jpg"
        dest_image = dest_images_dir / dest_filename
        if not dest_image.exists():
            shutil.copy2(source_image, dest_image)

        result["image_path"] = f"images/{dest_filename}"
        result["aspect_ratio"] = compute_aspect_ratio(dest_image)
    else:
        result["image_path"] = ""
        result["aspect_ratio"] = None

    # Carry over source paper info
    result["source_paper"] = result["id"]

    return result


# Field mapping: official → community
FIELD_MAP_DIAGRAM = {
    "id": "id",
    "content": "source_context",
    "visual_intent": "caption",
    "path_to_gt_image": "image_path",
    "category": "category",
}

FIELD_MAP_PLOT = {
    "id": "id",
    "content": "source_context",  # JSON data for plots
    "visual_intent": "caption",
    "path_to_gt_image": "image_path",
    "category": "category",
}

HF_DATASET_URL = (
    "https://huggingface.co/datasets/dwzhu/PaperBananaBench/resolve/main/PaperBananaBench.zip"
)


def import_references(
    bench_dir: Path,
    task: str,
    output_dir: Path,
    split: str = "ref",
) -> int:
    """Import references from PaperBananaBench into community format.

    Args:
        bench_dir: Path to extracted PaperBananaBench directory.
        task: 'diagram', 'plot', or 'both'.
        output_dir: Destination directory for index.json + images.
        split: Which JSON file to load ('ref' for ref.json).

    Returns:
        Number of examples imported.
    """
    tasks = ["diagram", "plot"] if task == "both" else [task]
    all_examples = []

    for t in tasks:
        task_dir = bench_dir / t
        ref_file = task_dir / f"{split}.json"

        if not ref_file.exists():
            print(f"  Skipping {t}: {ref_file} not found")
            continue

        with open(ref_file, encoding="utf-8") as f:
            entries = json.load(f)

        print(f"  Loading {len(entries)} {t} references from {ref_file}")

        images_dir = task_dir / "images"
        dest_images_dir = output_dir / "images"
        dest_images_dir.mkdir(parents=True, exist_ok=True)

        field_map = FIELD_MAP_DIAGRAM if t == "diagram" else FIELD_MAP_PLOT
        count = 0

        for entry in entries:
            # Prefix IDs to avoid collision between diagram and plot
            if task == "both":
                entry = {**entry, "id": f"{t}_{entry['id']}"}

            converted = convert_ref_entry(entry, field_map, images_dir, dest_images_dir, t)
            if converted:
                all_examples.append(converted)
                count += 1

        print(f"  Imported {count}/{len(entries)} {t} references")

    if not all_examples:
        print("Error: No examples imported.")
        return 0

    # Collect categories
    categories = sorted(set(e.get("category", "") for e in all_examples if e.get("category")))

    # Write index.json
    index_data = {
        "metadata": {
            "name": "paperbanana_bench",
            "description": (
                f"Reference set imported from official PaperBananaBench dataset. "
                f"Contains {len(all_examples)} examples across {len(categories)} categories."
            ),
            "version": "3.0.0",
            "source": "https://huggingface.co/datasets/dwzhu/PaperBananaBench",
            "categories": categories,
            "total_examples": len(all_examples),
        },
        "examples": all_examples,
    }

    index_path = output_dir / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)

    print(f"\n  Written {index_path} ({len(all_examples)} examples)")
    return len(all_examples)


def download_dataset(dest_dir: Path) -> Path:
    """Download PaperBananaBench from HuggingFace."""
    import subprocess

    zip_path = dest_dir / "PaperBananaBench.zip"

    print(f"Downloading PaperBananaBench to {zip_path}...")
    result = subprocess.run(
        ["wget", "-c", "-q", "--show-progress", "-O", str(zip_path), HF_DATASET_URL],
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError("Download failed. Try --local with a manually downloaded dataset.")

    print("Extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)

    extracted = dest_dir / "PaperBananaBench"
    if not extracted.exists():
        # Check for nested directory
        candidates = list(dest_dir.glob("*/PaperBananaBench"))
        if candidates:
            extracted = candidates[0]

    return extracted


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Import official PaperBananaBench references.\n\n"
            "Recommended: use 'paperbanana data download' instead."
        ),
    )
    parser.add_argument(
        "--local",
        type=str,
        default=None,
        help="Path to local PaperBananaBench directory (skips download)",
    )
    parser.add_argument(
        "--task",
        choices=["diagram", "plot", "both"],
        default="diagram",
        help="Which task's references to import (default: diagram)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: ~/.cache/paperbanana/reference_sets)",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Keep existing index.json and merge new entries",
    )
    args = parser.parse_args()

    # Default to cache dir if no output specified
    if args.output:
        output_dir = Path(args.output)
    else:
        from paperbanana.data.manager import default_cache_dir

        output_dir = default_cache_dir() / "reference_sets"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Backup existing index.json
    existing_index = output_dir / "index.json"
    if existing_index.exists() and not args.keep_existing:
        backup = output_dir / "index.json.bak"
        shutil.copy2(existing_index, backup)
        print(f"Backed up existing index.json to {backup}")

    if args.local:
        bench_dir = Path(args.local)
        if not bench_dir.exists():
            print(f"Error: {bench_dir} does not exist")
            sys.exit(1)
    else:
        tmp_dir = Path(tempfile.mkdtemp(prefix="paperbanana_"))
        try:
            bench_dir = download_dataset(tmp_dir)
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            sys.exit(1)

    print(f"\nImporting from {bench_dir}...")
    count = import_references(bench_dir, args.task, output_dir)

    if count > 0:
        print(f"\nDone! {count} references imported to {output_dir}/")
        print(f"Images saved to {output_dir}/images/")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
