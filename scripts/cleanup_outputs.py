#!/usr/bin/env python3
"""PaperBanana outputs cleanup utility.

Usage:
    python scripts/cleanup_outputs.py --dry-run          # preview what would be deleted
    python scripts/cleanup_outputs.py --keep-days 7      # delete runs older than 7 days
    python scripts/cleanup_outputs.py --keep-best 5      # keep only the 5 most recent runs
    python scripts/cleanup_outputs.py --slim              # remove intermediates from all runs
    python scripts/cleanup_outputs.py --keep-days 3 --slim  # combine: old runs deleted, rest slimmed
"""

import argparse
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"

KEEP_FILES = {"final_output.png", "metadata.json", "planning.json"}
INTERMEDIATE_PATTERNS = ["iter_*", "diagram_iter_*.png", "plot_iter_*.png"]


def parse_run_date(run_dir: Path) -> datetime | None:
    """Extract datetime from run directory name: run_YYYYMMDD_HHMMSS_HEXCODE."""
    name = run_dir.name
    if not name.startswith("run_") or len(name) < 20:
        return None
    try:
        return datetime.strptime(name[4:19], "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def get_run_dirs() -> list[Path]:
    """List all run directories sorted by name (chronological)."""
    if not OUTPUTS_DIR.exists():
        return []
    return sorted(
        [d for d in OUTPUTS_DIR.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda d: d.name,
    )


def dir_size(path: Path) -> int:
    """Total size of directory in bytes."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def fmt_size(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}KB"
    return f"{n / (1024 * 1024):.1f}MB"


def slim_run(run_dir: Path, dry_run: bool = False) -> int:
    """Remove intermediate files from a run directory. Returns bytes freed."""
    freed = 0
    for pattern in INTERMEDIATE_PATTERNS:
        for item in run_dir.glob(pattern):
            size = dir_size(item) if item.is_dir() else item.stat().st_size
            freed += size
            if dry_run:
                print(f"  would delete: {item.relative_to(OUTPUTS_DIR)} ({fmt_size(size)})")
            else:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
    return freed


def main():
    parser = argparse.ArgumentParser(description="Cleanup PaperBanana outputs")
    parser.add_argument("--keep-days", type=int, default=None, help="Keep runs from last N days")
    parser.add_argument("--keep-best", type=int, default=None, help="Keep N most recent runs")
    parser.add_argument("--slim", action="store_true", help="Remove intermediates from kept runs")
    parser.add_argument("--dry-run", action="store_true", help="Preview without deleting")
    args = parser.parse_args()

    if args.keep_days is None and args.keep_best is None and not args.slim:
        parser.print_help()
        return

    runs = get_run_dirs()
    if not runs:
        print("No run directories found.")
        return

    print(f"Found {len(runs)} runs in {OUTPUTS_DIR}")
    print(f"Total size: {fmt_size(sum(dir_size(r) for r in runs))}")
    print()

    to_delete = set()
    to_keep = set(runs)

    # Filter by age
    if args.keep_days is not None:
        cutoff = datetime.now() - timedelta(days=args.keep_days)
        for run in runs:
            run_date = parse_run_date(run)
            if run_date and run_date < cutoff:
                to_delete.add(run)
                to_keep.discard(run)

    # Filter by count (keep N most recent)
    if args.keep_best is not None:
        remaining = sorted(to_keep, key=lambda d: d.name, reverse=True)
        for run in remaining[args.keep_best :]:
            to_delete.add(run)
            to_keep.discard(run)

    # Report deletions
    total_freed = 0
    if to_delete:
        print(f"{'[DRY RUN] ' if args.dry_run else ''}Deleting {len(to_delete)} runs:")
        for run in sorted(to_delete, key=lambda d: d.name):
            size = dir_size(run)
            total_freed += size
            run_date = parse_run_date(run)
            date_str = run_date.strftime("%Y-%m-%d %H:%M") if run_date else "unknown"
            print(f"  {run.name}  ({date_str}, {fmt_size(size)})")
            if not args.dry_run:
                shutil.rmtree(run)

    # Slim remaining runs
    slim_freed = 0
    if args.slim and to_keep:
        print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Slimming {len(to_keep)} kept runs:")
        for run in sorted(to_keep, key=lambda d: d.name):
            freed = slim_run(run, dry_run=args.dry_run)
            slim_freed += freed
            if freed > 0:
                print(f"  {run.name}: freed {fmt_size(freed)}")

    # Summary
    print(f"\n--- Summary ---")
    print(f"Runs deleted: {len(to_delete)}")
    print(f"Runs kept: {len(to_keep)}")
    print(f"Space freed (deletions): {fmt_size(total_freed)}")
    if args.slim:
        print(f"Space freed (slimming): {fmt_size(slim_freed)}")
    print(f"Total freed: {fmt_size(total_freed + slim_freed)}")
    if args.dry_run:
        print("\n(dry run — no files were actually deleted)")


if __name__ == "__main__":
    main()
