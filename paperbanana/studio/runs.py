"""Discover and summarize prior pipeline runs under an output directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def list_run_ids(output_dir: str) -> list[str]:
    """Return run directory names (``run_*``), oldest first."""
    root = Path(output_dir)
    if not root.is_dir():
        return []
    runs = [d.name for d in root.iterdir() if d.is_dir() and d.name.startswith("run_")]
    runs.sort(
        key=lambda name: (Path(output_dir) / name).stat().st_mtime,
    )
    return runs


def list_batch_ids(output_dir: str) -> list[str]:
    """Return batch directory names (``batch_*``), oldest first."""
    root = Path(output_dir)
    if not root.is_dir():
        return []
    batches = [d.name for d in root.iterdir() if d.is_dir() and d.name.startswith("batch_")]
    batches.sort(
        key=lambda name: (Path(output_dir) / name).stat().st_mtime,
    )
    return batches


def _find_final_image(run_dir: Path) -> Optional[Path]:
    for ext in ("png", "jpg", "jpeg", "webp"):
        candidate = run_dir / f"final_output.{ext}"
        if candidate.is_file():
            return candidate
    return None


def load_run_summary(output_dir: str, run_id: str) -> dict[str, Any]:
    """Load paths and key fields for a single run."""
    run_dir = Path(output_dir) / run_id
    out: dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "exists": run_dir.is_dir(),
        "final_image": None,
        "metadata_path": None,
        "metadata_preview": "",
        "run_input_preview": "",
        "iteration_images": [],
    }
    if not run_dir.is_dir():
        out["error"] = "Run directory not found"
        return out

    final = _find_final_image(run_dir)
    if final:
        out["final_image"] = str(final.resolve())

    meta_path = run_dir / "metadata.json"
    if meta_path.is_file():
        out["metadata_path"] = str(meta_path)
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            out["metadata_preview"] = json.dumps(data, indent=2)[:12000]
        except (OSError, json.JSONDecodeError) as e:
            out["metadata_preview"] = f"(could not read metadata: {e})"

    inp_path = run_dir / "run_input.json"
    if inp_path.is_file():
        try:
            raw = json.loads(inp_path.read_text(encoding="utf-8"))
            out["run_input_preview"] = json.dumps(raw, indent=2)[:8000]
        except (OSError, json.JSONDecodeError) as e:
            out["run_input_preview"] = f"(could not read run_input: {e})"

    def _iter_sort_key(d: Path) -> int:
        parts = d.name.split("_", 1)
        if len(parts) < 2 or not parts[1].isdigit():
            return 0
        return int(parts[1])

    iter_dirs = sorted(
        [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("iter_")],
        key=_iter_sort_key,
    )
    images: list[str] = []
    for d in iter_dirs:
        for ext in ("png", "jpg", "jpeg", "webp"):
            p = d / f"output.{ext}"
            if p.is_file():
                images.append(str(p.resolve()))
                break
    out["iteration_images"] = images
    return out


def load_batch_summary(output_dir: str, batch_id: str) -> dict[str, Any]:
    """Load batch_report.json summary if present."""
    batch_dir = Path(output_dir) / batch_id
    out: dict[str, Any] = {
        "batch_id": batch_id,
        "batch_dir": str(batch_dir.resolve()),
        "exists": batch_dir.is_dir(),
        "report_preview": "",
    }
    if not batch_dir.is_dir():
        out["error"] = "Batch directory not found"
        return out
    report_path = batch_dir / "batch_report.json"
    if report_path.is_file():
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
            out["report_preview"] = json.dumps(data, indent=2)[:16000]
        except (OSError, json.JSONDecodeError) as e:
            out["report_preview"] = f"(could not read report: {e})"
    else:
        out["report_preview"] = "No batch_report.json in this directory yet."
    return out
