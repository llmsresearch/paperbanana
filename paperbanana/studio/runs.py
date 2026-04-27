"""Discover and summarize prior pipeline runs under an output directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def _read_json_file(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


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
    meta_data = _read_json_file(meta_path)
    if meta_data is not None:
        out["metadata_path"] = str(meta_path)
        out["metadata_preview"] = json.dumps(meta_data, indent=2)[:12000]
    elif meta_path.is_file():
        out["metadata_preview"] = "(could not read metadata)"

    inp_path = run_dir / "run_input.json"
    inp_data = _read_json_file(inp_path)
    if inp_data is not None:
        out["run_input_preview"] = json.dumps(inp_data, indent=2)[:8000]
    elif inp_path.is_file():
        out["run_input_preview"] = "(could not read run_input)"

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


def _collect_compare_fields(run_dir: Path) -> dict[str, Any]:
    meta = _read_json_file(run_dir / "metadata.json") or {}
    run_input = _read_json_file(run_dir / "run_input.json") or {}
    final_image = _find_final_image(run_dir)

    settings = meta.get("settings")
    if not isinstance(settings, dict):
        settings = {}

    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "final_image": str(final_image.resolve()) if final_image else None,
        "caption": run_input.get("communicative_intent"),
        "diagram_type": run_input.get("diagram_type"),
        "aspect_ratio": run_input.get("aspect_ratio"),
        "vlm_provider": settings.get("vlm_provider"),
        "vlm_model": settings.get("vlm_model"),
        "image_provider": settings.get("image_provider"),
        "image_model": settings.get("image_model"),
        "output_format": settings.get("output_format"),
        "refinement_iterations": settings.get("refinement_iterations"),
        "auto_refine": settings.get("auto_refine"),
        "max_iterations": settings.get("max_iterations"),
        "seed": settings.get("seed"),
        "created_at": meta.get("created_at") or meta.get("timestamp"),
        "duration_seconds": meta.get("duration_seconds"),
        "total_cost_usd": meta.get("total_cost_usd") or meta.get("cost_usd"),
    }


def compare_runs(output_dir: str, left_run_id: str, right_run_id: str) -> dict[str, Any]:
    """Return side-by-side run metadata and a compact diff summary."""
    root = Path(output_dir)
    left_dir = root / left_run_id
    right_dir = root / right_run_id
    if not left_dir.is_dir():
        return {"error": f"Run directory not found: {left_run_id}"}
    if not right_dir.is_dir():
        return {"error": f"Run directory not found: {right_run_id}"}

    left = _collect_compare_fields(left_dir)
    right = _collect_compare_fields(right_dir)

    keys = [
        "caption",
        "diagram_type",
        "aspect_ratio",
        "vlm_provider",
        "vlm_model",
        "image_provider",
        "image_model",
        "output_format",
        "refinement_iterations",
        "auto_refine",
        "max_iterations",
        "seed",
        "duration_seconds",
        "total_cost_usd",
    ]
    diffs: list[dict[str, Any]] = []
    for key in keys:
        if left.get(key) != right.get(key):
            diffs.append({"field": key, "left": left.get(key), "right": right.get(key)})

    return {"left": left, "right": right, "diffs": diffs}


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
            items = data.get("items", []) if isinstance(data, dict) else []
            status_counts: dict[str, int] = {}
            for item in items:
                status = item.get("status")
                if not status:
                    status = "success" if item.get("output_path") else "failed"
                status_counts[status] = status_counts.get(status, 0) + 1
            out["status_counts"] = status_counts
            out["can_resume"] = any(
                status in ("pending", "running", "failed") for status in status_counts
            )
        except (OSError, json.JSONDecodeError) as e:
            out["report_preview"] = f"(could not read report: {e})"
    else:
        out["report_preview"] = "No batch_report.json in this directory yet."
    return out
