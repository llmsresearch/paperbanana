"""Load analytics records from run/batch/orchestration artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from paperbanana.analytics.models import AnalyticsRecord


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_metadata_cost(payload: dict[str, Any]) -> float | None:
    candidates: list[Any] = []
    candidates.append(payload.get("total_cost_usd"))
    cost_tracking = payload.get("cost_tracking")
    if isinstance(cost_tracking, dict):
        candidates.append(cost_tracking.get("total_cost"))
        candidates.append(cost_tracking.get("total_cost_usd"))
    for item in candidates:
        if item is None:
            continue
        try:
            return float(item)
        except (TypeError, ValueError):
            continue
    return None


def _load_run_metadata(path: Path) -> list[AnalyticsRecord]:
    payload = _safe_load_json(path)
    if payload is None:
        return []

    timing = payload.get("timing")
    timing_seconds = 0.0
    if isinstance(timing, dict):
        timing_seconds = _as_float(timing.get("total_seconds"), 0.0)

    config_snapshot = payload.get("config_snapshot")
    vlm_provider: str | None = None
    image_provider: str | None = None
    if isinstance(config_snapshot, dict):
        if config_snapshot.get("vlm_provider"):
            vlm_provider = str(config_snapshot["vlm_provider"])
        if config_snapshot.get("image_provider"):
            image_provider = str(config_snapshot["image_provider"])

    if payload.get("vlm_provider"):
        vlm_provider = str(payload.get("vlm_provider"))
    if payload.get("image_provider"):
        image_provider = str(payload.get("image_provider"))

    run_id = str(payload.get("run_id") or path.parent.name)
    return [
        AnalyticsRecord(
            source_type="run",
            source_path=str(path),
            source_id=run_id,
            status="success",
            total_seconds=timing_seconds,
            cost_usd=_extract_metadata_cost(payload),
            vlm_provider=vlm_provider,
            image_provider=image_provider,
        )
    ]


def _load_batch_report(path: Path) -> list[AnalyticsRecord]:
    payload = _safe_load_json(path)
    if payload is None:
        return []
    batch_id = str(payload.get("batch_id") or path.parent.name)
    batch_seconds = _as_float(payload.get("total_seconds"), 0.0)
    items = payload.get("items")
    if not isinstance(items, list):
        return []
    if not items:
        return []
    avg_item_seconds = batch_seconds / len(items) if len(items) else 0.0
    records: list[AnalyticsRecord] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "item")
        status = str(item.get("status") or "unknown")
        if status == "running":
            status = "failed"
        records.append(
            AnalyticsRecord(
                source_type="batch_item",
                source_path=str(path),
                source_id=f"{batch_id}:{item_id}",
                status=status,
                total_seconds=avg_item_seconds,
                cost_usd=None,
            )
        )
    return records


def _load_orchestration_report(path: Path) -> list[AnalyticsRecord]:
    payload = _safe_load_json(path)
    if payload is None:
        return []
    orchestrate_id = str(payload.get("orchestration_id") or path.parent.name)
    total_seconds = _as_float(payload.get("total_seconds"), 0.0)
    generated_items = payload.get("generated_items")
    failures = payload.get("failures")
    if not isinstance(generated_items, list):
        generated_items = []
    if not isinstance(failures, list):
        failures = []
    total_items = len(generated_items) + len(failures)
    avg_item_seconds = total_seconds / total_items if total_items else 0.0

    records: list[AnalyticsRecord] = []
    for item in generated_items:
        if not isinstance(item, dict):
            continue
        records.append(
            AnalyticsRecord(
                source_type="orchestration_item",
                source_path=str(path),
                source_id=f"{orchestrate_id}:{item.get('id', 'item')}",
                status="success",
                total_seconds=avg_item_seconds,
                cost_usd=None,
            )
        )
    for item in failures:
        if not isinstance(item, dict):
            continue
        records.append(
            AnalyticsRecord(
                source_type="orchestration_item",
                source_path=str(path),
                source_id=f"{orchestrate_id}:{item.get('id', 'item')}",
                status="failed",
                total_seconds=avg_item_seconds,
                cost_usd=None,
            )
        )
    return records


def load_analytics_records(root_path: str | Path) -> list[AnalyticsRecord]:
    """Load normalized analytics records from an outputs root."""
    root = Path(root_path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root}")
    records: list[AnalyticsRecord] = []
    for path in root.rglob("metadata.json"):
        records.extend(_load_run_metadata(path))
    for path in root.rglob("batch_report.json"):
        records.extend(_load_batch_report(path))
    for path in root.rglob("figure_package.json"):
        records.extend(_load_orchestration_report(path))
    records.sort(key=lambda x: (x.source_type, x.source_id))
    return records
