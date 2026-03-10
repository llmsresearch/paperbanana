"""Batch generation: manifest loading and batch run id."""

from __future__ import annotations

import datetime
import uuid
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


def generate_batch_id() -> str:
    """Generate a unique batch run ID."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    return f"batch_{ts}_{short_uuid}"


def load_batch_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    """Load a batch manifest (YAML or JSON) and return a list of items.

    Each item is a dict with:
      - input: path to methodology text file (resolved relative to manifest parent)
      - caption: figure caption / communicative intent
      - id: optional string identifier for the item (default: index-based)

    Paths in the manifest are resolved relative to the manifest file's directory.
    """
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    parent = manifest_path.parent
    raw = manifest_path.read_text(encoding="utf-8")
    suffix = manifest_path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml

            data = yaml.safe_load(raw)
        except ImportError:
            raise RuntimeError(
                "PyYAML is required for YAML manifests. Install with: pip install pyyaml"
            )
    elif suffix == ".json":
        import json

        data = json.loads(raw)
    else:
        raise ValueError(f"Manifest must be .yaml, .yml, or .json. Got: {manifest_path.suffix}")

    if data is None:
        raise ValueError("Manifest is empty")
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "items" in data:
        items = data["items"]
    else:
        raise ValueError("Manifest must be a list of items or an object with an 'items' list")

    result = []
    for i, entry in enumerate(items):
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest item {i} must be an object, got {type(entry).__name__}")
        inp = entry.get("input")
        caption = entry.get("caption")
        if not inp or not caption:
            raise ValueError(f"Manifest item {i}: 'input' and 'caption' are required")
        input_path = Path(inp)
        if not input_path.is_absolute():
            input_path = (parent / input_path).resolve()
        result.append(
            {
                "input": str(input_path),
                "caption": str(caption),
                "id": entry.get("id", f"item_{i + 1}"),
            }
        )
    return result
