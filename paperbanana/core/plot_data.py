"""Load CSV/JSON files for the statistical plot pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _normalize_json_plot_payload(loaded: Any) -> Any:
    """Match legacy Studio/plot behavior for JSON files.

    - Top-level JSON array → used as-is (rows for the plot).
    - Top-level object with a ``data`` key → unwrap so ``raw_data={"data": payload}``
      is not double-nested; same as ``raw if isinstance(raw, list) else raw.get("data", raw)``.
    - Other top-level values (e.g. a single object without ``data``) → used as-is.
    """
    if isinstance(loaded, list):
        return loaded
    if isinstance(loaded, dict):
        return loaded.get("data", loaded)
    return loaded


def load_statistical_plot_payload(data_path: Path) -> tuple[str, Any]:
    """Read a data file and return (source_context, payload) for GenerationInput.

    ``payload`` is passed as ``raw_data={"data": payload}`` (CSV yields a list of rows).
    """
    data_path = Path(data_path).resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        import pandas as pd

        df = pd.read_csv(data_path)
        raw_data = df.to_dict(orient="records")
        source_context = (
            f"CSV data with columns: {list(df.columns)}\n"
            f"Rows: {len(df)}\nSample:\n{df.head().to_string()}"
        )
        return source_context, raw_data
    if suffix == ".json":
        loaded = json.loads(data_path.read_text(encoding="utf-8"))
        payload = _normalize_json_plot_payload(loaded)
        source_context = f"JSON data:\n{json.dumps(payload, indent=2)[:2000]}"
        return source_context, payload
    raise ValueError(f"Plot data must be .csv or .json, got: {data_path.suffix}")
