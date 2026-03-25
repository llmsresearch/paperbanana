"""Load CSV/JSON files for the statistical plot pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
        source_context = f"JSON data:\n{json.dumps(loaded, indent=2)[:2000]}"
        return source_context, loaded
    raise ValueError(f"Plot data must be .csv or .json, got: {data_path.suffix}")
