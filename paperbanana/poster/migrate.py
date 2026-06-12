"""IR schema migration: every deserialization goes through here.

v1 IRs (single global column grid) are converted to v2 (stacked bands)
on load, so old run directories stay resumable and editable without
maintaining two model families.
"""

from __future__ import annotations

import structlog

from paperbanana.poster.types import PosterIR

logger = structlog.get_logger()


def load_poster_ir(data: dict) -> PosterIR:
    """Deserialize a poster IR payload of any known schema version."""
    version = data.get("schema_version", 1)
    if version == 1:
        data = _migrate_v1_to_v2(dict(data))
        logger.info("Migrated poster IR v1 -> v2")
    return PosterIR(**data)


def _migrate_v1_to_v2(data: dict) -> dict:
    """v1 -> v2: synthesize a header band + one body band from the flat grid."""
    columns = data.pop("columns", 3)
    data["schema_version"] = 2
    data["bands"] = [
        {"id": "header", "kind": "header", "order": 0, "columns": 1},
        {"id": "body", "kind": "body", "order": 1, "columns": columns},
    ]
    for panel in data.get("panels", []):
        if panel.get("role") == "header":
            panel["band_id"] = "header"
            panel["column"] = 0
            panel["col_span"] = 1
        else:
            panel["band_id"] = "body"
            panel["column"] = panel.get("column") or 0
            panel["col_span"] = 1
        panel.setdefault("emphasis", "normal")
        # v1 placed header col_span implicitly; band columns=1 covers it.
    return data
