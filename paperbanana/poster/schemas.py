"""Layout schema retrieval: induced real-poster structure guides planning.

The schema library (``data/poster_schemas/schemas.json``) is built by
``scripts/build_poster_schemas.py`` from SciPostLayout's 7,855
human-annotated posters — column counts, section counts, visual-area
shares, and title-band heights as observed distributions, not opinions.
At generation time the planner retrieves the schema matching the venue's
orientation and receives it as structural priors. Rebuilding the library
from new corpora (e.g. ML-venue posters) upgrades every future run
without touching code: this is the data-learning layer of the poster
head.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()

DEFAULT_SCHEMA_PATH = Path("data/poster_schemas/schemas.json")


class Dist(BaseModel):
    p25: float
    median: float
    p75: float


class LayoutSchema(BaseModel):
    id: str
    orientation: str
    columns: int
    n_posters: int
    share_of_corpus: float
    sections: Optional[Dist] = None
    sections_per_column: Optional[float] = None
    visual_share: Dist
    title_band_frac: Optional[Dist] = None
    text_blocks: Optional[Dist] = None


class SchemaLibrary(BaseModel):
    source: str
    n_posters: int
    schemas: list[LayoutSchema] = Field(min_length=1)

    def for_orientation(self, orientation: str) -> list[LayoutSchema]:
        matches = [s for s in self.schemas if s.orientation == orientation]
        return sorted(matches, key=lambda s: -s.n_posters)


def load_schema_library(path: str | Path = DEFAULT_SCHEMA_PATH) -> SchemaLibrary:
    """Load the induced schema library.

    Raises:
        FileNotFoundError: The library ships with the package; a missing
            file means a broken installation or a bad path.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"poster schema library not found at {path}; rebuild it with "
            "scripts/build_poster_schemas.py or fix the path"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    library = SchemaLibrary(**data)
    logger.info(
        "Loaded poster schema library",
        schemas=len(library.schemas),
        corpus=library.n_posters,
    )
    return library


def select_schema(library: SchemaLibrary, orientation: str, min_columns: int = 1) -> LayoutSchema:
    """Most common real-poster schema for this orientation.

    ``min_columns`` lets the caller exclude degenerate single-column
    layouts for very wide posters.
    """
    candidates = [s for s in library.for_orientation(orientation) if s.columns >= min_columns]
    if not candidates:
        candidates = library.for_orientation(orientation)
    if not candidates:
        raise ValueError(f"schema library has no entries for orientation '{orientation}'")
    return candidates[0]


def format_layout_priors(library: SchemaLibrary, selected: LayoutSchema) -> str:
    """Prompt block describing what real posters of this shape look like."""
    alternatives = [
        s for s in library.for_orientation(selected.orientation) if s.id != selected.id
    ][:2]

    def describe(s: LayoutSchema) -> str:
        parts = [f"{s.columns} columns"]
        if s.sections:
            parts.append(
                f"{s.sections.p25:.0f}-{s.sections.p75:.0f} sections "
                f"(median {s.sections.median:.0f})"
            )
        parts.append(
            f"figures/tables covering {s.visual_share.p25 * 100:.0f}-"
            f"{s.visual_share.p75 * 100:.0f}% of the area "
            f"(median {s.visual_share.median * 100:.0f}%)"
        )
        if s.title_band_frac:
            parts.append(f"title band ~{s.title_band_frac.median * 100:.0f}% of height")
        return ", ".join(parts) + f" [{s.n_posters} real posters]"

    lines = [
        f"STRUCTURE OF REAL {selected.orientation.upper()} POSTERS "
        f"(induced from {library.n_posters} human-made scientific posters; "
        "treat as strong priors, not hard rules):",
        f"- Most common shape: {describe(selected)}",
    ]
    for alt in alternatives:
        lines.append(f"- Also common: {describe(alt)}")
    if selected.sections_per_column:
        lines.append(
            f"- Plan roughly {selected.sections_per_column:.0f} panels per column; "
            "panels beyond the median count make posters read as cluttered."
        )
    lines.append(
        "- If your figure/table share lands far below the observed range, grow the "
        "key figures or cut text rather than leaving sparse panels."
    )
    return "\n".join(lines)
