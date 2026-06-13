"""PosterMemory: the living corpus of layout exemplars.

Every poster the system sees — seeded from real-poster datasets,
ingested by the user (``paperbanana posters ingest``), or produced by
its own successful runs (self-promotion) — becomes a layout *skeleton*:
the band/column/span structure without any content. At generation time
the proposer retrieves the closest skeletons as few-shot exemplars, so
the system's design vocabulary grows with its corpus instead of being
hand-coded.

Stores are JSONL (one exemplar per line), mirroring the lessons store:
a read-only seed shipped at ``data/reference_sets/posters/exemplars.jsonl``
plus a per-user store under ``~/.config/paperbanana/poster_memory/``.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Literal, Optional

import structlog
from pydantic import BaseModel, Field

from paperbanana.poster.types import BandKind, Orientation, PanelEmphasis, PosterIR

logger = structlog.get_logger()

SEED_PATH = Path("data/reference_sets/posters/exemplars.jsonl")
MEMORY_DIR_ENV_VAR = "PAPERBANANA_POSTER_MEMORY_DIR"
DEFAULT_MEMORY_DIR = Path.home() / ".config" / "paperbanana" / "poster_memory"
MEMORY_FILENAME = "exemplars.jsonl"

ExemplarSource = Literal["scipostlayout", "paper2poster", "ingested", "self_generated"]


class SkeletonBand(BaseModel):
    kind: BandKind
    columns: int = Field(ge=1, le=6)
    height_frac: float = Field(gt=0, le=1)


class SkeletonPanel(BaseModel):
    band_index: int = Field(ge=0)
    column: int = Field(ge=0)
    col_span: int = Field(default=1, ge=1)
    height_frac: float = Field(gt=0, le=1, description="Of its band")
    has_figure: bool = False
    emphasis: PanelEmphasis = "normal"


class LayoutSkeleton(BaseModel):
    bands: list[SkeletonBand] = Field(min_length=1)
    panels: list[SkeletonPanel] = Field(min_length=1)


class PosterExemplar(BaseModel):
    id: str
    source: ExemplarSource
    venue: Optional[str] = None
    orientation: Orientation
    aspect_ratio: float = Field(gt=0, description="width / height")
    n_panels: int = Field(ge=1)
    n_figures: int = Field(ge=0)
    visual_share: float = Field(ge=0, le=1)
    skeleton: LayoutSkeleton
    quality: Optional[float] = Field(default=None, ge=1, le=5)
    thumbnail_path: Optional[str] = None
    created: str = ""
    tags: list[str] = Field(default_factory=list)


def resolve_memory_dir(memory_dir: Optional[str | Path] = None) -> Path:
    if memory_dir:
        return Path(memory_dir).expanduser()
    env = os.environ.get(MEMORY_DIR_ENV_VAR)
    return Path(env).expanduser() if env else DEFAULT_MEMORY_DIR


def _read_jsonl(path: Path) -> list[PosterExemplar]:
    if not path.is_file():
        return []
    exemplars = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            exemplars.append(PosterExemplar(**json.loads(line)))
    return exemplars


def load_exemplars(
    seed_path: str | Path = SEED_PATH,
    memory_dir: Optional[str | Path] = None,
) -> list[PosterExemplar]:
    """Seed + user exemplars; user entries shadow seed entries by id."""
    seed = _read_jsonl(Path(seed_path))
    user = _read_jsonl(resolve_memory_dir(memory_dir) / MEMORY_FILENAME)
    by_id = {e.id: e for e in seed}
    by_id.update({e.id: e for e in user})
    exemplars = list(by_id.values())
    logger.info("Loaded poster memory", seed=len(seed), user=len(user), total=len(exemplars))
    return exemplars


def append_exemplar(
    exemplar: PosterExemplar,
    memory_dir: Optional[str | Path] = None,
    max_self_generated: int = 50,
) -> Path:
    """Append to the user store; self-generated entries are FIFO-capped."""
    path = resolve_memory_dir(memory_dir) / MEMORY_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_jsonl(path)
    existing.append(exemplar)
    self_generated = [e for e in existing if e.source == "self_generated"]
    if len(self_generated) > max_self_generated:
        drop = {e.id for e in self_generated[: len(self_generated) - max_self_generated]}
        existing = [e for e in existing if e.id not in drop]
        logger.info("Poster memory FIFO cap applied", dropped=len(drop))
    path.write_text(
        "\n".join(json.dumps(json.loads(e.model_dump_json())) for e in existing) + "\n",
        encoding="utf-8",
    )
    return path


def purge_exemplars(
    memory_dir: Optional[str | Path] = None, self_generated_only: bool = False
) -> int:
    """Delete user exemplars; returns the number removed."""
    path = resolve_memory_dir(memory_dir) / MEMORY_FILENAME
    existing = _read_jsonl(path)
    if not existing:
        return 0
    if self_generated_only:
        kept = [e for e in existing if e.source != "self_generated"]
        removed = len(existing) - len(kept)
        path.write_text(
            "\n".join(json.dumps(json.loads(e.model_dump_json())) for e in kept) + "\n"
            if kept
            else "",
            encoding="utf-8",
        )
        return removed
    path.unlink()
    return len(existing)


def retrieve_exemplars(
    exemplars: list[PosterExemplar],
    orientation: Orientation,
    venue: Optional[str],
    n_figures: int,
    n_panels: int,
    aspect_ratio: float,
    k: int = 3,
) -> list[PosterExemplar]:
    """Closest real-poster skeletons for this generation task.

    Orientation is a hard filter. At most one self-generated exemplar is
    returned — the inbreeding guard: the system may learn from its own
    successes but never only from them.
    """

    def score(e: PosterExemplar) -> float:
        s = 0.0
        if venue and e.venue and e.venue.lower() == venue.lower():
            s += 3.0
        s += 2.0 * (1.0 - min(abs(e.aspect_ratio - aspect_ratio) / max(aspect_ratio, 0.1), 1.0))
        s += 1.0 - min(abs(e.n_figures - n_figures), 5) / 5.0
        s += 1.0 - min(abs(e.n_panels - n_panels), 6) / 6.0
        if e.source == "self_generated" and e.quality:
            s += 0.5 * (e.quality / 5.0)
        return s

    candidates = sorted(
        (e for e in exemplars if e.orientation == orientation), key=score, reverse=True
    )
    picked: list[PosterExemplar] = []
    self_used = 0
    for exemplar in candidates:
        if exemplar.source == "self_generated":
            if self_used >= 1:
                continue
            self_used += 1
        picked.append(exemplar)
        if len(picked) >= k:
            break
    return picked


def skeleton_from_ir(ir: PosterIR) -> LayoutSkeleton:
    """Deterministic inverse: a placed IR's structure, content-free."""
    bands = ir.bands_in_order()
    page_h = ir.size.height_mm
    band_index = {b.id: i for i, b in enumerate(bands)}
    skeleton_bands = [
        SkeletonBand(
            kind=b.kind,
            columns=b.columns,
            height_frac=max(0.01, min(1.0, (b.height_mm or page_h / len(bands)) / page_h)),
        )
        for b in bands
    ]
    panels = []
    for panel in ir.panels_in_order():
        band = ir.band(panel.band_id)
        band_h = band.height_mm or page_h / len(bands)
        panel_h = panel.bbox.h_mm if panel.bbox else band_h
        panels.append(
            SkeletonPanel(
                band_index=band_index[panel.band_id],
                column=panel.column,
                col_span=panel.col_span,
                height_frac=max(0.01, min(1.0, panel_h / band_h)),
                has_figure=any(el.kind == "figure" for el in panel.elements),
                emphasis=panel.emphasis,
            )
        )
    return LayoutSkeleton(bands=skeleton_bands, panels=panels)


def exemplar_from_run(
    ir: PosterIR,
    quality: Optional[float],
    run_id: str,
) -> PosterExemplar:
    """Package a successful run's structure for self-promotion."""
    n_figures = sum(1 for p in ir.panels for el in p.elements if el.kind == "figure")
    visual_share = 0.0
    page_area = ir.size.width_mm * ir.size.height_mm
    for panel in ir.panels:
        if panel.bbox and any(el.kind == "figure" for el in panel.elements):
            visual_share += (panel.bbox.w_mm * panel.bbox.h_mm) / page_area * 0.6
    return PosterExemplar(
        id=f"self_{run_id}",
        source="self_generated",
        venue=ir.venue,
        orientation=ir.orientation,
        aspect_ratio=ir.size.width_mm / ir.size.height_mm,
        n_panels=len(ir.panels),
        n_figures=n_figures,
        visual_share=round(min(1.0, visual_share), 3),
        skeleton=skeleton_from_ir(ir),
        quality=quality,
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        tags=["self"],
    )


def format_exemplars_block(exemplars: list[PosterExemplar]) -> str:
    """Compact prompt block: real poster structures as JSON skeletons."""
    if not exemplars:
        return ""
    lines = []
    for i, e in enumerate(exemplars, 1):
        meta = f"{e.orientation}, {e.n_panels} panels, {e.n_figures} figures, source={e.source}"
        if e.venue:
            meta += f", venue={e.venue}"
        if e.source == "self_generated":
            meta += f" (judge {e.quality})"
        lines.append(
            f"Exemplar {i} ({meta}):\n"
            + json.dumps(json.loads(e.skeleton.model_dump_json()), separators=(",", ":"))
        )
    return "\n".join(lines)
