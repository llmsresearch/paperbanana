"""PosterMemory tests: stores, retrieval scoring, promotion, skeletons."""

from __future__ import annotations

from pathlib import Path

from paperbanana.poster.layout import place_bands
from paperbanana.poster.memory import (
    LayoutSkeleton,
    PosterExemplar,
    SkeletonBand,
    SkeletonPanel,
    append_exemplar,
    exemplar_from_run,
    format_exemplars_block,
    load_exemplars,
    purge_exemplars,
    retrieve_exemplars,
    skeleton_from_ir,
)
from paperbanana.poster.types import PosterIR

REPO = Path(__file__).resolve().parents[2]
SEED = REPO / "data" / "reference_sets" / "posters" / "exemplars.jsonl"


def _exemplar(eid: str, **overrides) -> PosterExemplar:
    base = dict(
        id=eid,
        source="scipostlayout",
        orientation="landscape",
        aspect_ratio=1.33,
        n_panels=6,
        n_figures=3,
        visual_share=0.25,
        skeleton=LayoutSkeleton(
            bands=[
                SkeletonBand(kind="header", columns=1, height_frac=0.12),
                SkeletonBand(kind="body", columns=3, height_frac=0.88),
            ],
            panels=[
                SkeletonPanel(band_index=1, column=0, height_frac=0.5),
                SkeletonPanel(band_index=1, column=1, height_frac=0.5, has_figure=True),
            ],
        ),
    )
    base.update(overrides)
    return PosterExemplar(**base)


def test_shipped_seed_loads():
    exemplars = load_exemplars(seed_path=SEED, memory_dir="/nonexistent")
    assert len(exemplars) >= 200
    assert {e.orientation for e in exemplars} == {"landscape", "portrait"}
    assert all(e.source == "scipostlayout" for e in exemplars)


def test_user_store_roundtrip_and_fifo(tmp_path: Path):
    for i in range(4):
        append_exemplar(
            _exemplar(f"self_{i}", source="self_generated", quality=4.5),
            memory_dir=tmp_path,
            max_self_generated=2,
        )
    exemplars = load_exemplars(seed_path="/nonexistent", memory_dir=tmp_path)
    self_ids = [e.id for e in exemplars if e.source == "self_generated"]
    assert self_ids == ["self_2", "self_3"]  # FIFO capped at 2


def test_purge(tmp_path: Path):
    append_exemplar(_exemplar("a", source="ingested"), memory_dir=tmp_path)
    append_exemplar(_exemplar("b", source="self_generated", quality=4.2), memory_dir=tmp_path)
    assert purge_exemplars(memory_dir=tmp_path, self_generated_only=True) == 1
    remaining = load_exemplars(seed_path="/nonexistent", memory_dir=tmp_path)
    assert [e.id for e in remaining] == ["a"]
    assert purge_exemplars(memory_dir=tmp_path) == 1


def test_retrieval_orientation_filter_and_scoring():
    pool = [
        _exemplar("land_match", n_figures=3, n_panels=6),
        _exemplar("land_far", n_figures=8, n_panels=12, aspect_ratio=2.0),
        _exemplar("portrait", orientation="portrait", aspect_ratio=0.7),
        _exemplar("venue_match", venue="neurips"),
    ]
    picked = retrieve_exemplars(
        pool,
        orientation="landscape",
        venue="neurips",
        n_figures=3,
        n_panels=6,
        aspect_ratio=1.33,
        k=2,
    )
    ids = [e.id for e in picked]
    assert "portrait" not in ids
    assert ids[0] == "venue_match"  # venue bonus dominates


def test_retrieval_inbreeding_guard():
    pool = [
        _exemplar(f"self_{i}", source="self_generated", venue="neurips", quality=5.0)
        for i in range(3)
    ] + [_exemplar("real_1"), _exemplar("real_2")]
    picked = retrieve_exemplars(
        pool,
        orientation="landscape",
        venue="neurips",
        n_figures=3,
        n_panels=6,
        aspect_ratio=1.33,
        k=3,
    )
    assert sum(1 for e in picked if e.source == "self_generated") <= 1


def test_skeleton_from_ir_roundtrip(poster_ir: PosterIR):
    panels = [p.model_copy(update={"bbox": None}) for p in poster_ir.panels]
    ir = poster_ir.model_copy(update={"panels": panels})
    measured = {p.id: 120.0 if p.role != "header" else 80.0 for p in ir.panels}
    placed = place_bands(ir, measured)
    skeleton = skeleton_from_ir(placed)
    assert [b.kind for b in skeleton.bands] == ["header", "body"]
    assert skeleton.bands[1].columns == 3
    assert len(skeleton.panels) == len(placed.panels)
    assert any(p.has_figure for p in skeleton.panels)
    # Promotion packaging carries the structure + score.
    exemplar = exemplar_from_run(placed, quality=4.4, run_id="poster_test_1")
    assert exemplar.source == "self_generated"
    assert exemplar.quality == 4.4
    assert exemplar.skeleton.bands[1].columns == 3


def test_exemplars_block_formatting():
    block = format_exemplars_block([_exemplar("x", venue="iclr")])
    assert "Exemplar 1" in block and "venue=iclr" in block and '"bands"' in block
    assert format_exemplars_block([]) == ""
