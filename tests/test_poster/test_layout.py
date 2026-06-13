"""Skyline placer tests: spans, bands, determinism, overflow feedback."""

from __future__ import annotations

import pytest

from paperbanana.poster.layout import (
    BandOverflowError,
    LayoutError,
    panel_width_mm,
    place_bands,
)
from paperbanana.poster.types import Band, PosterIR

from .conftest import default_bands, make_panel


def _unplaced(poster_ir: PosterIR, panels: list, bands: list | None = None) -> PosterIR:
    for p in panels:
        p.bbox = None
    return poster_ir.model_copy(
        update={"panels": panels, "bands": bands or default_bands(columns=3)}
    )


def _measures(ir: PosterIR, default: float = 150.0, **overrides: float) -> dict[str, float]:
    return {p.id: overrides.get(p.id, default) for p in ir.panels}


def _header_and_body(n_body: int, columns: int = 3) -> tuple[list, list]:
    panels = [make_panel("header", 0, role="header", band_id="header")]
    for i in range(n_body):
        panels.append(make_panel(f"p{i}", i + 1, column=i % columns))
    return panels, default_bands(columns=columns)


def test_place_bands_places_everything(poster_ir: PosterIR):
    panels, bands = _header_and_body(5)
    ir = _unplaced(poster_ir, panels, bands)
    placed = place_bands(ir, _measures(ir, 120.0, header=80.0))
    assert placed.is_fully_placed()
    PosterIR(**placed.model_dump())  # invariants re-validated
    assert all(b.height_mm is not None for b in placed.bands)


def test_place_bands_is_deterministic(poster_ir: PosterIR):
    panels, bands = _header_and_body(5)
    ir = _unplaced(poster_ir, panels, bands)
    a = place_bands(ir, _measures(ir, 120.0, header=80.0))
    b = place_bands(ir, _measures(ir, 120.0, header=80.0))
    assert a.model_dump() == b.model_dump()


def test_header_band_is_content_sized(poster_ir: PosterIR):
    panels, bands = _header_and_body(3)
    ir = _unplaced(poster_ir, panels, bands)
    placed = place_bands(ir, _measures(ir, 200.0, header=90.0))
    header = next(p for p in placed.panels if p.role == "header")
    # measured 90 + 2x10 padding
    assert header.bbox.h_mm == pytest.approx(110.0, abs=0.1)
    assert header.bbox.w_mm == pytest.approx(placed.size.width_mm - 2 * placed.margin_mm)


def test_spanning_panel_sits_below_spanned_columns(poster_ir: PosterIR):
    panels = [
        make_panel("header", 0, role="header", band_id="header"),
        make_panel("a", 1, column=0),
        make_panel("b", 2, column=1),
        make_panel("hero", 3, column=0, col_span=2),
        make_panel("c", 4, column=2),
    ]
    ir = _unplaced(poster_ir, panels)
    placed = place_bands(ir, _measures(ir, 150.0, header=80.0, a=100.0, b=180.0))
    a = next(p for p in placed.panels if p.id == "a")
    b = next(p for p in placed.panels if p.id == "b")
    hero = next(p for p in placed.panels if p.id == "hero")
    # hero starts below the taller of columns 0/1
    assert hero.bbox.y_mm > max(a.bbox.y2_mm, b.bbox.y2_mm) - 0.01
    # hero spans two columns: wider than either single-column panel
    assert hero.bbox.w_mm > a.bbox.w_mm * 1.9
    # no overlap with either
    assert not hero.bbox.overlaps(a.bbox)
    assert not hero.bbox.overlaps(b.bbox)


def test_multi_band_stacking(poster_ir: PosterIR):
    bands = [
        Band(id="header", kind="header", order=0, columns=1),
        Band(id="banner", kind="banner", order=1, columns=1),
        Band(id="body", kind="body", order=2, columns=2),
    ]
    panels = [
        make_panel("header", 0, role="header", band_id="header"),
        make_panel(
            "take",
            1,
            role="takeaway",
            band_id="banner",
            elements=[
                __import__("paperbanana.poster.types", fromlist=["BannerElement"]).BannerElement(
                    content="One key message"
                )
            ],
        ),
        make_panel("left", 2, column=0),
        make_panel("right", 3, column=1),
    ]
    ir = _unplaced(poster_ir, panels, bands)
    placed = place_bands(ir, _measures(ir, 200.0, header=80.0, take=30.0))
    header, banner, left = (
        next(p for p in placed.panels if p.id == i) for i in ("header", "take", "left")
    )
    assert header.bbox.y2_mm <= banner.bbox.y_mm + 0.01
    assert banner.bbox.y2_mm <= left.bbox.y_mm + 0.01
    # banner capped at 8% of page height
    assert banner.bbox.h_mm <= placed.size.height_mm * 0.08 + 0.01


def test_band_overflow_carries_column_loads(poster_ir: PosterIR):
    panels, bands = _header_and_body(3)
    ir = _unplaced(poster_ir, panels, bands)
    with pytest.raises(BandOverflowError) as exc_info:
        place_bands(ir, _measures(ir, 2000.0, header=80.0))
    err = exc_info.value
    assert err.page_deficit_mm > 0
    assert err.overflows and all(o.band_id == "body" for o in err.overflows)
    assert err.overflows[0].required_mm > err.overflows[0].available_mm


def test_columns_fill_to_band_bottom(poster_ir: PosterIR):
    """Every body column bottom-aligns with its band — no dead space below
    content (the canvas contract every real poster satisfies)."""
    panels, bands = _header_and_body(3)
    ir = _unplaced(poster_ir, panels, bands)
    placed = place_bands(ir, _measures(ir, 100.0, header=60.0))
    body = next(b for b in placed.bands if b.kind == "body")
    band_bottom = max(p.bbox.y_mm + p.bbox.h_mm for p in placed.panels if p.band_id == "body")
    for col in range(body.columns):
        col_panels = [
            p
            for p in placed.panels
            if p.band_id == "body" and p.column <= col < p.column + p.col_span
        ]
        col_bottom = max(p.bbox.y_mm + p.bbox.h_mm for p in col_panels)
        assert col_bottom == pytest.approx(band_bottom, abs=1.0)


def test_height_frac_weights_slack_distribution(poster_ir: PosterIR):
    """The proposer's height_frac steers who gets the column's space:
    equal content, 3x the share -> visibly larger box."""
    panels = [make_panel("header", 0, role="header", band_id="header")]
    panels.append(make_panel("p0", 1, column=0))
    panels.append(make_panel("p1", 2, column=0))
    panels[1].height_frac = 0.75
    panels[2].height_frac = 0.25
    ir = _unplaced(poster_ir, panels, default_bands(columns=1))
    placed = place_bands(ir, _measures(ir, 100.0, header=60.0))
    placed_body = sorted((p for p in placed.panels if p.band_id == "body"), key=lambda p: p.order)
    assert placed_body[0].bbox.h_mm > placed_body[1].bbox.h_mm * 1.5


def test_missing_measurement_raises(poster_ir: PosterIR):
    panels, bands = _header_and_body(2)
    ir = _unplaced(poster_ir, panels, bands)
    measures = _measures(ir)
    del measures["p0"]
    with pytest.raises(LayoutError, match="no measurement"):
        place_bands(ir, measures)


def test_panel_width_mm_accounts_for_span(poster_ir: PosterIR):
    panels, bands = _header_and_body(3)
    panels[1] = make_panel("p0", 1, column=0, col_span=2)
    ir = _unplaced(poster_ir, panels, bands)
    single = panel_width_mm(ir, ir.panels[2])
    double = panel_width_mm(ir, ir.panels[1])
    assert double == pytest.approx(2 * single + ir.gutter_mm)
    header_w = panel_width_mm(ir, ir.panels[0])
    assert header_w == pytest.approx(ir.size.width_mm - 2 * ir.margin_mm)
