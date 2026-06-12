"""Layout solver tests."""

from __future__ import annotations

import pytest

from paperbanana.poster.layout import LayoutError, assign_bboxes
from paperbanana.poster.types import PosterIR

from .conftest import make_panel


def _unplaced(poster_ir: PosterIR, n_body: int = 5, columns: int = 3) -> PosterIR:
    panels = [
        make_panel("header", 0, role="header"),
        *[make_panel(f"p{i}", i + 1) for i in range(n_body)],
    ]
    return poster_ir.model_copy(update={"panels": panels, "columns": columns})


def test_assign_bboxes_places_everything(poster_ir: PosterIR):
    ir = _unplaced(poster_ir)
    placed = assign_bboxes(ir)
    assert placed.is_fully_placed()
    # PosterIR validation already guarantees no overlap and in-bounds; just
    # re-validate explicitly to be sure model_copy didn't skip it.
    PosterIR(**placed.model_dump())


def test_assign_bboxes_is_deterministic(poster_ir: PosterIR):
    ir = _unplaced(poster_ir)
    a = assign_bboxes(ir)
    b = assign_bboxes(ir)
    assert a.model_dump() == b.model_dump()


def test_header_spans_content_width(poster_ir: PosterIR):
    placed = assign_bboxes(_unplaced(poster_ir))
    header = next(p for p in placed.panels if p.role == "header")
    assert header.bbox is not None
    assert header.bbox.x_mm == placed.margin_mm
    assert header.bbox.w_mm == pytest.approx(placed.size.width_mm - 2 * placed.margin_mm)


def test_columns_filled_in_reading_order(poster_ir: PosterIR):
    placed = assign_bboxes(_unplaced(poster_ir, n_body=6, columns=3))
    body = sorted((p for p in placed.panels if p.role != "header"), key=lambda p: p.order)
    columns = [p.column for p in body]
    assert columns == sorted(columns), "reading order must walk columns left to right"
    assert set(columns) == {0, 1, 2}


def test_weights_drive_heights(poster_ir: PosterIR):
    panels = [
        make_panel("header", 0, role="header"),
        make_panel("small", 1),
        make_panel("big", 2),
        make_panel("c2", 3),
        make_panel("c3", 4),
    ]
    panels[1].weight = 1.0
    panels[2].weight = 3.0
    panels[3].weight = 2.0
    panels[4].weight = 2.0
    ir = poster_ir.model_copy(update={"panels": panels, "columns": 3})
    placed = assign_bboxes(ir)
    big = next(p for p in placed.panels if p.id == "big")
    small = next(p for p in placed.panels if p.id == "small")
    assert big.column == small.column == 0
    assert big.bbox is not None and small.bbox is not None
    assert big.bbox.h_mm / small.bbox.h_mm == pytest.approx(3.0, rel=0.05)


def test_columns_end_at_bottom_margin(poster_ir: PosterIR):
    placed = assign_bboxes(_unplaced(poster_ir, n_body=5, columns=3))
    for column in (0, 1, 2):
        col_panels = [p for p in placed.panels if p.column == column]
        bottom = max(p.bbox.y2_mm for p in col_panels)
        assert bottom == pytest.approx(placed.size.height_mm - placed.margin_mm, abs=0.01)


def test_requires_exactly_one_header(poster_ir: PosterIR):
    panels = [make_panel(f"p{i}", i) for i in range(4)]
    ir = poster_ir.model_copy(update={"panels": panels})
    with pytest.raises(LayoutError, match="header"):
        assign_bboxes(ir)


def test_too_few_panels_for_columns(poster_ir: PosterIR):
    panels = [make_panel("header", 0, role="header"), make_panel("only", 1)]
    ir = poster_ir.model_copy(update={"panels": panels, "columns": 3})
    with pytest.raises(LayoutError, match="columns"):
        assign_bboxes(ir)


def test_excessive_margins_raise(poster_ir: PosterIR):
    ir = _unplaced(poster_ir).model_copy(update={"margin_mm": 700.0})
    with pytest.raises(LayoutError):
        assign_bboxes(ir)
