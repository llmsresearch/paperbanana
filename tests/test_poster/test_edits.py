"""Edit op parsing and application tests."""

from __future__ import annotations

import pytest

from paperbanana.poster.edits import EditOpError, apply_edit_ops, parse_edit_ops
from paperbanana.poster.types import PosterIR


def test_parse_known_ops():
    ops = parse_edit_ops(
        [
            {"op": "rewrite_text", "panel_id": "method", "element_index": 1, "content": "x"},
            {"op": "set_type_scale", "level": "body", "pt": 30},
            {"op": "set_panel_weight", "panel_id": "results", "weight": 2.0},
            {"op": "move_panel_order", "panel_id": "results", "new_position": 1},
            {"op": "recolor_token", "token": "accent", "hex": "#FF0000"},
            {"op": "recurate_figure", "asset_id": "fig1", "note": "labels unreadable"},
        ]
    )
    assert len(ops) == 6


def test_parse_unknown_op_raises():
    with pytest.raises(EditOpError, match="invalid edit ops"):
        parse_edit_ops([{"op": "delete_everything"}])


def test_rewrite_text(poster_ir: PosterIR):
    ops = parse_edit_ops(
        [{"op": "rewrite_text", "panel_id": "method", "element_index": 1, "content": "- Tight"}]
    )
    new_ir, deferred = apply_edit_ops(poster_ir, ops)
    assert not deferred
    method = next(p for p in new_ir.panels if p.id == "method")
    assert method.elements[1].content == "- Tight"


def test_rewrite_text_on_figure_element_raises(poster_ir: PosterIR):
    ops = parse_edit_ops(
        [{"op": "rewrite_text", "panel_id": "method", "element_index": 2, "content": "x"}]
    )
    with pytest.raises(EditOpError, match="not text"):
        apply_edit_ops(poster_ir, ops)


def test_set_type_scale_respects_minima(poster_ir: PosterIR):
    ops = parse_edit_ops([{"op": "set_type_scale", "level": "body", "pt": 18}])
    with pytest.raises(EditOpError, match="below the hard minimum"):
        apply_edit_ops(poster_ir, ops, min_pt={"body": 24})
    new_ir, _ = apply_edit_ops(
        poster_ir,
        parse_edit_ops([{"op": "set_type_scale", "level": "body", "pt": 26}]),
        min_pt={"body": 24},
    )
    assert new_ir.style.type_scale_pt["body"] == 26


def test_weight_change_clears_layout(poster_ir: PosterIR):
    ops = parse_edit_ops([{"op": "set_panel_weight", "panel_id": "results", "weight": 2.5}])
    new_ir, _ = apply_edit_ops(poster_ir, ops)
    assert all(p.bbox is None for p in new_ir.panels)
    assert next(p for p in new_ir.panels if p.id == "results").weight == 2.5


def test_move_panel_order_renumbers(poster_ir: PosterIR):
    ops = parse_edit_ops([{"op": "move_panel_order", "panel_id": "conclusion", "new_position": 1}])
    new_ir, _ = apply_edit_ops(poster_ir, ops)
    ordered = [p.id for p in new_ir.panels_in_order()]
    assert ordered == ["header", "conclusion", "method", "results"]
    assert sorted(p.order for p in new_ir.panels) == [0, 1, 2, 3]


def test_recolor_token(poster_ir: PosterIR):
    ops = parse_edit_ops([{"op": "recolor_token", "token": "accent", "hex": "00FF00"}])
    new_ir, _ = apply_edit_ops(poster_ir, ops)
    assert new_ir.style.palette["accent"] == "#00FF00"


def test_recurate_is_deferred(poster_ir: PosterIR):
    ops = parse_edit_ops([{"op": "recurate_figure", "asset_id": "fig1", "note": "blurry"}])
    new_ir, deferred = apply_edit_ops(poster_ir, ops)
    assert len(deferred) == 1 and deferred[0].asset_id == "fig1"


def test_unknown_targets_raise(poster_ir: PosterIR):
    for payload, match in [
        ({"op": "rewrite_text", "panel_id": "nope", "element_index": 0, "content": "x"}, "panel"),
        ({"op": "set_panel_weight", "panel_id": "nope", "weight": 1.0}, "panel"),
        ({"op": "recolor_token", "token": "nope", "hex": "#000000"}, "token"),
        ({"op": "recurate_figure", "asset_id": "nope", "note": "x"}, "asset"),
    ]:
        with pytest.raises(EditOpError, match=match):
            apply_edit_ops(poster_ir, parse_edit_ops([payload]))
