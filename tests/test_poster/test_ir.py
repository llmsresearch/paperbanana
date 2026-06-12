"""PosterIR validation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from paperbanana.poster.types import (
    BBox,
    FigureProvenance,
    PhysicalSize,
    PosterIR,
    StyleTokens,
    inches_to_mm,
    mm_to_emu,
    mm_to_inches,
)


def test_unit_conversions():
    assert mm_to_inches(25.4) == pytest.approx(1.0)
    assert inches_to_mm(48) == pytest.approx(1219.2)
    assert mm_to_emu(1.0) == 36000
    assert mm_to_emu(914.4) == 914.4 * 36000


def test_physical_size_orientation():
    assert PhysicalSize(width_mm=1219.2, height_mm=914.4).orientation == "landscape"
    assert PhysicalSize(width_mm=841, height_mm=1189).orientation == "portrait"


def test_bbox_overlap():
    a = BBox(x_mm=0, y_mm=0, w_mm=100, h_mm=100)
    b = BBox(x_mm=50, y_mm=50, w_mm=100, h_mm=100)
    c = BBox(x_mm=100.05, y_mm=0, w_mm=100, h_mm=100)  # adjacent within tolerance
    assert a.overlaps(b)
    assert not a.overlaps(c)


def test_valid_ir_roundtrip(poster_ir: PosterIR):
    data = poster_ir.model_dump_json()
    restored = PosterIR.model_validate_json(data)
    assert restored.paper_title == poster_ir.paper_title
    assert restored.is_fully_placed()
    assert [p.id for p in restored.panels_in_order()] == [
        "header",
        "method",
        "results",
        "conclusion",
    ]


def test_ir_rejects_orientation_mismatch(poster_ir: PosterIR):
    with pytest.raises(ValidationError, match="orientation"):
        PosterIR(
            **{
                **poster_ir.model_dump(),
                "orientation": "portrait",
            }
        )


def test_ir_rejects_duplicate_panel_ids(poster_ir: PosterIR):
    data = poster_ir.model_dump()
    data["panels"][1]["id"] = "header"
    data["panels"][1]["order"] = 5
    with pytest.raises(ValidationError, match="duplicate panel ids"):
        PosterIR(**data)


def test_ir_rejects_unknown_asset_ref(poster_ir: PosterIR):
    data = poster_ir.model_dump()
    data["assets"] = {}
    with pytest.raises(ValidationError, match="unknown asset"):
        PosterIR(**data)


def test_ir_rejects_out_of_bounds_panel(poster_ir: PosterIR):
    data = poster_ir.model_dump()
    data["panels"][2]["bbox"] = {"x_mm": 1000, "y_mm": 150, "w_mm": 380, "h_mm": 700}
    data["panels"][3]["bbox"] = {"x_mm": 20, "y_mm": 860, "w_mm": 100, "h_mm": 40}
    with pytest.raises(ValidationError, match="exceeds page bounds"):
        PosterIR(**data)


def test_ir_rejects_overlapping_panels(poster_ir: PosterIR):
    data = poster_ir.model_dump()
    data["panels"][2]["bbox"] = dict(data["panels"][1]["bbox"])
    with pytest.raises(ValidationError, match="overlap"):
        PosterIR(**data)


def test_effective_dpi(figure_asset):
    # 2000 px placed at 380 mm wide -> 2000 / (380/25.4) ≈ 133.7 DPI
    assert figure_asset.effective_dpi(380) == pytest.approx(133.7, abs=0.1)


def test_provenance_requires_edit_instructions_for_reauthor():
    with pytest.raises(ValidationError, match="edit_instructions"):
        FigureProvenance(
            origin="paper",
            source_page=2,
            decision="reauthor",
            decision_reason="labels too small",
        )


def test_provenance_requires_source_page_for_paper_origin():
    with pytest.raises(ValidationError, match="source_page"):
        FigureProvenance(origin="paper", decision="reuse", decision_reason="ok")


def test_style_tokens_reject_missing_palette_key(style_tokens: StyleTokens):
    data = style_tokens.model_dump()
    del data["palette"]["accent"]
    with pytest.raises(ValidationError, match="accent"):
        StyleTokens(**data)


def test_style_tokens_reject_bad_hex(style_tokens: StyleTokens):
    data = style_tokens.model_dump()
    data["palette"]["primary"] = "blue"
    with pytest.raises(ValidationError, match="hex"):
        StyleTokens(**data)
