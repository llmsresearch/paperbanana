"""Preflight rule tests — every rule exercised both ways."""

from __future__ import annotations

import pytest

from paperbanana.poster.preflight import (
    contrast_ratio,
    render_preflight_markdown,
    run_preflight,
)
from paperbanana.poster.types import PosterIR, TextElement
from paperbanana.poster.venue_spec import (
    DimensionRule,
    FileRules,
    RequiredElementRule,
    TextRules,
    VenueSpec,
)


def _spec(**overrides) -> VenueSpec:
    base = dict(
        venue="neurips",
        year=2025,
        display_name="NeurIPS 2025",
        sources=["https://neurips.cc/Conferences/2025/PosterInstructions"],
        dimensions=DimensionRule(
            mode="max",
            width_mm=2438.4,
            height_mm=1219.2,
            default_width_mm=1219.2,
            default_height_mm=914.4,
        ),
        text_rules=TextRules(min_pt={}, min_image_dpi=100),
        required_elements=[
            RequiredElementRule(
                id="paper_title",
                description="Title present",
                rule="text_level_present",
                params={"level": "title"},
            )
        ],
    )
    base.update(overrides)
    return VenueSpec(**base)


def _failing(report, check_id_prefix):
    return [c for c in report.failures if c.id.startswith(check_id_prefix)]


def test_clean_poster_passes(poster_ir: PosterIR):
    report = run_preflight(poster_ir, _spec())
    assert report.passed, [f"{c.id}: {c.detail}" for c in report.failures]


def test_page_size_max_violation(poster_ir: PosterIR):
    spec = _spec(
        dimensions=DimensionRule(
            mode="max",
            width_mm=1000,
            height_mm=900,
            default_width_mm=1000,
            default_height_mm=900,
        )
    )
    report = run_preflight(poster_ir, spec)
    assert _failing(report, "page_size")


def test_page_size_exact_violation(poster_ir: PosterIR):
    spec = _spec(dimensions=DimensionRule(mode="exact", width_mm=841, height_mm=1189))
    report = run_preflight(poster_ir, spec)
    assert _failing(report, "page_size")


def test_orientation_violation(poster_ir: PosterIR):
    spec = _spec(
        dimensions=DimensionRule(
            mode="max",
            width_mm=2438.4,
            height_mm=1219.2,
            orientation="portrait",
            default_width_mm=900,
            default_height_mm=1200,
        )
    )
    report = run_preflight(poster_ir, spec)
    assert _failing(report, "orientation")


def test_font_minima_venue_rule(poster_ir: PosterIR):
    spec = _spec(text_rules=TextRules(min_pt={"body": 36}, min_image_dpi=100))
    report = run_preflight(poster_ir, spec)  # body is 28pt in the fixture
    assert _failing(report, "font_minima.body")


def test_font_minima_legibility_floor(poster_ir: PosterIR):
    data = poster_ir.model_dump()
    data["style"]["type_scale_pt"]["body"] = 12  # below the 24pt floor
    report = run_preflight(PosterIR(**data), _spec())
    assert _failing(report, "font_minima.body")


def test_image_dpi_violation(poster_ir: PosterIR):
    data = poster_ir.model_dump()
    data["assets"]["fig1"]["width_px"] = 500  # ~36 DPI at 360mm placement
    report = run_preflight(PosterIR(**data), _spec())
    assert _failing(report, "image_dpi.fig1")


def test_required_element_missing(poster_ir: PosterIR):
    data = poster_ir.model_dump()
    data["panels"][0]["elements"] = [
        e for e in data["panels"][0]["elements"] if e.get("level") != "title"
    ]
    report = run_preflight(PosterIR(**data), _spec())
    assert _failing(report, "required.paper_title")


def test_required_element_warn_severity(poster_ir: PosterIR):
    spec = _spec(
        required_elements=[
            RequiredElementRule(
                id="qr",
                description="QR code",
                rule="element_kind_present",
                params={"kind": "qr"},
                severity="warn",
            )
        ]
    )
    report = run_preflight(poster_ir, spec)
    assert report.passed
    assert any(c.id == "required.qr" for c in report.warnings)


def test_text_overflow_detected(poster_ir: PosterIR):
    data = poster_ir.model_dump()
    data["panels"][2]["elements"] = [
        TextElement(level="body", content="lorem ipsum " * 600).model_dump()
    ]
    report = run_preflight(PosterIR(**data), _spec())
    assert _failing(report, "text_overflow.results")


def test_contrast_violation(poster_ir: PosterIR):
    data = poster_ir.model_dump()
    data["style"]["palette"]["text"] = "#DDDDDD"  # light gray on light panel
    report = run_preflight(PosterIR(**data), _spec())
    assert _failing(report, "contrast")


def test_contrast_ratio_values():
    assert contrast_ratio("#000000", "#FFFFFF") == pytest.approx(21.0)
    assert contrast_ratio("#FFFFFF", "#FFFFFF") == pytest.approx(1.0)


def test_caption_anchoring_warns(poster_ir: PosterIR):
    data = poster_ir.model_dump()
    data["assets"]["fig1"]["provenance"]["caption_anchored"] = False
    report = run_preflight(PosterIR(**data), _spec())
    assert report.passed
    assert any(c.id == "caption_anchor.fig1" for c in report.warnings)


def test_reauthored_figure_must_be_verified(poster_ir: PosterIR):
    data = poster_ir.model_dump()
    prov = data["assets"]["fig1"]["provenance"]
    prov["decision"] = "reauthor"
    prov["edit_instructions"] = "enlarge axis labels"
    prov["faithfulness"] = "not_required"
    report = run_preflight(PosterIR(**data), _spec())
    assert _failing(report, "faithfulness.fig1")
    prov["faithfulness"] = "verified"
    report = run_preflight(PosterIR(**data), _spec())
    assert not _failing(report, "faithfulness.fig1")


def test_pdf_size_rule(poster_ir: PosterIR, tmp_path):
    big = tmp_path / "poster.pdf"
    big.write_bytes(b"0" * (3 * 1024 * 1024))
    spec = _spec(file_rules=FileRules(pdf_max_mb=2))
    report = run_preflight(poster_ir, spec, pdf_path=big)
    assert _failing(report, "pdf_file_size")


def test_png_pixel_rule(poster_ir: PosterIR, tmp_path):
    from PIL import Image

    png = tmp_path / "preview.png"
    Image.new("RGB", (5000, 3000), "white").save(png)
    spec = _spec(file_rules=FileRules(png_max_px=4000))
    report = run_preflight(poster_ir, spec, png_path=png)
    assert _failing(report, "png_pixel_limit")


def test_markdown_report_renders(poster_ir: PosterIR):
    report = run_preflight(poster_ir, _spec())
    md = render_preflight_markdown(report)
    assert "# Poster Preflight Report" in md
    assert "page_size" in md
