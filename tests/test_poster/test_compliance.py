"""Image-level venue compliance + generative pixel-sizing tests."""

from __future__ import annotations

from pathlib import Path

from paperbanana.poster.compliance import check_poster_compliance
from paperbanana.poster.venue_spec import load_venue_spec

SPEC_DIR = Path(__file__).resolve().parents[2] / "data" / "venue_specs"


def _spec(venue: str):
    return load_venue_spec(venue, None, extra_dir=str(SPEC_DIR))


def test_compliant_landscape_passes():
    spec = _spec("neurips")
    w_mm, h_mm = spec.dimensions.generation_size_mm()
    # 4K-ish raster at the venue aspect.
    report = check_poster_compliance(spec, 3840, 2160, w_mm, h_mm)
    assert report.passed
    assert any(c.id == "page_size" and c.status == "pass" for c in report.checks)
    assert any(c.id == "orientation" and c.status == "pass" for c in report.checks)


def test_wrong_orientation_fails():
    spec = _spec("aaai")  # AAAI is portrait
    w_mm, h_mm = spec.dimensions.generation_size_mm()
    # Feed a landscape size (swap) -> orientation fail.
    report = check_poster_compliance(spec, 3000, 2000, max(w_mm, h_mm), min(w_mm, h_mm))
    orient = next(c for c in report.checks if c.id == "orientation")
    assert orient.status == "fail"
    assert not report.passed


def test_low_dpi_warns_not_fails():
    spec = _spec("neurips")
    w_mm, h_mm = spec.dimensions.generation_size_mm()
    # Tiny raster on a big board -> low DPI but above 96 -> warn.
    report = check_poster_compliance(spec, 3840, 2160, w_mm, h_mm)
    dpi = next(c for c in report.checks if c.id == "print_dpi")
    assert dpi.status in ("warn", "pass")  # 3840px / 48in = 80 DPI -> warn


def test_oversize_fails():
    spec = _spec("neurips")
    # Exceed the venue's max physical size.
    report = check_poster_compliance(
        spec, 4000, 3000, spec.dimensions.width_mm + 500, spec.dimensions.height_mm + 500
    )
    page = next(c for c in report.checks if c.id == "page_size")
    assert page.status == "fail"
    assert not report.passed


def test_poster_pixels_respects_budget_and_aspect():
    from paperbanana.poster.generative import GenerativePosterPipeline

    # Don't construct providers; call the pure helper on the class.
    w_px, h_px = GenerativePosterPipeline._poster_pixels(
        object.__new__(GenerativePosterPipeline), 1219.2, 914.4
    )
    assert w_px % 16 == 0 and h_px % 16 == 0
    assert w_px * h_px <= 3840 * 2160 + 1
    # Aspect roughly preserved (4:3).
    assert abs((w_px / h_px) - (1219.2 / 914.4)) < 0.1
