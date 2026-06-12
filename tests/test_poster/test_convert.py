"""Conversion chain tests; real LibreOffice runs only when installed."""

from __future__ import annotations

from pathlib import Path

import pytest

from paperbanana.poster import convert
from paperbanana.poster.convert import (
    ConversionError,
    SofficeNotFoundError,
    find_soffice,
    pdf_to_png,
    pptx_to_pdf,
    render_panel_crops,
)
from paperbanana.poster.renderer import render_pptx
from paperbanana.poster.types import PosterIR, mm_to_inches


def _soffice_or_none() -> Path | None:
    try:
        return find_soffice()
    except SofficeNotFoundError:
        return None


needs_soffice = pytest.mark.skipif(_soffice_or_none() is None, reason="LibreOffice not installed")


def test_find_soffice_explicit_missing_raises():
    with pytest.raises(SofficeNotFoundError, match="brew install"):
        find_soffice("/no/such/binary")


def test_find_soffice_nothing_found(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    monkeypatch.setattr(convert, "KNOWN_SOFFICE_LOCATIONS", ("/no/such/place",))
    with pytest.raises(SofficeNotFoundError):
        find_soffice()


def test_pptx_to_pdf_failure_surfaces_stderr(tmp_path: Path):
    fake = tmp_path / "soffice"
    fake.write_text("#!/bin/sh\necho boom >&2\nexit 3\n")
    fake.chmod(0o755)
    src = tmp_path / "poster.pptx"
    src.write_bytes(b"not a pptx")
    with pytest.raises(ConversionError, match="boom"):
        pptx_to_pdf(src, tmp_path / "out", fake)


@needs_soffice
def test_full_conversion_chain(poster_ir: PosterIR, tmp_path: Path):
    import fitz

    pptx_path = render_pptx(poster_ir, tmp_path / "poster.pptx", tmp_path / "work")
    pdf_path = pptx_to_pdf(pptx_path, tmp_path / "pdf", find_soffice())
    assert pdf_path.is_file()

    # PDF physical page size must match the IR (within 1mm).
    with fitz.open(str(pdf_path)) as doc:
        page = doc[0]
        width_in = page.rect.width / 72
        height_in = page.rect.height / 72
    assert width_in == pytest.approx(mm_to_inches(poster_ir.size.width_mm), abs=0.05)
    assert height_in == pytest.approx(mm_to_inches(poster_ir.size.height_mm), abs=0.05)

    png_path = pdf_to_png(pdf_path, tmp_path / "preview.png", dpi=72)
    assert png_path.is_file()

    crops = render_panel_crops(pdf_path, poster_ir, tmp_path / "panels", dpi=100)
    assert set(crops) == {p.id for p in poster_ir.panels}
    for crop in crops.values():
        assert crop.is_file()
