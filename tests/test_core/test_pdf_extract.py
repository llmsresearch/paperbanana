"""PDF extraction and source loading (requires PyMuPDF)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fitz", reason="PyMuPDF not installed")

from paperbanana.core.pdf_text import extract_text_from_pdf
from paperbanana.core.source_loader import load_methodology_source


def test_extract_text_from_pdf(tmp_path: Path) -> None:
    import fitz

    pdf_path = tmp_path / "t.pdf"
    doc = fitz.open()
    p0 = doc.new_page()
    p0.insert_text((72, 72), "Hello from page one")
    p1 = doc.new_page()
    p1.insert_text((72, 72), "Second page content")
    doc.save(pdf_path)
    doc.close()

    full = extract_text_from_pdf(pdf_path)
    assert "Hello from page one" in full
    assert "Second page content" in full
    assert "--- Page 1 ---" in full
    assert "--- Page 2 ---" in full

    p1 = extract_text_from_pdf(pdf_path, pages_spec="1")
    assert "Hello from page one" in p1
    assert "Second page content" not in p1


def test_load_methodology_source_txt(tmp_path: Path) -> None:
    p = tmp_path / "m.txt"
    p.write_text("plain text", encoding="utf-8")
    assert load_methodology_source(p) == "plain text"


def test_load_methodology_source_txt_rejects_pdf_pages(tmp_path: Path) -> None:
    p = tmp_path / "m.txt"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="pdf_pages applies only"):
        load_methodology_source(p, pdf_pages="1-2")


def test_load_methodology_source_pdf(tmp_path: Path) -> None:
    import fitz

    pdf_path = tmp_path / "p.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "PDF methodology")
    doc.save(pdf_path)
    doc.close()

    out = load_methodology_source(pdf_path)
    assert "PDF methodology" in out
