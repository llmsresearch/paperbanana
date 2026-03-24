"""Unit tests for PDF page selection parsing (no PyMuPDF required)."""

from __future__ import annotations

import pytest

from paperbanana.core.pdf_text import parse_pdf_pages_spec


def test_parse_all_pages_when_empty() -> None:
    assert parse_pdf_pages_spec(None, 5) == [1, 2, 3, 4, 5]
    assert parse_pdf_pages_spec("", 3) == [1, 2, 3]
    assert parse_pdf_pages_spec("   ", 2) == [1, 2]


def test_parse_single_and_ranges() -> None:
    assert parse_pdf_pages_spec("2", 5) == [2]
    assert parse_pdf_pages_spec("5-3", 10) == [3, 4, 5]
    assert parse_pdf_pages_spec("1-2,4,6-7", 10) == [1, 2, 4, 6, 7]


def test_parse_deduplicates_and_sorts() -> None:
    assert parse_pdf_pages_spec("3,1,3,2", 5) == [1, 2, 3]


def test_parse_out_of_range() -> None:
    with pytest.raises(ValueError, match="out of range"):
        parse_pdf_pages_spec("10", 3)


def test_parse_empty_pdf() -> None:
    with pytest.raises(ValueError, match="no pages"):
        parse_pdf_pages_spec(None, 0)
