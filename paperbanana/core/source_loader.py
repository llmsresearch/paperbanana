"""Load methodology source text from plain files or PDF."""

from __future__ import annotations

from pathlib import Path

from paperbanana.core.pdf_text import extract_text_from_pdf, is_pdf_path


def load_methodology_source(path: Path, *, pdf_pages: str | None = None) -> str:
    """Read methodology context from *path*.

    For ``.pdf`` files, extracts text with PyMuPDF (optional extra ``paperbanana[pdf]``).
    *pdf_pages* selects 1-based pages (comma-separated and/or ranges); omitted means all pages.

    For non-PDF files, reads UTF-8 text. *pdf_pages* must not be set in that case.
    """
    path = Path(path)
    if is_pdf_path(path):
        return extract_text_from_pdf(path, pages_spec=pdf_pages)
    if pdf_pages is not None and str(pdf_pages).strip():
        raise ValueError("pdf_pages applies only to PDF inputs")
    return path.read_text(encoding="utf-8")
