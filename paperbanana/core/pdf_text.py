"""Extract plain text from PDF files (optional pymupdf dependency)."""

from __future__ import annotations

from pathlib import Path

_PDF_INSTALL_HINT = "Install PyMuPDF: pip install 'paperbanana[pdf]' or pip install pymupdf"


def parse_pdf_pages_spec(spec: str | None, page_count: int) -> list[int]:
    """Resolve a 1-based page selection into a sorted unique list of page numbers.

    *spec* is ``None``, empty, or whitespace only: all pages ``1 .. page_count``.
    Otherwise comma-separated tokens: single pages (``3``) or inclusive ranges (``2-5``).
    """
    if page_count < 1:
        raise ValueError("PDF has no pages")

    if spec is None or not str(spec).strip():
        return list(range(1, page_count + 1))

    seen: set[int] = set()
    for raw in str(spec).split(","):
        part = raw.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
        else:
            start = end = int(part)
        if start > end:
            start, end = end, start
        for p in range(start, end + 1):
            if p < 1 or p > page_count:
                raise ValueError(f"Page {p} is out of range for this PDF (1–{page_count})")
            seen.add(p)

    if not seen:
        return list(range(1, page_count + 1))

    return sorted(seen)


def extract_text_from_pdf(path: Path, pages_spec: str | None = None) -> str:
    """Open *path* and extract text from the selected pages (1-based *pages_spec*, see
    :func:`parse_pdf_pages_spec`). Pages are concatenated with clear separators.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise ImportError(f"PDF input requires PyMuPDF. {_PDF_INSTALL_HINT}") from e

    path = Path(path)
    doc = fitz.open(path)
    try:
        total = doc.page_count
        pages = parse_pdf_pages_spec(pages_spec, total)
        blocks: list[str] = []
        for p1 in pages:
            page = doc.load_page(p1 - 1)
            raw = page.get_text()
            text = raw.strip()
            if not text:
                text = "[no extractable text on this page]"
            blocks.append(f"--- Page {p1} ---\n\n{text}")
        return "\n\n".join(blocks)
    finally:
        doc.close()


def is_pdf_path(path: Path) -> bool:
    return path.suffix.lower() == ".pdf"
