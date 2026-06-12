"""pptx -> PDF -> PNG conversion via LibreOffice headless and pymupdf.

LibreOffice is a hard requirement of the poster head: if it cannot be
found the pipeline fails fast (before any API spend) with installation
instructions — there is no degraded raster-only path.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import structlog

from paperbanana.poster.types import PosterIR

logger = structlog.get_logger()

#: Well-known LibreOffice binary locations checked after PATH.
KNOWN_SOFFICE_LOCATIONS = (
    "/Applications/LibreOffice.app/Contents/MacOS/soffice",
    "/usr/bin/soffice",
    "/usr/local/bin/soffice",
    "/opt/homebrew/bin/soffice",
)

SOFFICE_TIMEOUT_S = 180


class SofficeNotFoundError(RuntimeError):
    """LibreOffice is required to convert pptx posters to PDF."""

    def __init__(self, explicit: str | None = None):
        checked = [explicit] if explicit else []
        checked += ["PATH (soffice)", *KNOWN_SOFFICE_LOCATIONS]
        super().__init__(
            "LibreOffice (soffice) was not found; it is required to convert the "
            f"poster pptx to a press-ready PDF. Checked: {', '.join(checked)}. "
            "Install it with 'brew install --cask libreoffice' (macOS) or your "
            "distribution's package manager, or set SOFFICE_PATH to the binary."
        )


class ConversionError(RuntimeError):
    """A document conversion subprocess failed."""


def find_soffice(explicit: str | None = None) -> Path:
    """Locate the LibreOffice binary.

    Precedence: explicit path (``SOFFICE_PATH`` setting) > ``soffice`` on
    PATH > well-known install locations.

    Raises:
        SofficeNotFoundError: If no binary is found. An explicit path that
            does not exist is an error, not a fall-through.
    """
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            return path
        raise SofficeNotFoundError(explicit)
    on_path = shutil.which("soffice")
    if on_path:
        return Path(on_path)
    for candidate in KNOWN_SOFFICE_LOCATIONS:
        if Path(candidate).is_file():
            return Path(candidate)
    raise SofficeNotFoundError()


def pptx_to_pdf(
    pptx_path: Path,
    out_dir: Path,
    soffice: Path,
    timeout_s: int = SOFFICE_TIMEOUT_S,
) -> Path:
    """Convert a pptx to PDF with LibreOffice headless.

    Raises:
        ConversionError: On non-zero exit, timeout, or missing output.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(soffice),
        "--headless",
        "--norestore",
        "--convert-to",
        "pdf",
        "--outdir",
        str(out_dir),
        str(pptx_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired as exc:
        raise ConversionError(
            f"LibreOffice conversion timed out after {timeout_s}s for {pptx_path}"
        ) from exc
    pdf_path = out_dir / f"{pptx_path.stem}.pdf"
    if result.returncode != 0 or not pdf_path.is_file():
        raise ConversionError(
            f"LibreOffice failed to convert {pptx_path} (exit {result.returncode}).\n"
            f"stdout: {result.stdout.strip()}\nstderr: {result.stderr.strip()}"
        )
    logger.info("Converted pptx to PDF", pdf=str(pdf_path))
    return pdf_path


def pdf_to_png(pdf_path: Path, out_path: Path, dpi: int = 150) -> Path:
    """Rasterize page 1 of a PDF to PNG at the given DPI (pymupdf)."""
    import fitz

    with fitz.open(str(pdf_path)) as doc:
        if doc.page_count < 1:
            raise ConversionError(f"PDF has no pages: {pdf_path}")
        page = doc[0]
        pix = page.get_pixmap(dpi=dpi)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(out_path))
    return out_path


def render_panel_crops(
    pdf_path: Path, ir: PosterIR, out_dir: Path, dpi: int = 200
) -> dict[str, Path]:
    """Render a zoomed crop of each panel from the converted PDF.

    These per-panel crops are what the VLM critic inspects — judging a
    full A0 canvas downscaled to model input resolution misses exactly
    the legibility defects posters fail on.
    """
    import fitz

    out_dir.mkdir(parents=True, exist_ok=True)
    crops: dict[str, Path] = {}
    with fitz.open(str(pdf_path)) as doc:
        page = doc[0]
        page_w_pt, page_h_pt = page.rect.width, page.rect.height
        for panel in ir.panels:
            box = panel.bbox
            if box is None:
                raise ValueError(f"panel '{panel.id}' has no bbox; cannot crop")
            clip = fitz.Rect(
                box.x_mm / ir.size.width_mm * page_w_pt,
                box.y_mm / ir.size.height_mm * page_h_pt,
                (box.x_mm + box.w_mm) / ir.size.width_mm * page_w_pt,
                (box.y_mm + box.h_mm) / ir.size.height_mm * page_h_pt,
            )
            pix = page.get_pixmap(dpi=dpi, clip=clip)
            crop_path = out_dir / f"panel_{panel.id}.png"
            pix.save(str(crop_path))
            crops[panel.id] = crop_path
    return crops
