"""Paper PDF ingestion: text, metadata, and figure extraction.

One extraction path, per the design: pages are rendered at detection DPI
and a VLM locates figure/table regions with captions; each detection is
then cross-checked against the PDF *text layer* (caption anchoring) and
cropped from a fresh high-DPI render. ``get_images()``-style embedded-
image extraction is deliberately not used — vector figures defeat it.
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path

import structlog
from PIL import Image

from paperbanana.core.orchestrate import split_paper_sections
from paperbanana.core.pdf_text import extract_text_from_pdf
from paperbanana.poster.agents.figure_detector import DetectedRegion, FigureDetectorAgent
from paperbanana.poster.agents.paper_metadata import PaperMetadataAgent
from paperbanana.poster.types import PaperAssets, PaperFigure, PaperSection

logger = structlog.get_logger()

DETECT_DPI = 200
#: Padding added around detected regions when cropping, in normalized coords.
CROP_PAD_NORM = 0.005
#: Vertical band (fraction of page height) around a region in which its
#: caption text must be found for the detection to count as anchored.
ANCHOR_BAND_NORM = 0.08

PAGE_CONCURRENCY = 4


class IngestError(RuntimeError):
    """Paper ingestion failed."""


def render_page(pdf_path: Path, page_index: int, dpi: int) -> Image.Image:
    """Render one PDF page (0-based) to a PIL image at the given DPI."""
    import fitz

    with fitz.open(str(pdf_path)) as doc:
        page = doc[page_index]
        pix = page.get_pixmap(dpi=dpi)
        return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


async def ingest_paper(
    pdf_path: Path,
    detector: FigureDetectorAgent,
    metadata_agent: PaperMetadataAgent,
    out_dir: Path,
    extract_dpi: int = 300,
) -> PaperAssets:
    """Ingest a paper PDF into :class:`PaperAssets`.

    Args:
        pdf_path: Source paper PDF.
        detector: VLM figure/table detector.
        metadata_agent: VLM bibliographic metadata extractor.
        out_dir: Run-scoped directory for extracted assets
            (``figures/`` is created inside it).
        extract_dpi: DPI of the final figure crops.
    """
    import fitz

    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise IngestError(f"paper PDF not found: {pdf_path}")

    with fitz.open(str(pdf_path)) as doc:
        page_count = doc.page_count
    if page_count < 1:
        raise IngestError(f"paper PDF has no pages: {pdf_path}")

    full_text = extract_text_from_pdf(pdf_path)
    sections = [
        PaperSection(heading=s["heading"], text=s["content"])
        for s in split_paper_sections(full_text)
    ]

    first_pages = extract_text_from_pdf(pdf_path, pages_spec=f"1-{min(2, page_count)}")
    metadata = await metadata_agent.run(first_pages_text=first_pages)

    sem = asyncio.Semaphore(PAGE_CONCURRENCY)

    async def detect(page_index: int) -> tuple[int, list[DetectedRegion]]:
        async with sem:
            image = await asyncio.to_thread(render_page, pdf_path, page_index, DETECT_DPI)
            regions = await detector.run(page_image=image, page_number=page_index + 1)
            return page_index, regions

    detections = await asyncio.gather(*(detect(i) for i in range(page_count)))

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figures: list[PaperFigure] = []
    counters = {"figure": 0, "table": 0}
    seen_numbers: dict[str, PaperFigure] = {}

    with fitz.open(str(pdf_path)) as doc:
        for page_index, regions in sorted(detections):
            page = doc[page_index]
            for region in regions:
                bbox_norm = region.bbox_norm()
                anchored = _caption_is_anchored(page, region, bbox_norm)
                number_key = (region.figure_number or "").strip().lower()
                if number_key and number_key in seen_numbers:
                    existing = seen_numbers[number_key]
                    logger.warning(
                        "Duplicate detection for numbered figure; keeping first",
                        figure=region.figure_number,
                        page=page_index + 1,
                        kept_page=existing.page,
                    )
                    continue
                counters[region.kind] += 1
                figure_id = ("fig" if region.kind == "figure" else "tab") + str(
                    counters[region.kind]
                )
                crop_path = figures_dir / f"{figure_id}.png"
                _crop_region(page, bbox_norm, extract_dpi, crop_path)
                figure = PaperFigure(
                    id=figure_id,
                    kind=region.kind,
                    page=page_index + 1,
                    bbox_norm=bbox_norm,
                    caption=region.caption or (region.figure_number or figure_id),
                    image_path=str(crop_path),
                    extract_dpi=extract_dpi,
                    caption_anchored=anchored,
                )
                figures.append(figure)
                if number_key:
                    seen_numbers[number_key] = figure

    assets = PaperAssets(
        pdf_path=str(pdf_path),
        page_count=page_count,
        title=metadata.title,
        authors=metadata.authors,
        affiliations=metadata.affiliations,
        abstract=metadata.abstract,
        sections=sections,
        figures=figures,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "paper_assets.json").write_text(assets.model_dump_json(indent=2), encoding="utf-8")
    logger.info(
        "Paper ingested",
        pages=page_count,
        figures=len([f for f in figures if f.kind == "figure"]),
        tables=len([f for f in figures if f.kind == "table"]),
        sections=len(sections),
    )
    return assets


def _caption_is_anchored(page, region: DetectedRegion, bbox_norm: list[float]) -> bool:
    """Cross-check a detected caption against the PDF text layer.

    The caption label (e.g. ``Figure 3``) must occur in the text layer
    within (or just below/above) the detected region. Detections without
    a numbered caption are never anchored.
    """
    label = (region.figure_number or "").strip()
    if not label:
        return False
    rects = page.search_for(label)
    if not rects:
        return False
    page_w = page.rect.width
    page_h = page.rect.height
    x0, y0, x1, y1 = bbox_norm
    band_top = max(0.0, y0 - ANCHOR_BAND_NORM)
    band_bottom = min(1.0, y1 + ANCHOR_BAND_NORM)
    for rect in rects:
        cx = (rect.x0 + rect.x1) / 2 / page_w
        cy = (rect.y0 + rect.y1) / 2 / page_h
        if x0 - 0.02 <= cx <= x1 + 0.02 and band_top <= cy <= band_bottom:
            return True
    return False


def _crop_region(page, bbox_norm: list[float], dpi: int, out_path: Path) -> None:
    """Crop a normalized region from a high-DPI page render."""
    import fitz

    x0, y0, x1, y1 = bbox_norm
    pad = CROP_PAD_NORM
    clip = fitz.Rect(
        max(0.0, x0 - pad) * page.rect.width,
        max(0.0, y0 - pad) * page.rect.height,
        min(1.0, x1 + pad) * page.rect.width,
        min(1.0, y1 + pad) * page.rect.height,
    )
    pix = page.get_pixmap(dpi=dpi, clip=clip)
    pix.save(str(out_path))
