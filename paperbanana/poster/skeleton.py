"""Cheap skeleton previews for best-of-N layout ranking.

A skeleton preview renders a candidate structure with the real text but
gray placeholder figures (true aspect ratios, id-stamped) — no image
generation, one fast render+convert per candidate. The judge ranks the
previews; only the winner gets the full-quality render.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import structlog
from PIL import Image, ImageDraw

from paperbanana.poster.convert import pdf_to_png, pptx_to_pdf
from paperbanana.poster.layout import BandOverflowError, place_bands
from paperbanana.poster.renderer import render_pptx
from paperbanana.poster.types import FigureAsset, PosterIR

logger = structlog.get_logger()

#: Placeholder raster long edge — measurement uses px *ratios* only, so
#: small placeholders keep geometry identical while rendering fast.
PLACEHOLDER_MAX_PX = 640


def placeholder_assets(assets: dict[str, FigureAsset], workdir: Path) -> dict[str, FigureAsset]:
    """Gray id-stamped stand-ins at each asset's true aspect ratio."""
    workdir.mkdir(parents=True, exist_ok=True)
    placeholders: dict[str, FigureAsset] = {}
    for asset_id, asset in assets.items():
        scale = PLACEHOLDER_MAX_PX / max(asset.width_px, asset.height_px)
        w = max(64, round(asset.width_px * min(1.0, scale)))
        h = max(64, round(asset.height_px * min(1.0, scale)))
        img = Image.new("RGB", (w, h), (210, 213, 219))
        draw = ImageDraw.Draw(img)
        draw.rectangle([2, 2, w - 3, h - 3], outline=(120, 124, 133), width=2)
        draw.line([2, 2, w - 3, h - 3], fill=(160, 164, 173), width=2)
        draw.line([2, h - 3, w - 3, 2], fill=(160, 164, 173), width=2)
        draw.text((w // 2 - 4 * len(asset_id), h // 2 - 8), asset_id, fill=(60, 63, 70))
        path = workdir / f"ph_{asset_id}.png"
        img.save(path)
        placeholders[asset_id] = asset.model_copy(update={"path": str(path)})
    return placeholders


def render_skeleton_preview(
    ir: PosterIR,
    measured: dict[str, float],
    workdir: Path,
    soffice: Path,
    label: str,
) -> tuple[Optional[Path], float]:
    """Place + render one candidate cheaply.

    Returns (preview_png_or_None, overflow_penalty_mm). A candidate that
    overflows is not rendered — its deficit is the penalty the ranking
    sees instead of a picture.
    """
    try:
        placed = place_bands(ir, measured)
    except BandOverflowError as exc:
        logger.info("Skeleton candidate overflows", label=label, deficit=exc.page_deficit_mm)
        return None, exc.page_deficit_mm
    workdir.mkdir(parents=True, exist_ok=True)
    pptx = render_pptx(placed, workdir / f"{label}.pptx", workdir / "work")
    pdf = pptx_to_pdf(pptx, workdir, soffice)
    png = pdf_to_png(pdf, workdir / f"{label}.png", dpi=72)
    return png, 0.0
