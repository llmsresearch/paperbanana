"""Real-figure embedding for the generative poster pipeline.

A poster figure must never be the image model's fabrication. So when
``--figures auto|real`` is set, the design is generated with bright
**magenta sentinel rectangles** as figure placeholders (reliably detected
by colour, not VLM bbox guessing), and each slot is then composited with
either the paper's real extracted figure or a faithfulness-gated
reauthored version. ``auto`` decides per figure which is better; ``real``
always uses the paper's crop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import structlog
from PIL import Image
from pydantic import BaseModel

from paperbanana.poster.agents.faithfulness import FaithfulnessAgent
from paperbanana.poster.agents.figure_curator import FigureCuratorAgent
from paperbanana.poster.types import PaperFigure

logger = structlog.get_logger()

#: Sentinel fills the model uses for placeholders (distinct so they don't mix).
SENTINEL_RGB = (255, 0, 255)  # magenta — figure slots
QR_SENTINEL_RGB = (0, 255, 255)  # cyan — QR slot
#: Min slot area as a fraction of the poster, to ignore stray sentinel pixels.
_MIN_SLOT_AREA_FRAC = 0.004
_MIN_QR_AREA_FRAC = 0.0006
#: Crops below this long-edge (px) are too low-res to feature legibly.
_MIN_FIGURE_LONG_PX = 500
#: Cover-fit a figure (cropping overflow) only when the cropped fraction is small.
_COVER_CROP_TOLERANCE = 0.12
FigureSource = Literal["real", "reauthored"]


class FigureChoice(BaseModel):
    figure_id: str
    source: FigureSource
    reason: str


class SlotBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


def rerender_high_dpi(
    pdf_path: Path, figure: PaperFigure, out_dir: Path, dpi: int = 700, max_long_px: int = 4200
) -> str:
    """Re-render a figure's PDF region at high DPI.

    Vector figures (most paper plots/diagrams) rasterize crisp at any DPI,
    so this yields a high-resolution, perfectly faithful crop with NO model
    in the loop — strictly better than VLM redraw for resolution. Raster
    source figures simply don't gain detail (capped by their own pixels).
    """
    import fitz

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{figure.id}_hires.png"
    with fitz.open(str(pdf_path)) as doc:
        page = doc[figure.page - 1]
        x0, y0, x1, y1 = figure.bbox_norm
        pad = 0.01
        clip = fitz.Rect(
            max(0.0, x0 - pad) * page.rect.width,
            max(0.0, y0 - pad) * page.rect.height,
            min(1.0, x1 + pad) * page.rect.width,
            min(1.0, y1 + pad) * page.rect.height,
        )
        region_long_in = max(clip.width, clip.height) / 72.0
        eff_dpi = min(dpi, max_long_px / max(region_long_in, 0.1))
        page.get_pixmap(dpi=int(eff_dpi), clip=clip).save(str(out))
    return str(out)


def select_poster_figures(figures: list[PaperFigure], max_figures: int = 3) -> list[PaperFigure]:
    """The figures worth featuring on the poster.

    Document order, figures before tables (a poster leads with visuals),
    capped, and skipping crops too low-resolution to be legible in print —
    a tiny crop blown up on a poster is unreadable, the user's complaint.
    """
    legible = []
    for f in figures:
        try:
            with Image.open(f.image_path) as im:
                if max(im.size) >= _MIN_FIGURE_LONG_PX:
                    legible.append(f)
        except Exception:
            continue
    ordered = sorted(legible, key=lambda f: (f.kind != "figure", f.page))
    return ordered[:max_figures]


def slot_spec(figures: list[PaperFigure]) -> str:
    """Prompt fragment describing the magenta placeholder boxes to leave."""
    lines = []
    for i, f in enumerate(figures, 1):
        with Image.open(f.image_path) as im:
            aspect = im.width / max(im.height, 1)
        shape = "wide" if aspect > 1.4 else ("tall" if aspect < 0.7 else "roughly square")
        lines.append(
            f'  - Placeholder {i} ({shape}, aspect ~{aspect:.1f}:1): for "{f.caption[:80]}"'
        )
    return (
        f"Leave EXACTLY {len(figures)} solid bright magenta (#FF00FF) rectangles as FIGURE "
        "PLACEHOLDERS — do not draw any figure, chart, or diagram yourself, only the magenta "
        "boxes (the real figures are composited in afterward). Make these figures PROMINENT: "
        "each box LARGE and integrated next to the section it supports (NOT crammed into one "
        "thin strip), sized to EXACTLY the stated aspect ratio with no padding inside the box "
        "so the figure fills it edge-to-edge:\n" + "\n".join(lines)
    )


def _detect_color_boxes(poster: Image.Image, mask, min_area_frac: float) -> list[SlotBox]:
    from scipy import ndimage

    if not mask.any():
        return []
    labels, n = ndimage.label(mask)
    total = poster.width * poster.height
    boxes: list[SlotBox] = []
    for idx in range(1, n + 1):
        ys, xs = np.where(labels == idx)
        if xs.size < min_area_frac * total:
            continue
        boxes.append(
            SlotBox(
                x=int(xs.min()),
                y=int(ys.min()),
                w=int(xs.max() - xs.min() + 1),
                h=int(ys.max() - ys.min() + 1),
            )
        )
    boxes.sort(key=lambda s: (round(s.y / max(1, poster.height // 12)), s.x))
    return boxes


def detect_slots(poster: Image.Image) -> list[SlotBox]:
    """Bounding boxes of the magenta figure placeholders, reading order."""
    arr = np.asarray(poster.convert("RGB"), dtype=np.int16)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mask = (r > 200) & (g < 80) & (b > 200)
    return _detect_color_boxes(poster, mask, _MIN_SLOT_AREA_FRAC)


def detect_qr_slot(poster: Image.Image) -> Optional[SlotBox]:
    """Bounding box of the cyan QR placeholder, if present (largest)."""
    arr = np.asarray(poster.convert("RGB"), dtype=np.int16)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mask = (r < 80) & (g > 200) & (b > 200)
    boxes = _detect_color_boxes(poster, mask, _MIN_QR_AREA_FRAC)
    return max(boxes, key=lambda s: s.w * s.h) if boxes else None


def scale_slot(slot: SlotBox, factor: float) -> SlotBox:
    return SlotBox(
        x=round(slot.x * factor),
        y=round(slot.y * factor),
        w=round(slot.w * factor),
        h=round(slot.h * factor),
    )


def composite_into_slot(poster: Image.Image, slot: SlotBox, figure: Image.Image) -> None:
    """Place the figure in the slot. Cover-fill (cropping a small overflow)
    when the aspect mismatch is minor — removing letterbox blank — else
    contain (letterbox) to avoid cutting figure data."""
    poster.paste("white", (slot.x, slot.y, slot.x + slot.w, slot.y + slot.h))
    fig = figure.convert("RGB")
    contain = min(slot.w / fig.width, slot.h / fig.height)
    cover = max(slot.w / fig.width, slot.h / fig.height)
    cropped_frac = 1.0 - (contain / cover)  # data lost if we cover-fill
    if cropped_frac <= _COVER_CROP_TOLERANCE:
        scaled = fig.resize(
            (max(1, round(fig.width * cover)), max(1, round(fig.height * cover))), Image.LANCZOS
        )
        left = (scaled.width - slot.w) // 2
        top = (scaled.height - slot.h) // 2
        poster.paste(scaled.crop((left, top, left + slot.w, top + slot.h)), (slot.x, slot.y))
    else:
        fw, fh = max(1, round(fig.width * contain)), max(1, round(fig.height * contain))
        fig = fig.resize((fw, fh), Image.LANCZOS)
        poster.paste(fig, (slot.x + (slot.w - fw) // 2, slot.y + (slot.h - fh) // 2))


async def prepare_figure(
    figure: PaperFigure,
    slot_width_mm: float,
    *,
    policy: str,
    curator: FigureCuratorAgent,
    faithfulness: FaithfulnessAgent,
    image_gen,
    palette: dict,
    reauthor_template: str,
    min_dpi: int,
    out_dir: Path,
    reauthor_px: Optional[tuple[int, int]] = None,
) -> tuple[Image.Image, FigureChoice]:
    """Choose and produce the image for one figure slot.

    ``real``: the paper's crop, always. ``auto``: the curator judges the
    crop's print quality; a poor one is reauthored (image-conditioned edit
    of the *real* figure) behind the faithfulness gate, and the real crop
    is kept whenever reauthoring is unnecessary or fails the gate — the
    paper's own figure is faithful by definition.
    """
    crop = Image.open(figure.image_path).convert("RGB")
    if policy == "real":
        return crop, FigureChoice(figure_id=figure.id, source="real", reason="policy=real")

    effective_dpi = crop.width / (slot_width_mm / 25.4) if slot_width_mm else 0.0
    decision = await curator.run(
        figure=figure,
        crop=crop,
        placed_width_mm=slot_width_mm,
        effective_dpi=effective_dpi,
        min_dpi=min_dpi,
    )
    if decision.decision == "reuse":
        return crop, FigureChoice(figure_id=figure.id, source="real", reason=decision.reason)

    # Reauthor: image-conditioned edit of the real figure, faithfulness-gated.
    try:
        prompt = reauthor_template.format(
            edit_instructions=decision.reason or "enlarge labels; clean legend for poster scale",
            primary=palette.get("primary", "#1A3A6B"),
            secondary=palette.get("secondary", "#4A6FA5"),
            accent=palette.get("accent", "#E8A33D"),
            background=palette.get("background", "#FFFFFF"),
            placed_width_mm=f"{slot_width_mm:.0f}",
        )
        # Reauthor at the slot's poster resolution (not the small crop size)
        # so the redrawn figure is crisp where it lands.
        gen_w, gen_h = reauthor_px or (crop.width, crop.height)
        reauthored = await image_gen.generate(
            prompt=prompt, images=[crop], width=gen_w, height=gen_h
        )
        verdict = await faithfulness.run(
            original=crop,
            reauthored=reauthored,
            caption=figure.caption,
        )
        if verdict.verdict == "pass":
            reauthored.save(out_dir / f"{figure.id}_reauthored.png")
            return reauthored, FigureChoice(
                figure_id=figure.id, source="reauthored", reason=decision.reason
            )
        logger.info(
            "Reauthor failed faithfulness; keeping real crop",
            figure=figure.id,
            differences=verdict.differences[:2],
        )
    except Exception as exc:  # provider lacks guided edits, etc. — real crop is always safe
        logger.warning("Reauthor unavailable; keeping real crop", figure=figure.id, error=str(exc))
    return crop, FigureChoice(
        figure_id=figure.id, source="real", reason="reauthor not faithful/available"
    )


class SlotCountError(RuntimeError):
    """The rendered design did not contain the expected number of slots."""
