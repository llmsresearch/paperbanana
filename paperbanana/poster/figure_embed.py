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
from typing import Literal

import numpy as np
import structlog
from PIL import Image
from pydantic import BaseModel

from paperbanana.poster.agents.faithfulness import FaithfulnessAgent
from paperbanana.poster.agents.figure_curator import FigureCuratorAgent
from paperbanana.poster.types import PaperFigure

logger = structlog.get_logger()

#: Sentinel fill the image model is told to use for figure placeholders.
SENTINEL_RGB = (255, 0, 255)
#: Min slot area as a fraction of the poster, to ignore stray magenta pixels.
_MIN_SLOT_AREA_FRAC = 0.004
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


def select_poster_figures(figures: list[PaperFigure], max_figures: int = 4) -> list[PaperFigure]:
    """The figures worth featuring on the poster.

    Document order, figures before tables (a poster leads with visuals),
    capped — a poster shows a handful of key visuals, not every crop.
    """
    ordered = sorted(figures, key=lambda f: (f.kind != "figure", f.page))
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
        "boxes (the real figures are composited in afterward). Place them where the figures "
        "belong in the layout and size each to its aspect ratio:\n" + "\n".join(lines)
    )


def detect_slots(poster: Image.Image) -> list[SlotBox]:
    """Bounding boxes of the magenta placeholder rectangles, reading order."""
    from scipy import ndimage

    arr = np.asarray(poster.convert("RGB"), dtype=np.int16)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mask = (r > 200) & (g < 80) & (b > 200)
    if not mask.any():
        return []
    labels, n = ndimage.label(mask)
    total = poster.width * poster.height
    boxes: list[SlotBox] = []
    for idx in range(1, n + 1):
        ys, xs = np.where(labels == idx)
        if xs.size < _MIN_SLOT_AREA_FRAC * total:
            continue
        boxes.append(
            SlotBox(
                x=int(xs.min()),
                y=int(ys.min()),
                w=int(xs.max() - xs.min() + 1),
                h=int(ys.max() - ys.min() + 1),
            )
        )
    # reading order: top-to-bottom in coarse rows, then left-to-right
    boxes.sort(key=lambda s: (round(s.y / max(1, poster.height // 12)), s.x))
    return boxes


def composite_into_slot(poster: Image.Image, slot: SlotBox, figure: Image.Image) -> None:
    """Paste the figure into the slot, aspect-kept, centered, white-backed."""
    poster.paste("white", (slot.x, slot.y, slot.x + slot.w, slot.y + slot.h))
    fig = figure.convert("RGB")
    scale = min(slot.w / fig.width, slot.h / fig.height)
    fw, fh = max(1, int(fig.width * scale)), max(1, int(fig.height * scale))
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
        reauthored = await image_gen.generate(
            prompt=prompt, images=[crop], width=crop.width, height=crop.height
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
