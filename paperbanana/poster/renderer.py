"""Deterministic PosterIR -> .pptx renderer.

The renderer is pure mechanism: every creative decision (content, order,
emphasis, palette, type scale) already lives in the IR. Text heights are
measured with the actual font files *before* any shape is written; text
that cannot fit raises :class:`TextOverflowError` so the pipeline can
ask for a rewrite — the renderer never silently shrinks fonts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import structlog
from matplotlib import font_manager
from PIL import Image, ImageFont

from paperbanana.poster.types import (
    FigureElement,
    Panel,
    PosterElement,
    PosterIR,
    QRElement,
    TextElement,
    mm_to_emu,
)

logger = structlog.get_logger()

MM_PER_PT = 25.4 / 72.0
#: Extra line height as a fraction of font size.
LINE_SPACING = 1.18
#: Vertical gap between consecutive elements inside a panel, in mm.
ELEMENT_GAP_MM = 6.0
#: Inner padding of a panel, in mm.
PANEL_PADDING_MM = 10.0
#: Safety margin applied to measured text heights (PIL vs PowerPoint
#: metric divergence), as a multiplicative factor.
MEASURE_SAFETY = 1.10
#: Fixed physical size of a rendered QR code, in mm.
QR_SIZE_MM = 80.0

#: Maximum pptx page dimension (PowerPoint limit: 56 inches).
MAX_PPTX_DIM_MM = 56 * 25.4


class TextOverflowError(ValueError):
    """A panel's content does not fit its box at the IR's type scale."""

    def __init__(self, panel_id: str, required_mm: float, available_mm: float):
        self.panel_id = panel_id
        self.required_mm = required_mm
        self.available_mm = available_mm
        super().__init__(
            f"Panel '{panel_id}' content needs {required_mm:.0f}mm but only "
            f"{available_mm:.0f}mm is available at the current type scale. "
            "Shorten the content, raise the panel weight, or adjust the storyboard."
        )


class PageSizeError(ValueError):
    """The poster page exceeds what the pptx format can represent."""


def required_print_scale(width_mm: float, height_mm: float) -> int:
    """Smallest integer design-scale divisor that fits the pptx page cap."""
    import math

    return max(1, math.ceil(max(width_mm, height_mm) / MAX_PPTX_DIM_MM))


@dataclass
class FigurePlacement:
    """Resolved physical placement of a figure inside a panel."""

    width_mm: float
    height_mm: float


def resolve_font_path(family: str) -> Path:
    """Resolve a font family to a concrete font file via matplotlib."""
    props = font_manager.FontProperties(family=family)
    return Path(font_manager.findfont(props, fallback_to_default=True))


def _load_font(family: str, size_pt: float, scale: float) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(resolve_font_path(family)), size=max(1, round(size_pt * scale)))


def measure_text_height_mm(
    content: str,
    family: str,
    size_pt: float,
    width_mm: float,
    line_spacing: float = LINE_SPACING,
) -> float:
    """Measured height of wrapped text at a physical width, in mm.

    Wrapping replicates the renderer's layout: paragraphs split on
    newlines, greedy word wrap measured with the actual font file.
    """
    # Measure at 4x point size for sub-pixel precision, then scale back.
    scale = 4.0
    font = _load_font(family, size_pt, scale)
    px_per_mm = scale / MM_PER_PT  # rendered pixels per physical mm at this scale
    max_width_px = width_mm * px_per_mm
    total_lines = 0
    for paragraph in content.split("\n"):
        words = paragraph.split() or [""]
        line = ""
        for word in words:
            candidate = f"{line} {word}".strip()
            if line and font.getlength(candidate) > max_width_px:
                total_lines += 1
                line = word
            else:
                line = candidate
        total_lines += 1
    line_height_mm = size_pt * MM_PER_PT * line_spacing
    return total_lines * line_height_mm


def resolve_figure_placement(
    panel: Panel, element: FigureElement, asset_width_px: int, asset_height_px: int
) -> FigurePlacement:
    """Physical figure size inside a panel: full inner width, aspect kept,
    clamped by ``max_height_frac`` of the panel's inner height."""
    if panel.bbox is None:
        raise ValueError(f"panel '{panel.id}' has no bbox; run the layout solver first")
    inner_w = panel.bbox.w_mm - 2 * PANEL_PADDING_MM
    inner_h = panel.bbox.h_mm - 2 * PANEL_PADDING_MM
    aspect = asset_height_px / asset_width_px
    width = inner_w
    height = width * aspect
    max_h = inner_h * element.max_height_frac
    if height > max_h:
        height = max_h
        width = height / aspect
    return FigurePlacement(width_mm=width, height_mm=height)


def measure_panel_required_height_mm(panel: Panel, ir: PosterIR) -> float:
    """Total inner height the panel's content requires, in mm."""
    if panel.bbox is None:
        raise ValueError(f"panel '{panel.id}' has no bbox; run the layout solver first")
    inner_w = panel.bbox.w_mm - 2 * PANEL_PADDING_MM
    required = 0.0
    if panel.title and panel.role != "header":
        required += (
            measure_text_height_mm(
                panel.title, ir.style.font_heading, ir.style.type_scale_pt["heading"], inner_w
            )
            + ELEMENT_GAP_MM
        )
    for element in panel.elements:
        required += _element_height_mm(panel, element, ir, inner_w) + ELEMENT_GAP_MM
    if required > 0:
        required -= ELEMENT_GAP_MM  # no gap after the last element
    return required * MEASURE_SAFETY


def _element_height_mm(panel: Panel, element: PosterElement, ir: PosterIR, inner_w: float) -> float:
    if isinstance(element, TextElement):
        family = (
            ir.style.font_heading if element.level in ("title", "heading") else ir.style.font_body
        )
        return measure_text_height_mm(
            element.content, family, ir.style.type_scale_pt[element.level], inner_w
        )
    if isinstance(element, FigureElement):
        asset = ir.assets[element.asset_id]
        placement = resolve_figure_placement(panel, element, asset.width_px, asset.height_px)
        height = placement.height_mm
        if element.caption:
            height += ELEMENT_GAP_MM / 2 + measure_text_height_mm(
                element.caption, ir.style.font_body, ir.style.type_scale_pt["caption"], inner_w
            )
        return height
    if isinstance(element, QRElement):
        height = QR_SIZE_MM
        if element.label:
            height += ELEMENT_GAP_MM / 2 + measure_text_height_mm(
                element.label, ir.style.font_body, ir.style.type_scale_pt["caption"], inner_w
            )
        return height
    raise TypeError(f"unknown element kind: {element!r}")


def check_overflow(ir: PosterIR) -> None:
    """Raise :class:`TextOverflowError` for the first panel that cannot fit."""
    for panel in ir.panels_in_order():
        if panel.bbox is None:
            raise ValueError(f"panel '{panel.id}' has no bbox; run the layout solver first")
        available = panel.bbox.h_mm - 2 * PANEL_PADDING_MM
        required = measure_panel_required_height_mm(panel, ir)
        if required > available:
            raise TextOverflowError(panel.id, required, available)


def render_pptx(ir: PosterIR, out_path: Path, workdir: Path) -> Path:
    """Render the IR to an editable .pptx at physical size.

    Args:
        ir: Fully-placed poster IR (all panels have bboxes).
        out_path: Destination .pptx path.
        workdir: Directory for generated raster assets (QR codes).

    Raises:
        PageSizeError: If the page exceeds the pptx 56-inch dimension cap.
        TextOverflowError: If any panel's content does not fit.
    """
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    from pptx.util import Emu, Pt

    scale = ir.print_scale
    if ir.size.width_mm / scale > MAX_PPTX_DIM_MM or ir.size.height_mm / scale > MAX_PPTX_DIM_MM:
        needed = required_print_scale(ir.size.width_mm, ir.size.height_mm)
        raise PageSizeError(
            f"page {ir.size.width_mm:.0f}x{ir.size.height_mm:.0f}mm at print_scale="
            f"{scale} exceeds the pptx maximum dimension of {MAX_PPTX_DIM_MM:.0f}mm "
            f"(56in); set print_scale={needed} (design at 1/{needed} size, "
            f"print at {needed * 100}%)"
        )
    if not ir.is_fully_placed():
        raise ValueError("IR is not fully placed; run the layout solver first")
    check_overflow(ir)

    def emu(mm: float):
        """Physical mm -> design-page EMU (divided by print_scale)."""
        return Emu(mm_to_emu(mm / scale))

    prs = Presentation()
    prs.slide_width = emu(ir.size.width_mm)
    prs.slide_height = emu(ir.size.height_mm)
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank layout

    def color(token: str) -> RGBColor:
        return RGBColor.from_string(ir.style.palette[token].lstrip("#").upper())

    # Poster background.
    bg = slide.shapes.add_shape(
        1, Emu(0), Emu(0), prs.slide_width, prs.slide_height
    )  # 1 = MSO_SHAPE.RECTANGLE
    bg.fill.solid()
    bg.fill.fore_color.rgb = color("background")
    bg.line.fill.background()
    bg.shadow.inherit = False

    workdir.mkdir(parents=True, exist_ok=True)

    for panel in ir.panels_in_order():
        box = panel.bbox
        assert box is not None
        is_header = panel.role == "header"
        panel_bg = "primary" if is_header else "panel_bg"
        text_token = "background" if is_header else "text"

        shape = slide.shapes.add_shape(
            1,
            emu(box.x_mm),
            emu(box.y_mm),
            emu(box.w_mm),
            emu(box.h_mm),
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = color(panel_bg)
        shape.line.color.rgb = color("primary")
        shape.line.width = Pt(1.5 / scale)
        shape.shadow.inherit = False

        inner_x = box.x_mm + PANEL_PADDING_MM
        inner_w = box.w_mm - 2 * PANEL_PADDING_MM

        # Distribute panel slack as bounded extra spacing between elements,
        # centering the remainder — content reads as deliberately composed
        # instead of pooling at the top of a tall box.
        required = measure_panel_required_height_mm(panel, ir)
        available = box.h_mm - 2 * PANEL_PADDING_MM
        slack = max(0.0, available - required)
        n_slots = len(panel.elements) + (1 if panel.title and not is_header else 0)
        if n_slots > 1:
            extra_gap = min(slack / (n_slots - 1), 2.5 * ELEMENT_GAP_MM)
        else:
            extra_gap = 0.0
        top_offset = (slack - extra_gap * max(n_slots - 1, 0)) / 2
        gap_mm = ELEMENT_GAP_MM + extra_gap
        cursor_y = box.y_mm + PANEL_PADDING_MM + top_offset

        def add_text(
            content: str,
            level: str,
            *,
            align_center: bool = False,
            text_color: str = text_token,
            x_mm: float = inner_x,
            w_mm: float = inner_w,
            y_mm: float | None = None,
        ) -> float:
            nonlocal cursor_y
            size_pt = ir.style.type_scale_pt[level]
            family = ir.style.font_heading if level in ("title", "heading") else ir.style.font_body
            height_mm = measure_text_height_mm(content, family, size_pt, w_mm) * MEASURE_SAFETY
            top = cursor_y if y_mm is None else y_mm
            tb = slide.shapes.add_textbox(
                emu(x_mm),
                emu(top),
                emu(w_mm),
                emu(height_mm),
            )
            tf = tb.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
            lines = content.split("\n")
            for i, line in enumerate(lines):
                para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                is_bullet = line.startswith("- ")
                para.text = ("•  " + line[2:]) if is_bullet else line
                para.alignment = PP_ALIGN.CENTER if align_center else PP_ALIGN.LEFT
                para.line_spacing = LINE_SPACING
                for run in para.runs:
                    run.font.size = Pt(size_pt / scale)
                    run.font.name = family
                    run.font.color.rgb = color(text_color)
                    run.font.bold = level in ("title", "heading")
            if y_mm is None:
                cursor_y = top + height_mm + gap_mm
            return height_mm

        if panel.title and not is_header:
            add_text(panel.title, "heading", text_color="primary")

        for element in panel.elements:
            if isinstance(element, TextElement):
                add_text(element.content, element.level, align_center=is_header)
            elif isinstance(element, FigureElement):
                asset = ir.assets[element.asset_id]
                placement = resolve_figure_placement(
                    panel, element, asset.width_px, asset.height_px
                )
                pic_x = inner_x + (inner_w - placement.width_mm) / 2
                slide.shapes.add_picture(
                    asset.path,
                    emu(pic_x),
                    emu(cursor_y),
                    emu(placement.width_mm),
                    emu(placement.height_mm),
                )
                cursor_y += placement.height_mm + gap_mm / 2
                if element.caption:
                    add_text(element.caption, "caption", align_center=True)
                else:
                    cursor_y += gap_mm / 2
            elif isinstance(element, QRElement):
                qr_path = _render_qr(element.url, workdir)
                qr_x = inner_x + (inner_w - QR_SIZE_MM) / 2
                slide.shapes.add_picture(
                    str(qr_path),
                    emu(qr_x),
                    emu(cursor_y),
                    emu(QR_SIZE_MM),
                    emu(QR_SIZE_MM),
                )
                cursor_y += QR_SIZE_MM + gap_mm / 2
                if element.label:
                    add_text(element.label, "caption", align_center=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_path))
    logger.info("Rendered poster pptx", path=str(out_path), panels=len(ir.panels))
    return out_path


def _render_qr(url: str, workdir: Path) -> Path:
    import hashlib

    import qrcode

    slug = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    path = workdir / f"qr_{slug}.png"
    img = qrcode.make(url)
    img = img.resize((1200, 1200), Image.NEAREST)
    img.save(path)
    return path
