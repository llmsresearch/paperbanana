"""Composite figure generation: stitch multiple images into a labeled grid."""

from __future__ import annotations

import string
from pathlib import Path
from typing import Any, Literal, Optional

import structlog
from PIL import Image, ImageDraw, ImageFont

logger = structlog.get_logger()

# Default settings
DEFAULT_SPACING = 20
DEFAULT_LABEL_FONT_SIZE = 32
DEFAULT_BG_COLOR = (255, 255, 255)
DEFAULT_LABEL_COLOR = (0, 0, 0)


def _auto_labels(count: int) -> list[str]:
    """Generate (a), (b), (c), ... labels."""
    return [f"({c})" for c in string.ascii_lowercase[:count]]


def _parse_layout(layout: str, image_count: int) -> tuple[int, int]:
    """Parse a layout string like '2x3' into (rows, cols).

    Also accepts 'auto' which picks a reasonable grid for the image count.
    """
    if layout.lower() == "auto":
        if image_count <= 3:
            return 1, image_count
        if image_count <= 4:
            return 2, 2
        if image_count <= 6:
            return 2, 3
        if image_count <= 9:
            return 3, 3
        cols = 4
        rows = (image_count + cols - 1) // cols
        return rows, cols

    parts = layout.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Layout must be 'RxC' (e.g. '2x3') or 'auto'. Got: {layout!r}")
    try:
        rows, cols = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(f"Layout must be 'RxC' with integers. Got: {layout!r}")
    if rows < 1 or cols < 1:
        raise ValueError(f"Layout rows and cols must be >= 1. Got: {rows}x{cols}")
    if rows * cols < image_count:
        raise ValueError(
            f"Layout {rows}x{cols} ({rows * cols} cells) cannot fit {image_count} images"
        )
    return rows, cols


def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to load a TrueType font, fall back to default."""
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("Arial Bold.ttf", size)
        except (OSError, IOError):
            return ImageFont.load_default()


def compose_images(
    image_paths: list[str | Path],
    *,
    layout: str = "auto",
    labels: Optional[list[str]] = None,
    auto_label: bool = True,
    spacing: int = DEFAULT_SPACING,
    label_position: Literal["top", "bottom"] = "bottom",
    label_font_size: int = DEFAULT_LABEL_FONT_SIZE,
    bg_color: tuple[int, int, int] = DEFAULT_BG_COLOR,
    label_color: tuple[int, int, int] = DEFAULT_LABEL_COLOR,
    output_path: Optional[str | Path] = None,
) -> Image.Image:
    """Compose multiple images into a single labeled grid.

    Args:
        image_paths: Paths to input images.
        layout: Grid layout as 'RxC' (e.g. '1x3', '2x2') or 'auto'.
        labels: Explicit labels per panel. Overrides auto_label.
        auto_label: If True and labels is None, generate (a), (b), (c), ...
        spacing: Pixel spacing between panels and around edges.
        label_position: Place labels 'top' or 'bottom' of each panel.
        label_font_size: Font size for labels.
        bg_color: Background color (RGB).
        label_color: Label text color (RGB).
        output_path: If provided, save the composite image to this path.

    Returns:
        The composite PIL Image.
    """
    if not image_paths:
        raise ValueError("At least one image path is required")

    # Load images
    images: list[Image.Image] = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        images.append(img)

    count = len(images)
    rows, cols = _parse_layout(layout, count)

    # Resolve labels
    panel_labels: list[str] | None = None
    if labels is not None:
        if len(labels) != count:
            raise ValueError(f"Expected {count} labels, got {len(labels)}")
        panel_labels = labels
    elif auto_label:
        panel_labels = _auto_labels(count)

    # Calculate label height
    label_height = 0
    font = _get_font(label_font_size)
    if panel_labels:
        label_height = label_font_size + 8  # text height + padding

    # Resize panels: equal height per row, preserving aspect ratio
    # First pass: determine target cell size
    # Scale all images to have the same height, then figure out column widths
    target_row_height = min(img.size[1] for img in images)
    # Cap at a reasonable maximum
    target_row_height = min(target_row_height, 1200)

    scaled: list[Image.Image] = []
    for img in images:
        w, h = img.size
        if h != target_row_height:
            scale = target_row_height / h
            new_w = max(1, round(w * scale))
            img = img.resize((new_w, target_row_height), Image.LANCZOS)
        scaled.append(img)

    # Determine column widths: max width in each column
    col_widths = [0] * cols
    for i, img in enumerate(scaled):
        col = i % cols
        col_widths[col] = max(col_widths[col], img.size[0])

    # Build the composite
    cell_height = target_row_height + label_height
    total_width = sum(col_widths) + spacing * (cols + 1)
    total_height = cell_height * rows + spacing * (rows + 1)

    composite = Image.new("RGB", (total_width, total_height), bg_color)
    draw = ImageDraw.Draw(composite)

    for i, img in enumerate(scaled):
        row = i // cols
        col = i % cols

        # Calculate position: center image within its cell column
        x_offset = spacing + sum(col_widths[:col]) + spacing * col
        x_center_offset = (col_widths[col] - img.size[0]) // 2
        y_offset = spacing + row * (cell_height + spacing)

        if label_position == "top" and panel_labels:
            img_y = y_offset + label_height
        else:
            img_y = y_offset

        composite.paste(img, (x_offset + x_center_offset, img_y))

        # Draw label
        if panel_labels and i < len(panel_labels):
            label_text = panel_labels[i]
            bbox = draw.textbbox((0, 0), label_text, font=font)
            text_w = bbox[2] - bbox[0]
            label_x = x_offset + (col_widths[col] - text_w) // 2

            if label_position == "top":
                label_y = y_offset + 2
            else:
                label_y = img_y + img.size[1] + 2

            draw.text((label_x, label_y), label_text, fill=label_color, font=font)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        composite.save(str(output_path))
        logger.info("Composite image saved", path=str(output_path), panels=count, layout=layout)

    return composite


def parse_composite_config(manifest_data: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Extract and validate the optional 'composite' section from a batch manifest.

    Returns None if no composite section is present.
    """
    composite = manifest_data.get("composite")
    if composite is None:
        return None
    if not isinstance(composite, dict):
        raise ValueError("'composite' must be a mapping")

    config: dict[str, Any] = {}

    layout = composite.get("layout", "auto")
    if not isinstance(layout, str):
        raise ValueError("composite.layout must be a string (e.g. '2x3' or 'auto')")
    config["layout"] = layout

    labels = composite.get("labels", "auto")
    if isinstance(labels, str) and labels.lower() == "auto":
        config["auto_label"] = True
        config["labels"] = None
    elif isinstance(labels, list):
        config["auto_label"] = False
        config["labels"] = [str(item) for item in labels]
    elif labels is None or labels is False:
        config["auto_label"] = False
        config["labels"] = None
    else:
        raise ValueError("composite.labels must be 'auto', a list of strings, or null")

    spacing = composite.get("spacing", DEFAULT_SPACING)
    if not isinstance(spacing, int) or spacing < 0:
        raise ValueError("composite.spacing must be a non-negative integer")
    config["spacing"] = spacing

    label_position = composite.get("label_position", "bottom")
    if label_position not in ("top", "bottom"):
        raise ValueError("composite.label_position must be 'top' or 'bottom'")
    config["label_position"] = label_position

    config["output"] = composite.get("output")

    return config
