"""Style guidelines for statistical plots."""

from __future__ import annotations

from pathlib import Path

import structlog

logger = structlog.get_logger()

DEFAULT_PLOT_GUIDELINES = """\
# NeurIPS 2025 Statistical Plot Aesthetics Guide

## 1. The "NeurIPS Look"

The prevailing aesthetic for 2025 is defined by precision, accessibility,
and high contrast. The "default" academic look has shifted away from
bare-bones styling toward a more graphic, publication-ready presentation.

- Vibe: Professional, clean, and information-dense.
- Backgrounds: Heavy bias toward stark white backgrounds for maximum contrast.
  "Seaborn-style" light grey background remains an accepted variant.
- Accessibility: Strong emphasis on distinguishing data not just by color,
  but by texture (patterns) and shape (markers).

## 2. Detailed Style Options

### Color Palettes

Categorical Data:
- Soft Pastels: Matte, low-saturation colors (salmon, sky blue, mint, lavender).
- Muted Earth Tones: Olive, beige, slate grey, and navy.
- High-Contrast Primaries: Used sparingly.
- Accessibility Mode: Combine color with geometric patterns (hatches, dots, stripes).

Sequential & Heatmaps:
- Perceptually Uniform: "Viridis" and "Magma/Plasma" are the standard.
- Diverging: "Coolwarm" for positive/negative splits.
- Avoid: Traditional "Jet/Rainbow" scale.

### Axes & Grids

- Grid lines: Fine dashed or dotted in light gray, rendered behind data.
- Spines: "Boxed" (all 4 sides) or "Open" (remove top and right).
- Ticks: Subtle, facing inward, or removed in favor of grid alignment.

### Layout & Typography

- Font: Exclusively Sans-Serif. Rotate x-labels 45 degrees only if needed.
- Legends: Float inside plot area or place horizontally above.
- Annotations: Direct labeling preferred over legend references.

## 3. Type-Specific Guidelines

### Bar Charts & Histograms
- Borders: Black outlines (high-contrast) or borderless (solid fills).
- Grouping: Bars grouped tightly, whitespace between categories.
- Error Bars: Black, flat caps.

### Line Charts
- Markers: Always include geometric markers at data points.
- Line Styles: Solid for primary, dashed for baselines/secondary.
- Uncertainty: Semi-transparent shaded bands for confidence intervals.

### Tree & Pie/Donut Charts
- Thick white borders to separate slices.
- Donut charts preferred over traditional pie charts.
- "Exploding" a slice to highlight key statistics.

### Scatter Plots
- Shape Coding: Different marker shapes for categorical dimensions.
- Fills: Solid and fully opaque markers.
- 3D: Walls with grids, drop-lines to floor.

### Heatmaps
- Cells: Strictly square aspect ratio.
- Annotation: Exact values inside cells (white or black text).
- Borders: Borderless or very thin white lines.

### Radar Charts
- Translucent polygon fills (alpha ~0.2).
- Solid darker perimeter line.

### Miscellaneous
- Dot Plots: "Lollipop" style (dots with thin connecting line to axis).

## 4. Common Pitfalls

- The "Excel Default" Look: 3D effects, shadows, serif fonts.
- The "Rainbow" Map: Jet/Rainbow is outdated and perceptually misleading.
- Ambiguous Lines: Always add markers to sparse line charts.
- Over-reliance on Color: Use patterns/shapes for accessibility.
- Cluttered Grids: Use light grey/dashed, never solid black.
"""


def load_plot_guidelines(guidelines_path: str | None = None) -> str:
    """Load statistical plot style guidelines.

    Args:
        guidelines_path: Path to custom guidelines file. If None, uses defaults.

    Returns:
        Guidelines text.
    """
    if guidelines_path:
        path = Path(guidelines_path) / "plot_style_guide.md"
        if path.exists():
            logger.info("Loading custom plot guidelines", path=str(path))
            return path.read_text(encoding="utf-8")

    return DEFAULT_PLOT_GUIDELINES
