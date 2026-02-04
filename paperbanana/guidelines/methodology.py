"""Style guidelines for methodology diagrams."""

from __future__ import annotations

from pathlib import Path

import structlog

logger = structlog.get_logger()

DEFAULT_METHODOLOGY_GUIDELINES = """\
# NeurIPS 2025 Method Diagram Aesthetics Guide

## 1. The "NeurIPS Look"

The prevailing aesthetic for 2025 is "Soft Tech & Scientific Pastels."
Gone are the days of harsh primary colors and sharp black boxes. The modern
NeurIPS diagram feels approachable yet precise. It utilizes high-value (light)
backgrounds to organize complexity, reserving saturation for the most critical
active elements. The vibe balances clean modularity (clear separation of parts)
with narrative flow (clear left-to-right progression).

## 2. Detailed Style Options

### A. Color Palettes

Design Philosophy: Use color to group logic, not just to decorate.
Avoid fully saturated backgrounds.

Background Fills (The "Zone" Strategy):
- Most papers use very light, desaturated pastels (Opacity ~10-15%).
- Aesthetically pleasing options: Cream/Beige (warm academic feel),
  Pale Blue/Ice (clean technical feel), Mint/Sage (soft organic feel),
  Pale Lavender (distinctive modern feel).
- Alternative (~20%): White backgrounds with colored dashed borders.

Functional Element Colors:
- For "Active" Modules: Medium saturation. Common pairings: Blue/Orange,
  Green/Purple, or Teal/Pink.
- Trainable Elements: Often Warm tones (Red, Orange, Deep Pink).
- Frozen/Static Elements: Often Cool tones (Grey, Ice Blue, Cyan).
- For Highlights/Results: High saturation reserved for "Error/Loss,"
  "Ground Truth," or final output.

### B. Shapes & Containers

Design Philosophy: "Softened Geometry." Sharp corners for data; rounded
corners for processes.

- Process Nodes: Rounded Rectangles (dominant ~80%).
- Tensors & Data: 3D Stacks/Cuboids for volume, Flat Squares/Grids
  for matrices/tokens.
- Cylinders: Reserved for Databases, Buffers, or Memory.
- Grouping: Solid light-colored containers with "Macro-Micro" pattern.
- Borders: Solid for physical components, Dashed for logical stages.

### C. Lines & Arrows

- Orthogonal/Elbow: For Network Architectures.
- Curved/Bezier: For System Logic, Feedback Loops.
- Solid Black/Grey: Standard data flow.
- Dashed Lines: Auxiliary flow (gradients, skip connections).
- Integrated Math: Operators placed directly on lines.

### D. Typography & Icons

- Labels: Sans-Serif (Arial, Roboto, Helvetica). Bold for headers.
- Variables: Serif, Italicized (LaTeX style).
- Icons: Trainable (Fire, Lightning), Frozen (Snowflake, Padlock),
  Operations (Gear, Magnifying Glass), Content (Document, Chat Bubble).

### E. Layout & Composition

- Flow: Left-to-right for sequential, top-to-bottom for hierarchical.
- Alignment: Snap to implicit grid.
- Spacing: Consistent gaps, closer within groups.
- Balance: Distribute visual weight evenly.
- Whitespace: Intentional separation of phases/concepts.

## 3. Common Pitfalls

- The "PowerPoint Default" Look with heavy black outlines.
- Font Mixing (Times New Roman for labels).
- Inconsistent Dimension mixing (2D and 3D without reason).
- Primary saturated backgrounds for grouping.
- Ambiguous arrows (same style for data flow and gradient flow).

## 4. Domain-Specific Styles

- AGENT/LLM Papers: Illustrative, narrative, cartoony. Chat bubbles,
  robots, document icons.
- COMPUTER VISION Papers: Spatial, dense, geometric. Frustums, rays,
  point clouds, RGB color coding.
- THEORETICAL Papers: Minimalist, abstract. Graph nodes, manifolds,
  grayscale with one highlight color.
- GENERATIVE/LEARNING Papers: Dynamic, process-oriented. Noise/denoising
  metaphors, gradual color transitions.
"""


def load_methodology_guidelines(guidelines_path: str | None = None) -> str:
    """Load methodology diagram style guidelines.

    Args:
        guidelines_path: Path to custom guidelines file. If None, uses defaults.

    Returns:
        Guidelines text.
    """
    if guidelines_path:
        path = Path(guidelines_path) / "methodology_style_guide.md"
        if path.exists():
            logger.info("Loading custom methodology guidelines", path=str(path))
            return path.read_text(encoding="utf-8")

    return DEFAULT_METHODOLOGY_GUIDELINES
