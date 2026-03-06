"""Style guidelines for presentation slides."""

from __future__ import annotations


DEFAULT_SLIDE_GUIDELINES = """\
# Presentation Slide Design Guidelines

## Layout Types

### Standard Layouts
- **title-hero**: Large centered title + subtitle. Use for covers and section breaks.
- **quote-callout**: Featured quote with attribution.
- **key-stat**: Single large number as focal point for metrics.
- **split-screen**: Half image, half text for feature highlights.
- **icon-grid**: Grid of icons with labels for feature lists.
- **two-columns**: Balanced column layout for paired information.
- **three-columns**: Triple content columns for categorization.
- **image-caption**: Full-bleed image + text overlay for storytelling.
- **bullet-list**: Structured bullet points for simple content.

### Infographic Layouts
- **linear-progression**: Sequential left-to-right flow for timelines, pipelines, steps.
- **binary-comparison**: A vs B side-by-side for before/after comparisons.
- **comparison-matrix**: Multi-factor grid for feature comparison.
- **hierarchical-layers**: Pyramid or stacked levels for priority visualization.
- **hub-spoke**: Central node with radiating items for concept maps.
- **bento-grid**: Varied-size tiles for overview dashboards.
- **funnel**: Narrowing stages for conversion processes.
- **dashboard**: Metrics with charts/numbers for KPIs.
- **venn-diagram**: Overlapping circles for relationships.
- **circular-flow**: Continuous cycle for recurring processes.
- **winding-roadmap**: Curved path with milestones for journeys.

## Quality Criteria

### Text Rendering
- All text must be clear, readable, and correctly spelled
- Title text: large, bold, immediately readable
- Body text: clear, legible, appropriate sizing
- Max 3-4 text elements per slide
- Font rendering must match the style aesthetic

### Visual Hierarchy
- Each slide conveys ONE clear message
- Most important element gets visual weight
- Generous margins and spacing (15%+ on each side)
- Consistent alignment for professionalism
- One clear focal point per slide

### Style Consistency
- Consistent color palette across all slides
- Same icon and illustration style throughout
- Same typography hierarchy
- Same layout grid system
- No slide numbers, page numbers, footers, headers, or logos

### Content Rules
- Headlines should be narrative (not labels)
- Avoid AI cliches: "dive into", "explore", "journey", "let's"
- Back covers should have meaningful content (CTA, takeaway), not just "Thank you"
- All statistics must cite sources
"""


def load_slide_guidelines() -> str:
    """Load slide design guidelines."""
    return DEFAULT_SLIDE_GUIDELINES
