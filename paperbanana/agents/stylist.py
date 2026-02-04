"""Stylist Agent: Optimizes diagram descriptions for visual aesthetics."""

from __future__ import annotations

import structlog

from paperbanana.agents.base import BaseAgent
from paperbanana.core.types import DiagramType
from paperbanana.providers.base import VLMProvider

logger = structlog.get_logger()


class StylistAgent(BaseAgent):
    """Refines a textual description to optimize visual aesthetics.

    Takes the Planner's output and enhances it with style-specific
    guidelines while preserving the content.
    """

    def __init__(
        self,
        vlm_provider: VLMProvider,
        guidelines: str = "",
        prompt_dir: str = "prompts",
    ):
        super().__init__(vlm_provider, prompt_dir)
        self.guidelines = guidelines

    @property
    def agent_name(self) -> str:
        return "stylist"

    async def run(
        self,
        description: str,
        guidelines: str | None = None,
        source_context: str = "",
        caption: str = "",
        diagram_type: DiagramType = DiagramType.METHODOLOGY,
    ) -> str:
        """Refine a description for optimal visual aesthetics.

        Args:
            description: The Planner's textual description.
            guidelines: Optional style guidelines (overrides instance default).
            source_context: Original methodology text from the paper.
            caption: Figure caption / communicative intent.
            diagram_type: Type of diagram being generated.

        Returns:
            Stylistically optimized description.
        """
        style_guidelines = guidelines or self.guidelines
        if not style_guidelines:
            style_guidelines = self._default_guidelines()

        prompt_type = "diagram" if diagram_type == DiagramType.METHODOLOGY else "plot"
        template = self.load_prompt(prompt_type)
        prompt = self.format_prompt(
            template,
            description=description,
            guidelines=style_guidelines,
            source_context=source_context,
            caption=caption,
        )

        logger.info("Running stylist agent", description_length=len(description))

        optimized = await self.vlm.generate(
            prompt=prompt,
            temperature=0.5,
            max_tokens=4096,
        )

        logger.info("Stylist refined description", length=len(optimized))
        return optimized

    def _default_guidelines(self) -> str:
        """Return default aesthetic guidelines if none provided."""
        return """
## Academic Illustration Style Guidelines

### Color Philosophy
- Use soft, muted, pastel tones — never fully saturated primaries
- Limit to 3-5 primary hues per diagram
- Each distinct color should map to a distinct concept or phase
- Use darker shades of the same hue for borders (not black)
- Describe colors in natural language (e.g., "soft sky blue", "warm peach")
- NEVER use hex codes, pixel dimensions, or point sizes in the description

### Typography
- Clean sans-serif fonts for all labels
- Visual hierarchy through size and weight: larger bold for titles,
  medium bold for components, smaller for annotations
- All text must be clear, readable English

### Layout
- Consistent spacing between elements
- Clear flow direction (left-to-right or top-to-bottom)
- Balanced composition with visual weight evenly distributed
- Use whitespace intentionally to separate phases and groups

### Visual Elements
- Rounded rectangles with soft pastel fills for components
- Solid arrows with dark gray color for primary data flow
- Dashed arrows for optional or conditional connections
- Semi-transparent colored backgrounds for grouping regions
- No gradients, no 3D effects, no drop shadows, no decorative borders
"""
