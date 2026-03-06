"""Polish Agent: Applies style guidelines to refine existing images.

Two-step process:
1. Analyze image against a style guide → generate improvement suggestions
2. Regenerate image with suggestions applied

Ported from PaperBanana-official/agents/polish_agent.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import structlog
from PIL import Image

from paperbanana.agents.base import BaseAgent
from paperbanana.core.types import DiagramType, PolishResult
from paperbanana.core.utils import image_to_base64, load_image, save_image
from paperbanana.providers.base import ImageGenProvider, VLMProvider

logger = structlog.get_logger()

DIAGRAM_SUGGESTION_PROMPT = """\
You are a senior art director for NeurIPS 2025. Your task is to critique a diagram \
against a provided style guide.
Provide up to 10 concise, actionable improvement suggestions. Focus on aesthetics \
(color, layout, fonts, icons).
Directly list the suggestions. Do not use filler phrases like "Based on the style guide...".
If the diagram is substantially compliant, output "No changes needed".\
"""

PLOT_SUGGESTION_PROMPT = """\
You are a senior data visualization expert for NeurIPS 2025. Your task is to critique \
a plot against a provided style guide.
Provide up to 10 concise, actionable improvement suggestions. Focus on aesthetics \
(color, layout, fonts).
Directly list the suggestions. Do not use filler phrases like "Based on the style guide...".
If the plot is substantially compliant, output "No changes needed".\
"""

DIAGRAM_POLISH_PROMPT = """\
## ROLE
You are a professional diagram polishing expert for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
You are given an existing diagram image and a list of specific improvement suggestions. \
Your task is to generate a polished version of this diagram by applying these suggestions \
while preserving the semantic logic and structure of the original diagram.

## OUTPUT
Generate a polished diagram image that maintains the original content while applying \
the improvement suggestions.\
"""

PLOT_POLISH_PROMPT = """\
## ROLE
You are a professional plot polishing expert for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
You are given an existing statistical plot image and a list of specific improvement \
suggestions. Your task is to generate a polished version of this plot by applying these \
suggestions while preserving all the data and quantitative information.

**Important Instructions:**
1. **Preserve Data:** Do NOT alter any data points, values, or quantitative information.
2. **Apply Suggestions:** Enhance the visual aesthetics according to the provided suggestions.
3. **Maintain Accuracy:** Ensure all numerical values and relationships remain accurate.
4. **Professional Quality:** Ensure the output meets publication standards.

## OUTPUT
Generate a polished plot image that maintains the original data while applying \
the improvement suggestions.\
"""


class PolishAgent(BaseAgent):
    """Refines existing images by applying style guide suggestions.

    Step 1: VLM analyzes the image against a style guide.
    Step 2: Image gen model produces a polished version.
    """

    def __init__(
        self,
        vlm_provider: VLMProvider,
        image_gen: ImageGenProvider,
        prompt_dir: str = "prompts",
        output_dir: str = "outputs",
        style_guides_dir: Optional[str] = None,
    ):
        super().__init__(vlm_provider, prompt_dir)
        self.image_gen = image_gen
        self.output_dir = Path(output_dir)
        self.style_guides_dir = Path(style_guides_dir) if style_guides_dir else None

    @property
    def agent_name(self) -> str:
        return "polish"

    async def run(
        self,
        image: Image.Image | str,
        diagram_type: DiagramType = DiagramType.METHODOLOGY,
        style_guide: Optional[str] = None,
        style_guide_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> PolishResult:
        """Polish an existing image using style guide analysis.

        Args:
            image: PIL Image or path to the image to polish.
            diagram_type: Type of diagram (affects suggestion prompts).
            style_guide: Style guide text (if already loaded).
            style_guide_path: Path to style guide file (alternative to style_guide).
            output_path: Where to save the polished image.

        Returns:
            PolishResult with suggestions, polished image path, etc.
        """
        # Load image if path provided
        if isinstance(image, str):
            image = load_image(image)

        # Load style guide
        guide_text = self._load_style_guide(diagram_type, style_guide, style_guide_path)

        # Step 1: Generate suggestions
        logger.info("Polish step 1: generating suggestions", diagram_type=diagram_type.value)
        suggestions = await self._generate_suggestions(image, guide_text, diagram_type)

        no_changes = not suggestions or "No changes needed" in suggestions
        if no_changes:
            logger.info("Polish: no changes needed, returning original image")
            if output_path is None:
                output_path = str(self.output_dir / "polished.png")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            save_image(image, output_path)
            return PolishResult(
                suggestions="No changes needed",
                polished_image_path=output_path,
                changed=False,
            )

        # Step 2: Generate polished image
        logger.info("Polish step 2: generating polished image", suggestions_length=len(suggestions))
        polished_path = await self._polish_image(image, suggestions, diagram_type, output_path)

        return PolishResult(
            suggestions=suggestions,
            polished_image_path=polished_path,
            changed=True,
        )

    async def _generate_suggestions(
        self,
        image: Image.Image,
        style_guide: str,
        diagram_type: DiagramType,
    ) -> str:
        """Step 1: Analyze image against style guide and generate suggestions."""
        system_prompt = (
            DIAGRAM_SUGGESTION_PROMPT
            if diagram_type == DiagramType.METHODOLOGY
            else PLOT_SUGGESTION_PROMPT
        )

        user_prompt = (
            f"Here is the style guide:\n{style_guide}\n\n"
            "Please analyze the provided image against this style guide and list "
            "up to 10 specific improvement suggestions to make the image visually "
            "more appealing. If the image is already perfect, just say 'No changes needed'."
        )

        response = await self.vlm.generate(
            prompt=user_prompt,
            images=[image],
            system_prompt=system_prompt,
            temperature=1.0,
            max_tokens=4096,
        )
        return response

    async def _polish_image(
        self,
        original_image: Image.Image,
        suggestions: str,
        diagram_type: DiagramType,
        output_path: Optional[str],
    ) -> str:
        """Step 2: Generate a polished version of the image with suggestions applied."""
        polish_prompt = (
            DIAGRAM_POLISH_PROMPT
            if diagram_type == DiagramType.METHODOLOGY
            else PLOT_POLISH_PROMPT
        )

        prompt = (
            f"{polish_prompt}\n\n"
            f"Please polish this image based on the following suggestions:\n\n"
            f"{suggestions}\n\n"
            f"Polished Image:"
        )

        # Use image generation provider to create polished version
        polished_image = await self.image_gen.generate(
            prompt=prompt,
            width=getattr(self, '_width', 1792),
            height=getattr(self, '_height', 1024),
        )

        if output_path is None:
            output_path = str(self.output_dir / "polished.png")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        save_image(polished_image, output_path)
        logger.info("Polished image saved", path=output_path)
        return output_path

    def _load_style_guide(
        self,
        diagram_type: DiagramType,
        style_guide: Optional[str],
        style_guide_path: Optional[str],
    ) -> str:
        """Load the appropriate style guide text."""
        if style_guide:
            return style_guide

        if style_guide_path:
            return Path(style_guide_path).read_text(encoding="utf-8")

        # Try default paths
        if self.style_guides_dir:
            filename = (
                "neurips2025_diagram_style_guide.md"
                if diagram_type == DiagramType.METHODOLOGY
                else "neurips2025_plot_style_guide.md"
            )
            default_path = self.style_guides_dir / filename
            if default_path.exists():
                return default_path.read_text(encoding="utf-8")

        # Minimal fallback
        return (
            "Follow NeurIPS 2025 publication standards: clean layout, "
            "consistent color scheme, legible fonts, proper labels."
        )
