"""Planner Agent: Generates detailed textual descriptions from source context."""

from __future__ import annotations

from pathlib import Path

import structlog

from paperbanana.agents.base import BaseAgent
from paperbanana.core.types import DiagramType, ReferenceExample
from paperbanana.core.utils import load_image
from PIL import Image
from paperbanana.providers.base import VLMProvider

logger = structlog.get_logger()


class PlannerAgent(BaseAgent):
    """Generates a comprehensive textual description for diagram generation.

    Uses in-context learning from retrieved reference examples (including
    their images) to create a detailed description that the Visualizer
    can render. Matches paper equation 4: P = VLM_plan(S, C, {(S_i, C_i, I_i)}).
    """

    def __init__(self, vlm_provider: VLMProvider, prompt_dir: str = "prompts"):
        super().__init__(vlm_provider, prompt_dir)

    @property
    def agent_name(self) -> str:
        return "planner"

    async def run(
        self,
        source_context: str,
        caption: str,
        examples: list[ReferenceExample],
        diagram_type: DiagramType = DiagramType.METHODOLOGY,
        input_images: list[Image.Image] | None = None,
    ) -> str:
        """Generate a detailed textual description of the target diagram.

        Args:
            source_context: Methodology text from the paper.
            caption: Communicative intent / figure caption.
            examples: Retrieved reference examples for in-context learning.
            diagram_type: Type of diagram being generated.
            input_images: Optional user-provided input images (sketches/charts).

        Returns:
            Detailed textual description for the Visualizer.
        """
        # Format input images info
        input_imgs = input_images or []
        num_inputs = len(input_imgs)
        input_images_text = ""
        if num_inputs > 0:
            input_images_text = f"\n**User Input Images**: [See images 1-{num_inputs} provided above]\n"

        # Format examples for in-context learning (start index after input images)
        examples_text = self._format_examples(examples, start_index=num_inputs)

        # Load reference images for visual in-context learning
        example_images = self._load_example_images(examples)
        
        # Combine images: User Inputs + Reference Examples
        all_images = input_imgs + example_images

        prompt_type = "diagram" if diagram_type == DiagramType.METHODOLOGY else "plot"
        template = self.load_prompt(prompt_type)
        prompt = self.format_prompt(
            template,
            source_context=source_context + input_images_text,
            caption=caption,
            examples=examples_text,
        )

        logger.info(
            "Running planner agent",
            num_examples=len(examples),
            num_images=len(example_images),
            context_length=len(source_context),
        )

        description = await self.vlm.generate(
            prompt=prompt,
            images=all_images if all_images else None,
            temperature=0.7,
            max_tokens=4096,
        )

        logger.info("Planner generated description", length=len(description))
        return description

    def _format_examples(self, examples: list[ReferenceExample], start_index: int = 0) -> str:
        """Format reference examples for the planner prompt.

        Args:
            examples: List of reference examples.
            start_index: Starting index for image references (after user inputs).
        """
        if not examples:
            return "(No reference examples available. Generate based on source context alone.)"

        lines = []
        img_index = start_index
        for i, ex in enumerate(examples, 1):
            has_image = self._has_valid_image(ex)
            image_ref = ""
            if has_image:
                img_index += 1
                image_ref = f"\n**Diagram**: [See reference image {img_index} above]"

            lines.append(
                f"### Example {i}\n"
                f"**Caption**: {ex.caption}\n"
                f"**Source Context**: {ex.source_context[:500]}"
                f"{image_ref}\n"
            )
        return "\n".join(lines)

    def _has_valid_image(self, example: ReferenceExample) -> bool:
        """Check if a reference example has a valid image file."""
        if not example.image_path:
            return False
        return Path(example.image_path).exists()

    def _load_example_images(self, examples: list[ReferenceExample]) -> list:
        """Load reference images from disk for in-context learning.

        Returns a list of PIL Image objects for examples that have valid images.
        """
        images = []
        for ex in examples:
            if not self._has_valid_image(ex):
                continue
            try:
                img = load_image(ex.image_path)
                images.append(img)
            except Exception as e:
                logger.warning(
                    "Failed to load reference image",
                    image_path=ex.image_path,
                    error=str(e),
                )
        return images
