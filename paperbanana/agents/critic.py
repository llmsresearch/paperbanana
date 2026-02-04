"""Critic Agent: Evaluates generated images and provides revision feedback."""

from __future__ import annotations

import json

import structlog

from paperbanana.agents.base import BaseAgent
from paperbanana.core.types import CritiqueResult, DiagramType
from paperbanana.core.utils import load_image
from paperbanana.providers.base import VLMProvider

logger = structlog.get_logger()


class CriticAgent(BaseAgent):
    """Evaluates generated diagrams and provides specific revision feedback.

    Compares the generated image against the source context to identify
    faithfulness, conciseness, readability, and aesthetic issues.
    """

    def __init__(self, vlm_provider: VLMProvider, prompt_dir: str = "prompts"):
        super().__init__(vlm_provider, prompt_dir)

    @property
    def agent_name(self) -> str:
        return "critic"

    async def run(
        self,
        image_path: str,
        description: str,
        source_context: str,
        caption: str,
        diagram_type: DiagramType = DiagramType.METHODOLOGY,
    ) -> CritiqueResult:
        """Evaluate a generated image and provide revision feedback.

        Args:
            image_path: Path to the generated image.
            description: The description used to generate the image.
            source_context: Original methodology text.
            caption: Figure caption / communicative intent.
            diagram_type: Type of diagram.

        Returns:
            CritiqueResult with evaluation and optional revised description.
        """
        # Load the image
        image = load_image(image_path)

        prompt_type = "diagram" if diagram_type == DiagramType.METHODOLOGY else "plot"
        template = self.load_prompt(prompt_type)
        prompt = self.format_prompt(
            template,
            source_context=source_context,
            caption=caption,
            description=description,
        )

        logger.info("Running critic agent", image_path=image_path)

        response = await self.vlm.generate(
            prompt=prompt,
            images=[image],
            temperature=0.3,
            max_tokens=4096,
            response_format="json",
        )

        critique = self._parse_response(response)
        logger.info(
            "Critic evaluation complete",
            needs_revision=critique.needs_revision,
            summary=critique.summary,
        )
        return critique

    def _parse_response(self, response: str) -> CritiqueResult:
        """Parse the VLM response into a CritiqueResult."""
        try:
            data = json.loads(response)
            return CritiqueResult(
                critic_suggestions=data.get("critic_suggestions", []),
                revised_description=data.get("revised_description"),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to parse critic response", error=str(e))
            # Conservative fallback: empty suggestions means no revision needed
            return CritiqueResult(
                critic_suggestions=[],
                revised_description=None,
            )
