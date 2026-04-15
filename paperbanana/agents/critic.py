"""Critic Agent: Evaluates generated images and provides revision feedback."""

from __future__ import annotations

import re
from typing import Optional

import structlog

from paperbanana.agents.base import BaseAgent
from paperbanana.core.types import CritiqueResult, DiagramType
from paperbanana.core.utils import extract_json, load_image
from paperbanana.providers.base import VLMProvider

logger = structlog.get_logger()


class CriticAgent(BaseAgent):
    """Evaluates generated diagrams and provides specific revision feedback.

    Compares the generated image against the source context to identify
    faithfulness, conciseness, readability, and aesthetic issues.
    """

    def __init__(
        self, vlm_provider: VLMProvider, prompt_dir: str = "prompts", prompt_recorder=None
    ):
        super().__init__(vlm_provider, prompt_dir, prompt_recorder=prompt_recorder)

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
        user_feedback: Optional[str] = None,
    ) -> CritiqueResult:
        """Evaluate a generated image and provide revision feedback.

        Args:
            image_path: Path to the generated image.
            description: The description used to generate the image.
            source_context: Original methodology text.
            caption: Figure caption / communicative intent.
            diagram_type: Type of diagram.
            user_feedback: Optional user comments for the critic to consider.

        Returns:
            CritiqueResult with evaluation and optional revised description.
        """
        # Load the image
        image = load_image(image_path)

        prompt_type = "diagram" if diagram_type == DiagramType.METHODOLOGY else "plot"
        template = self.load_prompt(prompt_type)
        prompt_label = self._prompt_label_from_image_path(image_path) or "critic"
        # Build prompt manually so we record once after appending user_feedback.
        prompt = template.format(
            source_context=source_context,
            caption=caption,
            description=description,
        )

        if user_feedback:
            prompt += (
                f"\n\nAdditional user feedback to consider in your evaluation:\n{user_feedback}"
            )

        # Record the exact prompt sent to the model (including user_feedback in continue-run flows)
        if self._prompt_recorder is not None:
            try:
                self._prompt_recorder.record(
                    agent_name=self.agent_name,
                    label=prompt_label,
                    prompt=prompt,
                )
            except Exception:
                logger.warning("Prompt recording failed", agent=self.agent_name, label=prompt_label)

        json_ok = getattr(self.vlm, "supports_json_mode", True)
        logger.info("Running critic agent", image_path=image_path, json_mode=json_ok)
        response = await self.vlm.generate(
            prompt=prompt,
            images=[image],
            temperature=0.3,
            max_tokens=4096,
            response_format="json" if json_ok else None,
        )
        critique = self._parse_response(response)
        logger.info(
            "Critic evaluation complete",
            needs_revision=critique.needs_revision,
            summary=critique.summary,
        )
        return critique

    @staticmethod
    def _prompt_label_from_image_path(image_path: str) -> str | None:
        m = re.search(r"(?:diagram|plot)_iter_(\d+)\.", image_path)
        return f"critic_iter_{m.group(1)}" if m else None

    def _parse_response(self, response: str) -> CritiqueResult:
        """Parse VLM response into a CritiqueResult."""
        data = extract_json(response)
        if isinstance(data, dict):
            try:
                return CritiqueResult(
                    critic_suggestions=data.get("critic_suggestions", []),
                    revised_description=data.get("revised_description"),
                )
            except (KeyError, TypeError) as e:
                logger.warning("Failed to build CritiqueResult", error=str(e))
        logger.warning("Failed to parse critic response as JSON")
        return CritiqueResult(critic_suggestions=[], revised_description=None)
