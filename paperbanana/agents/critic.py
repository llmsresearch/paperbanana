"""Critic Agent: Evaluates generated images and provides revision feedback."""

from __future__ import annotations

import json
import re

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
        """Parse the VLM response into a CritiqueResult.

        Uses a multi-layer fallback strategy:
        1. Standard json.loads
        2. json-repair library (handles unterminated strings, trailing commas, etc.)
        3. Regex extraction of key fields
        4. Conservative default (no revision)
        """
        data = self._try_parse_json(response)

        if data is not None:
            suggestions = data.get("critic_suggestions", [])
            # Normalize: sometimes returned as a single string instead of list
            if isinstance(suggestions, str):
                suggestions = [s.strip() for s in suggestions.split(";") if s.strip()]
            revised = data.get("revised_description")
            # "No changes needed." means no revision
            if revised and "no changes needed" in revised.lower():
                revised = None
            return CritiqueResult(
                critic_suggestions=suggestions,
                revised_description=revised,
            )

        # Layer 3: Regex extraction
        logger.warning("All JSON parsers failed, attempting regex extraction")
        return self._regex_extract(response)

    def _try_parse_json(self, response: str) -> dict | None:
        """Try parsing JSON with standard lib, then json-repair."""
        # Layer 1: Standard json.loads
        try:
            data = json.loads(response)
            if isinstance(data, list) and len(data) > 0:
                data = data[0]
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, KeyError, AttributeError):
            pass

        # Layer 2: json-repair
        try:
            from json_repair import repair_json

            repaired = repair_json(response, return_objects=True)
            if isinstance(repaired, list) and len(repaired) > 0:
                repaired = repaired[0]
            if isinstance(repaired, dict):
                logger.debug("JSON repaired successfully by json-repair")
                return repaired
        except Exception as e:
            logger.debug("json-repair also failed", error=str(e))

        return None

    def _regex_extract(self, response: str) -> CritiqueResult:
        """Last-resort extraction using regex patterns."""
        suggestions = []
        revised = None

        # Try to find critic_suggestions content
        sugg_match = re.search(
            r'"critic_suggestions"\s*:\s*\[([^\]]*)',
            response,
            re.DOTALL,
        )
        if sugg_match:
            raw = sugg_match.group(1)
            suggestions = [
                s.strip().strip('"').strip("'")
                for s in raw.split(",")
                if s.strip().strip('"').strip("'")
            ]

        # Try to find revised_description
        rev_match = re.search(
            r'"revised_description"\s*:\s*"((?:[^"\\]|\\.)*)',
            response,
            re.DOTALL,
        )
        if rev_match:
            revised = rev_match.group(1)
            if "no changes needed" in revised.lower():
                revised = None

        if not suggestions and revised is None:
            logger.warning("Regex extraction found nothing, defaulting to no-revision")

        return CritiqueResult(
            critic_suggestions=suggestions,
            revised_description=revised,
        )
