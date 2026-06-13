"""Faithfulness gate for re-authored figures."""

from __future__ import annotations

from typing import Any, Literal

import structlog
from PIL import Image
from pydantic import BaseModel, Field

from paperbanana.agents.base import BaseAgent
from paperbanana.core.utils import extract_json

logger = structlog.get_logger()


class FaithfulnessVerdict(BaseModel):
    verdict: Literal["pass", "fail"]
    differences: list[str] = Field(default_factory=list)


class FaithfulnessAgent(BaseAgent):
    """Compares a re-authored figure against the original for data fidelity.

    This is the load-bearing control of the re-authoring feature: a
    restyled figure that changes data is worse than an ugly one.
    """

    @property
    def agent_name(self) -> str:
        return "faithfulness"

    async def run(
        self,
        original: Image.Image,
        reauthored: Image.Image,
        caption: str,
        **kwargs: Any,
    ) -> FaithfulnessVerdict:
        template = self.load_prompt("poster")
        prompt = self.format_prompt(template, caption=caption)
        # Some VLMs loop restating the same "difference" until the response
        # is cut mid-JSON, quasi-deterministically at low temperature — the
        # bounded retries climb the temperature to break the loop.
        raw = ""
        for attempt, temperature in enumerate((0.1, 0.4, 0.7)):
            raw = await self.vlm.generate(
                prompt=prompt,
                images=[original, reauthored],
                response_format="json",
                temperature=temperature,
            )
            data = extract_json(raw)
            if isinstance(data, dict):
                return FaithfulnessVerdict(**data)
            logger.warning(
                "Faithfulness verdict garbled",
                attempt=attempt + 1,
                raw_preview=raw[:200],
            )
        raise ValueError(
            f"faithfulness agent returned no JSON object after 3 attempts: {raw[:400]!r}"
        )
