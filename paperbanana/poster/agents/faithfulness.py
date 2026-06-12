"""Faithfulness gate for re-authored figures."""

from __future__ import annotations

from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, Field

from paperbanana.agents.base import BaseAgent
from paperbanana.core.utils import extract_json


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
        raw = await self.vlm.generate(
            prompt=prompt,
            images=[original, reauthored],
            response_format="json",
            temperature=0.1,
        )
        data = extract_json(raw)
        if not isinstance(data, dict):
            raise ValueError(f"faithfulness agent returned no JSON object: {raw[:400]!r}")
        return FaithfulnessVerdict(**data)
