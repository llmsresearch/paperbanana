"""Skeleton ranking: pick the best of N candidate structures."""

from __future__ import annotations

from typing import Any

import structlog
from PIL import Image
from pydantic import BaseModel, Field

from paperbanana.agents.base import BaseAgent
from paperbanana.core.utils import extract_json

logger = structlog.get_logger()


class SkeletonRanking(BaseModel):
    winner: int = Field(ge=0, description="Index of the best candidate")
    scores: dict[str, float] = Field(
        default_factory=dict, description="Candidate index -> score (1-5)"
    )
    rationale: str = ""


class SkeletonJudgeAgent(BaseAgent):
    """Ranks skeleton previews of candidate layouts in one call."""

    @property
    def agent_name(self) -> str:
        return "skeleton_judge"

    async def run(
        self,
        previews: list[Image.Image],
        storyboard_summary: str,
        penalties: dict[int, float],
        **kwargs: Any,
    ) -> SkeletonRanking:
        template = self.load_prompt("poster")
        listing = "\n".join(f"Image {i + 1}: candidate layout {i}." for i in range(len(previews)))
        penalty_note = (
            "\n".join(
                f"- candidate {i} OVERFLOWED the page by {p:.0f}mm (no image; score it lowest)"
                for i, p in penalties.items()
            )
            or "(all candidates fit)"
        )
        prompt = self.format_prompt(
            template,
            listing=listing,
            penalty_note=penalty_note,
            storyboard_summary=storyboard_summary,
            n=len(previews) + len(penalties),
        )
        raw = await self.vlm.generate(
            prompt=prompt, images=previews, response_format="json", temperature=0.2
        )
        data = extract_json(raw)
        if not isinstance(data, dict):
            raise ValueError(f"skeleton judge returned no JSON object: {raw[:400]!r}")
        ranking = SkeletonRanking(**data)
        logger.info("Skeletons ranked", winner=ranking.winner, scores=ranking.scores)
        return ranking
