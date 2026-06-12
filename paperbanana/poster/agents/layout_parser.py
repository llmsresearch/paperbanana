"""VLM layout parser: any poster image -> a layout skeleton for memory."""

from __future__ import annotations

from typing import Any, Optional

import structlog
from PIL import Image
from pydantic import BaseModel, Field

from paperbanana.agents.base import BaseAgent
from paperbanana.core.utils import extract_json
from paperbanana.poster.memory import LayoutSkeleton

logger = structlog.get_logger()


class ParsedPoster(BaseModel):
    skeleton: LayoutSkeleton
    n_figures: int = Field(ge=0)
    visual_share: float = Field(ge=0, le=1)
    venue: Optional[str] = None


class LayoutParserAgent(BaseAgent):
    """Parses a poster image into its band/column/span structure.

    This is how the system learns from ANY sample: ingest a poster you
    admire and its structure joins the exemplar memory.
    """

    @property
    def agent_name(self) -> str:
        return "layout_parser"

    async def run(
        self, poster_image: Image.Image, source_name: str = "", **kwargs: Any
    ) -> ParsedPoster:
        template = self.load_prompt("poster")
        prompt = self.format_prompt(template, prompt_label=source_name or "poster")
        raw = await self.vlm.generate(
            prompt=prompt, images=[poster_image], response_format="json", temperature=0.2
        )
        data = extract_json(raw)
        if not isinstance(data, dict):
            raise ValueError(
                f"layout parser returned no JSON object for '{source_name}': {raw[:400]!r}"
            )
        parsed = ParsedPoster(**data)
        logger.info(
            "Poster parsed",
            source=source_name,
            bands=[(b.kind, b.columns) for b in parsed.skeleton.bands],
            panels=len(parsed.skeleton.panels),
        )
        return parsed
