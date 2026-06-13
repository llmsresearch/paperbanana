"""VLM layout parser: any poster image -> a layout skeleton for memory."""

from __future__ import annotations

from typing import Any, Optional

import structlog
from PIL import Image
from pydantic import BaseModel, Field, ValidationError

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
        parsed: Optional[ParsedPoster] = None
        raw, problem = "", ""
        for attempt, temperature in enumerate((0.2, 0.5, 0.8)):
            raw = await self.vlm.generate(
                prompt=prompt,
                images=[poster_image],
                response_format="json",
                temperature=temperature,
            )
            data = extract_json(raw)
            if isinstance(data, dict):
                try:
                    parsed = ParsedPoster(**_normalize_parser_payload(data))
                    break
                except ValidationError as exc:
                    problem = f"payload failed validation: {str(exc)[:200]}"
            else:
                problem = "no JSON object in response"
            logger.warning(
                "Layout parse attempt garbled",
                source=source_name,
                attempt=attempt + 1,
                problem=problem,
                raw_preview=raw[:200],
            )
        if parsed is None:
            raise ValueError(
                f"layout parser failed for '{source_name}' after 3 attempts "
                f"({problem}): {raw[:400]!r}"
            )
        logger.info(
            "Poster parsed",
            source=source_name,
            bands=[(b.kind, b.columns) for b in parsed.skeleton.bands],
            panels=len(parsed.skeleton.panels),
        )
        return parsed


def _normalize_parser_payload(data: dict) -> dict:
    """Accept the model's common shape drift around the ``skeleton`` nesting.

    The contract nests bands/panels under ``skeleton``; models frequently
    emit them at the top level instead. Lift them so the structure the
    model actually described is honored rather than discarded.
    """
    if "skeleton" not in data and ("bands" in data or "panels" in data):
        data = dict(data)
        data["skeleton"] = {
            "bands": data.pop("bands", []),
            "panels": data.pop("panels", []),
        }
    return data
