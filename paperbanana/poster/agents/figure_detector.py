"""VLM figure/table detection on rendered PDF pages."""

from __future__ import annotations

from typing import Any, Literal, Optional

import structlog
from PIL import Image
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from paperbanana.agents.base import BaseAgent
from paperbanana.core.utils import extract_json

logger = structlog.get_logger()


class DetectedRegion(BaseModel):
    """One figure/table region detected on a page render."""

    kind: Literal["figure", "table"]
    figure_number: Optional[str] = Field(
        default=None, description='e.g. "Figure 3" or "Table 1"; null if unnumbered'
    )
    caption: str = ""
    bbox: list[int] = Field(description="[x0, y0, x1, y1] in 0..1000 image coordinates")

    def bbox_norm(self) -> list[float]:
        x0, y0, x1, y1 = self.bbox
        clamp = lambda v: max(0.0, min(1.0, v / 1000.0))  # noqa: E731
        nx0, ny0, nx1, ny1 = clamp(x0), clamp(y0), clamp(x1), clamp(y1)
        if nx1 <= nx0 or ny1 <= ny0:
            raise ValueError(f"degenerate detection bbox: {self.bbox}")
        return [nx0, ny0, nx1, ny1]


_REGIONS_ADAPTER: TypeAdapter[list[DetectedRegion]] = TypeAdapter(list[DetectedRegion])


class FigureDetectorAgent(BaseAgent):
    """Detects figure/table regions and captions on a rendered page."""

    @property
    def agent_name(self) -> str:
        return "figure_detector"

    async def run(
        self, page_image: Image.Image, page_number: int, **kwargs: Any
    ) -> list[DetectedRegion]:
        template = self.load_prompt("poster")
        prompt = self.format_prompt(
            template,
            page_number=page_number,
            prompt_label=f"page_{page_number}",
        )
        regions = None
        raw, problem = "", ""
        # Some VLMs degenerate into repeating hallucinated regions until the
        # output is cut mid-array, quasi-deterministically at low temperature
        # — so the bounded retries climb the temperature to break the loop.
        for attempt, temperature in enumerate((0.2, 0.5, 0.8)):
            raw = await self.vlm.generate(
                prompt=prompt,
                images=[page_image],
                response_format="json",
                temperature=temperature,
            )
            data = extract_json(raw)
            if isinstance(data, list):
                try:
                    regions = _REGIONS_ADAPTER.validate_python(data)
                    break
                except ValidationError as exc:
                    problem = f"regions failed validation: {str(exc)[:200]}"
            else:
                problem = "no JSON array in response"
            logger.warning(
                "Figure detection attempt garbled",
                page=page_number,
                attempt=attempt + 1,
                problem=problem,
                raw_preview=raw[:200],
            )
        if regions is None:
            raise ValueError(
                f"figure detector failed for page {page_number} after 3 attempts "
                f"({problem}): {raw[:400]!r}"
            )
        logger.info("Detected regions", page=page_number, count=len(regions))
        return regions
