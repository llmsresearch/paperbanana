"""Per-figure curation: reuse, re-author, or generate new."""

from __future__ import annotations

from typing import Any

from PIL import Image

from paperbanana.agents.base import BaseAgent
from paperbanana.core.utils import extract_json
from paperbanana.poster.types import FigureDecisionResult, PaperFigure


class FigureCuratorAgent(BaseAgent):
    """Decides, per figure, how it should appear on the poster.

    The curator sees the actual crop plus the *computed* effective DPI at
    the planned placement size — the decision is grounded in print
    physics, not vibes. Its verdict is advisory for quality (preflight
    re-verifies DPI deterministically) but authoritative for intent.
    """

    @property
    def agent_name(self) -> str:
        return "figure_curator"

    async def run(
        self,
        figure: PaperFigure,
        crop: Image.Image,
        placed_width_mm: float,
        effective_dpi: float,
        min_dpi: int,
        **kwargs: Any,
    ) -> FigureDecisionResult:
        template = self.load_prompt("poster")
        prompt = self.format_prompt(
            template,
            figure_id=figure.id,
            kind=figure.kind,
            caption=figure.caption,
            placed_width_mm=f"{placed_width_mm:.0f}",
            effective_dpi=f"{effective_dpi:.0f}",
            min_dpi=min_dpi,
            prompt_label=figure.id,
        )
        raw = await self.vlm.generate(
            prompt=prompt, images=[crop], response_format="json", temperature=0.3
        )
        data = extract_json(raw)
        if not isinstance(data, dict):
            raise ValueError(
                f"figure curator returned no JSON object for {figure.id}: {raw[:400]!r}"
            )
        data["figure_id"] = figure.id
        # Models routinely echo the full contract with unused keys as ""
        # (seen live: reset_table with chart_kind="") — blank means absent.
        for key in ("edit_instructions", "generate_brief", "chart_kind"):
            if isinstance(data.get(key), str) and not data[key].strip():
                data[key] = None
        return FigureDecisionResult(**data)
