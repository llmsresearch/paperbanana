"""Poster storyboard planning: what goes on the poster, in what order."""

from __future__ import annotations

from typing import Any, Optional

from paperbanana.agents.base import BaseAgent
from paperbanana.core.utils import extract_json
from paperbanana.poster.types import PaperAssets, Storyboard

_SECTION_CHAR_BUDGET = 2200


class PosterContentAgent(BaseAgent):
    """Distills the paper into a panel storyboard (no geometry, no style)."""

    @property
    def agent_name(self) -> str:
        return "content_planner"

    async def run(
        self,
        assets: PaperAssets,
        venue_display: str,
        venue_notes: str,
        columns_hint: int,
        qr_url: Optional[str] = None,
        **kwargs: Any,
    ) -> Storyboard:
        template = self.load_prompt("poster")
        figures_listing = (
            "\n".join(
                f"- {f.id} ({f.kind}, page {f.page}): {f.caption[:200]}" for f in assets.figures
            )
            or "(no figures detected)"
        )
        sections_text = "\n\n".join(
            f"## {s.heading}\n{s.text[:_SECTION_CHAR_BUDGET]}" for s in assets.sections
        )
        prompt = self.format_prompt(
            template,
            title=assets.title,
            abstract=assets.abstract,
            sections_text=sections_text,
            figures_listing=figures_listing,
            venue_display=venue_display,
            venue_notes=venue_notes or "(none)",
            columns_hint=columns_hint,
            qr_url=qr_url or "(none provided)",
        )
        raw = await self.vlm.generate(prompt=prompt, response_format="json", temperature=0.4)
        data = extract_json(raw)
        if not isinstance(data, dict):
            raise ValueError(f"content planner returned no JSON object: {raw[:400]!r}")
        data.setdefault("qr_url", qr_url)
        storyboard = Storyboard(**data)
        known = {f.id for f in assets.figures}
        for panel in storyboard.panels:
            for fid in panel.figure_ids:
                if not fid.startswith("new:") and fid not in known:
                    raise ValueError(
                        f"content planner referenced unknown figure '{fid}' "
                        f"(known: {sorted(known)})"
                    )
        return storyboard
