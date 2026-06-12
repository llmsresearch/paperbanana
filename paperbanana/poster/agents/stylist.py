"""Poster style tokens: palette and typography."""

from __future__ import annotations

from typing import Any

from paperbanana.agents.base import BaseAgent
from paperbanana.core.utils import extract_json
from paperbanana.poster.types import StyleTokens

#: Families available to the stylist — present on macOS/Linux render hosts
#: and in LibreOffice, so PIL measurement and pptx->PDF conversion agree.
SAFE_FONT_FAMILIES = ("Helvetica", "Arial", "Times New Roman", "Georgia", "Verdana")


class PosterStylistAgent(BaseAgent):
    """Produces StyleTokens consistent with venue guidance and hard minima."""

    @property
    def agent_name(self) -> str:
        return "stylist"

    async def run(
        self,
        title: str,
        abstract: str,
        venue_display: str,
        venue_fonts: list[str] | None,
        min_pt_floor: dict[str, float],
        design_guidelines: str = "",
        **kwargs: Any,
    ) -> StyleTokens:
        template = self.load_prompt("poster")
        preferred = [f for f in (venue_fonts or []) if f in SAFE_FONT_FAMILIES]
        prompt = self.format_prompt(
            template,
            title=title,
            abstract=abstract[:1200],
            venue_display=venue_display,
            allowed_fonts=", ".join(SAFE_FONT_FAMILIES),
            venue_fonts=", ".join(preferred) or "(no venue preference)",
            min_pt_floor="; ".join(f"{k}: {v:.0f}pt" for k, v in sorted(min_pt_floor.items())),
            design_guidelines=design_guidelines or "(none)",
        )
        raw = await self.vlm.generate(prompt=prompt, response_format="json", temperature=0.5)
        data = extract_json(raw)
        if not isinstance(data, dict):
            raise ValueError(f"stylist returned no JSON object: {raw[:400]!r}")
        tokens = StyleTokens(**data)
        for field in ("font_heading", "font_body"):
            family = getattr(tokens, field)
            if family not in SAFE_FONT_FAMILIES:
                raise ValueError(
                    f"stylist chose unsupported font '{family}'; allowed: {SAFE_FONT_FAMILIES}"
                )
        for level, floor in min_pt_floor.items():
            actual = tokens.type_scale_pt.get(level)
            if actual is not None and actual < floor:
                raise ValueError(f"stylist set {level}={actual}pt below the hard minimum {floor}pt")
        return tokens
