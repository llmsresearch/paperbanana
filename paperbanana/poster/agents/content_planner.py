"""Poster storyboard planning: what goes on the poster, in what order."""

from __future__ import annotations

from typing import Any, Optional

import structlog

from paperbanana.agents.base import BaseAgent
from paperbanana.core.utils import extract_json
from paperbanana.poster.types import PaperAssets, Storyboard

logger = structlog.get_logger()

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
        design_guidelines: str = "",
        layout_patterns: str = "",
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
            design_guidelines=design_guidelines or "(none)",
            layout_patterns=layout_patterns or "",
        )
        raw = await self.vlm.generate(prompt=prompt, response_format="json", temperature=0.4)
        data = extract_json(raw)
        if not isinstance(data, dict):
            raise ValueError(f"content planner returned no JSON object: {raw[:400]!r}")
        data.setdefault("qr_url", qr_url)
        # Models sometimes list the QR code as a pseudo-figure; QR placement
        # is driven by panel role + qr_url, so normalize those entries away.
        # Likewise, only 'header' and 'qr' roles carry structural meaning —
        # an inventive section role degrades cleanly to 'custom'.
        from typing import get_args

        from paperbanana.poster.types import PanelRole

        valid_roles = set(get_args(PanelRole))
        for panel in data.get("panels", []):
            if not isinstance(panel, dict):
                continue
            if "figure_ids" in panel:
                panel["figure_ids"] = [
                    fid for fid in panel["figure_ids"] if str(fid).lower() not in ("qr", "qrcode")
                ]
            role = str(panel.get("role", "custom")).lower()
            panel["role"] = role if role in valid_roles else "custom"
        # Key stats must be short enough to render as huge callouts; an
        # over-long "stat" is a sentence, not a number — drop it (it still
        # appears in the panel text), and log the decision.
        kept_stats = []
        for stat in data.get("key_stats") or []:
            value = str(stat.get("value", "")).strip()
            if 0 < len(value) <= 16 and len(str(stat.get("label", ""))) <= 80:
                kept_stats.append(stat)
            else:
                logger.warning("Dropping unusable key_stat", stat=stat)
        data["key_stats"] = kept_stats
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
