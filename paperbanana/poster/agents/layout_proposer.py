"""Learned layout proposer: the model designs the poster's geometry."""

from __future__ import annotations

import json
from typing import Any

import structlog

from paperbanana.agents.base import BaseAgent
from paperbanana.core.utils import extract_json
from paperbanana.poster.proposal import LayoutProposal
from paperbanana.poster.types import FigureAsset, PhysicalSize, Storyboard

logger = structlog.get_logger()


class LayoutProposerAgent(BaseAgent):
    """Proposes the full poster structure as a validated JSON contract.

    Conditioned on the storyboard (content), page geometry, induced
    corpus priors, retrieved real-poster skeletons, and — on re-proposal
    — explicit violation/overflow feedback. The proposer decides
    structure only; it cannot author text or numbers.
    """

    @property
    def agent_name(self) -> str:
        return "layout_proposer"

    async def run(
        self,
        storyboard: Storyboard,
        assets: dict[str, FigureAsset],
        size: PhysicalSize,
        margin_mm: float,
        gutter_mm: float,
        layout_patterns: str = "",
        exemplars_block: str = "",
        violation_feedback: str = "",
        proposal_index: int = 0,
        **kwargs: Any,
    ) -> LayoutProposal:
        template = self.load_prompt("poster")
        panels_summary = []
        for panel in sorted(storyboard.panels, key=lambda p: p.order):
            chars = sum(len(b) for b in panel.text_blocks)
            figs = []
            for fid in panel.figure_ids:
                asset_id = fid[4:] if fid.startswith("new:") else fid
                asset = assets.get(asset_id)
                aspect = f"{asset.width_px / asset.height_px:.2f}" if asset else "?"
                figs.append(f"{asset_id}(w/h={aspect})")
            panels_summary.append(
                f"- {panel.id} (role={panel.role}): ~{chars} chars text, "
                f"figures: {', '.join(figs) or 'none'}"
            )
        key_stats = (
            "\n".join(
                f'- id="{s.id}": {s.value} — {s.label} (from panel {s.source_panel})'
                for s in storyboard.key_stats
            )
            or "(none — callouts are not possible)"
        )
        # Deterministic measurement the model must not be trusted to derive:
        # the per-column width each column count yields on THIS page.
        width_table = ", ".join(
            f"{n} cols -> {(size.width_mm - 2 * margin_mm - (n - 1) * gutter_mm) / n:.0f}mm wide"
            for n in range(2, 7)
        )
        page_desc = (
            f"{size.width_mm:.0f}x{size.height_mm:.0f}mm {size.orientation} "
            f"(aspect {size.width_mm / size.height_mm:.2f}:1). "
            f"Column widths on this page: {width_table}. "
            "Pick the column count whose width lands in the 300-550mm range; on pages "
            "wider than 2:1 that means 5-6 columns. Remember every figure renders at "
            "its column's width — wider columns make figures TALLER."
        )
        prompt = self.format_prompt(
            template,
            page=page_desc,
            margin_mm=f"{margin_mm:.0f}",
            gutter_mm=f"{gutter_mm:.0f}",
            panels_summary="\n".join(panels_summary),
            takeaway=storyboard.takeaway or "(none — a banner is not possible)",
            key_stats=key_stats,
            figure_ids=", ".join(sorted(assets)) or "(none)",
            layout_patterns=layout_patterns or "(no corpus priors available)",
            exemplars_block=exemplars_block or "(no exemplars retrieved)",
            violation_feedback=violation_feedback or "(first attempt)",
            example=proposal_to_json_example(),
            prompt_label=f"proposal_{proposal_index}",
        )
        raw = await self.vlm.generate(
            prompt=prompt, response_format="json", temperature=0.7, max_tokens=4096
        )
        data = extract_json(raw)
        if not isinstance(data, dict):
            raise ValueError(f"layout proposer returned no JSON object: {raw[:400]!r}")
        proposal = LayoutProposal(**data)
        logger.info(
            "Layout proposed",
            bands=[(b.kind, b.columns) for b in proposal.bands],
            banner=proposal.use_banner,
            hero=proposal.hero_figure_id,
            callouts=len(proposal.callouts),
        )
        return proposal


def format_overflow_feedback(overflows, page_deficit_mm: float) -> str:
    """Render measured overflow loads as re-proposal feedback."""
    total_required = sum(o.required_mm for o in overflows)
    lines = [
        f"YOUR PREVIOUS STRUCTURE OVERFLOWED THE PAGE by {page_deficit_mm:.0f}mm "
        "after content was measured. The content volume is FIXED — you must "
        "spread it over more parallel columns, not more stacked bands. "
        "Concretely: use ONE body band with MORE COLUMNS THAN YOUR LAST PROPOSAL "
        "(narrower columns also make figures shorter), avoid multiple stacked "
        "body bands, shrink the hero span to 2 columns or drop it, and skip the "
        "banner if space is tight. Measured loads:",
    ]
    for o in overflows:
        lines.append(
            f"- band '{o.band_id}' column {o.column}: content needs {o.required_mm:.0f}mm, "
            f"only {o.available_mm:.0f}mm available"
        )
    lines.append(f"(total overflowing content: {total_required:.0f}mm)")
    return "\n".join(lines)


def format_violation_feedback(violations) -> str:
    """Render fatal violations as re-proposal feedback."""
    lines = ["YOUR PREVIOUS PROPOSAL WAS INVALID. Fix exactly these problems:"]
    for v in violations:
        lines.append(f"- {v.code} ({v.target}): {v.detail}")
    return "\n".join(lines)


def proposal_to_json_example() -> str:
    """Format-only contract example (deliberately plain — see prompt note)."""
    return json.dumps(
        {
            "bands": [
                {"id": "header", "kind": "header", "columns": 1},
                {"id": "main", "kind": "body", "columns": 4},
            ],
            "placements": [
                {
                    "panel_id": "<id>",
                    "band_id": "main",
                    "column": 0,
                    "col_span": 1,
                    "height_frac": 0.55,
                },
                {
                    "panel_id": "<id>",
                    "band_id": "main",
                    "column": 0,
                    "col_span": 1,
                    "height_frac": 0.45,
                },
                {
                    "panel_id": "<id>",
                    "band_id": "main",
                    "column": 1,
                    "col_span": 1,
                    "height_frac": 1.0,
                },
                {
                    "panel_id": "<id>",
                    "band_id": "main",
                    "column": 2,
                    "col_span": 2,
                    "height_frac": 0.6,
                },
            ],
            "use_banner": False,
            "hero_figure_id": None,
            "callouts": [],
            "rationale": "why this structure fits this content",
        },
        indent=2,
    )
