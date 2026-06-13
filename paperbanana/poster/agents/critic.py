"""VLM poster critique over per-panel zoom renders."""

from __future__ import annotations

from typing import Any

from PIL import Image

from paperbanana.agents.base import BaseAgent
from paperbanana.core.utils import extract_json
from paperbanana.poster.types import PosterCritique, PosterIR, PreflightReport

#: Cap on panel crops sent to the critic in one call.
MAX_PANEL_IMAGES = 10


class PosterCriticAgent(BaseAgent):
    """Critiques the rendered poster and proposes structured edit ops.

    Receives the full preview *and* per-panel zoom crops: a downscaled
    A0 canvas hides exactly the legibility defects that matter, so the
    panel crops are where fine-grained judgment happens. Deterministic
    preflight failures are passed in as known defects the critique must
    address.
    """

    @property
    def agent_name(self) -> str:
        return "critic"

    async def run(
        self,
        ir: PosterIR,
        preview: Image.Image,
        panel_crops: dict[str, Image.Image],
        preflight: PreflightReport,
        iteration: int,
        anchor_images: list[Image.Image] | None = None,
        comprehension_gaps: list[str] | None = None,
        **kwargs: Any,
    ) -> PosterCritique:
        template = self.load_prompt("poster")
        anchors = anchor_images or []
        panel_ids = list(panel_crops)[:MAX_PANEL_IMAGES]
        images = anchors + [preview] + [panel_crops[pid] for pid in panel_ids]
        offset = len(anchors)
        anchor_listing = "".join(
            f"Image {i + 1}: a REFERENCE poster previously judged excellent (calibration "
            "anchor — this is what 5/5 looks like; do not critique it).\n"
            for i in range(len(anchors))
        )
        image_listing = (
            anchor_listing
            + f"Image {offset + 1}: full poster preview (the poster under review).\n"
            + "\n".join(
                f"Image {offset + i + 2}: zoom of panel '{pid}'." for i, pid in enumerate(panel_ids)
            )
        )
        preflight_failures = (
            "\n".join(
                f"- {c.id}: {c.value} (required {c.threshold}) — {c.detail}"
                for c in preflight.failures
            )
            or "(none — all deterministic checks pass)"
        )
        panels_summary = "\n".join(
            f"- {p.id} (role={p.role}, order={p.order}, weight={p.weight}): "
            + "; ".join(
                el.content[:60] if el.kind == "text" else f"[{el.kind}]" for el in p.elements
            )
            for p in ir.panels_in_order()
        )
        gaps_block = (
            "\n".join(f"- {g}" for g in comprehension_gaps)
            if comprehension_gaps
            else "(none — a fresh reader answered every quiz question from the poster)"
        )
        prompt = self.format_prompt(
            template,
            iteration=iteration,
            image_listing=image_listing,
            preflight_failures=preflight_failures,
            panels_summary=panels_summary,
            comprehension_gaps=gaps_block,
            prompt_label=f"iter_{iteration}",
        )
        raw = await self.vlm.generate(
            prompt=prompt, images=images, response_format="json", temperature=0.3
        )
        data = extract_json(raw)
        if not isinstance(data, dict):
            raise ValueError(f"critic returned no JSON object: {raw[:400]!r}")
        return PosterCritique(**data)
