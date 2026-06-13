"""Figure curation execution: REUSE, RE-AUTHOR, or GENERATE.

This module is the poster head's differentiator: figures are never just
copied from the paper. Each one is judged against print physics (the
computed DPI at its planned placement) and poster legibility, then
either reused, re-rendered via image-conditioned generation under a
strict faithfulness gate, or replaced by a newly generated diagram.
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Awaitable, Callable, Optional

import structlog
from PIL import Image
from pydantic import ValidationError

from paperbanana.poster.agents.faithfulness import FaithfulnessAgent
from paperbanana.poster.agents.figure_curator import FigureCuratorAgent
from paperbanana.poster.renderer import PANEL_PADDING_MM
from paperbanana.poster.tables import (
    parse_table,
    render_table_matplotlib,
    table_to_plot_payload,
)
from paperbanana.poster.types import (
    FigureAsset,
    FigureDecisionResult,
    FigureProvenance,
    PaperAssets,
    PaperFigure,
    Storyboard,
    mm_to_inches,
)

logger = structlog.get_logger()

FIGURE_CONCURRENCY = 3
#: Gemini image generation aspect ratios.
_ALLOWED_RATIOS = ("1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9")
# Conservative cap across providers (gpt-image-2 rejects edges > 3840).
_MAX_GEN_DIM_PX = 3840

#: Async callable producing a new diagram image: (slug, brief, context) -> path.
DiagramGenerator = Callable[[str, str, str], Awaitable[Path]]


class PosterFigureError(RuntimeError):
    """A figure could not be prepared for the poster."""


def estimate_placed_width_mm(
    poster_width_mm: float, columns: int, margin_mm: float, gutter_mm: float
) -> float:
    """Inner width of one column — the planned placement width of figures."""
    col_w = (poster_width_mm - 2 * margin_mm - (columns - 1) * gutter_mm) / columns
    return col_w - 2 * PANEL_PADDING_MM


def _closest_ratio(width: int, height: int) -> str:
    target = width / height

    def ratio_value(r: str) -> float:
        w, h = r.split(":")
        return int(w) / int(h)

    return min(_ALLOWED_RATIOS, key=lambda r: abs(ratio_value(r) - target))


def _supports_guided_edits(image_gen) -> bool:
    return "images" in inspect.signature(image_gen.generate).parameters


def _snap_dim(px: int, multiple: int = 16, minimum: int = 256) -> int:
    """Snap a pixel dimension down to the nearest API-safe multiple."""
    return max(minimum, (px // multiple) * multiple)


async def curate_figures(
    assets: PaperAssets,
    storyboard: Storyboard,
    curator: FigureCuratorAgent,
    faithfulness: FaithfulnessAgent,
    image_gen,
    diagram_generator: DiagramGenerator,
    chart_generator,
    reauthor_prompt_template: str,
    palette: dict[str, str],
    placed_width_mm: float,
    min_dpi: int,
    out_dir: Path,
    overrides: Optional[dict[str, str]] = None,
    max_reauthor_attempts: int = 2,
    generation_context: str = "",
) -> tuple[dict[str, FigureAsset], list[FigureDecisionResult]]:
    """Curate every figure the storyboard places, plus requested new figures.

    Returns:
        (assets by id, decision log). Asset ids match the storyboard's
        ``figure_ids`` (``new:<slug>`` requests yield id ``<slug>``).

    Raises:
        PosterFigureError: When a re-authored figure cannot pass the
            faithfulness gate within the attempt budget, or a referenced
            figure does not exist.
    """
    overrides = overrides or {}
    referenced = [fid for panel in storyboard.panels for fid in panel.figure_ids]
    by_id = {f.id: f for f in assets.figures}
    for fid in referenced:
        if not fid.startswith("new:") and fid not in by_id:
            raise PosterFigureError(f"storyboard references unknown figure '{fid}'")

    out_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(FIGURE_CONCURRENCY)

    async def prepare_paper_figure(fid: str) -> tuple[FigureAsset, FigureDecisionResult]:
        figure = by_id[fid]
        crop = Image.open(figure.image_path).convert("RGB")
        effective_dpi = crop.width / mm_to_inches(placed_width_mm)
        async with sem:
            if fid in overrides:
                decision = FigureDecisionResult(
                    figure_id=fid,
                    decision=overrides[fid],  # type: ignore[arg-type]
                    reason="explicit user override (--figure-decision)",
                    edit_instructions=(
                        "Enlarge all text for poster legibility; simplify the legend."
                        if overrides[fid] == "reauthor"
                        else None
                    ),
                    generate_brief=(
                        f"A poster-friendly redesign of: {figure.caption}"
                        if overrides[fid] == "generate"
                        else None
                    ),
                )
            else:
                decision = await curator.run(
                    figure=figure,
                    crop=crop,
                    placed_width_mm=placed_width_mm,
                    effective_dpi=effective_dpi,
                    min_dpi=min_dpi,
                )
            asset = await _execute_decision(
                figure,
                crop,
                decision,
                faithfulness=faithfulness,
                image_gen=image_gen,
                diagram_generator=diagram_generator,
                chart_generator=chart_generator,
                reauthor_prompt_template=reauthor_prompt_template,
                palette=palette,
                placed_width_mm=placed_width_mm,
                out_dir=out_dir,
                max_reauthor_attempts=max_reauthor_attempts,
                generation_context=generation_context,
            )
            return asset, decision

    async def prepare_new_figure(slug: str, brief: str) -> tuple[FigureAsset, FigureDecisionResult]:
        async with sem:
            decision = FigureDecisionResult(
                figure_id=slug,
                decision="generate",
                reason="storyboard requested a figure the paper lacks",
                generate_brief=brief,
            )
            path = await diagram_generator(slug, brief, generation_context)
            with Image.open(path) as img:
                width_px, height_px = img.size
            asset = FigureAsset(
                id=slug,
                path=str(path),
                width_px=width_px,
                height_px=height_px,
                provenance=FigureProvenance(
                    origin="generated",
                    decision="generate",
                    decision_reason=decision.reason,
                ),
            )
            return asset, decision

    tasks = []
    seen: set[str] = set()
    for fid in referenced:
        if fid in seen:
            continue
        seen.add(fid)
        if fid.startswith("new:"):
            slug = fid[4:]
            tasks.append(prepare_new_figure(slug, storyboard.new_figure_briefs[slug]))
        else:
            tasks.append(prepare_paper_figure(fid))

    results = await asyncio.gather(*tasks)
    asset_map = {a.id: a for a, _ in results}
    decisions = [d for _, d in results]
    logger.info(
        "Figure curation complete",
        total=len(decisions),
        reuse=sum(d.decision == "reuse" for d in decisions),
        reauthor=sum(d.decision == "reauthor" for d in decisions),
        generate=sum(d.decision == "generate" for d in decisions),
    )
    return asset_map, decisions


async def _execute_decision(
    figure: PaperFigure,
    crop: Image.Image,
    decision: FigureDecisionResult,
    *,
    faithfulness: FaithfulnessAgent,
    image_gen,
    diagram_generator: DiagramGenerator,
    chart_generator,
    reauthor_prompt_template: str,
    palette: dict[str, str],
    placed_width_mm: float,
    out_dir: Path,
    max_reauthor_attempts: int,
    generation_context: str,
) -> FigureAsset:
    if decision.decision == "reuse":
        return FigureAsset(
            id=figure.id,
            path=figure.image_path,
            width_px=crop.width,
            height_px=crop.height,
            provenance=FigureProvenance(
                origin="paper",
                paper_figure_id=figure.id,
                source_page=figure.page,
                source_bbox_norm=figure.bbox_norm,
                decision="reuse",
                decision_reason=decision.reason,
                caption_anchored=figure.caption_anchored,
            ),
        )

    if decision.decision == "generate":
        brief = decision.generate_brief or ""
        path = await diagram_generator(figure.id, brief, generation_context)
        with Image.open(path) as img:
            width_px, height_px = img.size
        return FigureAsset(
            id=figure.id,
            path=str(path),
            width_px=width_px,
            height_px=height_px,
            provenance=FigureProvenance(
                origin="generated",
                paper_figure_id=figure.id,
                source_page=figure.page,
                source_bbox_norm=figure.bbox_norm,
                decision="generate",
                decision_reason=decision.reason,
                caption_anchored=figure.caption_anchored,
            ),
        )

    if decision.decision in ("rechart", "reset_table"):
        last_differences: list[str] = []
        for attempt in range(1, max_reauthor_attempts + 1):
            # A malformed parse (model ignored the contract, or the crop is
            # not actually a table) is a failed attempt like any other —
            # retried, then escalated with override guidance, never a crash.
            try:
                # Climb temperature across attempts: a parse truncated or
                # looping at low temp does so deterministically, so a retry
                # at the same temp repeats the failure.
                table = await parse_table(
                    crop,
                    faithfulness.vlm,
                    faithfulness.prompt_dir,
                    caption=figure.caption,
                    temperature=0.1 + 0.3 * (attempt - 1),
                )
            except (ValueError, ValidationError) as exc:
                last_differences = [f"table parse failed: {str(exc)[:200]}"]
                logger.warning(
                    "Table parse failed",
                    figure=figure.id,
                    decision=decision.decision,
                    attempt=attempt,
                    error=str(exc)[:300],
                )
                continue
            if decision.decision == "reset_table":
                path = out_dir / f"{figure.id}_reset.png"
                render_table_matplotlib(table, palette, path, placed_width_mm)
            else:
                intent = (
                    f"A clean {decision.chart_kind or 'bar'} chart for a conference poster "
                    f"showing: {table.title or figure.caption}. Use EXACTLY the values in "
                    "the raw data (no rounding); highlight the row marked highlight_row "
                    "with the accent color; large axis labels readable from 2 meters."
                )
                path = await chart_generator(
                    table_to_plot_payload(table, decision.chart_kind),
                    intent,
                    out_dir / f"{figure.id}_chart.png",
                )
            with Image.open(path) as rendered_check:
                rendered = rendered_check.convert("RGB")
            verdict = await faithfulness.run(
                original=crop,
                reauthored=rendered,
                caption=(
                    figure.caption
                    + " [the second image is a deliberate "
                    + ("chart conversion" if decision.decision == "rechart" else "re-typeset")
                    + " of the original table: layout/format differences are expected; "
                    "verify ONLY that every numeric value and label matches the original]"
                ),
            )
            if verdict.verdict == "pass":
                logger.info(
                    "Table transformed",
                    figure=figure.id,
                    decision=decision.decision,
                    attempt=attempt,
                )
                return FigureAsset(
                    id=figure.id,
                    path=str(path),
                    width_px=rendered.width,
                    height_px=rendered.height,
                    provenance=FigureProvenance(
                        origin="paper",
                        paper_figure_id=figure.id,
                        source_page=figure.page,
                        source_bbox_norm=figure.bbox_norm,
                        decision=decision.decision,
                        decision_reason=decision.reason,
                        caption_anchored=figure.caption_anchored,
                        faithfulness="verified",
                    ),
                )
            last_differences = verdict.differences
            logger.warning(
                "Table transformation failed faithfulness gate",
                figure=figure.id,
                decision=decision.decision,
                attempt=attempt,
                differences=verdict.differences,
            )
        raise PosterFigureError(
            f"figure '{figure.id}' ({decision.decision}) failed the faithfulness gate "
            f"{max_reauthor_attempts} time(s); last violations: {last_differences}. "
            f"Re-run with --figure-decision {figure.id}=reuse to place the original crop."
        )

    # decision == "reauthor"
    if not _supports_guided_edits(image_gen):
        raise PosterFigureError(
            f"figure '{figure.id}' needs re-authoring but image provider "
            f"'{getattr(image_gen, 'name', type(image_gen).__name__)}' does not support "
            "image-conditioned generation; use a provider with guided edits or re-run "
            f"with --figure-decision {figure.id}=reuse"
        )

    # Image APIs constrain dimensions (gpt-image-2: divisible by 16); snap down.
    target_w = _snap_dim(min(_MAX_GEN_DIM_PX, round(mm_to_inches(placed_width_mm) * 300)))
    target_h = _snap_dim(min(_MAX_GEN_DIM_PX, round(target_w * crop.height / crop.width)))
    instructions = decision.edit_instructions or ""
    last_differences: list[str] = []

    for attempt in range(1, max_reauthor_attempts + 1):
        prompt = reauthor_prompt_template.format(
            edit_instructions=instructions
            + (
                "\n\nPrevious attempt failed the fidelity audit; these violations "
                "MUST be corrected:\n" + "\n".join(f"- {d}" for d in last_differences)
                if last_differences
                else ""
            ),
            primary=palette["primary"],
            secondary=palette["secondary"],
            accent=palette["accent"],
            background=palette["background"],
            placed_width_mm=f"{placed_width_mm:.0f}",
        )
        reauthored = await image_gen.generate(
            prompt=prompt,
            images=[crop],
            width=target_w,
            height=target_h,
            aspect_ratio=_closest_ratio(crop.width, crop.height),
        )
        verdict = await faithfulness.run(
            original=crop, reauthored=reauthored, caption=figure.caption
        )
        if verdict.verdict == "pass":
            path = out_dir / f"{figure.id}_reauthored.png"
            reauthored.save(path)
            logger.info(
                "Figure re-authored", figure=figure.id, attempt=attempt, size=reauthored.size
            )
            return FigureAsset(
                id=figure.id,
                path=str(path),
                width_px=reauthored.width,
                height_px=reauthored.height,
                provenance=FigureProvenance(
                    origin="paper",
                    paper_figure_id=figure.id,
                    source_page=figure.page,
                    source_bbox_norm=figure.bbox_norm,
                    decision="reauthor",
                    decision_reason=decision.reason,
                    edit_instructions=decision.edit_instructions,
                    caption_anchored=figure.caption_anchored,
                    faithfulness="verified",
                ),
            )
        last_differences = verdict.differences
        logger.warning(
            "Re-authored figure failed faithfulness gate",
            figure=figure.id,
            attempt=attempt,
            differences=verdict.differences,
        )

    raise PosterFigureError(
        f"figure '{figure.id}' failed the faithfulness gate {max_reauthor_attempts} "
        f"time(s); last violations: {last_differences}. Re-run with "
        f"--figure-decision {figure.id}=reuse to place the original crop instead."
    )
