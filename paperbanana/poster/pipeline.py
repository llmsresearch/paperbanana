"""End-to-end poster generation pipeline.

Stages::

    ingest -> storyboard -> style -> figure curation -> build IR
    -> loop: layout -> fit text -> render pptx -> PDF -> preview/crops
             -> preflight -> critic -> apply edit ops
    -> final artifacts (pptx, press-ready PDF, preview PNG, preflight report)

LibreOffice is verified at construction time, before any API spend.
Stage outputs are persisted to the run directory as they complete, and
``resume_dir`` reloads them so an interrupted run continues where it
stopped.
"""

from __future__ import annotations

import datetime
import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

import structlog
from PIL import Image

from paperbanana.core.config import Settings
from paperbanana.core.cost_tracker import CostTracker
from paperbanana.core.prompt_recorder import PromptRecorder
from paperbanana.core.utils import extract_json, find_prompt_dir
from paperbanana.guidelines.venues import UnknownVenueError, VenuePack, resolve_venue
from paperbanana.poster.agents import (
    FaithfulnessAgent,
    FigureCuratorAgent,
    FigureDetectorAgent,
    PaperMetadataAgent,
    PosterContentAgent,
    PosterCriticAgent,
    PosterStylistAgent,
)
from paperbanana.poster.convert import find_soffice, pdf_to_png, pptx_to_pdf, render_panel_crops
from paperbanana.poster.edits import EditOpError, apply_edit_ops, parse_edit_ops
from paperbanana.poster.figures import curate_figures, estimate_placed_width_mm
from paperbanana.poster.ingest import ingest_paper
from paperbanana.poster.layout import (
    DEFAULT_HEADER_FRAC,
    LayoutError,
    assign_bboxes,
    body_column_height_mm,
)
from paperbanana.poster.lessons import format_lessons_block, load_lessons, record_lessons
from paperbanana.poster.preflight import (
    DEFAULT_LEGIBILITY_MIN_PT,
    render_preflight_markdown,
    run_preflight,
)
from paperbanana.poster.renderer import (
    PANEL_PADDING_MM,
    TextOverflowError,
    measure_panel_required_height_mm,
    render_pptx,
    required_print_scale,
)
from paperbanana.poster.schemas import (
    format_layout_priors,
    load_schema_library,
    select_schema,
)
from paperbanana.poster.style_knowledge import load_poster_style_guide
from paperbanana.poster.types import (
    FigureAsset,
    FigureDecisionResult,
    FigureElement,
    Panel,
    PaperAssets,
    PhysicalSize,
    PosterIR,
    PosterOutput,
    QRElement,
    Storyboard,
    StyleTokens,
    TextElement,
)
from paperbanana.poster.venue_spec import VenueSpec, load_venue_spec

logger = structlog.get_logger()

#: Attempts to shorten an overflowing panel's text before giving up.
TEXT_FIT_ATTEMPTS = 2
#: Deterministic weight-rebalancing rounds after text shortening.
WEIGHT_FIT_ATTEMPTS = 3
#: Column-count escalation ceiling for dense posters.
MAX_FIT_COLUMNS = 5
#: Narrowest acceptable body column.
MIN_COLUMN_WIDTH_MM = 200.0


def _generate_poster_run_id() -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"poster_{ts}_{uuid.uuid4().hex[:6]}"


class PosterPipeline:
    """Generates a venue-compliant conference poster from a paper PDF."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        progress_callback: Optional[Callable[[str, dict], None]] = None,
        vlm_client=None,
        image_gen_client=None,
    ):
        self.settings = settings or Settings()
        self.run_id = _generate_poster_run_id()
        self._header_frac = DEFAULT_HEADER_FRAC
        self._progress_callback = progress_callback
        self._run_dir = Path(self.settings.output_dir) / self.run_id

        # Hard requirement, checked before any API spend.
        self._soffice = find_soffice(self.settings.soffice_path)

        self._prompt_recorder = None
        if self.settings.save_prompts:
            self._prompt_recorder = PromptRecorder(run_dir_provider=lambda: self._run_dir)

        if vlm_client is not None and image_gen_client is not None:
            self._vlm = vlm_client
            self._image_gen = image_gen_client
        else:
            from paperbanana.providers.registry import ProviderRegistry

            self._vlm = ProviderRegistry.create_vlm(self.settings)
            self._image_gen = ProviderRegistry.create_image_gen(self.settings)
        self._cost_tracker = CostTracker(budget=self.settings.budget_usd)
        for provider in (self._vlm, self._image_gen):
            if hasattr(provider, "cost_tracker"):
                provider.cost_tracker = self._cost_tracker

        prompt_dir = self.settings.prompt_dir or find_prompt_dir()
        self._prompt_dir = prompt_dir
        kwargs = {"prompt_dir": prompt_dir, "prompt_recorder": self._prompt_recorder}
        self.detector = FigureDetectorAgent(self._vlm, **kwargs)
        self.metadata_agent = PaperMetadataAgent(self._vlm, **kwargs)
        self.content_agent = PosterContentAgent(self._vlm, **kwargs)
        self.curator = FigureCuratorAgent(self._vlm, **kwargs)
        self.stylist = PosterStylistAgent(self._vlm, **kwargs)
        self.critic = PosterCriticAgent(self._vlm, **kwargs)
        self.faithfulness = FaithfulnessAgent(self._vlm, **kwargs)

        logger.info(
            "Poster pipeline initialized",
            run_id=self.run_id,
            soffice=str(self._soffice),
            vlm=getattr(self._vlm, "name", "custom"),
            image_gen=getattr(self._image_gen, "name", "custom"),
        )

    def _emit(self, event: str, **payload: Any) -> None:
        logger.info("poster_progress", progress_event=event, **payload)
        if self._progress_callback is not None:
            try:
                self._progress_callback(event, payload)
            except Exception:
                logger.warning("Progress callback failed", progress_event=event)

    # ------------------------------------------------------------------
    # Stage persistence / resume

    def _stage_path(self, name: str) -> Path:
        return self._run_dir / f"{name}.json"

    def _load_stage(self, name: str, resume: bool):
        path = self._stage_path(name)
        if resume and path.is_file():
            logger.info("Resuming stage from checkpoint", stage=name)
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def _save_stage(self, name: str, payload: str) -> None:
        self._run_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._stage_path(name).with_suffix(".json.tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(self._stage_path(name))

    # ------------------------------------------------------------------

    async def generate(
        self,
        paper_pdf: Path,
        venue: Optional[str] = None,
        year: Optional[int] = None,
        qr_url: Optional[str] = None,
        figure_overrides: Optional[dict[str, str]] = None,
        iterations: Optional[int] = None,
        resume_dir: Optional[Path] = None,
    ) -> PosterOutput:
        """Generate a poster from a paper PDF.

        Args:
            paper_pdf: Source paper.
            venue: Venue name (defaults to ``settings.venue``).
            year: Venue spec year (latest available when omitted).
            qr_url: Link rendered as a QR code on the poster.
            figure_overrides: Per-figure decision overrides
                (``{"fig3": "reuse"}``) — explicit user authority over the
                curator.
            iterations: Critic refinement iterations
                (defaults to ``settings.poster_refinement_iterations``).
            resume_dir: A previous run directory to resume from.
        """
        venue_name = (venue or self.settings.venue).strip().lower()
        max_iterations = (
            iterations if iterations is not None else self.settings.poster_refinement_iterations
        )
        resume = False
        if resume_dir is not None:
            self._run_dir = Path(resume_dir)
            self.run_id = self._run_dir.name
            resume = True
        self._run_dir.mkdir(parents=True, exist_ok=True)

        spec = load_venue_spec(venue_name, year, extra_dir=self.settings.venue_spec_dir)
        venue_pack = self._resolve_style_pack(venue_name)
        design_guide = load_poster_style_guide(
            self.settings.guidelines_path, venue=venue_name, venue_dir=self.settings.venue_dir
        )

        # Learning layers: induced real-poster structure + lessons from
        # previous runs on this machine.
        gen_w_mm, _ = spec.dimensions.generation_size_mm()
        schema_library = load_schema_library()
        layout_schema = select_schema(
            schema_library,
            spec.dimensions.generation_orientation(),
            min_columns=3 if gen_w_mm >= 1100 else 2,
        )
        layout_patterns = format_layout_priors(schema_library, layout_schema)
        lessons = load_lessons(venue=venue_name)
        design_guide_with_lessons = design_guide + format_lessons_block(lessons)
        self._emit(
            "knowledge_loaded",
            schema=layout_schema.id,
            schema_posters=layout_schema.n_posters,
            lessons=len(lessons),
        )
        run_lessons: list[str] = []
        self._shorten_events = 0
        min_pt_floor = dict(DEFAULT_LEGIBILITY_MIN_PT)
        for level, value in spec.text_rules.min_pt.items():
            min_pt_floor[level] = max(min_pt_floor.get(level, 0), value)

        self._save_stage(
            "run_input",
            json.dumps(
                {
                    "paper_pdf": str(paper_pdf),
                    "venue": venue_name,
                    "spec_year": spec.year,
                    "qr_url": qr_url,
                    "figure_overrides": figure_overrides or {},
                    "iterations": max_iterations,
                },
                indent=2,
            ),
        )

        # Stage 1: ingest -------------------------------------------------
        self._emit("ingest_started", paper=str(paper_pdf))
        cached = self._load_stage("paper_assets/paper_assets", resume)
        if cached is not None:
            assets = PaperAssets(**cached)
        else:
            assets = await ingest_paper(
                Path(paper_pdf),
                self.detector,
                self.metadata_agent,
                self._run_dir / "paper_assets",
                extract_dpi=self.settings.poster_extract_dpi,
            )
        self._emit("ingest_complete", figures=len(assets.figures), title=assets.title)

        # Stage 2: storyboard ----------------------------------------------
        cached = self._load_stage("storyboard", resume)
        if cached is not None:
            storyboard = Storyboard(**cached)
        else:
            storyboard = await self.content_agent.run(
                assets=assets,
                venue_display=spec.display_name,
                venue_notes=spec.notes or "",
                columns_hint=layout_schema.columns,
                qr_url=qr_url,
                design_guidelines=design_guide_with_lessons,
                layout_patterns=layout_patterns,
            )
            self._save_stage("storyboard", storyboard.model_dump_json(indent=2))
        self._emit("storyboard_complete", panels=len(storyboard.panels))

        # Stage 3: style ----------------------------------------------------
        cached = self._load_stage("style", resume)
        if cached is not None:
            style = StyleTokens(**cached)
        else:
            style = await self.stylist.run(
                title=assets.title,
                abstract=assets.abstract,
                venue_display=spec.display_name,
                venue_fonts=venue_pack.config.fonts if venue_pack else None,
                min_pt_floor=min_pt_floor,
                design_guidelines=design_guide_with_lessons,
            )
            self._save_stage("style", style.model_dump_json(indent=2))
        self._emit("style_complete", fonts=[style.font_heading, style.font_body])

        # Stage 4: figure curation -------------------------------------------
        gen_w, gen_h = spec.dimensions.generation_size_mm()
        placed_width = estimate_placed_width_mm(gen_w, storyboard.columns, 20.0, 10.0)
        cached = self._load_stage("figure_assets", resume)
        if cached is not None:
            asset_map = {k: FigureAsset(**v) for k, v in cached["assets"].items()}
            decisions = [FigureDecisionResult(**d) for d in cached["decisions"]]
        else:
            reauthor_template = (Path(self._prompt_dir) / "poster" / "reauthor_edit.txt").read_text(
                encoding="utf-8"
            )
            asset_map, decisions = await curate_figures(
                assets,
                storyboard,
                self.curator,
                self.faithfulness,
                self._image_gen,
                self._make_diagram_generator(),
                reauthor_template,
                style.palette,
                placed_width_mm=placed_width,
                min_dpi=spec.text_rules.min_image_dpi,
                out_dir=self._run_dir / "curated_figures",
                overrides=figure_overrides,
                max_reauthor_attempts=self.settings.poster_reauthor_max_attempts,
                generation_context=self._generation_context(assets),
            )
            self._save_stage(
                "figure_assets",
                json.dumps(
                    {
                        "assets": {
                            k: json.loads(v.model_dump_json()) for k, v in asset_map.items()
                        },
                        "decisions": [json.loads(d.model_dump_json()) for d in decisions],
                    },
                    indent=2,
                ),
            )
        self._emit(
            "figures_complete",
            decisions={d.figure_id: d.decision for d in decisions},
        )

        # Stage 5: build IR ----------------------------------------------------
        ir = self._build_ir(assets, storyboard, style, asset_map, spec, qr_url)
        self._header_frac = self._tune_header_band(ir, layout_schema)
        # Column escalation during fitting may only be undone down to the
        # storyboard's own choice — never below it.
        self._columns_floor = max(2, ir.columns)

        # Stage 6: refinement loop ----------------------------------------------
        iteration = 0
        preflight = None
        pptx_path = pdf_path = preview_path = None
        while True:
            iteration += 1
            iter_dir = self._run_dir / f"iter_{iteration}"
            iter_dir.mkdir(parents=True, exist_ok=True)

            if not ir.is_fully_placed():
                ir = assign_bboxes(ir, header_frac=self._header_frac)
            ir = await self._fit_content(ir, min_pt_floor)
            (iter_dir / "poster_ir.json").write_text(ir.model_dump_json(indent=2), encoding="utf-8")

            pptx_path = render_pptx(ir, iter_dir / "poster.pptx", self._run_dir / "work")
            pdf_path = pptx_to_pdf(pptx_path, iter_dir, self._soffice)
            preview_path = pdf_to_png(pdf_path, iter_dir / "preview.png", dpi=96)
            crops = render_panel_crops(pdf_path, ir, iter_dir / "panels", dpi=150)
            preflight = run_preflight(ir, spec, pdf_path=pdf_path, png_path=preview_path)
            (iter_dir / "preflight.json").write_text(
                preflight.model_dump_json(indent=2), encoding="utf-8"
            )
            self._emit(
                "iteration_rendered",
                iteration=iteration,
                preflight_passed=preflight.passed,
                failures=[c.id for c in preflight.failures],
            )

            if iteration > max_iterations:
                break

            critique = await self.critic.run(
                ir=ir,
                preview=Image.open(preview_path).convert("RGB"),
                panel_crops={k: Image.open(v).convert("RGB") for k, v in crops.items()},
                preflight=preflight,
                iteration=iteration,
            )
            (iter_dir / "critique.json").write_text(
                critique.model_dump_json(indent=2), encoding="utf-8"
            )
            self._emit(
                "critique_complete",
                iteration=iteration,
                blocking=critique.blocking,
                ops=len(critique.edit_ops),
            )
            if critique.blocking and critique.summary:
                run_lessons.append(
                    f"Critic flagged (iteration {iteration}): {critique.summary[:300]}"
                )
            if not critique.blocking and not critique.edit_ops and preflight.passed:
                break

            # Apply ops individually: an illegal op from the critic (e.g.
            # shrinking type below the legibility floor) is rejected and
            # reported, never silently obeyed — and never kills the run.
            deferred = []
            rejected: list[str] = []
            applied = 0
            for raw_op in critique.edit_ops:
                try:
                    ops = parse_edit_ops([raw_op])
                    ir, deferred_one = apply_edit_ops(ir, ops, min_pt=min_pt_floor)
                    deferred.extend(deferred_one)
                    applied += 1
                except EditOpError as exc:
                    rejected.append(str(exc))
            if rejected:
                self._emit(
                    "edit_ops_rejected",
                    iteration=iteration,
                    applied=applied,
                    rejected=rejected,
                )
                run_lessons.extend(f"Critic proposed an illegal edit: {r[:200]}" for r in rejected)
            if deferred:
                logger.warning(
                    "recurate_figure ops are not re-executed within the loop in v1; "
                    "re-run with --figure-decision to override these figures",
                    assets=[op.asset_id for op in deferred],
                )

        # Final artifacts -------------------------------------------------------
        assert pptx_path and pdf_path and preview_path and preflight is not None
        final_pptx = self._run_dir / "poster.pptx"
        final_pdf = self._run_dir / "poster.pdf"
        final_preview = self._run_dir / "preview.png"
        shutil.copy2(pptx_path, final_pptx)
        shutil.copy2(pdf_path, final_pdf)
        shutil.copy2(preview_path, final_preview)
        ir_path = self._run_dir / "poster_ir.json"
        ir_path.write_text(ir.model_dump_json(indent=2), encoding="utf-8")
        preflight = run_preflight(ir, spec, pdf_path=final_pdf, png_path=final_preview)
        (self._run_dir / "preflight_report.json").write_text(
            preflight.model_dump_json(indent=2), encoding="utf-8"
        )
        (self._run_dir / "preflight_report.md").write_text(
            render_preflight_markdown(preflight), encoding="utf-8"
        )
        output = PosterOutput(
            run_dir=str(self._run_dir),
            pptx_path=str(final_pptx),
            pdf_path=str(final_pdf),
            preview_path=str(final_preview),
            ir_path=str(ir_path),
            preflight=preflight,
            iterations=iteration,
            figure_decisions=decisions,
            metadata={
                "venue": venue_name,
                "spec_year": spec.year,
                "spec_display_name": spec.display_name,
                "size_mm": [ir.size.width_mm, ir.size.height_mm],
                "print_scale": ir.print_scale,
                "cost_usd": getattr(self._cost_tracker, "total_cost", None),
            },
        )
        (self._run_dir / "poster_output.json").write_text(
            output.model_dump_json(indent=2), encoding="utf-8"
        )
        for check in preflight.failures:
            run_lessons.append(f"Final preflight failure {check.id}: {check.detail[:200]}")
        if self._shorten_events >= 3:
            run_lessons.append(
                f"Storyboard overpacked panels: {self._shorten_events} overflow rewrites were "
                "needed; plan fewer/shorter bullets per panel from the start."
            )
        record_lessons(self.run_id, venue_name, run_lessons)
        self._emit(
            "poster_complete",
            run_dir=str(self._run_dir),
            preflight_passed=preflight.passed,
            iterations=iteration,
            lessons_recorded=len(run_lessons),
        )
        return output

    # ------------------------------------------------------------------

    def _tune_header_band(self, ir: PosterIR, schema) -> float:
        """Size the header band to its measured content, bounded by what
        real posters do (the schema's title-band p25-p75 range)."""
        from paperbanana.poster.types import BBox

        header = next(p for p in ir.panels if p.role == "header")
        content_w = ir.size.width_mm - 2 * ir.margin_mm
        content_h = ir.size.height_mm - 2 * ir.margin_mm
        probe = header.model_copy(
            update={
                "bbox": BBox(x_mm=ir.margin_mm, y_mm=ir.margin_mm, w_mm=content_w, h_mm=content_h)
            }
        )
        required = measure_panel_required_height_mm(probe, ir) + 2 * PANEL_PADDING_MM
        needed_frac = required / content_h
        if schema.title_band_frac is not None:
            lo, hi = schema.title_band_frac.p25, schema.title_band_frac.p75
        else:
            lo, hi = 0.10, 0.20
        frac = min(max(needed_frac, lo), max(hi, needed_frac))
        self._emit(
            "header_band_tuned",
            needed_frac=round(needed_frac, 3),
            corpus_range=[lo, hi],
            chosen=round(frac, 3),
        )
        return frac

    def _resolve_style_pack(self, venue_name: str) -> Optional[VenuePack]:
        """Style packs are optional supplements to the mandatory venue spec."""
        try:
            return resolve_venue(
                venue_name,
                builtin_dir=self.settings.guidelines_path,
                extra_dir=self.settings.venue_dir,
            )
        except UnknownVenueError:
            logger.info("No style pack for venue (poster spec only)", venue=venue_name)
            return None

    def _generation_context(self, assets: PaperAssets) -> str:
        """Method-centric context handed to nested diagram generation."""
        preferred = [
            s
            for s in assets.sections
            if any(
                key in s.heading.lower()
                for key in ("method", "approach", "architecture", "framework", "model")
            )
        ]
        sections = preferred or assets.sections
        text = "\n\n".join(f"## {s.heading}\n{s.text}" for s in sections)
        return f"{assets.title}\n\n{assets.abstract}\n\n{text}"[:24000]

    def _make_diagram_generator(self):
        """Nested PaperBanana diagram pipeline for GENERATE decisions."""

        async def generate(slug: str, brief: str, context: str) -> Path:
            from paperbanana.core.pipeline import PaperBananaPipeline
            from paperbanana.core.types import DiagramType, GenerationInput

            sub_settings = self.settings.model_copy(
                update={
                    "output_dir": str(self._run_dir / "subruns"),
                    "generate_caption": False,
                    "export_tikz": False,
                    "vector_export": "none",
                    "optimize_inputs": False,
                }
            )
            self._emit("diagram_generation_started", figure=slug)
            pipeline = PaperBananaPipeline(settings=sub_settings)
            result = await pipeline.generate(
                GenerationInput(
                    source_context=context,
                    communicative_intent=brief,
                    diagram_type=DiagramType.METHODOLOGY,
                )
            )
            self._emit("diagram_generation_complete", figure=slug, path=result.image_path)
            return Path(result.image_path)

        return generate

    def _build_ir(
        self,
        assets: PaperAssets,
        storyboard: Storyboard,
        style: StyleTokens,
        asset_map: dict[str, FigureAsset],
        spec: VenueSpec,
        qr_url: Optional[str],
    ) -> PosterIR:
        gen_w, gen_h = spec.dimensions.generation_size_mm()
        captions = {f.id: f.caption for f in assets.figures}
        panels: list[Panel] = [
            Panel(
                id="header",
                role="header",
                order=0,
                elements=[
                    TextElement(level="title", content=assets.title),
                    TextElement(level="authors", content=", ".join(assets.authors) or " "),
                    *(
                        [
                            TextElement(
                                level="affiliation",
                                content=" · ".join(assets.affiliations),
                            )
                        ]
                        if assets.affiliations
                        else []
                    ),
                ],
            )
        ]
        effective_qr = storyboard.qr_url or qr_url
        for sb_panel in storyboard.panels:
            elements: list = [
                TextElement(level="body", content=block)
                for block in sb_panel.text_blocks
                if block.strip()
            ]
            # Several figures must share the panel's height budget; a panel
            # with text also needs room for it.
            n_figures = len(sb_panel.figure_ids)
            height_budget = 0.85 if not sb_panel.text_blocks else 0.6
            max_height_frac = min(0.6, height_budget / n_figures) if n_figures else 0.6
            for fid in sb_panel.figure_ids:
                asset_id = fid[4:] if fid.startswith("new:") else fid
                caption = captions.get(asset_id)
                short_caption = caption.split(". ")[0].strip() if caption else None
                elements.append(
                    FigureElement(
                        asset_id=asset_id,
                        caption=short_caption,
                        max_height_frac=max_height_frac,
                    )
                )
            if sb_panel.role == "qr" and effective_qr:
                elements.append(QRElement(url=effective_qr, label="Paper & code"))
            if not elements:
                elements = [TextElement(level="body", content=sb_panel.title or sb_panel.id)]
            panels.append(
                Panel(
                    id=sb_panel.id,
                    role=sb_panel.role,
                    title=sb_panel.title,
                    order=sb_panel.order,
                    weight=sb_panel.weight,
                    elements=elements,
                )
            )
        # A QR code is an element, not a panel: a panel containing only QR
        # elements wastes a column slot, so fold it into the preceding
        # content panel; likewise, a missing QR is appended to the last one.
        body = sorted([p for p in panels if p.role != "header"], key=lambda p: p.order)
        qr_only = [p for p in body if all(el.kind == "qr" for el in p.elements) and len(body) > 1]
        for panel in qr_only:
            target = next(b for b in reversed(body) if b is not panel)
            target.elements = list(target.elements) + list(panel.elements)
            panels.remove(panel)
            body.remove(panel)
        for i, panel in enumerate(sorted(panels, key=lambda p: p.order)):
            panel.order = i
        if effective_qr and not any(el.kind == "qr" for p in panels for el in p.elements):
            last = max((p for p in panels if p.role != "header"), key=lambda p: p.order)
            last.elements = list(last.elements) + [
                QRElement(url=effective_qr, label="Paper & code")
            ]

        size = PhysicalSize(width_mm=gen_w, height_mm=gen_h)
        return PosterIR(
            venue=spec.venue,
            venue_spec_year=spec.year,
            size=size,
            orientation=size.orientation,
            print_scale=required_print_scale(gen_w, gen_h),
            columns=storyboard.columns,
            style=style,
            panels=panels,
            assets=asset_map,
            paper_title=assets.title,
            authors=assets.authors,
            affiliations=assets.affiliations,
        )

    def _overflowing_panels(self, ir: PosterIR) -> list[tuple[Panel, float, float]]:
        """(panel, required_mm, available_mm) for every overflowing placed panel."""
        overflowing = []
        for panel in ir.panels_in_order():
            if panel.bbox is None:
                continue
            available = panel.bbox.h_mm - 2 * PANEL_PADDING_MM
            required = measure_panel_required_height_mm(panel, ir)
            if required > available:
                overflowing.append((panel, required, available))
        return overflowing

    async def _fit_content(self, ir: PosterIR, min_pt_floor: dict[str, float]) -> PosterIR:
        """Make every panel's content fit its box, without shrinking type.

        Two bounded mechanisms, alternating:
        1. VLM text shortening (content is the model's domain).
        2. Deterministic rebalancing: every panel's weight is set to its
           measured content height and the columns are re-split
           (geometry is code's domain) — this fixes figure-heavy panels
           that no amount of text-cutting can save, while further
           shortening handles a genuinely global deficit.

        Raises:
            TextOverflowError: If a panel still cannot fit after both
                budgets — a genuine impossibility the storyboard must fix.
        """
        ir = await self._shorten_text_rounds(ir)
        for columns in range(ir.columns, MAX_FIT_COLUMNS + 1):
            if columns != ir.columns:
                col_w = (
                    ir.size.width_mm - 2 * ir.margin_mm - (columns - 1) * ir.gutter_mm
                ) / columns
                if col_w < MIN_COLUMN_WIDTH_MM:
                    break
                data = ir.model_dump()
                data["columns"] = columns
                for p in data["panels"]:
                    p["bbox"] = None
                    p["column"] = None
                ir = assign_bboxes(PosterIR(**data), header_frac=self._header_frac)
                self._emit("columns_increased", columns=columns)
            for _ in range(WEIGHT_FIT_ATTEMPTS):
                if not self._overflowing_panels(ir):
                    return self._try_fewer_columns(ir)
                # Measured heights shift after each re-layout (text rewraps
                # at the column width, figure clamps change), so iterate.
                ir = self._rebalance_by_measured_heights(ir)
            if not self._overflowing_panels(ir):
                return self._try_fewer_columns(ir)
            ir = await self._shorten_text_rounds(ir, rounds=1)
            ir = self._rebalance_by_measured_heights(ir)
            if not self._overflowing_panels(ir):
                return self._try_fewer_columns(ir)
        overflowing = self._overflowing_panels(ir)
        if overflowing:
            panel, required, available = overflowing[0]
            raise TextOverflowError(panel.id, required, available)
        return self._try_fewer_columns(ir)

    def _try_fewer_columns(self, ir: PosterIR) -> PosterIR:
        """Undo column escalation when the content also fits in fewer,
        wider columns — escalation must not be a ratchet (a lone panel
        marooned in a sparse extra column is worse than denser columns)."""
        floor = getattr(self, "_columns_floor", 2)
        while ir.columns > floor:
            data = ir.model_dump()
            data["columns"] = ir.columns - 1
            for p in data["panels"]:
                p["bbox"] = None
                p["column"] = None
            try:
                candidate = assign_bboxes(PosterIR(**data), header_frac=self._header_frac)
            except LayoutError:
                break
            for _ in range(2):
                candidate = self._rebalance_by_measured_heights(candidate)
            if self._overflowing_panels(candidate):
                break
            ir = candidate
            self._emit("columns_reduced", columns=ir.columns)
        return ir

    def _rebalance_by_measured_heights(self, ir: PosterIR) -> PosterIR:
        """Set every body panel's weight to its measured content height.

        Weights are relative shares, so weight == required height makes
        each panel's allocation proportional to what its content actually
        needs, and the column splitter balances total required height
        across columns. Incremental per-panel bumping cannot do this:
        when all panels in a column overflow, scaling them together just
        renormalizes the same shares.
        """
        required_by_id: dict[str, float] = {}
        for panel in ir.panels_in_order():
            if panel.role == "header" or panel.bbox is None:
                continue
            required_by_id[panel.id] = (
                measure_panel_required_height_mm(panel, ir) + 2 * PANEL_PADDING_MM
            )
        data = ir.model_dump()
        for p in data["panels"]:
            if p["id"] in required_by_id:
                p["weight"] = round(required_by_id[p["id"]], 1)
            p["bbox"] = None
            p["column"] = None
        self._emit(
            "panels_rebalanced",
            weights={k: round(v) for k, v in required_by_id.items()},
        )
        new_ir = PosterIR(**data)
        return assign_bboxes(
            new_ir,
            header_frac=self._header_frac,
            column_capacity_mm=body_column_height_mm(new_ir, self._header_frac),
        )

    async def _shorten_text_rounds(self, ir: PosterIR, rounds: int = TEXT_FIT_ATTEMPTS) -> PosterIR:
        """Bounded VLM text-shortening passes over overflowing panels."""
        template = (Path(self._prompt_dir) / "poster" / "shorten.txt").read_text(encoding="utf-8")
        for _ in range(rounds):
            overflowing = self._overflowing_panels(ir)
            if not overflowing:
                return ir
            data = ir.model_dump()
            panels_by_id = {p["id"]: p for p in data["panels"]}
            for panel, required, available in overflowing:
                text_indices = [
                    i for i, el in enumerate(panel.elements) if isinstance(el, TextElement)
                ]
                if not text_indices:
                    continue  # figure-only panel: nothing to shorten here
                blocks = [panel.elements[i].content for i in text_indices]
                target_ratio = max(30, int(available / required * 90))
                prompt = template.format(
                    panel_title=panel.title or panel.id,
                    text_blocks=json.dumps(blocks, ensure_ascii=False, indent=2),
                    required_mm=f"{required:.0f}",
                    available_mm=f"{available:.0f}",
                    target_ratio=target_ratio,
                )
                raw = await self._vlm.generate(
                    prompt=prompt, response_format="json", temperature=0.3
                )
                shortened = extract_json(raw)
                if not isinstance(shortened, list) or len(shortened) != len(blocks):
                    raise ValueError(
                        f"text-fit rewrite for panel '{panel.id}' returned an invalid "
                        f"response (expected {len(blocks)} blocks): {raw[:300]!r}"
                    )
                for i, content in zip(text_indices, shortened):
                    if not str(content).strip():
                        raise ValueError(f"text-fit rewrite for panel '{panel.id}' emptied a block")
                    panels_by_id[panel.id]["elements"][i]["content"] = str(content)
                self._shorten_events = getattr(self, "_shorten_events", 0) + 1
                self._emit(
                    "panel_text_shortened",
                    panel=panel.id,
                    required_mm=round(required),
                    available_mm=round(available),
                )
            ir = PosterIR(**data)
        return ir
