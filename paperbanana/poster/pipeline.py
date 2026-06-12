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
    LayoutProposerAgent,
    PaperMetadataAgent,
    PosterContentAgent,
    PosterCriticAgent,
    PosterStylistAgent,
)
from paperbanana.poster.agents.layout_proposer import (
    format_overflow_feedback,
    format_violation_feedback,
)
from paperbanana.poster.agents.skeleton_judge import SkeletonJudgeAgent
from paperbanana.poster.convert import find_soffice, pdf_to_png, pptx_to_pdf, render_panel_crops
from paperbanana.poster.edits import EditOpError, apply_edit_ops, parse_edit_ops
from paperbanana.poster.figures import curate_figures, estimate_placed_width_mm
from paperbanana.poster.ingest import ingest_paper
from paperbanana.poster.layout import BandOverflowError, panel_width_mm, place_bands
from paperbanana.poster.legalizer import (
    LayoutProposalError,
    build_ir_from_proposal,
    repair,
    validate_proposal,
)
from paperbanana.poster.lessons import format_lessons_block, load_lessons, record_lessons
from paperbanana.poster.memory import (
    append_exemplar,
    exemplar_from_run,
    format_exemplars_block,
    load_exemplars,
    retrieve_exemplars,
)
from paperbanana.poster.preflight import (
    DEFAULT_LEGIBILITY_MIN_PT,
    render_preflight_markdown,
    run_preflight,
)
from paperbanana.poster.proposal import LayoutProposal, RepairAction
from paperbanana.poster.renderer import (
    TextOverflowError,
    measure_panel_required_at_width,
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
    Band,
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

#: Text-shortening passes per structural round before restructuring.
TEXT_FIT_ATTEMPTS = 2
#: Structural adjustment rounds (column bump now; learned re-proposal in Phase 2).
MAX_STRUCTURE_ROUNDS = 3
#: Column-count ceiling for dense posters.
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
        self.layout_proposer = LayoutProposerAgent(self._vlm, **kwargs)
        self.skeleton_judge = SkeletonJudgeAgent(self._vlm, **kwargs)
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

        # Stage 5: build IR + learned structure ---------------------------------
        ir = self._build_ir(assets, storyboard, style, asset_map, spec, qr_url)
        exemplars = []
        if self.settings.poster_exemplar_top_k > 0:
            exemplars = retrieve_exemplars(
                load_exemplars(memory_dir=self.settings.poster_memory_dir),
                orientation=ir.orientation,
                venue=venue_name,
                n_figures=len(asset_map),
                n_panels=len(storyboard.panels) + 1,
                aspect_ratio=ir.size.width_mm / ir.size.height_mm,
                k=self.settings.poster_exemplar_top_k,
            )
            self._emit(
                "exemplars_retrieved",
                ids=[e.id for e in exemplars],
                sources=[e.source for e in exemplars],
            )
        self._proposer_ctx = {
            "storyboard": storyboard,
            "layout_patterns": layout_patterns,
            "exemplars": exemplars,
        }
        ir = await self._propose_structure(ir, storyboard, layout_patterns, resume=resume)

        # Stage 6: refinement loop ----------------------------------------------
        iteration = 0
        preflight = None
        pptx_path = pdf_path = preview_path = None
        while True:
            iteration += 1
            iter_dir = self._run_dir / f"iter_{iteration}"
            iter_dir.mkdir(parents=True, exist_ok=True)

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
        if self.settings.poster_memory_promote and preflight.passed:
            await self._maybe_promote(ir, assets, final_preview)
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

        # Default band structure (header + one body band). The Phase-2
        # LayoutProposerAgent replaces this with a learned proposal
        # (multiple bands, spans, banner, callouts).
        bands = [
            Band(id="header", kind="header", order=0, columns=1),
            Band(id="body", kind="body", order=1, columns=storyboard.columns),
        ]
        for panel in panels:
            panel.band_id = "header" if panel.role == "header" else "body"
            panel.column = 0
            panel.col_span = 1

        size = PhysicalSize(width_mm=gen_w, height_mm=gen_h)
        return PosterIR(
            venue=spec.venue,
            venue_spec_year=spec.year,
            size=size,
            orientation=size.orientation,
            print_scale=required_print_scale(gen_w, gen_h),
            bands=bands,
            style=style,
            panels=panels,
            assets=asset_map,
            paper_title=assets.title,
            authors=assets.authors,
            affiliations=assets.affiliations,
        )

    async def _maybe_promote(self, ir: PosterIR, assets: PaperAssets, preview_path: Path) -> None:
        """Self-promotion: a judged-good poster's structure joins memory.

        The run's own success feeds the exemplar store (capped, tagged,
        purgeable) so future proposals learn from it — the living-beacon
        loop closing on the system's best work.
        """
        from paperbanana.poster.evaluation import evaluate_poster

        try:
            evaluation = await evaluate_poster(
                self._vlm,
                preview_path=preview_path,
                paper_context=f"{assets.title}\n\n{assets.abstract}"[:6000],
                prompt_dir=Path(self._prompt_dir),
            )
        except Exception as exc:
            # Promotion is an optimization loop, not an output guarantee:
            # a judge hiccup must not fail a finished poster. Logged, never silent.
            logger.warning("Self-promotion judging failed; skipping", error=str(exc))
            self._emit("memory_promotion_skipped", reason=str(exc)[:200])
            return
        threshold = self.settings.poster_memory_promote_min_score
        if evaluation.overall >= threshold:
            exemplar = exemplar_from_run(ir, evaluation.overall, self.run_id)
            append_exemplar(
                exemplar,
                memory_dir=self.settings.poster_memory_dir,
                max_self_generated=self.settings.poster_memory_max_self,
            )
            self._emit("memory_promoted", exemplar=exemplar.id, judge_overall=evaluation.overall)
        else:
            self._emit(
                "memory_promotion_declined",
                judge_overall=evaluation.overall,
                threshold=threshold,
            )

    async def _propose_structure(
        self,
        draft_ir: PosterIR,
        storyboard: Storyboard,
        layout_patterns: str,
        resume: bool = False,
    ) -> PosterIR:
        """Obtain a legal learned structure and apply it to the draft IR."""
        cached = self._load_stage("layout_proposal", resume)
        if cached is not None:
            proposal = LayoutProposal(**cached["proposal"])
            repairs = [RepairAction(**r) for r in cached["repairs"]]
            rounds = cached.get("rounds", 0)
        else:
            proposal, repairs, rounds, rank_scores = await self._best_of_n_proposals(
                storyboard, draft_ir, layout_patterns
            )
            self._save_stage(
                "layout_proposal",
                json.dumps(
                    {
                        "proposal": json.loads(proposal.model_dump_json()),
                        "repairs": [json.loads(r.model_dump_json()) for r in repairs],
                        "rounds": rounds,
                        "rank_scores": rank_scores,
                    },
                    indent=2,
                ),
            )
        ir = build_ir_from_proposal(
            draft_ir,
            proposal,
            storyboard,
            repairs,
            reproposal_rounds=rounds,
            exemplar_ids=[e.id for e in self._proposer_ctx.get("exemplars", [])],
        )
        self._emit(
            "layout_proposed",
            bands=[(b.kind, b.columns) for b in ir.bands_in_order()],
            banner=proposal.use_banner,
            hero=proposal.hero_figure_id,
            callouts=len(proposal.callouts),
            repairs=len(repairs),
            rounds=rounds,
        )
        return ir

    async def _best_of_n_proposals(
        self,
        storyboard: Storyboard,
        draft_ir: PosterIR,
        layout_patterns: str,
    ) -> tuple[LayoutProposal, list[RepairAction], int, Optional[dict]]:
        """Generate N structures, rank cheap skeleton previews, keep the best.

        Candidates with fatal violations are dropped (logged); if none
        survive, the single-proposal re-prompt loop takes over. With one
        survivor there is nothing to rank.
        """
        from paperbanana.poster.skeleton import placeholder_assets, render_skeleton_preview

        n = self.settings.poster_layout_proposals
        candidates: list[tuple[LayoutProposal, list[RepairAction]]] = []
        for i in range(n):
            proposal = await self.layout_proposer.run(
                storyboard=storyboard,
                assets=draft_ir.assets,
                size=draft_ir.size,
                margin_mm=draft_ir.margin_mm,
                gutter_mm=draft_ir.gutter_mm,
                layout_patterns=layout_patterns,
                exemplars_block=format_exemplars_block(
                    getattr(self, "_proposer_ctx", {}).get("exemplars", [])
                ),
                proposal_index=i,
            )
            violations = validate_proposal(
                proposal,
                storyboard,
                set(draft_ir.assets),
                draft_ir.size.width_mm,
                draft_ir.margin_mm,
                draft_ir.gutter_mm,
            )
            fatal = [v for v in violations if v.fatal]
            if fatal:
                self._emit(
                    "candidate_dropped",
                    index=i,
                    fatal=[f"{v.code}:{v.target}" for v in fatal],
                )
                continue
            candidates.append(
                repair(
                    proposal,
                    violations,
                    draft_ir.size.width_mm,
                    draft_ir.margin_mm,
                    draft_ir.gutter_mm,
                )
            )
        if not candidates:
            proposal, repairs, rounds = await self._obtain_legal_proposal(
                storyboard, draft_ir, layout_patterns, feedback=""
            )
            return proposal, repairs, rounds, None
        if len(candidates) == 1:
            proposal, repairs = candidates[0]
            return proposal, repairs, 0, None

        # Rank: skeleton previews with placeholder figures, one judge call.
        skeleton_dir = self._run_dir / "skeletons"
        placeholders = placeholder_assets(draft_ir.assets, skeleton_dir / "assets")
        previews: list = []
        preview_indices: list[int] = []
        penalties: dict[int, float] = {}
        for i, (proposal, repairs) in enumerate(candidates):
            candidate_ir = build_ir_from_proposal(
                draft_ir.model_copy(update={"assets": placeholders}),
                proposal,
                storyboard,
                repairs,
            )
            measured = self._measure_all(candidate_ir)
            png, penalty = render_skeleton_preview(
                candidate_ir, measured, skeleton_dir, self._soffice, label=f"candidate_{i}"
            )
            if png is None:
                penalties[i] = penalty
            else:
                previews.append(Image.open(png).convert("RGB"))
                preview_indices.append(i)
        summary = "; ".join(
            f"{p.id}({p.role}, figs={len(p.figure_ids)})" for p in storyboard.panels
        )
        if not previews:
            # Every candidate overflowed: keep the least-bad one; the fit
            # loop's shortening + re-proposal machinery takes it from here.
            best = min(penalties, key=penalties.get)
            self._emit("skeleton_rank_all_overflow", chosen=best, penalties=penalties)
            proposal, repairs = candidates[best]
            return proposal, repairs, 0, {str(k): -v for k, v in penalties.items()}
        ranking = await self.skeleton_judge.run(
            previews=previews,
            storyboard_summary=summary,
            penalties=penalties,
        )
        # The judge ranks the *rendered* candidates by image position; map back.
        winner_pos = min(max(ranking.winner, 0), len(preview_indices) - 1)
        winner = preview_indices[winner_pos]
        self._save_stage(
            "skeleton_rank",
            json.dumps(
                {
                    "winner": winner,
                    "scores": ranking.scores,
                    "penalties": penalties,
                    "rationale": ranking.rationale,
                },
                indent=2,
            ),
        )
        self._emit("skeletons_ranked", winner=winner, scores=ranking.scores)
        proposal, repairs = candidates[winner]
        return proposal, repairs, 0, ranking.scores

    async def _obtain_legal_proposal(
        self,
        storyboard: Storyboard,
        draft_ir: PosterIR,
        layout_patterns: str,
        feedback: str,
    ) -> tuple[LayoutProposal, list[RepairAction], int]:
        """Propose, validate, repair; fatal violations re-prompt, bounded."""
        budget = self.settings.poster_proposal_repair_rounds
        fatal: list = []
        for attempt in range(budget + 1):
            ctx = getattr(self, "_proposer_ctx", {})
            proposal = await self.layout_proposer.run(
                storyboard=storyboard,
                assets=draft_ir.assets,
                size=draft_ir.size,
                margin_mm=draft_ir.margin_mm,
                gutter_mm=draft_ir.gutter_mm,
                layout_patterns=layout_patterns,
                exemplars_block=format_exemplars_block(ctx.get("exemplars", [])),
                violation_feedback=feedback,
                proposal_index=attempt,
            )
            violations = validate_proposal(
                proposal,
                storyboard,
                set(draft_ir.assets),
                draft_ir.size.width_mm,
                draft_ir.margin_mm,
                draft_ir.gutter_mm,
            )
            fatal = [v for v in violations if v.fatal]
            if not fatal:
                repaired, repairs = repair(
                    proposal,
                    violations,
                    draft_ir.size.width_mm,
                    draft_ir.margin_mm,
                    draft_ir.gutter_mm,
                )
                return repaired, repairs, attempt
            feedback = format_violation_feedback(fatal)
            self._emit(
                "proposal_rejected",
                attempt=attempt,
                fatal=[f"{v.code}:{v.target}" for v in fatal],
            )
        raise LayoutProposalError(fatal, budget + 1)

    def _measure_all(self, ir: PosterIR) -> dict[str, float]:
        """Measured content height per panel at its structural width."""
        return {
            panel.id: measure_panel_required_at_width(panel, ir, panel_width_mm(ir, panel))
            for panel in ir.panels
        }

    async def _fit_content(self, ir: PosterIR, min_pt_floor: dict[str, float]) -> PosterIR:
        """Fit measured content into the proposed structure, without
        shrinking type.

        Bounded mechanisms, in order per structural round: measurement +
        skyline placement; VLM text shortening targeted at overflowing
        columns; then a LEARNED structural re-proposal — the proposer is
        re-prompted with the measured per-column loads and restructures
        (more columns, moved panels, dropped hero span). No hand-coded
        structural adjustment remains.

        Raises:
            TextOverflowError: When content cannot fit after all budgets.
        """
        last_overflow: Optional[BandOverflowError] = None
        for structural_round in range(MAX_STRUCTURE_ROUNDS):
            for _ in range(TEXT_FIT_ATTEMPTS + 1):
                measured = self._measure_all(ir)
                try:
                    return place_bands(ir, measured)
                except BandOverflowError as exc:
                    last_overflow = exc
                    ir = await self._shorten_overflowing(ir, measured, exc)
            ctx = getattr(self, "_proposer_ctx", None)
            if ctx is None or last_overflow is None:
                break
            feedback = format_overflow_feedback(
                last_overflow.overflows, last_overflow.page_deficit_mm
            )
            try:
                proposal, repairs, _ = await self._obtain_legal_proposal(
                    ctx["storyboard"], ir, ctx["layout_patterns"], feedback=feedback
                )
            except LayoutProposalError:
                break
            ir = build_ir_from_proposal(
                ir,
                proposal,
                ctx["storyboard"],
                repairs,
                reproposal_rounds=structural_round + 1,
            )
            self._emit(
                "structure_reproposed",
                round=structural_round + 1,
                bands=[(b.kind, b.columns) for b in ir.bands_in_order()],
            )
        if last_overflow is not None and last_overflow.overflows:
            worst = last_overflow.overflows[0]
            raise TextOverflowError(
                f"{worst.band_id}/col{worst.column}", worst.required_mm, worst.available_mm
            )
        raise TextOverflowError("poster", 0.0, 0.0)

    async def _shorten_overflowing(
        self,
        ir: PosterIR,
        measured: dict[str, float],
        overflow: BandOverflowError,
    ) -> PosterIR:
        """One VLM text-shortening pass over panels in overflowing columns."""
        template = (Path(self._prompt_dir) / "poster" / "shorten.txt").read_text(encoding="utf-8")
        hot = {(o.band_id, o.column) for o in overflow.overflows}
        ratios = {
            (o.band_id, o.column): (
                max(0.3, o.available_mm / o.required_mm) if o.required_mm > 0 else 1.0
            )
            for o in overflow.overflows
        }
        data = ir.model_dump()
        panels_by_id = {p["id"]: p for p in data["panels"]}
        shortened_any = False
        for panel in ir.panels_in_order():
            spanned = {
                (panel.band_id, c) for c in range(panel.column, panel.column + panel.col_span)
            }
            touched = spanned & hot
            if not touched:
                continue
            text_indices = [i for i, el in enumerate(panel.elements) if isinstance(el, TextElement)]
            if not text_indices:
                continue  # figure-only panel: nothing to shorten here
            ratio = min(ratios[key] for key in touched)
            required = measured[panel.id]
            available = required * ratio
            blocks = [panel.elements[i].content for i in text_indices]
            prompt = template.format(
                panel_title=panel.title or panel.id,
                text_blocks=json.dumps(blocks, ensure_ascii=False, indent=2),
                required_mm=f"{required:.0f}",
                available_mm=f"{available:.0f}",
                target_ratio=max(30, int(ratio * 90)),
            )
            raw = await self._vlm.generate(prompt=prompt, response_format="json", temperature=0.3)
            rewritten = extract_json(raw)
            if not isinstance(rewritten, list) or len(rewritten) != len(blocks):
                raise ValueError(
                    f"text-fit rewrite for panel '{panel.id}' returned an invalid "
                    f"response (expected {len(blocks)} blocks): {raw[:300]!r}"
                )
            for i, content in zip(text_indices, rewritten):
                if not str(content).strip():
                    raise ValueError(f"text-fit rewrite for panel '{panel.id}' emptied a block")
                panels_by_id[panel.id]["elements"][i]["content"] = str(content)
            shortened_any = True
            self._shorten_events = getattr(self, "_shorten_events", 0) + 1
            self._emit(
                "panel_text_shortened",
                panel=panel.id,
                required_mm=round(required),
                available_mm=round(available),
            )
        if not shortened_any:
            return ir
        for p in data["panels"]:
            p["bbox"] = None
        return PosterIR(**data)
