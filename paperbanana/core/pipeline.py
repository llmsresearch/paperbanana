"""Main PaperBanana pipeline orchestration."""

from __future__ import annotations

import asyncio
import datetime
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import structlog
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from paperbanana.agents.caption import CaptionAgent
from paperbanana.agents.critic import CriticAgent
from paperbanana.agents.ir_planner import IRPlannerAgent
from paperbanana.agents.optimizer import InputOptimizerAgent
from paperbanana.agents.planner import PlannerAgent
from paperbanana.agents.retriever import RetrieverAgent
from paperbanana.agents.structurer import StructurerAgent
from paperbanana.agents.stylist import StylistAgent
from paperbanana.agents.tikz_exporter import TikZExporterAgent
from paperbanana.agents.visualizer import VisualizerAgent
from paperbanana.core.config import Settings
from paperbanana.core.cost_tracker import CostTracker
from paperbanana.core.diagram_ir import (
    extract_diagram_ir,
    format_diagram_ir_for_regeneration,
    save_raster_wrapped_svg,
    save_svg_from_ir,
)
from paperbanana.core.prompt_recorder import PromptRecorder
from paperbanana.core.types import (
    CritiqueResult,
    DiagramIR,
    DiagramType,
    GenerationInput,
    GenerationOutput,
    IterationRecord,
    PipelineProgressEvent,
    PipelineProgressStage,
    ReferenceExample,
    RunMetadata,
)
from paperbanana.core.utils import (
    ensure_dir,
    find_prompt_dir,
    generate_run_id,
    load_image,
    save_image,
    save_json,
)
from paperbanana.guidelines.methodology import load_methodology_guidelines
from paperbanana.guidelines.plots import load_plot_guidelines
from paperbanana.guidelines.venues import VenuePack, resolve_venue, select_aspect_ratio
from paperbanana.providers.registry import ProviderRegistry
from paperbanana.reference.exemplar_retrieval import (
    ExemplarRetrievalError,
    ExternalExemplarRetriever,
    map_external_hits_to_examples,
)
from paperbanana.reference.store import ReferenceStore
from paperbanana.vector.graphviz_render import (
    diagram_ir_to_dot,
    find_dot_executable,
    render_dot_to_file,
)

logger = structlog.get_logger()

_ssl_skip_applied = False


def _get_version() -> str:
    """Return the installed PaperBanana version, or 'unknown' if unavailable."""
    try:
        from importlib.metadata import version

        return version("paperbanana")
    except Exception:
        return "unknown"


def _emit_progress(
    callback: Optional[Callable[[PipelineProgressEvent], None]],
    event: PipelineProgressEvent,
) -> None:
    """Invoke progress callback if set; swallow errors so pipeline is not affected."""
    if callback is None:
        return
    try:
        callback(event)
    except Exception:
        logger.warning("Progress callback failed", stage=event.stage, exc_info=True)


async def _call_with_retry(label, fn, *args, max_attempts=3, **kwargs):
    """Retry an async agent call with exponential backoff.

    Complements provider-level retries by catching agent-level failures
    (e.g. response parsing errors, unexpected formats) that survive
    the lower-level HTTP retry layer.
    """
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(min=2, max=30),
        reraise=True,
    ):
        with attempt:
            attempt_num = attempt.retry_state.attempt_number
            if attempt_num > 1:
                logger.warning(
                    f"Retrying {label}",
                    attempt=attempt_num,
                    max_attempts=max_attempts,
                )
            return await fn(*args, **kwargs)


def _apply_ssl_skip():
    """Disable SSL verification globally for corporate proxy environments."""
    global _ssl_skip_applied
    if _ssl_skip_applied:
        return
    _ssl_skip_applied = True

    import ssl

    logger.warning("SSL verification disabled via SKIP_SSL_VERIFICATION=true")

    # Handle stdlib ssl (urllib, http.client)
    ssl._create_default_https_context = ssl._create_unverified_context

    # Handle httpx
    try:
        import httpx

        _orig_client_init = httpx.Client.__init__
        _orig_async_init = httpx.AsyncClient.__init__

        def _patched_client_init(self, *args, **kwargs):
            kwargs["verify"] = False
            _orig_client_init(self, *args, **kwargs)

        def _patched_async_init(self, *args, **kwargs):
            kwargs["verify"] = False
            _orig_async_init(self, *args, **kwargs)

        httpx.Client.__init__ = _patched_client_init
        httpx.AsyncClient.__init__ = _patched_async_init
    except ImportError:
        pass

    # Suppress urllib3 InsecureRequestWarning
    try:
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except ImportError:
        pass


class PaperBananaPipeline:
    """Main orchestration pipeline for academic illustration generation.

    Implements the two-phase process:
    1. Linear Planning: Retriever -> Planner -> Stylist
    2. Iterative Refinement: Visualizer <-> Critic (up to N iterations)
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        vlm_client=None,
        image_gen_fn=None,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        """Initialize the pipeline.

        Args:
            settings: Configuration settings. If None, loads from env/defaults.
            vlm_client: Optional pre-configured VLM client (for HF Spaces demo).
            image_gen_fn: Optional image generation function (for HF Spaces demo).
        """
        self.settings = settings or Settings()
        self.run_id = generate_run_id()
        self._progress_callback = progress_callback

        if self.settings.skip_ssl_verification:
            _apply_ssl_skip()

        # Prompt recorder (writes formatted prompts to outputs/<run_id>/prompts/)
        self._prompt_recorder = None
        if self.settings.save_prompts:
            self._prompt_recorder = PromptRecorder(run_dir_provider=lambda: self._run_dir)

        # Initialize providers
        if vlm_client is not None:
            # Demo mode: use provided clients
            self._vlm = vlm_client
            self._image_gen = image_gen_fn
            self._demo_mode = True
        else:
            self._vlm = ProviderRegistry.create_vlm(self.settings)
            self._image_gen = ProviderRegistry.create_image_gen(self.settings)
            self._demo_mode = False

        # Cost tracking (optional — active when budget is set or always for reporting)
        self._cost_tracker: CostTracker | None = None
        if not self._demo_mode:
            self._cost_tracker = CostTracker(budget=self.settings.budget_usd)
            if hasattr(self._vlm, "cost_tracker"):
                self._vlm.cost_tracker = self._cost_tracker
            if hasattr(self._image_gen, "cost_tracker"):
                self._image_gen.cost_tracker = self._cost_tracker

        # Load reference store (resolves cache → built-in fallback)
        self.reference_store = ReferenceStore.from_settings(self.settings)
        self._external_exemplar_retriever: ExternalExemplarRetriever | None = None
        if self.settings.exemplar_retrieval_enabled and self.settings.exemplar_retrieval_endpoint:
            self._external_exemplar_retriever = ExternalExemplarRetriever(
                endpoint=self.settings.exemplar_retrieval_endpoint,
                timeout_seconds=self.settings.exemplar_retrieval_timeout_seconds,
                max_retries=self.settings.exemplar_retrieval_max_retries,
            )

        # Load guidelines (venue-aware resolution: built-in packs, then user packs)
        guidelines_path = self.settings.guidelines_path
        venue = self.settings.venue
        venue_dir = self.settings.venue_dir
        self._venue_pack: VenuePack | None = None
        if venue and venue != "custom":
            self._venue_pack = resolve_venue(
                venue, builtin_dir=guidelines_path, extra_dir=venue_dir
            )
        self._methodology_guidelines = load_methodology_guidelines(
            guidelines_path, venue=venue, venue_dir=venue_dir
        )
        self._plot_guidelines = load_plot_guidelines(
            guidelines_path, venue=venue, venue_dir=venue_dir
        )
        if self._venue_pack and self._venue_pack.config.fonts:
            font_note = (
                "\n\n## Venue Font Preferences\n\n"
                f"Preferred font families for this venue: "
                f"{', '.join(self._venue_pack.config.fonts)}.\n"
            )
            self._methodology_guidelines += font_note
            self._plot_guidelines += font_note

        # Initialize agents
        prompt_dir = self._find_prompt_dir()
        self._prompt_dir = prompt_dir
        self.optimizer = InputOptimizerAgent(
            self._vlm, prompt_dir=prompt_dir, prompt_recorder=self._prompt_recorder
        )
        self.retriever = RetrieverAgent(
            self._vlm, prompt_dir=prompt_dir, prompt_recorder=self._prompt_recorder
        )
        self.planner = PlannerAgent(
            self._vlm, prompt_dir=prompt_dir, prompt_recorder=self._prompt_recorder
        )
        self.ir_planner = IRPlannerAgent(
            self._vlm, prompt_dir=prompt_dir, prompt_recorder=self._prompt_recorder
        )
        self.stylist = StylistAgent(
            self._vlm,
            guidelines=self._methodology_guidelines,
            prompt_dir=prompt_dir,
            prompt_recorder=self._prompt_recorder,
        )
        self.structurer = StructurerAgent(
            self._vlm, prompt_dir=prompt_dir, prompt_recorder=self._prompt_recorder
        )
        self.visualizer = VisualizerAgent(
            self._image_gen,
            self._vlm,
            prompt_dir=prompt_dir,
            output_dir=str(self._run_dir),
            prompt_recorder=self._prompt_recorder,
            output_resolution=self.settings.output_resolution,
            image_quality=self.settings.image_quality,
        )
        self.critic = CriticAgent(
            self._vlm, prompt_dir=prompt_dir, prompt_recorder=self._prompt_recorder
        )
        self.caption_agent = CaptionAgent(
            self._vlm, prompt_dir=prompt_dir, prompt_recorder=self._prompt_recorder
        )
        self.tikz_exporter = TikZExporterAgent(
            self._vlm, prompt_dir=prompt_dir, prompt_recorder=self._prompt_recorder
        )
        self._sync_run_scoped_agents()

        logger.info(
            "Pipeline initialized",
            run_id=self.run_id,
            vlm=getattr(self._vlm, "name", "custom"),
            image_gen=getattr(self._image_gen, "name", "custom"),
        )

    def _emit_progress(self, event: str, **payload: Any) -> None:
        """Emit a structured progress event.

        Events are best-effort: any callback error is logged and ignored so that
        progress consumers cannot break the main pipeline.
        """
        # structlog uses the positional message as the "event" field internally;
        # avoid passing a keyword named "event" to prevent collisions.
        logger.info("progress_event", progress_event=event, **payload)
        if self._progress_callback is not None:
            try:
                self._progress_callback(event, payload)
            except Exception:
                logger.warning("Progress callback raised", progress_event=event)

    def _check_budget(self, context: str, iteration: int | None = None) -> bool:
        """Return True if the cost tracker is over budget, logging a warning."""
        if not (self._cost_tracker and self._cost_tracker.is_over_budget):
            return False
        if iteration is not None:
            logger.warning(
                f"Budget exceeded {context}, stopping early",
                iteration=iteration,
            )
        else:
            logger.warning(f"Budget exceeded {context}, skipping iterations")
        return True

    @property
    def _run_dir(self) -> Path:
        """Directory for this run's outputs."""
        return ensure_dir(Path(self.settings.output_dir) / self.run_id)

    def _find_prompt_dir(self) -> str:
        """Find the prompts directory, preferring settings.prompt_dir if set."""
        if self.settings.prompt_dir:
            return self.settings.prompt_dir
        return find_prompt_dir()

    def _sync_run_scoped_agents(self) -> None:
        self.visualizer.set_output_dir(self._run_dir)

    async def _generate_caption(
        self,
        *,
        image_path: str,
        source_context: str,
        intent: str,
        description: str,
        diagram_type: DiagramType,
        progress_callback: Optional[Callable[[PipelineProgressEvent], None]],
    ) -> tuple[Optional[str], float]:
        """Run the CaptionAgent if ``generate_caption`` is enabled.

        Returns:
            (generated_caption, caption_seconds).  Both default to
            ``(None, 0.0)`` when the setting is off or the agent fails.
        """
        if not self.settings.generate_caption:
            return None, 0.0

        _emit_progress(
            progress_callback,
            PipelineProgressEvent(
                stage=PipelineProgressStage.CAPTION_START,
                message="Generating figure caption",
            ),
        )
        self._emit_progress("caption_started")
        generated_caption: Optional[str] = None
        caption_start = time.perf_counter()
        try:
            generated_caption = await self.caption_agent.run(
                image_path=image_path,
                source_context=source_context,
                intent=intent,
                description=description,
                diagram_type=diagram_type,
            )
        except Exception as e:
            logger.warning("Caption generation failed", error=str(e))
        caption_seconds = time.perf_counter() - caption_start
        _emit_progress(
            progress_callback,
            PipelineProgressEvent(
                stage=PipelineProgressStage.CAPTION_END,
                message="Caption generated",
                seconds=caption_seconds,
                extra={"caption": generated_caption},
            ),
        )
        self._emit_progress(
            "caption_completed",
            seconds=round(caption_seconds, 1),
            caption=generated_caption,
        )
        return generated_caption, caption_seconds

    def _build_final_output(
        self,
        iterations: list[IterationRecord],
        run_dir: Path,
        empty_warning: str,
    ) -> str:
        """Derive the final output image path from the last iteration.

        Resolves the output format and file extension, constructs the
        output path, and — for raster formats — loads the last
        iteration's image and saves it in the requested format.  SVG
        output requires caller-side handling after this method returns.

        Returns:
            The output file path, or ``""`` when *iterations* is empty.
        """
        output_format = getattr(self.settings, "output_format", "png").lower()
        ext = "jpg" if output_format == "jpeg" else output_format
        final_output_path = str(run_dir / f"final_output.{ext}")

        if iterations:
            if output_format != "svg":
                final_image = iterations[-1].image_path
                img = load_image(final_image)
                save_image(img, final_output_path, format=output_format)
        else:
            final_output_path = ""
            logger.warning(empty_warning, run_id=self.run_id)

        return final_output_path

    async def _visualize_with_rollback(
        self,
        *,
        iteration: int,
        rollback_to: Optional[int],
        run_dir: Path,
        visualizer: Optional[VisualizerAgent] = None,
        **visualizer_kwargs: Any,
    ) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Run the visualizer for one refinement iteration, rolling back on failure.

        When the visualizer fails after retries (or returns no image) and a
        previous best image exists (``rollback_to`` is not ``None``), the
        failure is logged and recorded, and ``(None, rollback_info)`` is
        returned so the caller stops the loop and keeps the previous best
        image as the final output.  Without a prior image (first round) the
        exception propagates, preserving existing error behavior.

        ``visualizer`` selects the agent instance to run (defaults to
        ``self.visualizer``); multi-candidate fan-out passes a per-branch
        instance so concurrent branches never share output paths.
        """
        visualizer = visualizer or self.visualizer
        try:
            image_path = await _call_with_retry(
                "visualizer",
                visualizer.run,
                iteration=iteration,
                **visualizer_kwargs,
            )
            if not image_path:
                raise RuntimeError("Visualizer returned no image path")
        except Exception as e:
            if rollback_to is None:
                raise
            rollback_info: Dict[str, Any] = {
                "rollback_occurred": True,
                "failed_iteration": iteration,
                "rolled_back_to_iteration": rollback_to,
                "stage": "visualizer",
                "error": str(e),
            }
            logger.warning(
                "Visualizer failed after retries; rolling back to previous best image",
                iteration=iteration,
                rolled_back_to_iteration=rollback_to,
                error=str(e),
            )
            self._emit_progress(
                "visualizer_rollback",
                iteration=iteration,
                rolled_back_to_iteration=rollback_to,
                error=str(e),
            )
            if self.settings.save_iterations:
                save_json(
                    {
                        "failed": True,
                        "stage": "visualizer",
                        "error": str(e),
                        "rolled_back_to_iteration": rollback_to,
                    },
                    ensure_dir(run_dir / f"iter_{iteration}") / "failure.json",
                )
            return None, rollback_info
        return image_path, None

    async def _run_refinement_branch(
        self,
        *,
        input: GenerationInput,
        initial_description: str,
        effective_ratio: Optional[str],
        vector_formats: Optional[list[str]],
        total_iters: int,
        run_dir: Path,
        visualizer: VisualizerAgent,
        seed: Optional[int],
        candidate_index: Optional[int] = None,
        progress_callback: Optional[Callable[[PipelineProgressEvent], None]] = None,
    ) -> Dict[str, Any]:
        """Run one Phase-2 Visualizer<->Critic refinement loop.

        Used both for the standard single run (``candidate_index=None``,
        ``run_dir=self._run_dir``, ``visualizer=self.visualizer``) and for
        each branch of multi-candidate fan-out (per-candidate run dir,
        per-branch ``VisualizerAgent`` instance, per-candidate seed).

        The shared ``CostTracker`` is safe to accumulate from concurrent
        branches (single event loop, synchronous appends), so the budget
        guard stops every branch at its next checkpoint once exceeded.
        Per-agent cost attribution may interleave across branches; totals
        and budget enforcement are unaffected.
        """

        def _extra(**payload: Any) -> Dict[str, Any]:
            if candidate_index is not None:
                payload["candidate"] = candidate_index
            return payload

        current_description = initial_description
        iterations: list[IterationRecord] = []
        iteration_timings: list[dict[str, float | int]] = []
        rollback_info: Optional[Dict[str, Any]] = None

        # Check budget after pre-iteration phases (retriever, planner, stylist)
        budget_exceeded = self._check_budget("after planning phases")

        for i in range(total_iters):
            if budget_exceeded:
                break

            iter_index = i + 1
            logger.info(
                f"Phase 2: Iteration {iter_index}/{total_iters}"
                + (" (auto)" if self.settings.auto_refine else "")
                + (f" [candidate {candidate_index}]" if candidate_index is not None else "")
            )
            self._emit_progress(
                "iteration_started",
                **_extra(
                    iteration=iter_index,
                    total_iterations=total_iters,
                    auto=self.settings.auto_refine,
                ),
            )

            # Step 4: Visualizer — generate image
            if self._cost_tracker:
                self._cost_tracker.set_agent("visualizer")
            _emit_progress(
                progress_callback,
                PipelineProgressEvent(
                    stage=PipelineProgressStage.VISUALIZER_START,
                    message=f"Generating image (iteration {iter_index}/{total_iters})",
                    iteration=iter_index,
                    extra=_extra(total_iterations=total_iters),
                ),
            )
            visualizer_start = time.perf_counter()
            image_path, rollback_info = await self._visualize_with_rollback(
                iteration=iter_index,
                rollback_to=iterations[-1].iteration if iterations else None,
                run_dir=run_dir,
                visualizer=visualizer,
                description=current_description,
                diagram_type=input.diagram_type,
                raw_data=input.raw_data,
                seed=seed,
                aspect_ratio=effective_ratio,
                vector_formats=vector_formats,
                sketch_guided=bool(input.input_images),
            )
            visualizer_seconds = time.perf_counter() - visualizer_start
            if image_path is None:
                _emit_progress(
                    progress_callback,
                    PipelineProgressEvent(
                        stage=PipelineProgressStage.VISUALIZER_END,
                        message=(
                            f"Visualizer iteration {iter_index} failed; keeping previous best image"
                        ),
                        seconds=visualizer_seconds,
                        iteration=iter_index,
                        extra=_extra(rollback=True, error=rollback_info["error"]),
                    ),
                )
                break
            _emit_progress(
                progress_callback,
                PipelineProgressEvent(
                    stage=PipelineProgressStage.VISUALIZER_END,
                    message=f"Visualizer iteration {iter_index} done",
                    seconds=visualizer_seconds,
                    iteration=iter_index,
                    extra=_extra(),
                ),
            )
            logger.info(
                f"[Visualizer] Iteration {iter_index}/{total_iters} done",
                seconds=round(visualizer_seconds, 1),
            )
            self._emit_progress(
                "visualizer_completed",
                **_extra(iteration=iter_index, seconds=round(visualizer_seconds, 1)),
            )

            # Step 5: Critic — evaluate and provide feedback
            if self._cost_tracker:
                self._cost_tracker.set_agent("critic")
            _emit_progress(
                progress_callback,
                PipelineProgressEvent(
                    stage=PipelineProgressStage.CRITIC_START,
                    message="Critic reviewing",
                    iteration=iter_index,
                    extra=_extra(),
                ),
            )
            critic_start = time.perf_counter()
            try:
                critique = await _call_with_retry(
                    "critic",
                    self.critic.run,
                    image_path=image_path,
                    description=current_description,
                    source_context=input.source_context,
                    caption=input.communicative_intent,
                    diagram_type=input.diagram_type,
                )
            except Exception:
                logger.warning(
                    "Critic failed after retries, accepting current image",
                    iteration=iter_index,
                    exc_info=True,
                )
                critique = CritiqueResult()
            critic_seconds = time.perf_counter() - critic_start
            _emit_progress(
                progress_callback,
                PipelineProgressEvent(
                    stage=PipelineProgressStage.CRITIC_END,
                    message="Critic done",
                    seconds=critic_seconds,
                    iteration=iter_index,
                    extra=_extra(
                        needs_revision=critique.needs_revision,
                        summary=critique.summary,
                        critic_suggestions=critique.critic_suggestions[:3],
                    ),
                ),
            )
            self._emit_progress(
                "critic_completed",
                **_extra(
                    iteration=iter_index,
                    seconds=round(critic_seconds, 1),
                    needs_revision=critique.needs_revision,
                ),
            )

            iteration_record = IterationRecord(
                iteration=iter_index,
                description=current_description,
                image_path=image_path,
                critique=critique,
            )
            iteration_timings.append(
                {
                    "iteration": iter_index,
                    "visualizer_seconds": visualizer_seconds,
                    "critic_seconds": critic_seconds,
                }
            )
            iterations.append(iteration_record)

            # Save iteration artifacts
            if self.settings.save_iterations:
                iter_dir = ensure_dir(run_dir / f"iter_{iter_index}")
                save_json(
                    {
                        "description": current_description,
                        "critique": critique.model_dump(),
                    },
                    iter_dir / "details.json",
                )

            # Check if revision needed
            if critique.needs_revision and critique.revised_description:
                logger.info(
                    "Revision needed",
                    iteration=iter_index,
                    summary=critique.summary,
                )
                current_description = critique.revised_description
            else:
                logger.info(
                    "No further revision needed",
                    iteration=iter_index,
                    summary=critique.summary,
                )
                self._emit_progress(
                    "iteration_completed",
                    **_extra(
                        iteration=iter_index,
                        total_iterations=len(iterations),
                        needs_revision=critique.needs_revision,
                    ),
                )
                break

            self._emit_progress(
                "iteration_completed",
                **_extra(
                    iteration=iter_index,
                    total_iterations=len(iterations),
                    needs_revision=critique.needs_revision,
                ),
            )

            # Check budget between iterations
            if self._check_budget("between iterations", iteration=iter_index):
                budget_exceeded = True
                break

        return {
            "iterations": iterations,
            "iteration_timings": iteration_timings,
            "rollback_info": rollback_info,
            "final_description": current_description,
            "budget_exceeded": budget_exceeded,
            "vector_paths": dict(visualizer._last_vector_paths),
        }

    async def _fan_out_candidates(
        self,
        *,
        input: GenerationInput,
        initial_description: str,
        effective_ratio: Optional[str],
        vector_formats: Optional[list[str]],
        total_iters: int,
        num_candidates: int,
        progress_callback: Optional[Callable[[PipelineProgressEvent], None]] = None,
    ) -> tuple[Dict[str, Any], int, list[Dict[str, Any]]]:
        """Run Phase 2 ``num_candidates`` times in parallel.

        Each branch gets its own output directory
        (``<run_dir>/candidates/cand_<i>``), its own ``VisualizerAgent``
        instance (the agent keeps per-run mutable state: ``output_dir`` and
        ``_last_vector_paths``), and a deterministic seed offset
        (``settings.seed + i - 1`` when a seed is set, else ``None``).

        A failed branch does not affect the others; if every branch fails,
        a ``RuntimeError`` is raised. Returns ``(primary_branch,
        primary_index, candidates_meta)`` where the primary is the
        lowest-index successful candidate (candidate 1 unless it failed).
        """
        candidates_root = ensure_dir(self._run_dir / "candidates")
        seeds: list[Optional[int]] = []
        tasks = []
        for idx in range(1, num_candidates + 1):
            cand_dir = ensure_dir(candidates_root / f"cand_{idx}")
            seed = self.settings.seed + (idx - 1) if self.settings.seed is not None else None
            seeds.append(seed)
            branch_visualizer = VisualizerAgent(
                self._image_gen,
                self._vlm,
                prompt_dir=self._prompt_dir,
                output_dir=str(cand_dir),
                prompt_recorder=self._prompt_recorder,
                output_resolution=self.settings.output_resolution,
                image_quality=self.settings.image_quality,
            )
            tasks.append(
                self._run_refinement_branch(
                    input=input,
                    initial_description=initial_description,
                    effective_ratio=effective_ratio,
                    vector_formats=vector_formats,
                    total_iters=total_iters,
                    run_dir=cand_dir,
                    visualizer=branch_visualizer,
                    seed=seed,
                    candidate_index=idx,
                    progress_callback=progress_callback,
                )
            )

        self._emit_progress("candidates_fanout_started", num_candidates=num_candidates)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Candidate final outputs are always raster; the run-root final
        # output carries the canonical requested format (including svg).
        output_format = getattr(self.settings, "output_format", "png").lower()
        cand_format = "png" if output_format == "svg" else output_format
        cand_ext = "jpg" if cand_format == "jpeg" else cand_format

        candidates_meta: list[Dict[str, Any]] = []
        errors: list[tuple[int, BaseException]] = []
        primary: Optional[Dict[str, Any]] = None
        primary_index = 0
        for idx, result in enumerate(results, start=1):
            if isinstance(result, BaseException):
                errors.append((idx, result))
                logger.warning(
                    "Candidate branch failed",
                    candidate=idx,
                    error=str(result),
                )
                self._emit_progress("candidate_failed", candidate=idx, error=str(result))
                candidates_meta.append(
                    {
                        "index": idx,
                        "seed": seeds[idx - 1],
                        "image_path": None,
                        "iterations": 0,
                        "critic_satisfied": False,
                        "error": str(result),
                    }
                )
                continue

            image_path: Optional[str] = None
            if result["iterations"]:
                cand_final = str(candidates_root / f"cand_{idx}" / f"final_output.{cand_ext}")
                img = load_image(result["iterations"][-1].image_path)
                save_image(img, cand_final, format=cand_format)
                image_path = cand_final
            last_critique = result["iterations"][-1].critique if result["iterations"] else None
            entry: Dict[str, Any] = {
                "index": idx,
                "seed": seeds[idx - 1],
                "image_path": image_path,
                "iterations": len(result["iterations"]),
                "critic_satisfied": bool(last_critique and not last_critique.needs_revision),
                "budget_exceeded": result["budget_exceeded"],
                "error": None,
            }
            if result["rollback_info"] is not None:
                entry["rollback"] = result["rollback_info"]
            candidates_meta.append(entry)
            self._emit_progress(
                "candidate_completed",
                candidate=idx,
                iterations=len(result["iterations"]),
                image_path=image_path,
            )
            if primary is None:
                primary = result
                primary_index = idx

        if primary is None:
            detail = "; ".join(f"cand_{i}: {e}" for i, e in errors)
            raise RuntimeError(f"All {num_candidates} candidate branches failed: {detail}")

        return primary, primary_index, candidates_meta

    def _effective_vector_export(self, input: GenerationInput) -> str:
        """Resolve vector export mode from input override or settings."""
        if input.vector_export is not None:
            return input.vector_export
        return getattr(self.settings, "vector_export", "none") or "none"

    async def _maybe_export_methodology_vector(
        self,
        *,
        vector_mode: str,
        diagram_type: DiagramType,
        final_description: str,
        source_context: str,
        caption: str,
        run_dir: Path,
        progress_callback: Optional[Callable[[PipelineProgressEvent], None]],
    ) -> tuple[Optional[str], Optional[str]]:
        """Optional SVG/PDF export for methodology diagrams (Graphviz)."""
        if diagram_type != DiagramType.METHODOLOGY:
            return None, None
        if vector_mode == "none" or not final_description.strip():
            return None, None

        struct_start = time.perf_counter()
        if self._cost_tracker:
            self._cost_tracker.set_agent("structurer")
        _emit_progress(
            progress_callback,
            PipelineProgressEvent(
                stage=PipelineProgressStage.STRUCTURER_START,
                message="Building vector diagram (structurer)",
            ),
        )
        self._emit_progress("vector_export_started", mode=vector_mode)

        try:
            ir_model = await self.structurer.run(
                description=final_description,
                source_context=source_context,
                caption=caption,
            )
        except Exception as e:
            logger.warning("Vector structurer failed", error=str(e))
            save_json({"error": str(e)}, run_dir / "vector_export_error.json")
            struct_seconds = time.perf_counter() - struct_start
            _emit_progress(
                progress_callback,
                PipelineProgressEvent(
                    stage=PipelineProgressStage.STRUCTURER_END,
                    message="Structurer failed",
                    seconds=struct_seconds,
                    extra={"error": str(e)},
                ),
            )
            self._emit_progress("vector_export_failed", error=str(e))
            return None, None

        save_json(ir_model.model_dump(), run_dir / "diagram_ir.json")
        dot_src = diagram_ir_to_dot(ir_model)
        (run_dir / "diagram.dot").write_text(dot_src, encoding="utf-8")

        svg_path: Optional[str] = None
        pdf_path: Optional[str] = None
        if vector_mode in ("svg", "both"):
            candidate = run_dir / "final_output.svg"
            if render_dot_to_file(dot_src, str(candidate), "svg"):
                svg_path = str(candidate)
        if vector_mode in ("pdf", "both"):
            candidate = run_dir / "final_output.pdf"
            if render_dot_to_file(dot_src, str(candidate), "pdf"):
                pdf_path = str(candidate)

        struct_seconds = time.perf_counter() - struct_start
        _emit_progress(
            progress_callback,
            PipelineProgressEvent(
                stage=PipelineProgressStage.STRUCTURER_END,
                message="Vector export done",
                seconds=struct_seconds,
                extra={
                    "svg": svg_path,
                    "pdf": pdf_path,
                    "graphviz_on_path": find_dot_executable() is not None,
                },
            ),
        )
        self._emit_progress(
            "vector_export_completed",
            seconds=round(struct_seconds, 2),
            svg=svg_path,
            pdf=pdf_path,
        )
        return svg_path, pdf_path

    async def _resolve_retrieval_candidates(
        self, input: GenerationInput, candidates: list[ReferenceExample]
    ) -> tuple[list[ReferenceExample], str, list[str]]:
        """Resolve candidate pool based on exemplar-retrieval settings."""
        if not self.settings.exemplar_retrieval_enabled:
            return candidates, "disabled", []

        if self._external_exemplar_retriever is None:
            logger.warning(
                "Exemplar retrieval enabled but endpoint is not configured; "
                "using baseline retrieval"
            )
            return candidates, "fallback_no_endpoint", []

        try:
            hits = await self._external_exemplar_retriever.retrieve(
                source_context=input.source_context,
                caption=input.communicative_intent,
                diagram_type=input.diagram_type,
                top_k=self.settings.exemplar_retrieval_top_k,
            )
        except ExemplarRetrievalError as e:
            logger.warning(
                "External exemplar retrieval failed; using baseline retrieval",
                error=str(e),
            )
            return candidates, "fallback_error", []

        if not hits:
            logger.warning("External exemplar retrieval returned no hits; using baseline retrieval")
            return candidates, "fallback_empty", []

        mapped = map_external_hits_to_examples(hits, self.reference_store)
        mode = self.settings.exemplar_retrieval_mode
        if mode == "external_only":
            return mapped, "external_only", [e.id for e in mapped]
        return mapped, "external_then_rerank", [e.id for e in mapped]

    async def regenerate_from_ir(
        self,
        *,
        diagram_ir: DiagramIR,
        source_context: str,
        caption: str,
        aspect_ratio: Optional[str] = None,
        progress_callback: Optional[Callable[[PipelineProgressEvent], None]] = None,
    ) -> GenerationOutput:
        """Regenerate a methodology figure from edited DiagramIR with lock hints."""
        total_start = time.perf_counter()
        current_description = format_diagram_ir_for_regeneration(diagram_ir)
        iterations: list[IterationRecord] = []
        iteration_timings: list[dict[str, float | int]] = []
        rollback_info: Optional[Dict[str, Any]] = None
        budget_exceeded = self._check_budget("before regenerate-from-ir iterations")
        vector_formats = ["svg", "pdf"] if self.settings.vector_export else None
        total_iters = (
            self.settings.max_iterations
            if self.settings.auto_refine
            else self.settings.refinement_iterations
        )

        if self.settings.save_iterations:
            save_json(
                {
                    "source_context": source_context,
                    "communicative_intent": caption,
                    "diagram_type": DiagramType.METHODOLOGY.value,
                    "raw_data": None,
                    "aspect_ratio": aspect_ratio,
                    "regeneration_mode": "diagram_ir_locked",
                    "locked_nodes": diagram_ir.locks.locked_node_ids,
                    "locked_edges": diagram_ir.locks.locked_edge_refs,
                    "locked_groups": diagram_ir.locks.locked_group_ids,
                },
                self._run_dir / "run_input.json",
            )
            save_json(diagram_ir.model_dump(), self._run_dir / "diagram_ir_input.json")

        for i in range(total_iters):
            if budget_exceeded:
                break
            iter_index = i + 1

            if self._cost_tracker:
                self._cost_tracker.set_agent("visualizer")
            _emit_progress(
                progress_callback,
                PipelineProgressEvent(
                    stage=PipelineProgressStage.VISUALIZER_START,
                    message=f"Generating image (iteration {iter_index}/{total_iters})",
                    iteration=iter_index,
                    extra={"total_iterations": total_iters, "mode": "regenerate_ir"},
                ),
            )
            visualizer_start = time.perf_counter()
            image_path, rollback_info = await self._visualize_with_rollback(
                iteration=iter_index,
                rollback_to=iterations[-1].iteration if iterations else None,
                run_dir=self._run_dir,
                description=current_description,
                diagram_type=DiagramType.METHODOLOGY,
                raw_data=None,
                seed=self.settings.seed,
                aspect_ratio=aspect_ratio,
                vector_formats=vector_formats,
            )
            visualizer_seconds = time.perf_counter() - visualizer_start
            if image_path is None:
                _emit_progress(
                    progress_callback,
                    PipelineProgressEvent(
                        stage=PipelineProgressStage.VISUALIZER_END,
                        message=(
                            f"Visualizer iteration {iter_index} failed; keeping previous best image"
                        ),
                        seconds=visualizer_seconds,
                        iteration=iter_index,
                        extra={"rollback": True, "error": rollback_info["error"]},
                    ),
                )
                break
            _emit_progress(
                progress_callback,
                PipelineProgressEvent(
                    stage=PipelineProgressStage.VISUALIZER_END,
                    message=f"Visualizer iteration {iter_index} done",
                    seconds=visualizer_seconds,
                    iteration=iter_index,
                ),
            )

            if self._cost_tracker:
                self._cost_tracker.set_agent("critic")
            _emit_progress(
                progress_callback,
                PipelineProgressEvent(
                    stage=PipelineProgressStage.CRITIC_START,
                    message="Critic reviewing",
                    iteration=iter_index,
                ),
            )
            critic_start = time.perf_counter()
            try:
                critique = await _call_with_retry(
                    "critic",
                    self.critic.run,
                    image_path=image_path,
                    description=current_description,
                    source_context=source_context,
                    caption=caption,
                    diagram_type=DiagramType.METHODOLOGY,
                )
            except Exception:
                logger.warning(
                    "Critic failed after retries, accepting current image",
                    iteration=iter_index,
                    exc_info=True,
                )
                critique = CritiqueResult()
            critic_seconds = time.perf_counter() - critic_start
            _emit_progress(
                progress_callback,
                PipelineProgressEvent(
                    stage=PipelineProgressStage.CRITIC_END,
                    message="Critic done",
                    seconds=critic_seconds,
                    iteration=iter_index,
                    extra={
                        "needs_revision": critique.needs_revision,
                        "summary": critique.summary,
                        "critic_suggestions": critique.critic_suggestions[:3],
                    },
                ),
            )

            iteration_timings.append(
                {
                    "iteration": iter_index,
                    "visualizer_seconds": visualizer_seconds,
                    "critic_seconds": critic_seconds,
                }
            )
            iterations.append(
                IterationRecord(
                    iteration=iter_index,
                    description=current_description,
                    image_path=image_path,
                    critique=critique,
                )
            )
            if self.settings.save_iterations:
                iter_dir = ensure_dir(self._run_dir / f"iter_{iter_index}")
                save_json(
                    {
                        "description": current_description,
                        "critique": critique.model_dump(),
                        "mode": "regenerate_ir",
                    },
                    iter_dir / "details.json",
                )

            if critique.needs_revision and critique.revised_description:
                # Keep lock constraints attached to every revised prompt.
                current_description = (
                    critique.revised_description.strip()
                    + "\n\n"
                    + format_diagram_ir_for_regeneration(diagram_ir)
                )
            else:
                break

            if self._check_budget("between regenerate-from-ir iterations", iteration=iter_index):
                budget_exceeded = True
                break

        output_format = getattr(self.settings, "output_format", "png").lower()
        ext = "jpg" if output_format == "jpeg" else output_format
        final_output_path = str(self._run_dir / f"final_output.{ext}")

        if iterations:
            final_image = iterations[-1].image_path
            if output_format == "svg":
                save_json(diagram_ir.model_dump(), self._run_dir / "diagram_ir.json")
                save_svg_from_ir(diagram_ir, final_output_path)
            else:
                img = load_image(final_image)
                save_image(img, final_output_path, format=output_format)
        else:
            final_output_path = ""

        generated_caption, caption_seconds = await self._generate_caption(
            image_path=final_output_path,
            source_context=source_context,
            intent=caption,
            description=current_description,
            diagram_type=DiagramType.METHODOLOGY,
            progress_callback=progress_callback,
        )

        total_seconds = time.perf_counter() - total_start
        metadata = RunMetadata(
            run_id=self.run_id,
            timestamp=datetime.datetime.now().isoformat(),
            vlm_provider=getattr(self._vlm, "name", "custom"),
            vlm_model=getattr(self._vlm, "model_name", "custom"),
            image_provider=getattr(self._image_gen, "name", "custom"),
            image_model=getattr(self._image_gen, "model_name", "custom"),
            refinement_iterations=len(iterations),
            seed=self.settings.seed,
            config_snapshot=self.settings.model_dump(
                exclude={
                    "google_api_key",
                    "openai_api_key",
                    "openrouter_api_key",
                    "anthropic_api_key",
                    "atlascloud_api_key",
                    "litellm_api_key",
                }
            ),
        )
        metadata_dict = metadata.model_dump()
        metadata_dict["timing"] = {
            "total_seconds": total_seconds,
            "caption_seconds": caption_seconds,
            "iterations": iteration_timings,
        }
        metadata_dict["regeneration"] = {
            "mode": "diagram_ir_locked",
            "locked_nodes": len(diagram_ir.locks.locked_node_ids),
            "locked_edges": len(diagram_ir.locks.locked_edge_refs),
            "locked_groups": len(diagram_ir.locks.locked_group_ids),
        }
        if rollback_info is not None:
            metadata_dict["rollback"] = rollback_info
        if generated_caption is not None:
            metadata_dict["generated_caption"] = generated_caption
        if self._cost_tracker:
            cost_summary = self._cost_tracker.summary()
            cost_summary["budget_exceeded"] = budget_exceeded
            if self.settings.budget_usd is not None:
                cost_summary["budget_usd"] = self.settings.budget_usd
            metadata_dict["cost"] = cost_summary

        if self.settings.vector_export and self.visualizer._last_vector_paths:
            metadata_dict["vector_output_paths"] = self.visualizer._last_vector_paths

        save_json(metadata_dict, self._run_dir / "metadata.json")
        return GenerationOutput(
            image_path=final_output_path,
            description=current_description,
            iterations=iterations,
            metadata=metadata_dict,
            generated_caption=generated_caption,
        )

    async def generate(
        self,
        input: GenerationInput,
        progress_callback: Optional[Callable[[PipelineProgressEvent], None]] = None,
    ) -> GenerationOutput:
        """Run the full generation pipeline.

        Args:
            input: Generation input with source context and caption.

        Returns:
            GenerationOutput with final image and metadata.
        """
        total_start = time.perf_counter()

        self._emit_progress(
            "generation_started",
            run_id=self.run_id,
            diagram_type=input.diagram_type.value,
            context_length=len(input.source_context),
            input_image_count=len(input.input_images),
        )

        # Save input for resume/continue support
        if self.settings.save_iterations:
            save_json(
                {
                    "source_context": input.source_context,
                    "communicative_intent": input.communicative_intent,
                    "diagram_type": input.diagram_type.value,
                    "raw_data": input.raw_data,
                    "aspect_ratio": input.aspect_ratio,
                    "vector_export": self._effective_vector_export(input),
                    "input_images": input.input_images,
                },
                self._run_dir / "run_input.json",
            )

        logger.info(
            "Starting generation",
            run_id=self.run_id,
            diagram_type=input.diagram_type.value,
            context_length=len(input.source_context),
        )

        # Select guidelines based on diagram type
        guidelines = (
            self._methodology_guidelines
            if input.diagram_type == DiagramType.METHODOLOGY
            else self._plot_guidelines
        )

        # ── Phase 0: Input Optimization (optional) ───────────────────
        optimize_seconds = 0.0
        if self.settings.optimize_inputs:
            logger.info("Phase 0: Optimizing inputs (parallel)")
            if self._cost_tracker:
                self._cost_tracker.set_agent("optimizer")
            _emit_progress(
                progress_callback,
                PipelineProgressEvent(
                    stage=PipelineProgressStage.OPTIMIZER_START,
                    message="Optimizing inputs (parallel)",
                ),
            )
            self._emit_progress("phase0_optimize_started")
            optimize_start = time.perf_counter()
            try:
                optimized = await self.optimizer.run(
                    source_context=input.source_context,
                    caption=input.communicative_intent,
                    diagram_type=input.diagram_type,
                )
                optimize_seconds = time.perf_counter() - optimize_start
                _emit_progress(
                    progress_callback,
                    PipelineProgressEvent(
                        stage=PipelineProgressStage.OPTIMIZER_END,
                        message="Optimizer done",
                        seconds=optimize_seconds,
                    ),
                )
                logger.info(
                    "[Optimizer] done",
                    seconds=round(optimize_seconds, 1),
                )
                self._emit_progress(
                    "phase0_optimize_completed",
                    seconds=round(optimize_seconds, 1),
                )

                # Save originals and apply optimized versions
                if self.settings.save_iterations:
                    save_json(
                        {
                            "original_context": input.source_context,
                            "original_caption": input.communicative_intent,
                            "optimized_context": optimized["optimized_context"],
                            "optimized_caption": optimized["optimized_caption"],
                        },
                        self._run_dir / "optimization.json",
                    )

                input = GenerationInput(
                    source_context=optimized["optimized_context"],
                    communicative_intent=optimized["optimized_caption"],
                    diagram_type=input.diagram_type,
                    raw_data=input.raw_data,
                    aspect_ratio=input.aspect_ratio,
                    input_images=input.input_images,
                )
            except Exception:
                optimize_seconds = time.perf_counter() - optimize_start
                logger.warning(
                    "Optimizer failed, continuing with original input",
                    seconds=round(optimize_seconds, 1),
                    exc_info=True,
                )
                _emit_progress(
                    progress_callback,
                    PipelineProgressEvent(
                        stage=PipelineProgressStage.OPTIMIZER_END,
                        message="Optimizer failed, using original input",
                        seconds=optimize_seconds,
                    ),
                )

        # ── Phase 1: Linear Planning ─────────────────────────────────

        # Step 1: Retriever — find relevant examples (timer includes external call when enabled)
        logger.info("Phase 1: Retrieval")
        if self._cost_tracker:
            self._cost_tracker.set_agent("retriever")
        _emit_progress(
            progress_callback,
            PipelineProgressEvent(
                stage=PipelineProgressStage.RETRIEVER_START,
                message="Retrieving examples",
            ),
        )
        self._emit_progress("phase1_retrieval_started")
        retrieval_start = time.perf_counter()

        if input.reference_ids:
            # Manual override: look up each ID, skip automatic retrieval
            examples = []
            missing_ids = []
            for ref_id in input.reference_ids:
                ref = self.reference_store.get_by_id(ref_id)
                if ref is not None:
                    examples.append(ref)
                else:
                    missing_ids.append(ref_id)
            if missing_ids:
                raise ValueError(
                    f"Unknown reference IDs: {', '.join(missing_ids)}. "
                    "Use 'paperbanana references list' to see available IDs."
                )
            retrieval_mode = "manual_override"
            external_candidate_ids: list[str] = list(input.reference_ids)
            logger.info(
                "Using manual reference ID override",
                ids=input.reference_ids,
                resolved=len(examples),
            )
        else:
            if self.settings.reference_category:
                candidates = self.reference_store.get_by_categories(
                    self.settings.reference_category
                )
                logger.info(
                    "Filtered candidates by category",
                    categories=self.settings.reference_category,
                    count=len(candidates),
                )
            else:
                candidates = self.reference_store.get_all()
            (
                candidates,
                retrieval_mode,
                external_candidate_ids,
            ) = await self._resolve_retrieval_candidates(input, candidates)
            if retrieval_mode == "external_only":
                examples = candidates[: self.settings.num_retrieval_examples]
            else:
                examples = await _call_with_retry(
                    "retriever",
                    self.retriever.run,
                    source_context=input.source_context,
                    caption=input.communicative_intent,
                    candidates=candidates,
                    num_examples=self.settings.num_retrieval_examples,
                    diagram_type=input.diagram_type,
                )

        retrieval_seconds = time.perf_counter() - retrieval_start
        _emit_progress(
            progress_callback,
            PipelineProgressEvent(
                stage=PipelineProgressStage.RETRIEVER_END,
                message="Retriever done",
                seconds=retrieval_seconds,
                extra={"examples_count": len(examples), "retrieval_mode": retrieval_mode},
            ),
        )
        logger.info(
            "[Retriever] done",
            seconds=round(retrieval_seconds, 1),
            examples_found=len(examples),
            retrieval_mode=retrieval_mode,
        )
        self._emit_progress(
            "phase1_retrieval_completed",
            seconds=round(retrieval_seconds, 1),
            examples_found=len(examples),
            retrieval_mode=retrieval_mode,
        )

        # Step 2: Planner — generate textual description
        logger.info("Phase 1: Planning")
        if self._cost_tracker:
            self._cost_tracker.set_agent("planner")
        _emit_progress(
            progress_callback,
            PipelineProgressEvent(
                stage=PipelineProgressStage.PLANNER_START,
                message="Planning description",
            ),
        )
        self._emit_progress("phase1_planning_started")
        planning_start = time.perf_counter()
        description, planner_ratio = await _call_with_retry(
            "planner",
            self.planner.run,
            source_context=input.source_context,
            caption=input.communicative_intent,
            examples=examples,
            diagram_type=input.diagram_type,
            supported_ratios=getattr(self.visualizer.image_gen, "supported_ratios", None),
            input_images=input.input_images,
        )
        planning_seconds = time.perf_counter() - planning_start
        _emit_progress(
            progress_callback,
            PipelineProgressEvent(
                stage=PipelineProgressStage.PLANNER_END,
                message="Planner done",
                seconds=planning_seconds,
                extra={"recommended_ratio": planner_ratio},
            ),
        )
        self._emit_progress(
            "phase1_planning_completed",
            seconds=round(planning_seconds, 1),
            recommended_ratio=planner_ratio,
        )

        # Step 3: Stylist — optimize description aesthetics
        logger.info("Phase 1: Styling")
        if self._cost_tracker:
            self._cost_tracker.set_agent("stylist")
        _emit_progress(
            progress_callback,
            PipelineProgressEvent(
                stage=PipelineProgressStage.STYLIST_START,
                message="Styling description",
            ),
        )
        self._emit_progress("phase1_styling_started")
        styling_start = time.perf_counter()
        try:
            optimized_description = await _call_with_retry(
                "stylist",
                self.stylist.run,
                description=description,
                guidelines=guidelines,
                source_context=input.source_context,
                caption=input.communicative_intent,
                diagram_type=input.diagram_type,
            )
        except Exception:
            logger.warning(
                "Stylist failed after retries, falling back to planner output",
                exc_info=True,
            )
            optimized_description = description
        styling_seconds = time.perf_counter() - styling_start
        _emit_progress(
            progress_callback,
            PipelineProgressEvent(
                stage=PipelineProgressStage.STYLIST_END,
                message="Stylist done",
                seconds=styling_seconds,
            ),
        )
        self._emit_progress(
            "phase1_styling_completed",
            seconds=round(styling_seconds, 1),
        )

        # Save planning outputs
        if self.settings.save_iterations:
            save_json(
                {
                    "retrieved_examples": [e.id for e in examples],
                    "input_images": input.input_images,
                    "initial_description": description,
                    "optimized_description": optimized_description,
                    "planner_recommended_ratio": planner_ratio,
                },
                self._run_dir / "planning.json",
            )

        # ── Phase 2: Iterative Refinement ─────────────────────────────

        # Aspect ratio priority: user-specified > venue default (venue.yaml)
        # > planner-recommended > default (None)
        venue_ratio = self._venue_pack.config.aspect_ratio if self._venue_pack else None
        effective_ratio, ratio_source = select_aspect_ratio(
            input.aspect_ratio, venue_ratio, planner_ratio
        )
        if effective_ratio:
            logger.info(
                "Using aspect ratio",
                source=ratio_source,
                ratio=effective_ratio,
            )
            self._emit_progress(
                "aspect_ratio_selected",
                ratio=effective_ratio,
                source=ratio_source,
            )

        vector_formats = ["svg", "pdf"] if self.settings.vector_export != "none" else None

        if self.settings.auto_refine:
            total_iters = self.settings.max_iterations
        else:
            total_iters = self.settings.refinement_iterations

        num_candidates = self.settings.num_candidates
        if not 1 <= num_candidates <= 8:
            raise ValueError(f"num_candidates must be between 1 and 8, got {num_candidates}")

        candidates_meta: Optional[list[Dict[str, Any]]] = None
        primary_candidate_index: Optional[int] = None
        if num_candidates == 1:
            branch = await self._run_refinement_branch(
                input=input,
                initial_description=optimized_description,
                effective_ratio=effective_ratio,
                vector_formats=vector_formats,
                total_iters=total_iters,
                run_dir=self._run_dir,
                visualizer=self.visualizer,
                seed=self.settings.seed,
                progress_callback=progress_callback,
            )
        else:
            branch, primary_candidate_index, candidates_meta = await self._fan_out_candidates(
                input=input,
                initial_description=optimized_description,
                effective_ratio=effective_ratio,
                vector_formats=vector_formats,
                total_iters=total_iters,
                num_candidates=num_candidates,
                progress_callback=progress_callback,
            )

        iterations: list[IterationRecord] = branch["iterations"]
        iteration_timings = branch["iteration_timings"]
        rollback_info: Optional[Dict[str, Any]] = branch["rollback_info"]
        current_description = branch["final_description"]
        budget_exceeded = branch["budget_exceeded"]
        phase2_vector_paths: dict[str, str] = branch["vector_paths"]

        # Final output (with multi-candidate fan-out, the run-root output is
        # the primary candidate: candidate 1 unless it failed)
        output_format = getattr(self.settings, "output_format", "png").lower()
        final_output_path = self._build_final_output(
            iterations,
            self._run_dir,
            "No iterations completed — budget exceeded during planning phases",
        )
        ir_planner_status: str | None = None
        ir_planner_error: str | None = None

        if iterations and output_format == "svg":
            if input.diagram_type == DiagramType.METHODOLOGY:
                try:
                    diagram_ir = await self.ir_planner.run(
                        source_context=input.source_context,
                        caption=input.communicative_intent,
                        styled_description=current_description,
                    )
                    ir_planner_status = "success"
                    logger.info("IR planner produced structured diagram IR")
                except Exception as e:
                    ir_planner_status = "fallback"
                    ir_planner_error = str(e)
                    logger.warning(
                        "IR planner failed; falling back to heuristic IR",
                        error=str(e),
                    )
                    diagram_ir = extract_diagram_ir(
                        current_description,
                        title=input.communicative_intent or "Methodology Diagram",
                    )
                save_json(diagram_ir.model_dump(), self._run_dir / "diagram_ir.json")
                save_svg_from_ir(diagram_ir, final_output_path)
            else:
                save_raster_wrapped_svg(iterations[-1].image_path, final_output_path)

        # ── Caption Generation (optional) ─────────────────────────────
        generated_caption, caption_seconds = await self._generate_caption(
            image_path=final_output_path,
            source_context=input.source_context,
            intent=input.communicative_intent,
            description=current_description,
            diagram_type=input.diagram_type,
            progress_callback=progress_callback,
        )

        # ── Optional: TikZ / PGFPlots export ─────────────────────────
        tikz_path: str | None = None
        should_export_tikz = (
            input.diagram_type == DiagramType.METHODOLOGY and self.settings.export_tikz
        ) or (input.diagram_type == DiagramType.STATISTICAL_PLOT and self.settings.export_pgfplots)

        if should_export_tikz and final_output_path:
            if self._cost_tracker:
                self._cost_tracker.set_agent("tikz_exporter")
            _emit_progress(
                progress_callback,
                PipelineProgressEvent(
                    stage=PipelineProgressStage.TIKZ_EXPORTER_START,
                    message="Exporting to LaTeX/TikZ",
                ),
            )
            self._emit_progress("tikz_export_started")
            tikz_start = time.perf_counter()
            tikz_error: str | None = None
            try:
                tikz_source = await self.tikz_exporter.run(
                    image_path=final_output_path,
                    source_context=input.source_context,
                    caption=input.communicative_intent,
                    diagram_type=input.diagram_type,
                    description=current_description,
                    source_label=str(self._run_dir),
                    model_label=getattr(self._vlm, "model_name", ""),
                    venue=self.settings.venue,
                    version=_get_version(),
                )
                tex_path = Path(final_output_path).with_suffix(".tex")
                tex_path.write_text(tikz_source, encoding="utf-8")
                tikz_path = str(tex_path)
                logger.info("TikZ export saved", path=tikz_path)
            except Exception as e:
                tikz_error = str(e)
                logger.warning("TikZ export failed", exc_info=True)
            tikz_seconds = time.perf_counter() - tikz_start
            _emit_progress(
                progress_callback,
                PipelineProgressEvent(
                    stage=PipelineProgressStage.TIKZ_EXPORTER_END,
                    message="TikZ export failed (image is unaffected)"
                    if tikz_error
                    else "TikZ export done",
                    seconds=tikz_seconds,
                    extra={"tikz_path": tikz_path, "error": tikz_error},
                ),
            )
            if tikz_error:
                self._emit_progress(
                    "tikz_export_failed",
                    seconds=round(tikz_seconds, 1),
                    error=tikz_error,
                )
            else:
                self._emit_progress(
                    "tikz_export_completed",
                    seconds=round(tikz_seconds, 1),
                    tikz_path=tikz_path,
                )

        vector_mode = self._effective_vector_export(input)
        structurer_seconds = 0.0
        ve_start = time.perf_counter()
        vector_svg_path, vector_pdf_path = await self._maybe_export_methodology_vector(
            vector_mode=vector_mode,
            diagram_type=input.diagram_type,
            final_description=current_description,
            source_context=input.source_context,
            caption=input.communicative_intent,
            run_dir=self._run_dir,
            progress_callback=progress_callback,
        )
        structurer_seconds = time.perf_counter() - ve_start

        total_seconds = time.perf_counter() - total_start
        logger.info(
            "Total generation time",
            run_id=self.run_id,
            total_seconds=total_seconds,
        )
        self._emit_progress(
            "generation_completed",
            run_id=self.run_id,
            total_seconds=total_seconds,
            iterations=len(iterations),
        )

        # Build metadata
        metadata = RunMetadata(
            run_id=self.run_id,
            timestamp=datetime.datetime.now().isoformat(),
            vlm_provider=getattr(self._vlm, "name", "custom"),
            vlm_model=getattr(self._vlm, "model_name", "custom"),
            image_provider=getattr(self._image_gen, "name", "custom"),
            image_model=getattr(self._image_gen, "model_name", "custom"),
            refinement_iterations=len(iterations),
            seed=self.settings.seed,
            config_snapshot=self.settings.model_dump(
                exclude={
                    "google_api_key",
                    "openai_api_key",
                    "openrouter_api_key",
                    "anthropic_api_key",
                    "atlascloud_api_key",
                    "litellm_api_key",
                }
            ),
        )

        metadata_dict = metadata.model_dump()

        metadata_dict["timing"] = {
            "total_seconds": total_seconds,
            "optimize_seconds": optimize_seconds,
            "retrieval_seconds": retrieval_seconds,
            "planning_seconds": planning_seconds,
            "styling_seconds": styling_seconds,
            "caption_seconds": caption_seconds,
            "iterations": iteration_timings,
        }
        if input.diagram_type == DiagramType.METHODOLOGY and vector_mode != "none":
            metadata_dict["timing"]["structurer_seconds"] = structurer_seconds
        metadata_dict["retrieval"] = {
            "mode": retrieval_mode,
            "external_enabled": self.settings.exemplar_retrieval_enabled,
            "external_candidate_ids": external_candidate_ids,
        }
        metadata_dict["input_images"] = {"count": len(input.input_images)}
        if rollback_info is not None:
            metadata_dict["rollback"] = rollback_info
        if candidates_meta is not None:
            metadata_dict["num_candidates"] = num_candidates
            metadata_dict["primary_candidate"] = primary_candidate_index
            metadata_dict["candidates"] = candidates_meta
        if generated_caption is not None:
            metadata_dict["generated_caption"] = generated_caption
        if ir_planner_status is not None:
            metadata_dict["ir_planner"] = {
                "status": ir_planner_status,
                "fallback_used": ir_planner_status == "fallback",
                "error": ir_planner_error,
            }
        metadata_dict["vector_export"] = {
            "mode": vector_mode,
            "svg_path": vector_svg_path,
            "pdf_path": vector_pdf_path,
            "graphviz_available": find_dot_executable() is not None,
        }

        if self._cost_tracker:
            cost_summary = self._cost_tracker.summary()
            cost_summary["budget_exceeded"] = budget_exceeded
            if self.settings.budget_usd is not None:
                cost_summary["budget_usd"] = self.settings.budget_usd
            metadata_dict["cost"] = cost_summary

        # Include vector output paths when vector export was requested
        # (from the primary Phase-2 branch's visualizer instance)
        if self.settings.vector_export != "none" and phase2_vector_paths:
            metadata_dict["vector_output_paths"] = phase2_vector_paths

        # Always write metadata (including cost) to disk for every run
        if tikz_path:
            metadata_dict["tikz_path"] = tikz_path
        save_json(metadata_dict, self._run_dir / "metadata.json")

        output = GenerationOutput(
            image_path=final_output_path,
            description=current_description,
            iterations=iterations,
            metadata=metadata_dict,
            generated_caption=generated_caption,
            tikz_path=tikz_path,
            vector_svg_path=vector_svg_path,
            vector_pdf_path=vector_pdf_path,
        )

        logger.info(
            "Generation complete",
            run_id=self.run_id,
            output=final_output_path,
            total_iterations=len(iterations),
        )

        return output

    async def continue_run(
        self,
        resume_state,
        additional_iterations: Optional[int] = None,
        user_feedback: Optional[str] = None,
        progress_callback: Optional[Callable[[PipelineProgressEvent], None]] = None,
    ) -> GenerationOutput:
        """Continue a previous run with more iterations.

        Args:
            resume_state: ResumeState loaded from a previous run.
            additional_iterations: Number of extra iterations (or use settings).
            user_feedback: Optional user comments for the critic to consider.

        Returns:
            GenerationOutput with final image and metadata.
        """

        total_start = time.perf_counter()

        # Override run dir to write into the existing run
        run_dir = Path(resume_state.run_dir)
        self.run_id = resume_state.run_id
        self._sync_run_scoped_agents()

        if self.settings.auto_refine:
            total_iters = self.settings.max_iterations
        else:
            total_iters = additional_iterations or self.settings.refinement_iterations

        start_iter = resume_state.last_iteration
        current_description = resume_state.last_description

        logger.info(
            "Continuing run",
            run_id=self.run_id,
            from_iteration=start_iter,
            additional_iterations=total_iters,
            has_feedback=user_feedback is not None,
        )
        self._emit_progress(
            "continue_started",
            run_id=self.run_id,
            from_iteration=start_iter,
            additional_iterations=total_iters,
            has_feedback=user_feedback is not None,
        )

        iterations: list[IterationRecord] = []
        iteration_timings = []
        budget_exceeded = False
        rollback_info: Optional[Dict[str, Any]] = None
        vector_formats = ["svg", "pdf"] if self.settings.vector_export != "none" else None

        for i in range(total_iters):
            if budget_exceeded:
                break

            iter_num = start_iter + i + 1
            logger.info(
                f"Phase 2: Iteration {iter_num}" + (" (auto)" if self.settings.auto_refine else "")
            )
            self._emit_progress(
                "iteration_started",
                iteration=iter_num,
                total_iterations=start_iter + total_iters,
                auto=self.settings.auto_refine,
                mode="continue",
            )

            # Visualizer — generate image
            if self._cost_tracker:
                self._cost_tracker.set_agent("visualizer")
            _emit_progress(
                progress_callback,
                PipelineProgressEvent(
                    stage=PipelineProgressStage.VISUALIZER_START,
                    message=f"Generating image (iteration {iter_num})",
                    iteration=iter_num,
                    extra={"total_iterations": total_iters},
                ),
            )
            if iterations:
                rollback_to = iterations[-1].iteration
            elif resume_state.last_image_path:
                # The run being continued already produced an image.
                rollback_to = start_iter
            else:
                rollback_to = None
            visualizer_start = time.perf_counter()
            image_path, rollback_info = await self._visualize_with_rollback(
                iteration=iter_num,
                rollback_to=rollback_to,
                run_dir=run_dir,
                description=current_description,
                diagram_type=resume_state.diagram_type,
                raw_data=resume_state.raw_data,
                seed=self.settings.seed,
                aspect_ratio=resume_state.aspect_ratio,
                vector_formats=vector_formats,
            )
            visualizer_seconds = time.perf_counter() - visualizer_start
            if image_path is None:
                _emit_progress(
                    progress_callback,
                    PipelineProgressEvent(
                        stage=PipelineProgressStage.VISUALIZER_END,
                        message=(
                            f"Visualizer iteration {iter_num} failed; keeping previous best image"
                        ),
                        seconds=visualizer_seconds,
                        iteration=iter_num,
                        extra={"rollback": True, "error": rollback_info["error"]},
                    ),
                )
                break
            _emit_progress(
                progress_callback,
                PipelineProgressEvent(
                    stage=PipelineProgressStage.VISUALIZER_END,
                    message=f"Visualizer iteration {iter_num} done",
                    seconds=visualizer_seconds,
                    iteration=iter_num,
                ),
            )
            logger.info(
                f"[Visualizer] Iteration {iter_num} done",
                seconds=round(visualizer_seconds, 1),
            )
            self._emit_progress(
                "visualizer_completed",
                iteration=iter_num,
                seconds=round(visualizer_seconds, 1),
                mode="continue",
            )

            # Critic — evaluate with optional user feedback
            if self._cost_tracker:
                self._cost_tracker.set_agent("critic")
            _emit_progress(
                progress_callback,
                PipelineProgressEvent(
                    stage=PipelineProgressStage.CRITIC_START,
                    message="Critic reviewing",
                    iteration=iter_num,
                ),
            )
            critic_start = time.perf_counter()
            try:
                critique = await _call_with_retry(
                    "critic",
                    self.critic.run,
                    image_path=image_path,
                    description=current_description,
                    source_context=resume_state.source_context,
                    caption=resume_state.communicative_intent,
                    diagram_type=resume_state.diagram_type,
                    user_feedback=user_feedback,
                )
            except Exception:
                logger.warning(
                    "Critic failed after retries, accepting current image",
                    iteration=iter_num,
                    exc_info=True,
                )
                critique = CritiqueResult()
            critic_seconds = time.perf_counter() - critic_start
            _emit_progress(
                progress_callback,
                PipelineProgressEvent(
                    stage=PipelineProgressStage.CRITIC_END,
                    message="Critic done",
                    seconds=critic_seconds,
                    iteration=iter_num,
                    extra={
                        "needs_revision": critique.needs_revision,
                        "summary": critique.summary,
                        "critic_suggestions": critique.critic_suggestions[:3],
                    },
                ),
            )
            logger.info(
                "[Critic] done",
                seconds=round(critic_seconds, 1),
                needs_revision=critique.needs_revision,
            )
            self._emit_progress(
                "critic_completed",
                iteration=iter_num,
                seconds=round(critic_seconds, 1),
                needs_revision=critique.needs_revision,
                mode="continue",
            )

            iteration_record = IterationRecord(
                iteration=iter_num,
                description=current_description,
                image_path=image_path,
                critique=critique,
            )
            iteration_timings.append(
                {
                    "iteration": iter_num,
                    "visualizer_seconds": visualizer_seconds,
                    "critic_seconds": critic_seconds,
                }
            )
            iterations.append(iteration_record)

            if self.settings.save_iterations:
                iter_dir = ensure_dir(run_dir / f"iter_{iter_num}")
                save_json(
                    {
                        "description": current_description,
                        "critique": critique.model_dump(),
                        "user_feedback": user_feedback,
                    },
                    iter_dir / "details.json",
                )

            if critique.needs_revision and critique.revised_description:
                logger.info(
                    "Revision needed",
                    iteration=iter_num,
                    summary=critique.summary,
                )
                current_description = critique.revised_description
            else:
                logger.info(
                    "No further revision needed",
                    iteration=iter_num,
                    summary=critique.summary,
                )
                break

            self._emit_progress(
                "iteration_completed",
                iteration=iter_num,
                total_iterations=start_iter + len(iterations),
                needs_revision=critique.needs_revision,
                mode="continue",
            )

            # Check budget between iterations
            if self._check_budget("between iterations", iteration=iter_num):
                budget_exceeded = True
                break

        # Final output
        output_format = getattr(self.settings, "output_format", "png").lower()
        final_iterations = iterations
        if not iterations and rollback_info is not None and resume_state.last_image_path:
            # Rolled back before any new image was produced: keep the
            # previous run's best image as the final output.
            final_iterations = [
                IterationRecord(
                    iteration=start_iter,
                    description=current_description,
                    image_path=resume_state.last_image_path,
                )
            ]
        final_output_path = self._build_final_output(
            final_iterations,
            run_dir,
            "No iterations completed — budget exceeded before first iteration",
        )
        ir_planner_status: str | None = None
        ir_planner_error: str | None = None

        if iterations and output_format == "svg":
            if resume_state.diagram_type == DiagramType.METHODOLOGY:
                try:
                    diagram_ir = await self.ir_planner.run(
                        source_context=resume_state.source_context,
                        caption=resume_state.communicative_intent,
                        styled_description=current_description,
                    )
                    ir_planner_status = "success"
                    logger.info("IR planner produced structured diagram IR")
                except Exception as e:
                    ir_planner_status = "fallback"
                    ir_planner_error = str(e)
                    logger.warning(
                        "IR planner failed; falling back to heuristic IR",
                        error=str(e),
                    )
                    diagram_ir = extract_diagram_ir(
                        current_description,
                        title=resume_state.communicative_intent or "Methodology Diagram",
                    )
                save_json(diagram_ir.model_dump(), run_dir / "diagram_ir.json")
                save_svg_from_ir(diagram_ir, final_output_path)
            else:
                save_raster_wrapped_svg(iterations[-1].image_path, final_output_path)

        # ── Caption Generation (optional) ─────────────────────────────
        generated_caption, caption_seconds = await self._generate_caption(
            image_path=final_output_path,
            source_context=resume_state.source_context,
            intent=resume_state.communicative_intent,
            description=current_description,
            diagram_type=resume_state.diagram_type,
            progress_callback=progress_callback,
        )

        vector_mode = getattr(self.settings, "vector_export", "none") or "none"
        ve_start = time.perf_counter()
        vector_svg_path, vector_pdf_path = await self._maybe_export_methodology_vector(
            vector_mode=vector_mode,
            diagram_type=resume_state.diagram_type,
            final_description=current_description,
            source_context=resume_state.source_context,
            caption=resume_state.communicative_intent,
            run_dir=run_dir,
            progress_callback=progress_callback,
        )
        structurer_seconds = time.perf_counter() - ve_start

        total_seconds = time.perf_counter() - total_start
        logger.info(
            "Continue run complete",
            run_id=self.run_id,
            total_seconds=total_seconds,
            new_iterations=len(iterations),
        )
        self._emit_progress(
            "continue_completed",
            run_id=self.run_id,
            total_seconds=total_seconds,
            new_iterations=len(iterations),
        )

        # Update metadata
        metadata = RunMetadata(
            run_id=self.run_id,
            timestamp=datetime.datetime.now().isoformat(),
            vlm_provider=getattr(self._vlm, "name", "custom"),
            vlm_model=getattr(self._vlm, "model_name", "custom"),
            image_provider=getattr(self._image_gen, "name", "custom"),
            image_model=getattr(self._image_gen, "model_name", "custom"),
            refinement_iterations=start_iter + len(iterations),
            seed=self.settings.seed,
            config_snapshot=self.settings.model_dump(
                exclude={
                    "google_api_key",
                    "openai_api_key",
                    "openrouter_api_key",
                    "anthropic_api_key",
                    "atlascloud_api_key",
                    "litellm_api_key",
                }
            ),
        )

        metadata_dict = metadata.model_dump()
        metadata_dict["timing"] = {
            "continue_total_seconds": total_seconds,
            "caption_seconds": caption_seconds,
            "iterations": iteration_timings,
        }
        if resume_state.diagram_type == DiagramType.METHODOLOGY and vector_mode != "none":
            metadata_dict["timing"]["structurer_seconds"] = structurer_seconds
        metadata_dict["continued_from_iteration"] = start_iter
        if rollback_info is not None:
            metadata_dict["rollback"] = rollback_info
        if ir_planner_status is not None:
            metadata_dict["ir_planner"] = {
                "status": ir_planner_status,
                "fallback_used": ir_planner_status == "fallback",
                "error": ir_planner_error,
            }
        if user_feedback:
            metadata_dict["user_feedback"] = user_feedback
        if generated_caption is not None:
            metadata_dict["generated_caption"] = generated_caption
        metadata_dict["vector_export"] = {
            "mode": vector_mode,
            "svg_path": vector_svg_path,
            "pdf_path": vector_pdf_path,
            "graphviz_available": find_dot_executable() is not None,
        }

        if self._cost_tracker:
            cost_summary = self._cost_tracker.summary()
            cost_summary["budget_exceeded"] = budget_exceeded
            if self.settings.budget_usd is not None:
                cost_summary["budget_usd"] = self.settings.budget_usd
            metadata_dict["cost"] = cost_summary

        if self.settings.vector_export != "none" and self.visualizer._last_vector_paths:
            metadata_dict["vector_output_paths"] = self.visualizer._last_vector_paths

        # Always write metadata (including cost) to disk for every run
        save_json(metadata_dict, run_dir / "metadata_continued.json")

        output = GenerationOutput(
            image_path=final_output_path,
            description=current_description,
            iterations=iterations,
            metadata=metadata_dict,
            generated_caption=generated_caption,
            vector_svg_path=vector_svg_path,
            vector_pdf_path=vector_pdf_path,
        )

        return output
