"""Main PaperBanana pipeline orchestration."""

from __future__ import annotations

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
from paperbanana.agents.stylist import StylistAgent
from paperbanana.agents.visualizer import VisualizerAgent
from paperbanana.core.config import Settings
from paperbanana.core.cost_tracker import CostTracker
from paperbanana.core.diagram_ir import (
    extract_diagram_ir,
    save_raster_wrapped_svg,
    save_svg_from_ir,
)
from paperbanana.core.prompt_recorder import PromptRecorder
from paperbanana.core.types import (
    CritiqueResult,
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
from paperbanana.providers.registry import ProviderRegistry
from paperbanana.reference.exemplar_retrieval import (
    ExemplarRetrievalError,
    ExternalExemplarRetriever,
    map_external_hits_to_examples,
)
from paperbanana.reference.store import ReferenceStore

logger = structlog.get_logger()

_ssl_skip_applied = False


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

        # Load guidelines (venue-aware resolution)
        guidelines_path = self.settings.guidelines_path
        venue = self.settings.venue
        self._methodology_guidelines = load_methodology_guidelines(guidelines_path, venue=venue)
        self._plot_guidelines = load_plot_guidelines(guidelines_path, venue=venue)

        # Initialize agents
        prompt_dir = self._find_prompt_dir()
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
        self.visualizer = VisualizerAgent(
            self._image_gen,
            self._vlm,
            prompt_dir=prompt_dir,
            output_dir=str(self._run_dir),
            prompt_recorder=self._prompt_recorder,
        )
        self.critic = CriticAgent(
            self._vlm, prompt_dir=prompt_dir, prompt_recorder=self._prompt_recorder
        )
        self.caption_agent = CaptionAgent(
            self._vlm, prompt_dir=prompt_dir, prompt_recorder=self._prompt_recorder
        )

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
                    "initial_description": description,
                    "optimized_description": optimized_description,
                    "planner_recommended_ratio": planner_ratio,
                },
                self._run_dir / "planning.json",
            )

        # ── Phase 2: Iterative Refinement ─────────────────────────────

        # Aspect ratio priority: user-specified > planner-recommended > default (None)
        effective_ratio = input.aspect_ratio or planner_ratio
        if effective_ratio:
            ratio_source = "user" if input.aspect_ratio else "planner"
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

        current_description = optimized_description
        iterations: list[IterationRecord] = []
        iteration_timings = []
        vector_formats = ["svg", "pdf"] if self.settings.vector_export else None

        if self.settings.auto_refine:
            total_iters = self.settings.max_iterations
        else:
            total_iters = self.settings.refinement_iterations

        # Check budget after pre-iteration phases (retriever, planner, stylist)
        budget_exceeded = self._check_budget("after planning phases")

        for i in range(total_iters):
            if budget_exceeded:
                break

            iter_index = i + 1
            logger.info(
                f"Phase 2: Iteration {iter_index}/{total_iters}"
                + (" (auto)" if self.settings.auto_refine else "")
            )
            self._emit_progress(
                "iteration_started",
                iteration=iter_index,
                total_iterations=total_iters,
                auto=self.settings.auto_refine,
            )

            # Step 4: Visualizer — generate image
            if self._cost_tracker:
                self._cost_tracker.set_agent("visualizer")
            _emit_progress(
                progress_callback,
                PipelineProgressEvent(
                    stage=PipelineProgressStage.VISUALIZER_START,
                    message=f"Generating image (iteration {i + 1}/{total_iters})",
                    iteration=i + 1,
                    extra={"total_iterations": total_iters},
                ),
            )
            visualizer_start = time.perf_counter()
            image_path = await _call_with_retry(
                "visualizer",
                self.visualizer.run,
                description=current_description,
                diagram_type=input.diagram_type,
                raw_data=input.raw_data,
                iteration=iter_index,
                seed=self.settings.seed,
                aspect_ratio=effective_ratio,
                vector_formats=vector_formats,
            )
            visualizer_seconds = time.perf_counter() - visualizer_start
            _emit_progress(
                progress_callback,
                PipelineProgressEvent(
                    stage=PipelineProgressStage.VISUALIZER_END,
                    message=f"Visualizer iteration {i + 1} done",
                    seconds=visualizer_seconds,
                    iteration=i + 1,
                ),
            )
            logger.info(
                f"[Visualizer] Iteration {iter_index}/{total_iters} done",
                seconds=round(visualizer_seconds, 1),
            )
            self._emit_progress(
                "visualizer_completed",
                iteration=iter_index,
                seconds=round(visualizer_seconds, 1),
            )

            # Step 5: Critic — evaluate and provide feedback
            if self._cost_tracker:
                self._cost_tracker.set_agent("critic")
            _emit_progress(
                progress_callback,
                PipelineProgressEvent(
                    stage=PipelineProgressStage.CRITIC_START,
                    message="Critic reviewing",
                    iteration=i + 1,
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
                    iteration=i + 1,
                    extra={
                        "needs_revision": critique.needs_revision,
                        "summary": critique.summary,
                        "critic_suggestions": critique.critic_suggestions[:3],
                    },
                ),
            )
            self._emit_progress(
                "critic_completed",
                iteration=iter_index,
                seconds=round(critic_seconds, 1),
                needs_revision=critique.needs_revision,
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
                iter_dir = ensure_dir(self._run_dir / f"iter_{i + 1}")
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
                    iteration=iter_index,
                    total_iterations=len(iterations),
                    needs_revision=critique.needs_revision,
                )
                break

            self._emit_progress(
                "iteration_completed",
                iteration=iter_index,
                total_iterations=len(iterations),
                needs_revision=critique.needs_revision,
            )

            # Check budget between iterations
            if self._check_budget("between iterations", iteration=iter_index):
                budget_exceeded = True
                break

        # Final output
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
                exclude={"google_api_key", "openai_api_key", "openrouter_api_key"}
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
        metadata_dict["retrieval"] = {
            "mode": retrieval_mode,
            "external_enabled": self.settings.exemplar_retrieval_enabled,
            "external_candidate_ids": external_candidate_ids,
        }
        if generated_caption is not None:
            metadata_dict["generated_caption"] = generated_caption
        if ir_planner_status is not None:
            metadata_dict["ir_planner"] = {
                "status": ir_planner_status,
                "fallback_used": ir_planner_status == "fallback",
                "error": ir_planner_error,
            }

        if self._cost_tracker:
            cost_summary = self._cost_tracker.summary()
            cost_summary["budget_exceeded"] = budget_exceeded
            if self.settings.budget_usd is not None:
                cost_summary["budget_usd"] = self.settings.budget_usd
            metadata_dict["cost"] = cost_summary

        # Include vector output paths when vector export was requested
        if self.settings.vector_export and self.visualizer._last_vector_paths:
            metadata_dict["vector_output_paths"] = self.visualizer._last_vector_paths

        # Always write metadata (including cost) to disk for every run
        save_json(metadata_dict, self._run_dir / "metadata.json")

        output = GenerationOutput(
            image_path=final_output_path,
            description=current_description,
            iterations=iterations,
            metadata=metadata_dict,
            generated_caption=generated_caption,
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
        vector_formats = ["svg", "pdf"] if self.settings.vector_export else None

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
            visualizer_start = time.perf_counter()
            image_path = await _call_with_retry(
                "visualizer",
                self.visualizer.run,
                description=current_description,
                diagram_type=resume_state.diagram_type,
                raw_data=resume_state.raw_data,
                iteration=iter_num,
                seed=self.settings.seed,
                aspect_ratio=resume_state.aspect_ratio,
                vector_formats=vector_formats,
            )
            visualizer_seconds = time.perf_counter() - visualizer_start
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
        final_output_path = self._build_final_output(
            iterations,
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
                exclude={"google_api_key", "openai_api_key", "openrouter_api_key"}
            ),
        )

        metadata_dict = metadata.model_dump()
        metadata_dict["timing"] = {
            "continue_total_seconds": total_seconds,
            "caption_seconds": caption_seconds,
            "iterations": iteration_timings,
        }
        metadata_dict["continued_from_iteration"] = start_iter
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

        if self._cost_tracker:
            cost_summary = self._cost_tracker.summary()
            cost_summary["budget_exceeded"] = budget_exceeded
            if self.settings.budget_usd is not None:
                cost_summary["budget_usd"] = self.settings.budget_usd
            metadata_dict["cost"] = cost_summary

        if self.settings.vector_export and self.visualizer._last_vector_paths:
            metadata_dict["vector_output_paths"] = self.visualizer._last_vector_paths

        # Always write metadata (including cost) to disk for every run
        save_json(metadata_dict, run_dir / "metadata_continued.json")

        output = GenerationOutput(
            image_path=final_output_path,
            description=current_description,
            iterations=iterations,
            metadata=metadata_dict,
            generated_caption=generated_caption,
        )

        return output
