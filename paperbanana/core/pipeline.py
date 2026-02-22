"""Main PaperBanana pipeline orchestration."""

from __future__ import annotations

import datetime
import time
from pathlib import Path
from typing import Optional

import structlog

from paperbanana.agents.critic import CriticAgent
from paperbanana.agents.optimizer import InputOptimizerAgent
from paperbanana.agents.planner import PlannerAgent
from paperbanana.agents.retriever import RetrieverAgent
from paperbanana.agents.stylist import StylistAgent
from paperbanana.agents.visualizer import VisualizerAgent
from paperbanana.core.config import Settings
from paperbanana.core.types import (
    DiagramType,
    GenerationInput,
    GenerationOutput,
    IterationRecord,
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
from paperbanana.reference.store import ReferenceStore

logger = structlog.get_logger()

_ssl_skip_applied = False


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
    ):
        """Initialize the pipeline.

        Args:
            settings: Configuration settings. If None, loads from env/defaults.
            vlm_client: Optional pre-configured VLM client (for HF Spaces demo).
            image_gen_fn: Optional image generation function (for HF Spaces demo).
        """
        self.settings = settings or Settings()
        self.run_id = generate_run_id()

        if self.settings.skip_ssl_verification:
            _apply_ssl_skip()

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

        # Load reference store
        self.reference_store = ReferenceStore(self.settings.reference_set_path)

        # Load guidelines
        guidelines_path = self.settings.guidelines_path
        self._methodology_guidelines = load_methodology_guidelines(guidelines_path)
        self._plot_guidelines = load_plot_guidelines(guidelines_path)

        # Initialize agents
        prompt_dir = self._find_prompt_dir()
        self.optimizer = InputOptimizerAgent(self._vlm, prompt_dir=prompt_dir)
        self.retriever = RetrieverAgent(self._vlm, prompt_dir=prompt_dir)
        self.planner = PlannerAgent(self._vlm, prompt_dir=prompt_dir)
        self.stylist = StylistAgent(
            self._vlm, guidelines=self._methodology_guidelines, prompt_dir=prompt_dir
        )
        self.visualizer = VisualizerAgent(
            self._image_gen,
            self._vlm,
            prompt_dir=prompt_dir,
            output_dir=str(self._run_dir),
        )
        self.critic = CriticAgent(self._vlm, prompt_dir=prompt_dir)

        logger.info(
            "Pipeline initialized",
            run_id=self.run_id,
            vlm=getattr(self._vlm, "name", "custom"),
            image_gen=getattr(self._image_gen, "name", "custom"),
        )

    @property
    def _run_dir(self) -> Path:
        """Directory for this run's outputs."""
        return ensure_dir(Path(self.settings.output_dir) / self.run_id)

    def _find_prompt_dir(self) -> str:
        """Find the prompts directory relative to the package."""
        return find_prompt_dir()

    async def generate(self, input: GenerationInput) -> GenerationOutput:
        """Run the full generation pipeline.

        Args:
            input: Generation input with source context and caption.

        Returns:
            GenerationOutput with final image and metadata.
        """
        total_start = time.perf_counter()

        # Save input for resume/continue support
        if self.settings.save_iterations:
            save_json(
                {
                    "source_context": input.source_context,
                    "communicative_intent": input.communicative_intent,
                    "diagram_type": input.diagram_type.value,
                    "raw_data": input.raw_data,
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
            optimize_start = time.perf_counter()
            optimized = await self.optimizer.run(
                source_context=input.source_context,
                caption=input.communicative_intent,
                diagram_type=input.diagram_type,
            )
            optimize_seconds = time.perf_counter() - optimize_start
            logger.info(
                "[Optimizer] done",
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
            )

        # ── Phase 1: Linear Planning ─────────────────────────────────

        # Step 1: Retriever — find relevant examples
        logger.info("Phase 1: Retrieval")
        candidates = self.reference_store.get_all()
        retrieval_start = time.perf_counter()
        examples = await self.retriever.run(
            source_context=input.source_context,
            caption=input.communicative_intent,
            candidates=candidates,
            num_examples=self.settings.num_retrieval_examples,
            diagram_type=input.diagram_type,
        )
        retrieval_seconds = time.perf_counter() - retrieval_start
        logger.info(
            "[Retriever] done",
            seconds=round(retrieval_seconds, 1),
            examples_found=len(examples),
        )

        # Step 2: Planner — generate textual description
        logger.info("Phase 1: Planning")
        planning_start = time.perf_counter()
        description = await self.planner.run(
            source_context=input.source_context,
            caption=input.communicative_intent,
            examples=examples,
            diagram_type=input.diagram_type,
        )
        planning_seconds = time.perf_counter() - planning_start
        logger.info(
            "[Planner] done",
            seconds=round(planning_seconds, 1),
        )

        # Step 3: Stylist — optimize description aesthetics
        logger.info("Phase 1: Styling")
        styling_start = time.perf_counter()
        optimized_description = await self.stylist.run(
            description=description,
            guidelines=guidelines,
            source_context=input.source_context,
            caption=input.communicative_intent,
            diagram_type=input.diagram_type,
        )
        styling_seconds = time.perf_counter() - styling_start
        logger.info(
            "[Stylist] done",
            seconds=round(styling_seconds, 1),
        )

        # Save planning outputs
        if self.settings.save_iterations:
            save_json(
                {
                    "retrieved_examples": [e.id for e in examples],
                    "initial_description": description,
                    "optimized_description": optimized_description,
                },
                self._run_dir / "planning.json",
            )

        # ── Phase 2: Iterative Refinement ─────────────────────────────

        current_description = optimized_description
        iterations: list[IterationRecord] = []
        iteration_timings = []

        if self.settings.auto_refine:
            total_iters = self.settings.max_iterations
        else:
            total_iters = self.settings.refinement_iterations

        for i in range(total_iters):
            logger.info(
                f"Phase 2: Iteration {i + 1}/{total_iters}"
                + (" (auto)" if self.settings.auto_refine else "")
            )

            # Step 4: Visualizer — generate image
            visualizer_start = time.perf_counter()
            image_path = await self.visualizer.run(
                description=current_description,
                diagram_type=input.diagram_type,
                raw_data=input.raw_data,
                iteration=i + 1,
            )
            visualizer_seconds = time.perf_counter() - visualizer_start
            logger.info(
                f"[Visualizer] Iteration {i + 1}/{total_iters} done",
                seconds=round(visualizer_seconds, 1),
            )

            # Step 5: Critic — evaluate and provide feedback
            critic_start = time.perf_counter()
            critique = await self.critic.run(
                image_path=image_path,
                description=current_description,
                source_context=input.source_context,
                caption=input.communicative_intent,
                diagram_type=input.diagram_type,
            )
            critic_seconds = time.perf_counter() - critic_start
            logger.info(
                "[Critic] done",
                seconds=round(critic_seconds, 1),
                needs_revision=critique.needs_revision,
            )

            iteration_record = IterationRecord(
                iteration=i + 1,
                description=current_description,
                image_path=image_path,
                critique=critique,
            )
            iteration_timings.append(
                {
                    "iteration": i + 1,
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
                    iteration=i + 1,
                    summary=critique.summary,
                )
                current_description = critique.revised_description
            else:
                logger.info(
                    "No further revision needed",
                    iteration=i + 1,
                    summary=critique.summary,
                )
                break

        # Final output
        final_image = iterations[-1].image_path
        output_format = getattr(self.settings, "output_format", "png").lower()
        ext = "jpg" if output_format == "jpeg" else output_format
        final_output_path = str(self._run_dir / f"final_output.{ext}")

        # Load and save in desired format (handles PNG→JPEG/WebP conversion)
        img = load_image(final_image)
        save_image(img, final_output_path, format=output_format)

        total_seconds = time.perf_counter() - total_start
        logger.info(
            "Total generation time",
            run_id=self.run_id,
            total_seconds=total_seconds,
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
            "iterations": iteration_timings,
        }

        if self.settings.save_iterations:
            save_json(metadata_dict, self._run_dir / "metadata.json")

        output = GenerationOutput(
            image_path=final_output_path,
            description=current_description,
            iterations=iterations,
            metadata=metadata_dict,
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

        iterations: list[IterationRecord] = []
        iteration_timings = []

        for i in range(total_iters):
            iter_num = start_iter + i + 1
            logger.info(
                f"Phase 2: Iteration {iter_num}" + (" (auto)" if self.settings.auto_refine else "")
            )

            # Visualizer — generate image
            visualizer_start = time.perf_counter()
            image_path = await self.visualizer.run(
                description=current_description,
                diagram_type=resume_state.diagram_type,
                raw_data=resume_state.raw_data,
                iteration=iter_num,
            )
            visualizer_seconds = time.perf_counter() - visualizer_start
            logger.info(
                f"[Visualizer] Iteration {iter_num} done",
                seconds=round(visualizer_seconds, 1),
            )

            # Critic — evaluate with optional user feedback
            critic_start = time.perf_counter()
            critique = await self.critic.run(
                image_path=image_path,
                description=current_description,
                source_context=resume_state.source_context,
                caption=resume_state.communicative_intent,
                diagram_type=resume_state.diagram_type,
                user_feedback=user_feedback,
            )
            critic_seconds = time.perf_counter() - critic_start
            logger.info(
                "[Critic] done",
                seconds=round(critic_seconds, 1),
                needs_revision=critique.needs_revision,
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

        # Final output
        final_image = iterations[-1].image_path
        output_format = getattr(self.settings, "output_format", "png").lower()
        ext = "jpg" if output_format == "jpeg" else output_format
        final_output_path = str(run_dir / f"final_output.{ext}")

        img = load_image(final_image)
        save_image(img, final_output_path, format=output_format)

        total_seconds = time.perf_counter() - total_start
        logger.info(
            "Continue run complete",
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
            config_snapshot=self.settings.model_dump(
                exclude={"google_api_key", "openai_api_key", "openrouter_api_key"}
            ),
        )

        metadata_dict = metadata.model_dump()
        metadata_dict["timing"] = {
            "continue_total_seconds": total_seconds,
            "iterations": iteration_timings,
        }
        metadata_dict["continued_from_iteration"] = start_iter
        if user_feedback:
            metadata_dict["user_feedback"] = user_feedback

        if self.settings.save_iterations:
            save_json(metadata_dict, run_dir / "metadata_continued.json")

        output = GenerationOutput(
            image_path=final_output_path,
            description=current_description,
            iterations=iterations,
            metadata=metadata_dict,
        )

        return output
