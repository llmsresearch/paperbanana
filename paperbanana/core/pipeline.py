"""Main PaperBanana pipeline orchestration."""

from __future__ import annotations

import asyncio
import datetime
import shutil
from pathlib import Path
from typing import AsyncIterator, Optional

import structlog

from paperbanana.agents.critic import CriticAgent
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
from paperbanana.core.utils import ensure_dir, generate_run_id, save_json
from paperbanana.guidelines.methodology import load_methodology_guidelines
from paperbanana.guidelines.plots import load_plot_guidelines
from paperbanana.providers.registry import ProviderRegistry
from paperbanana.reference.store import ReferenceStore

logger = structlog.get_logger()

_ssl_skip_applied = False

# Valid experiment modes (compatible with official PaperBanana)
VALID_MODES = {
    "vanilla", "planner", "planner_stylist", "planner_critic",
    "full", "polish",
    # Official naming aliases
    "dev_planner", "dev_planner_stylist", "dev_planner_critic",
    "dev_full", "demo_full", "dev_polish",
}


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


def _normalize_mode(mode: str) -> str:
    """Normalize official mode names to simplified internal names."""
    aliases = {
        "dev_planner": "planner",
        "dev_planner_stylist": "planner_stylist",
        "dev_planner_critic": "planner_critic",
        "demo_planner_critic": "planner_critic",
        "dev_full": "full",
        "demo_full": "full",
        "dev_polish": "polish",
    }
    return aliases.get(mode, mode)


class PaperBananaPipeline:
    """Main orchestration pipeline for academic illustration generation.

    Supports multiple experiment modes:
    - vanilla: Direct generation (no retrieval/planning)
    - planner: Retriever -> Planner -> Visualizer
    - planner_stylist: Retriever -> Planner -> Stylist -> Visualizer
    - planner_critic: Retriever -> Planner -> Visualizer -> Critic loop
    - full: Retriever -> Planner -> Stylist -> Visualizer -> Critic loop (default)
    - polish: Style-guide-based refinement of existing images
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

        # Initialize core agents
        prompt_dir = self._find_prompt_dir()
        self._prompt_dir = prompt_dir
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

        # Lazy-initialized agents and providers (only created when needed)
        self._vanilla_agent = None
        self._polish_agent = None
        self._polish_image_gen = None

        logger.info(
            "Pipeline initialized",
            run_id=self.run_id,
            mode=self.settings.exp_mode,
            vlm=getattr(self._vlm, "name", "custom"),
            image_gen=getattr(self._image_gen, "name", "custom"),
        )

    @property
    def _run_dir(self) -> Path:
        """Directory for this run's outputs."""
        return ensure_dir(Path(self.settings.output_dir) / self.run_id)

    def _find_prompt_dir(self) -> str:
        """Find the prompts directory relative to the package."""
        candidates = [
            Path("prompts"),
            Path(__file__).parent.parent.parent / "prompts",
        ]
        for p in candidates:
            if p.exists():
                return str(p)
        return "prompts"

    def _get_vanilla_agent(self):
        """Lazy-initialize the VanillaAgent."""
        if self._vanilla_agent is None:
            from paperbanana.agents.vanilla import VanillaAgent

            self._vanilla_agent = VanillaAgent(
                vlm_provider=self._vlm,
                image_gen=self._image_gen,
                prompt_dir=self._prompt_dir,
                output_dir=str(self._run_dir),
            )
        return self._vanilla_agent

    def _get_polish_image_gen(self):
        """Lazy-initialize the polish-specific image gen (supports image editing)."""
        if self._polish_image_gen is None:
            if self._demo_mode:
                self._polish_image_gen = self._image_gen
            else:
                self._polish_image_gen = ProviderRegistry.create_polish_image_gen(
                    self.settings
                )
        return self._polish_image_gen

    def _get_polish_agent(self):
        """Lazy-initialize the PolishAgent."""
        if self._polish_agent is None:
            from paperbanana.agents.polish import PolishAgent

            style_guides_dir = None
            # Try to locate style guides
            for candidate in [
                Path("data/style_guides"),
                Path(self.settings.guidelines_path).parent / "style_guides",
            ]:
                if candidate.exists():
                    style_guides_dir = str(candidate)
                    break

            self._polish_agent = PolishAgent(
                vlm_provider=self._vlm,
                image_gen=self._get_polish_image_gen(),
                prompt_dir=self._prompt_dir,
                output_dir=str(self._run_dir),
                style_guides_dir=style_guides_dir,
            )
        return self._polish_agent

    # ── Public API ────────────────────────────────────────────────────

    async def generate(self, input: GenerationInput) -> GenerationOutput:
        """Run the generation pipeline in the configured experiment mode.

        Args:
            input: Generation input with source context and caption.

        Returns:
            GenerationOutput with final image and metadata.
        """
        mode = _normalize_mode(self.settings.exp_mode)

        # Auto VLM selection (DeferredVLMProvider)
        if hasattr(self._vlm, "select_model"):
            selected = self._vlm.select_model(input)
            logger.info("Auto VLM resolved", model=selected)

        logger.info(
            "Starting generation",
            run_id=self.run_id,
            mode=mode,
            diagram_type=input.diagram_type.value,
            context_length=len(input.source_context),
            vlm_model=getattr(self._vlm, "model_name", "unknown"),
        )

        if mode == "vanilla":
            return await self._run_vanilla(input)
        elif mode == "planner":
            return await self._run_planner(input)
        elif mode == "planner_stylist":
            return await self._run_planner_stylist(input)
        elif mode == "planner_critic":
            return await self._run_planner_critic(input)
        elif mode == "polish":
            return await self._run_polish(input)
        elif mode == "full":
            return await self._run_full(input)
        else:
            raise ValueError(
                f"Unknown experiment mode: {self.settings.exp_mode}. "
                f"Available: {', '.join(sorted(VALID_MODES))}"
            )

    async def batch_generate(
        self,
        inputs: list[GenerationInput],
        max_concurrent: Optional[int] = None,
    ) -> AsyncIterator[GenerationOutput]:
        """Process multiple inputs with concurrency control.

        Args:
            inputs: List of generation inputs.
            max_concurrent: Max concurrent tasks (defaults to settings.batch_concurrent).

        Yields:
            GenerationOutput for each completed input (in completion order).
        """
        concurrent = max_concurrent or self.settings.batch_concurrent
        semaphore = asyncio.Semaphore(concurrent)

        async def _process_one(inp: GenerationInput) -> GenerationOutput:
            async with semaphore:
                # Each batch item gets its own run_id
                saved = self.run_id
                self.run_id = generate_run_id()
                try:
                    return await self.generate(inp)
                finally:
                    self.run_id = saved

        tasks = [asyncio.create_task(_process_one(inp)) for inp in inputs]

        for future in asyncio.as_completed(tasks):
            result = await future
            yield result

    # ── Mode implementations ──────────────────────────────────────────

    async def _run_vanilla(self, input: GenerationInput) -> GenerationOutput:
        """Vanilla mode: direct generation without retrieval or planning."""
        vanilla = self._get_vanilla_agent()

        image_path = await vanilla.run(
            source_context=input.source_context,
            visual_intent=input.communicative_intent,
            diagram_type=input.diagram_type,
            raw_data=input.raw_data,
            output_path=str(self._run_dir / "vanilla_output.png"),
        )

        final_path = str(self._run_dir / "final_output.png")
        shutil.copy2(image_path, final_path)

        return self._build_output(
            final_path=final_path,
            description=input.communicative_intent,
            iterations=[IterationRecord(
                iteration=1,
                description=input.communicative_intent,
                image_path=image_path,
            )],
        )

    async def _run_planner(self, input: GenerationInput) -> GenerationOutput:
        """Planner mode: Retriever -> Planner -> Visualizer."""
        examples, description = await self._phase1_retrieve_and_plan(input)

        image_path = await self.visualizer.run(
            description=description,
            diagram_type=input.diagram_type,
            raw_data=input.raw_data,
            iteration=1,
        )

        final_path = str(self._run_dir / "final_output.png")
        shutil.copy2(image_path, final_path)

        if self.settings.save_iterations:
            save_json(
                {"retrieved_examples": [e.id for e in examples],
                 "description": description},
                self._run_dir / "planning.json",
            )

        return self._build_output(
            final_path=final_path,
            description=description,
            iterations=[IterationRecord(
                iteration=1, description=description, image_path=image_path,
            )],
        )

    async def _run_planner_stylist(self, input: GenerationInput) -> GenerationOutput:
        """Planner+Stylist mode: Retriever -> Planner -> Stylist -> Visualizer."""
        examples, description = await self._phase1_retrieve_and_plan(input)
        guidelines = self._get_guidelines(input.diagram_type)

        optimized = await self.stylist.run(
            description=description,
            guidelines=guidelines,
            source_context=input.source_context,
            caption=input.communicative_intent,
            diagram_type=input.diagram_type,
        )

        image_path = await self.visualizer.run(
            description=optimized,
            diagram_type=input.diagram_type,
            raw_data=input.raw_data,
            iteration=1,
        )

        final_path = str(self._run_dir / "final_output.png")
        shutil.copy2(image_path, final_path)

        if self.settings.save_iterations:
            save_json(
                {"retrieved_examples": [e.id for e in examples],
                 "initial_description": description,
                 "optimized_description": optimized},
                self._run_dir / "planning.json",
            )

        return self._build_output(
            final_path=final_path,
            description=optimized,
            iterations=[IterationRecord(
                iteration=1, description=optimized, image_path=image_path,
            )],
        )

    async def _run_planner_critic(self, input: GenerationInput) -> GenerationOutput:
        """Planner+Critic mode: Retriever -> Planner -> Visualizer -> Critic loop."""
        examples, description = await self._phase1_retrieve_and_plan(input)

        if self.settings.save_iterations:
            save_json(
                {"retrieved_examples": [e.id for e in examples],
                 "description": description},
                self._run_dir / "planning.json",
            )

        # Critic iteration loop with rollback
        return await self._critic_iteration_loop(input, description)

    async def _run_full(self, input: GenerationInput) -> GenerationOutput:
        """Full mode: Retriever -> Planner -> Stylist -> Visualizer -> Critic loop."""
        guidelines = self._get_guidelines(input.diagram_type)

        # Phase 1: Linear planning
        logger.info("Phase 1: Retrieval")
        candidates = self.reference_store.get_all()
        examples = await self.retriever.run(
            source_context=input.source_context,
            caption=input.communicative_intent,
            candidates=candidates,
            num_examples=self.settings.num_retrieval_examples,
            diagram_type=input.diagram_type,
        )

        logger.info("Phase 1: Planning")
        description = await self.planner.run(
            source_context=input.source_context,
            caption=input.communicative_intent,
            examples=examples,
            diagram_type=input.diagram_type,
        )

        logger.info("Phase 1: Styling")
        optimized = await self.stylist.run(
            description=description,
            guidelines=guidelines,
            source_context=input.source_context,
            caption=input.communicative_intent,
            diagram_type=input.diagram_type,
        )

        if self.settings.save_iterations:
            save_json(
                {"retrieved_examples": [e.id for e in examples],
                 "initial_description": description,
                 "optimized_description": optimized},
                self._run_dir / "planning.json",
            )

        # Phase 2: Critic iteration loop with rollback
        return await self._critic_iteration_loop(input, optimized)

    async def _run_polish(self, input: GenerationInput) -> GenerationOutput:
        """Polish mode: refine an existing image with style guide suggestions."""
        polish = self._get_polish_agent()

        # Polish requires a ground truth image path in raw_data
        gt_path = None
        if input.raw_data and "gt_image_path" in input.raw_data:
            gt_path = input.raw_data["gt_image_path"]

        if gt_path is None:
            raise ValueError(
                "Polish mode requires raw_data={'gt_image_path': '...'} "
                "pointing to the ground truth image to polish."
            )

        result = await polish.run(
            image=gt_path,
            diagram_type=input.diagram_type,
            output_path=str(self._run_dir / "polished.png"),
        )

        final_path = str(self._run_dir / "final_output.png")
        shutil.copy2(result.polished_image_path, final_path)

        return self._build_output(
            final_path=final_path,
            description=f"Polished: {result.suggestions[:200]}",
            iterations=[IterationRecord(
                iteration=1,
                description=result.suggestions,
                image_path=result.polished_image_path,
            )],
            extra_metadata={"changed": result.changed, "suggestions": result.suggestions},
        )

    # ── Shared helpers ────────────────────────────────────────────────

    async def _phase1_retrieve_and_plan(self, input: GenerationInput):
        """Run retrieval + planning (shared by planner/planner_stylist/planner_critic)."""
        logger.info("Phase 1: Retrieval")
        candidates = self.reference_store.get_all()
        examples = await self.retriever.run(
            source_context=input.source_context,
            caption=input.communicative_intent,
            candidates=candidates,
            num_examples=self.settings.num_retrieval_examples,
            diagram_type=input.diagram_type,
        )

        logger.info("Phase 1: Planning")
        description = await self.planner.run(
            source_context=input.source_context,
            caption=input.communicative_intent,
            examples=examples,
            diagram_type=input.diagram_type,
        )

        return examples, description

    async def _critic_iteration_loop(
        self,
        input: GenerationInput,
        initial_description: str,
    ) -> GenerationOutput:
        """Run Visualizer <-> Critic loop with rollback on generation failure.

        Tracks `current_best_image_path` across iterations. If the Visualizer
        fails to produce a valid image in a round, we roll back to the
        previous best instead of crashing.
        """
        max_rounds = self.settings.max_critic_rounds
        current_description = initial_description
        iterations: list[IterationRecord] = []
        current_best_image_path: Optional[str] = None

        for i in range(max_rounds):
            logger.info(f"Phase 2: Iteration {i + 1}/{max_rounds}")

            # Visualizer: generate image
            try:
                image_path = await self.visualizer.run(
                    description=current_description,
                    diagram_type=input.diagram_type,
                    raw_data=input.raw_data,
                    iteration=i + 1,
                )

                # Validate output exists
                if image_path and Path(image_path).exists():
                    current_best_image_path = image_path
                else:
                    logger.warning(
                        "Visualizer output missing, rolling back",
                        iteration=i + 1,
                        expected=image_path,
                    )
                    if current_best_image_path:
                        break
                    raise RuntimeError(f"First visualization failed: {image_path}")
            except Exception as e:
                logger.error(
                    "Visualizer failed, rolling back to previous best",
                    iteration=i + 1,
                    error=str(e),
                )
                if current_best_image_path:
                    break
                raise

            # Critic: evaluate and provide feedback
            critique = await self.critic.run(
                image_path=current_best_image_path,
                description=current_description,
                source_context=input.source_context,
                caption=input.communicative_intent,
                diagram_type=input.diagram_type,
            )

            iteration_record = IterationRecord(
                iteration=i + 1,
                description=current_description,
                image_path=current_best_image_path,
                critique=critique,
            )
            iterations.append(iteration_record)

            # Save iteration artifacts
            if self.settings.save_iterations:
                iter_dir = ensure_dir(self._run_dir / f"iter_{i + 1}")
                save_json(
                    {"description": current_description,
                     "critique": critique.model_dump()},
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
        final_image = current_best_image_path or iterations[-1].image_path
        final_output_path = str(self._run_dir / "final_output.png")
        shutil.copy2(final_image, final_output_path)

        return self._build_output(
            final_path=final_output_path,
            description=current_description,
            iterations=iterations,
        )

    def _get_guidelines(self, diagram_type: DiagramType) -> str:
        """Get guidelines text for the given diagram type."""
        return (
            self._methodology_guidelines
            if diagram_type == DiagramType.METHODOLOGY
            else self._plot_guidelines
        )

    def _build_output(
        self,
        final_path: str,
        description: str,
        iterations: list[IterationRecord],
        extra_metadata: Optional[dict] = None,
    ) -> GenerationOutput:
        """Build the final GenerationOutput with metadata."""
        metadata = RunMetadata(
            run_id=self.run_id,
            timestamp=datetime.datetime.now().isoformat(),
            vlm_provider=getattr(self._vlm, "name", "custom"),
            vlm_model=getattr(self._vlm, "model_name", "custom"),
            image_provider=getattr(self._image_gen, "name", "custom"),
            image_model=getattr(self._image_gen, "model_name", "custom"),
            refinement_iterations=len(iterations),
            config_snapshot=self.settings.model_dump(
                exclude={"google_api_key"}
            ),
        )

        if self.settings.save_iterations:
            meta_dict = metadata.model_dump()
            if extra_metadata:
                meta_dict.update(extra_metadata)
            save_json(meta_dict, self._run_dir / "metadata.json")

        output = GenerationOutput(
            image_path=final_path,
            description=description,
            iterations=iterations,
            metadata=metadata.model_dump(),
        )

        logger.info(
            "Generation complete",
            run_id=self.run_id,
            mode=self.settings.exp_mode,
            output=final_path,
            total_iterations=len(iterations),
        )

        return output
