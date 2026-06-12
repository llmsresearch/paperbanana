"""End-to-end benchmark harness for PaperBananaBench.

Runs generation + evaluation across the full benchmark dataset (or a filtered
subset) and produces an aggregated report with win rates, per-dimension scores,
and per-category breakdowns.
"""

from __future__ import annotations

import datetime
import os
import time
from pathlib import Path
from typing import Callable, Optional, TypeVar, Union

import structlog
from pydantic import BaseModel, Field

from paperbanana.agents.visualizer import VisualizerAgent
from paperbanana.core.config import Settings
from paperbanana.core.pipeline import PaperBananaPipeline
from paperbanana.core.types import (
    DiagramType,
    GenerationInput,
    ReferenceExample,
    TestCase,
)
from paperbanana.core.utils import ensure_dir, save_json
from paperbanana.evaluation.judge import DIMENSIONS, VLMJudge
from paperbanana.evaluation.metrics import scores_to_dict
from paperbanana.providers.registry import ProviderRegistry
from paperbanana.reference.store import ReferenceStore

logger = structlog.get_logger()


# ── Report models ────────────────────────────────────────────────


class BenchmarkEntryResult(BaseModel):
    """Result for a single benchmark entry."""

    id: str
    category: str = ""
    task: str = "diagram"
    difficulty: Optional[str] = None
    run_id: Optional[str] = None
    image_path: Optional[str] = None
    iteration_count: int = 0
    generation_seconds: float = 0.0
    evaluation: Optional[dict] = None
    error: Optional[str] = None


class BenchmarkReport(BaseModel):
    """Full benchmark run report."""

    created_at: str
    run_dir: Optional[str] = None  # Directory where report and partial results were written
    split: str = "reference"  # Entry source: curated 'reference' pool or official 'test' split
    task: Optional[str] = None  # 'diagram' or 'plot' for official test-split runs
    mode: str = "full"  # 'full' pipeline or 'vanilla' (single direct visualizer call)
    settings_snapshot: dict = Field(default_factory=dict)
    total_entries: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    total_seconds: float = 0.0
    entries: list[BenchmarkEntryResult] = Field(default_factory=list)
    summary: dict = Field(default_factory=dict)


# ── Filtering ────────────────────────────────────────────────────

BenchmarkEntry = Union[ReferenceExample, TestCase]
EntryT = TypeVar("EntryT", ReferenceExample, TestCase)


def filter_examples(
    examples: list[EntryT],
    *,
    category: Optional[str] = None,
    ids: Optional[list[str]] = None,
    limit: Optional[int] = None,
) -> list[EntryT]:
    """Filter benchmark entries (reference or test split) by category, IDs, or subset size."""
    result = examples

    if ids:
        id_set = set(ids)
        result = [e for e in result if e.id in id_set]

    if category:
        result = [e for e in result if e.category == category]

    if limit and limit > 0:
        result = result[:limit]

    return result


# ── Aggregation ──────────────────────────────────────────────────


def aggregate_results(entries: list[BenchmarkEntryResult]) -> dict:
    """Compute win rates, mean scores, and per-category breakdowns."""
    scored = [e for e in entries if e.evaluation is not None]
    if not scored:
        return {}

    # Overall win rates (defensive .get for malformed evaluation dicts)
    def _winner(e: BenchmarkEntryResult) -> str:
        return (e.evaluation or {}).get("overall_winner", "")

    def _score(e: BenchmarkEntryResult) -> float:
        return float((e.evaluation or {}).get("overall_score", 0.0))

    model_wins = sum(1 for e in scored if _winner(e) == "Model")
    human_wins = sum(1 for e in scored if _winner(e) == "Human")
    ties = len(scored) - model_wins - human_wins

    overall_scores = [_score(e) for e in scored]
    mean_overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0

    # Per-dimension means
    dimension_means: dict[str, float] = {}
    for dim in DIMENSIONS:
        key = f"{dim}_score"
        values = [float(e.evaluation[key]) for e in scored if e.evaluation and key in e.evaluation]
        if values:
            dimension_means[dim] = round(sum(values) / len(values), 1)

    # Per-category breakdown
    categories: dict[str, list[BenchmarkEntryResult]] = {}
    for e in scored:
        cat = e.category or "uncategorized"
        categories.setdefault(cat, []).append(e)

    category_breakdown: dict[str, dict] = {}
    for cat, cat_entries in sorted(categories.items()):
        cat_scores = [_score(e) for e in cat_entries]
        cat_model = sum(1 for e in cat_entries if _winner(e) == "Model")
        n = len(cat_entries)
        category_breakdown[cat] = {
            "count": n,
            "model_win_rate": round(cat_model / n * 100, 1) if n else 0.0,
            "mean_score": round(sum(cat_scores) / n, 1) if n else 0.0,
        }

    # Per-difficulty breakdown (official plot test split carries difficulty labels)
    difficulty_breakdown: dict[str, dict] = {}
    by_difficulty: dict[str, list[BenchmarkEntryResult]] = {}
    for e in scored:
        if e.difficulty:
            by_difficulty.setdefault(e.difficulty, []).append(e)
    for diff, diff_entries in sorted(by_difficulty.items()):
        diff_scores = [_score(e) for e in diff_entries]
        diff_model = sum(1 for e in diff_entries if _winner(e) == "Model")
        n = len(diff_entries)
        difficulty_breakdown[diff] = {
            "count": n,
            "model_win_rate": round(diff_model / n * 100, 1) if n else 0.0,
            "mean_score": round(sum(diff_scores) / n, 1) if n else 0.0,
        }

    gen_times = [e.generation_seconds for e in scored if e.generation_seconds > 0]
    mean_gen_time = round(sum(gen_times) / len(gen_times), 1) if gen_times else 0.0

    summary = {
        "evaluated": len(scored),
        "model_wins": model_wins,
        "human_wins": human_wins,
        "ties": ties,
        "model_win_rate": round(model_wins / len(scored) * 100, 1),
        "mean_overall_score": round(mean_overall, 1),
        "dimension_means": dimension_means,
        "category_breakdown": category_breakdown,
        "mean_generation_seconds": mean_gen_time,
    }
    if difficulty_breakdown:
        summary["difficulty_breakdown"] = difficulty_breakdown
    return summary


# ── Runner ───────────────────────────────────────────────────────


class BenchmarkRunner:
    """Runs generation + evaluation across PaperBananaBench entries."""

    def __init__(
        self,
        settings: Settings,
        *,
        pipeline_factory: Callable[[Settings], PaperBananaPipeline] = PaperBananaPipeline,
        judge_factory: Optional[Callable[[Settings], VLMJudge]] = None,
        visualizer_factory: Optional[Callable[[Settings], VisualizerAgent]] = None,
    ):
        self.settings = settings
        self.pipeline_factory = pipeline_factory
        self.judge_factory = judge_factory or self._default_judge_factory
        # Used only in vanilla mode: a single direct Visualizer call per entry.
        self.visualizer_factory = visualizer_factory or self._default_visualizer_factory
        # Concurrency for processing benchmark entries (generation + evaluation).
        # Defaults to 1 to preserve existing sequential behaviour unless
        # explicitly overridden by the caller.
        self.concurrency: int = max(1, getattr(settings, "benchmark_concurrency", 1))

    def _default_judge_factory(self, settings: Settings) -> VLMJudge:
        from paperbanana.core.utils import find_prompt_dir

        judge_settings = settings
        if settings.judge_vlm_provider:
            judge_settings = settings.model_copy(
                update={"vlm_provider": settings.judge_vlm_provider}
            )
        vlm = ProviderRegistry.create_vlm(judge_settings, model_override=settings.judge_vlm_model)
        return VLMJudge(vlm, prompt_dir=find_prompt_dir())

    def _default_visualizer_factory(self, settings: Settings) -> VisualizerAgent:
        from paperbanana.core.utils import find_prompt_dir

        vlm = ProviderRegistry.create_vlm(settings)
        image_gen = ProviderRegistry.create_image_gen(settings)
        prompt_dir = settings.prompt_dir or find_prompt_dir()
        return VisualizerAgent(
            image_gen,
            vlm,
            prompt_dir=prompt_dir,
            output_resolution=settings.output_resolution,
            image_quality=settings.image_quality,
        )

    def load_test_entries(
        self,
        task: str = "diagram",
        *,
        category: Optional[str] = None,
        ids: Optional[list[str]] = None,
        limit: Optional[int] = None,
    ) -> list[TestCase]:
        """Load the official PaperBananaBench test split with optional filtering."""
        from paperbanana.data.manager import DatasetManager

        manager = DatasetManager()
        cases = manager.load_test_split(task)
        filtered = filter_examples(cases, category=category, ids=ids, limit=limit)
        logger.info(
            "Loaded official test split",
            task=task,
            total=len(cases),
            filtered=len(filtered),
        )
        return filtered

    def load_entries(
        self,
        *,
        category: Optional[str] = None,
        ids: Optional[list[str]] = None,
        limit: Optional[int] = None,
    ) -> list[ReferenceExample]:
        """Load benchmark entries from the reference store with optional filtering."""
        store = ReferenceStore.from_settings(self.settings)
        examples = store.get_all()

        if not examples:
            raise ValueError("No benchmark entries found. Run 'paperbanana data download' first.")

        filtered = filter_examples(examples, category=category, ids=ids, limit=limit)
        logger.info(
            "Loaded benchmark entries",
            total=len(examples),
            filtered=len(filtered),
        )
        return filtered

    async def run(
        self,
        entries: list[BenchmarkEntry],
        *,
        output_dir: Optional[Path] = None,
        eval_only_dir: Optional[str] = None,
        mode: str = "full",
        split: str = "reference",
        task: Optional[str] = None,
    ) -> BenchmarkReport:
        """Run the benchmark across all entries.

        Args:
            entries: Benchmark entries to process (curated ReferenceExample pool
                or official-test-split TestCase entries).
            output_dir: Directory for outputs. Auto-generated if None.
            eval_only_dir: If set, skip generation and evaluate existing images
                from this directory. Expects <entry_id>/final_output.png layout.
            mode: 'full' runs the complete agentic pipeline; 'vanilla' makes a
                single direct Visualizer call (no retrieval/planning/styling/critique)
                for ablation parity with the upstream baseline.
            split: Label recorded in the report ('reference' or 'test').
            task: Label recorded in the report ('diagram' or 'plot') for test-split runs.
        """
        if mode not in ("full", "vanilla"):
            raise ValueError(f"mode must be 'full' or 'vanilla', got: {mode}")
        run_dir = (
            Path(output_dir)
            if output_dir is not None
            else Path(self.settings.output_dir) / f"benchmark_{_timestamp()}"
        )
        run_dir = ensure_dir(Path(run_dir))

        judge = self.judge_factory(self.settings)
        results: list[BenchmarkEntryResult] = []
        total_start = time.perf_counter()

        # Process entries with bounded concurrency so that network-bound
        # generation/evaluation can run in parallel without overwhelming
        # providers or local resources.
        import asyncio

        semaphore = asyncio.Semaphore(self.concurrency)
        results_lock = asyncio.Lock()

        async def _run_single(idx: int, entry: BenchmarkEntry) -> None:
            async with semaphore:
                logger.info(
                    f"Benchmark entry {idx + 1}/{len(entries)}",
                    id=entry.id,
                    category=entry.category,
                )
                result = await self._process_entry(
                    entry, judge=judge, run_dir=run_dir, eval_only_dir=eval_only_dir, mode=mode
                )
            # Append and save partials under a lock so that writes remain
            # consistent even when many tasks complete at once.
            async with results_lock:
                results.append(result)
                _save_partial(results, run_dir)

        tasks = [_run_single(i, entry) for i, entry in enumerate(entries)]
        if tasks:
            await asyncio.gather(*tasks)

        total_seconds = time.perf_counter() - total_start

        completed = [r for r in results if r.evaluation is not None]
        failed = [r for r in results if r.error is not None]
        skipped = [r for r in results if r.error is None and r.evaluation is None]

        report = BenchmarkReport(
            created_at=datetime.datetime.now().isoformat(),
            run_dir=str(run_dir),
            split=split,
            task=task,
            mode=mode,
            settings_snapshot=self.settings.model_dump(
                exclude={
                    "google_api_key",
                    "openai_api_key",
                    "openrouter_api_key",
                    "anthropic_api_key",
                }
            ),
            total_entries=len(entries),
            completed=len(completed),
            failed=len(failed),
            skipped=len(skipped),
            total_seconds=round(total_seconds, 1),
            entries=results,
            summary=aggregate_results(results),
        )

        save_json(report.model_dump(), run_dir / "benchmark_report.json")
        logger.info(
            "Benchmark complete",
            completed=len(completed),
            failed=len(failed),
            total_seconds=round(total_seconds, 1),
        )
        return report

    async def _process_entry(
        self,
        entry: BenchmarkEntry,
        *,
        judge: VLMJudge,
        run_dir: Path,
        eval_only_dir: Optional[str] = None,
        mode: str = "full",
    ) -> BenchmarkEntryResult:
        """Generate and evaluate a single benchmark entry."""
        is_test_case = isinstance(entry, TestCase)
        reference_path = entry.gt_image_path if is_test_case else entry.image_path
        diagram_type = (
            DiagramType.STATISTICAL_PLOT
            if is_test_case and entry.task == "plot"
            else DiagramType.METHODOLOGY
        )
        aspect_ratio = entry.aspect_ratio if is_test_case else None
        raw_data = entry.raw_data if is_test_case else None

        result = BenchmarkEntryResult(
            id=entry.id,
            category=entry.category or "",
            task=entry.task if is_test_case else "diagram",
            difficulty=entry.difficulty if is_test_case else None,
        )

        # Skip entries without reference images (can't evaluate)
        if not reference_path or not Path(reference_path).exists():
            result.error = "reference image not found"
            logger.warning("Skipping entry: no reference image", id=entry.id)
            return result

        # Prevent path traversal: entry.id is used as a single path component
        # for eval-only lookups and vanilla-mode output directories.
        if (eval_only_dir or mode == "vanilla") and (".." in entry.id or os.sep in entry.id):
            result.error = "invalid entry id for filesystem path"
            logger.warning("Skipping entry: invalid id", id=entry.id)
            return result

        # Generation (or locate existing output for eval-only mode)
        image_path: Optional[str] = None

        if eval_only_dir:
            base = Path(eval_only_dir)
            candidate = base / entry.id / "final_output.png"
            if not candidate.exists():
                candidate = base / f"{entry.id}.png"
            if candidate.exists():
                image_path = str(candidate)
            else:
                result.error = f"generated image not found in {eval_only_dir}"
                logger.warning("Skipping eval-only entry: image not found", id=entry.id)
                return result
        elif mode == "vanilla":
            try:
                gen_start = time.perf_counter()
                image_path = await self._vanilla_generate(
                    entry_id=entry.id,
                    source_context=entry.source_context,
                    caption=entry.caption,
                    diagram_type=diagram_type,
                    raw_data=raw_data,
                    aspect_ratio=aspect_ratio,
                    run_dir=run_dir,
                )
                result.generation_seconds = round(time.perf_counter() - gen_start, 1)
                result.image_path = image_path
                result.iteration_count = 1
            except Exception as e:
                result.error = str(e)
                logger.error("Vanilla generation failed", id=entry.id, error=str(e))
                return result
        else:
            try:
                gen_start = time.perf_counter()
                pipeline = self.pipeline_factory(self.settings)
                gen_output = await pipeline.generate(
                    GenerationInput(
                        source_context=entry.source_context,
                        communicative_intent=entry.caption,
                        diagram_type=diagram_type,
                        raw_data=raw_data,
                        aspect_ratio=aspect_ratio,
                    )
                )
                result.generation_seconds = round(time.perf_counter() - gen_start, 1)
                result.run_id = gen_output.metadata.get("run_id")
                result.image_path = gen_output.image_path
                result.iteration_count = len(gen_output.iterations)
                image_path = gen_output.image_path
            except Exception as e:
                result.error = str(e)
                logger.error("Generation failed", id=entry.id, error=str(e))
                return result

        # Evaluation
        try:
            scores = await judge.evaluate(
                image_path=image_path,
                source_context=entry.source_context,
                caption=entry.caption,
                reference_path=reference_path,
                task=diagram_type,
            )
            result.evaluation = scores_to_dict(scores)
        except Exception as e:
            result.error = f"evaluation failed: {e}"
            logger.error("Evaluation failed", id=entry.id, error=str(e))

        return result

    async def _vanilla_generate(
        self,
        *,
        entry_id: str,
        source_context: str,
        caption: str,
        diagram_type: DiagramType,
        raw_data: Optional[dict],
        aspect_ratio: Optional[str],
        run_dir: Path,
    ) -> str:
        """No-pipeline baseline: one direct Visualizer call per entry.

        Skips Retriever/Planner/Stylist/Critic entirely. The visualizer receives
        the raw caption + source context (for plots, the raw data table is
        appended by the visualizer itself), mirroring the upstream 'vanilla'
        ablation mode.
        """
        visualizer = self.visualizer_factory(self.settings)
        out_dir = ensure_dir(run_dir / entry_id)
        visualizer.set_output_dir(out_dir)

        if diagram_type == DiagramType.STATISTICAL_PLOT:
            # Raw data is appended to the description by the visualizer.
            description = caption or source_context
        else:
            description = f"{caption}\n\n{source_context}" if caption else source_context

        return await visualizer.run(
            description=description,
            diagram_type=diagram_type,
            raw_data=raw_data,
            output_path=str(out_dir / "final_output.png"),
            iteration=1,
            seed=self.settings.seed,
            aspect_ratio=aspect_ratio,
        )


# ── Helpers ──────────────────────────────────────────────────────


def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _save_partial(results: list[BenchmarkEntryResult], run_dir: Path) -> None:
    """Save incremental results so partial runs survive crashes."""
    partial = [r.model_dump() for r in results]
    save_json(partial, run_dir / "partial_results.json")
