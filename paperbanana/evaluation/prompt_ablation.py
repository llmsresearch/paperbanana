"""A/B comparison runner for prompt variant evaluation.

Runs the same inputs through two prompt configurations (baseline vs variant),
scores both with VLMJudge, and produces a side-by-side report with
per-dimension deltas and win rates.
"""

from __future__ import annotations

import datetime
import time
from pathlib import Path
from typing import Callable, Optional

import structlog
from pydantic import BaseModel, Field

from paperbanana.core.config import Settings
from paperbanana.core.pipeline import PaperBananaPipeline
from paperbanana.core.types import (
    EvaluationScore,
    GenerationInput,
    ReferenceExample,
)
from paperbanana.core.utils import find_prompt_dir
from paperbanana.evaluation.judge import DIMENSIONS, VLMJudge
from paperbanana.evaluation.metrics import scores_to_dict
from paperbanana.providers.registry import ProviderRegistry

logger = structlog.get_logger()


# ── Report models ────────────────────────────────────────────────


class PromptVariantConfig(BaseModel):
    """Identifies a prompt variant by name and directory."""

    name: str
    prompt_dir: str


class PromptVariantResult(BaseModel):
    """Generation + evaluation result for a single (entry, variant) pair."""

    entry_id: str
    variant_name: str
    run_id: str = ""
    image_path: str = ""
    iteration_count: int = 0
    generation_seconds: float = 0.0
    evaluation: Optional[dict] = None
    error: Optional[str] = None


class PromptComparisonEntry(BaseModel):
    """Side-by-side comparison for a single benchmark entry."""

    entry_id: str
    category: str = ""
    baseline: Optional[PromptVariantResult] = None
    variant: Optional[PromptVariantResult] = None
    dimension_deltas: dict[str, float] = Field(default_factory=dict)
    overall_delta: float = 0.0
    winner: str = ""


class PromptAblationReport(BaseModel):
    """Full A/B prompt comparison report."""

    created_at: str
    baseline: PromptVariantConfig
    variant: PromptVariantConfig
    seed: Optional[int] = None
    total_entries: int = 0
    compared: int = 0
    failed: int = 0
    total_seconds: float = 0.0
    entries: list[PromptComparisonEntry] = Field(default_factory=list)
    summary: dict = Field(default_factory=dict)


# ── Comparison logic ─────────────────────────────────────────────


def compute_dimension_deltas(
    baseline_eval: dict, variant_eval: dict
) -> tuple[dict[str, float], float]:
    """Compute per-dimension score deltas (variant - baseline).

    Positive deltas mean the variant scored higher.
    """
    deltas: dict[str, float] = {}
    for dim in DIMENSIONS:
        key = f"{dim}_score"
        b_score = float(baseline_eval.get(key, 0.0))
        v_score = float(variant_eval.get(key, 0.0))
        deltas[dim] = round(v_score - b_score, 1)

    b_overall = float(baseline_eval.get("overall_score", 0.0))
    v_overall = float(variant_eval.get("overall_score", 0.0))
    overall_delta = round(v_overall - b_overall, 1)

    return deltas, overall_delta


def build_summary(entries: list[PromptComparisonEntry]) -> dict:
    """Aggregate comparison results into win rates and mean deltas."""
    scored = [
        e
        for e in entries
        if e.baseline
        and e.variant
        and e.baseline.evaluation is not None
        and e.variant.evaluation is not None
    ]
    if not scored:
        return {}

    variant_wins = sum(1 for e in scored if e.winner == "variant")
    baseline_wins = sum(1 for e in scored if e.winner == "baseline")
    ties = len(scored) - variant_wins - baseline_wins

    mean_deltas: dict[str, float] = {}
    for dim in DIMENSIONS:
        values = [e.dimension_deltas.get(dim, 0.0) for e in scored]
        mean_deltas[dim] = round(sum(values) / len(values), 1)

    overall_deltas = [e.overall_delta for e in scored]
    mean_overall_delta = round(sum(overall_deltas) / len(overall_deltas), 1)

    baseline_scores = [float(e.baseline.evaluation.get("overall_score", 0.0)) for e in scored]
    variant_scores = [float(e.variant.evaluation.get("overall_score", 0.0)) for e in scored]

    return {
        "scored": len(scored),
        "variant_wins": variant_wins,
        "baseline_wins": baseline_wins,
        "ties": ties,
        "variant_win_rate": round(variant_wins / len(scored) * 100, 1),
        "baseline_win_rate": round(baseline_wins / len(scored) * 100, 1),
        "mean_dimension_deltas": mean_deltas,
        "mean_overall_delta": mean_overall_delta,
        "mean_baseline_score": round(sum(baseline_scores) / len(baseline_scores), 1),
        "mean_variant_score": round(sum(variant_scores) / len(variant_scores), 1),
    }


# ── Runner ───────────────────────────────────────────────────────


def validate_prompt_dir(prompt_dir: str) -> None:
    """Check that a prompt directory looks valid before running."""
    p = Path(prompt_dir)
    if not p.is_dir():
        raise ValueError(f"Prompt directory does not exist: {prompt_dir}")
    if not (p / "diagram").is_dir() and not (p / "plot").is_dir():
        raise ValueError(
            f"Prompt directory missing expected subdirectory (diagram/ or plot/): {prompt_dir}"
        )


class PromptAblationRunner:
    """Runs A/B comparisons between two prompt configurations."""

    def __init__(
        self,
        base_settings: Settings,
        *,
        baseline_prompt_dir: Optional[str] = None,
        variant_prompt_dir: str,
        baseline_name: str = "baseline",
        variant_name: str = "variant",
        pipeline_factory: Callable[[Settings], PaperBananaPipeline] = PaperBananaPipeline,
        judge_factory: Optional[Callable[[Settings], VLMJudge]] = None,
    ):
        self.base_settings = base_settings
        self.baseline_prompt_dir = baseline_prompt_dir or find_prompt_dir()
        self.variant_prompt_dir = variant_prompt_dir
        self.baseline_name = baseline_name
        self.variant_name = variant_name
        self.pipeline_factory = pipeline_factory
        self.judge_factory = judge_factory or self._default_judge_factory

    def _default_judge_factory(self, settings: Settings) -> VLMJudge:
        vlm = ProviderRegistry.create_vlm(settings)
        return VLMJudge(vlm, prompt_dir=find_prompt_dir())

    def _settings_for_variant(self, prompt_dir: str) -> Settings:
        """Create a settings copy pointing to the given prompt directory."""
        return self.base_settings.model_copy(update={"prompt_dir": prompt_dir})

    async def _run_variant(
        self,
        variant_name: str,
        prompt_dir: str,
        input_data: GenerationInput,
        entry_id: str,
    ) -> PromptVariantResult:
        """Generate a single entry with a specific prompt configuration."""
        result = PromptVariantResult(entry_id=entry_id, variant_name=variant_name)
        try:
            settings = self._settings_for_variant(prompt_dir)
            pipeline = self.pipeline_factory(settings)
            gen_start = time.perf_counter()
            output = await pipeline.generate(input_data)
            result.generation_seconds = round(time.perf_counter() - gen_start, 1)
            result.run_id = str(output.metadata.get("run_id", ""))
            result.image_path = output.image_path
            result.iteration_count = len(output.iterations)
        except Exception as e:
            result.error = str(e)
            logger.error(
                "Prompt ablation generation failed",
                entry=entry_id,
                variant=variant_name,
                error=str(e),
            )
        return result

    async def _evaluate_result(
        self,
        result: PromptVariantResult,
        entry: ReferenceExample,
        judge: VLMJudge,
    ) -> None:
        """Score a generation result against the entry's reference image."""
        if result.error or not result.image_path:
            return
        if not entry.image_path or not Path(entry.image_path).exists():
            result.error = "reference image not found"
            return
        try:
            scores: EvaluationScore = await judge.evaluate(
                image_path=result.image_path,
                source_context=entry.source_context,
                caption=entry.caption,
                reference_path=entry.image_path,
            )
            result.evaluation = scores_to_dict(scores)
        except Exception as e:
            result.error = f"evaluation failed: {e}"
            logger.error("Evaluation failed", entry=entry.id, error=str(e))

    async def _process_entry(
        self,
        entry: ReferenceExample,
        judge: VLMJudge,
    ) -> PromptComparisonEntry:
        """Run both variants for a single entry and compute deltas."""
        input_data = GenerationInput(
            source_context=entry.source_context,
            communicative_intent=entry.caption,
        )

        comparison = PromptComparisonEntry(
            entry_id=entry.id,
            category=entry.category or "",
        )

        baseline_result = await self._run_variant(
            self.baseline_name,
            self.baseline_prompt_dir,
            input_data,
            entry.id,
        )
        variant_result = await self._run_variant(
            self.variant_name,
            self.variant_prompt_dir,
            input_data,
            entry.id,
        )

        await self._evaluate_result(baseline_result, entry, judge)
        await self._evaluate_result(variant_result, entry, judge)

        comparison.baseline = baseline_result
        comparison.variant = variant_result

        if baseline_result.evaluation and variant_result.evaluation:
            deltas, overall_delta = compute_dimension_deltas(
                baseline_result.evaluation, variant_result.evaluation
            )
            comparison.dimension_deltas = deltas
            comparison.overall_delta = overall_delta

            if overall_delta > 0:
                comparison.winner = "variant"
            elif overall_delta < 0:
                comparison.winner = "baseline"
            else:
                comparison.winner = "tie"

        return comparison

    async def run(
        self,
        entries: list[ReferenceExample],
    ) -> PromptAblationReport:
        """Run the A/B comparison across all entries."""
        validate_prompt_dir(self.baseline_prompt_dir)
        validate_prompt_dir(self.variant_prompt_dir)

        judge = self.judge_factory(self.base_settings)
        results: list[PromptComparisonEntry] = []
        total_start = time.perf_counter()

        for idx, entry in enumerate(entries):
            logger.info(
                f"Prompt ablation {idx + 1}/{len(entries)}",
                entry=entry.id,
                category=entry.category,
            )
            comparison = await self._process_entry(entry, judge)
            results.append(comparison)

        total_seconds = time.perf_counter() - total_start
        compared = sum(1 for e in results if e.baseline and e.variant)
        failed = sum(
            1
            for e in results
            if (e.baseline and e.baseline.error) or (e.variant and e.variant.error)
        )

        return PromptAblationReport(
            created_at=datetime.datetime.now().isoformat(),
            baseline=PromptVariantConfig(
                name=self.baseline_name,
                prompt_dir=self.baseline_prompt_dir,
            ),
            variant=PromptVariantConfig(
                name=self.variant_name,
                prompt_dir=self.variant_prompt_dir,
            ),
            seed=self.base_settings.seed,
            total_entries=len(entries),
            compared=compared,
            failed=failed,
            total_seconds=round(total_seconds, 1),
            entries=results,
            summary=build_summary(results),
        )

    @staticmethod
    def save_report(report: PromptAblationReport, path: str | Path) -> Path:
        """Write the report to a JSON file."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        return output_path
