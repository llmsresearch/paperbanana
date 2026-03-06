"""Ablation runner for baseline vs retrieval-augmented generation."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Callable, Optional

import structlog
from pydantic import BaseModel, Field

from paperbanana.core.config import Settings
from paperbanana.core.pipeline import PaperBananaPipeline
from paperbanana.core.types import EvaluationScore, GenerationInput, GenerationOutput
from paperbanana.evaluation.judge import VLMJudge
from paperbanana.evaluation.metrics import scores_to_dict
from paperbanana.providers.registry import ProviderRegistry

logger = structlog.get_logger()


class AblationVariant(BaseModel):
    """Single ablation variant configuration."""

    name: str
    retrieval_enabled: bool
    top_k: int
    retrieval_mode: str


class AblationVariantResult(BaseModel):
    """Single ablation variant result summary."""

    name: str
    retrieval_enabled: bool
    top_k: int
    retrieval_mode: str
    run_id: str
    image_path: str
    iteration_count: int
    critic_suggestion_count: int
    component_alignment_proxy_score: float
    total_seconds: float
    retrieval_seconds: float
    metric_mode: str
    component_alignment_metric: str
    human_preference_proxy: Optional[dict] = None


class AblationReport(BaseModel):
    """Retrieval ablation report."""

    created_at: str
    source_context_chars: int
    caption: str
    ablation_seed: int
    reference_image: Optional[str] = None
    metric_notes: dict[str, str] = Field(default_factory=dict)
    variants: list[AblationVariantResult] = Field(default_factory=list)
    summary: dict[str, str | float | int | None] = Field(default_factory=dict)


def parse_top_k_values(csv: str) -> list[int]:
    """Parse top-k values from a comma-separated string."""
    if not csv.strip():
        raise ValueError("top-k list must not be empty")

    values: list[int] = []
    seen: set[int] = set()
    for token in csv.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as e:
            raise ValueError(f"Invalid top-k value: {token}") from e
        if value < 1:
            raise ValueError(f"top-k must be >= 1, got {value}")
        if value in seen:
            continue
        seen.add(value)
        values.append(value)

    if not values:
        raise ValueError("No valid top-k values found")
    return values


def _count_critic_suggestions(output: GenerationOutput) -> int:
    count = 0
    for it in output.iterations:
        if it.critique:
            count += len(it.critique.critic_suggestions)
    return count


def _alignment_proxy_score(critic_suggestion_count: int) -> float:
    """Map fewer critique suggestions to a higher alignment proxy score."""
    score = 100.0 - (10.0 * critic_suggestion_count)
    if score < 0.0:
        return 0.0
    return score


def _extract_timing(output: GenerationOutput) -> tuple[float, float]:
    timing = output.metadata.get("timing", {})
    total_seconds = float(timing.get("total_seconds", 0.0))
    retrieval_seconds = float(timing.get("retrieval_seconds", 0.0))
    return total_seconds, retrieval_seconds


class RetrievalAblationRunner:
    """Runs baseline vs retrieval variants and produces a structured report."""

    def __init__(
        self,
        base_settings: Settings,
        *,
        reference_image_path: Optional[str] = None,
        pipeline_factory: Callable[[Settings], PaperBananaPipeline] = PaperBananaPipeline,
        judge_factory: Optional[Callable[[Settings], VLMJudge]] = None,
    ):
        self.base_settings = base_settings
        self.reference_image_path = reference_image_path
        self.pipeline_factory = pipeline_factory
        self.judge_factory = judge_factory or self._default_judge_factory

    def _default_judge_factory(self, settings: Settings) -> VLMJudge:
        from paperbanana.core.utils import find_prompt_dir

        vlm = ProviderRegistry.create_vlm(settings)
        return VLMJudge(vlm, prompt_dir=find_prompt_dir())

    def _build_variants(self, top_k_values: list[int]) -> list[AblationVariant]:
        variants = [
            AblationVariant(
                name="baseline",
                retrieval_enabled=False,
                top_k=self.base_settings.num_retrieval_examples,
                retrieval_mode="disabled",
            )
        ]
        for k in top_k_values:
            variants.append(
                AblationVariant(
                    name=f"retrieval_k{k}",
                    retrieval_enabled=True,
                    top_k=k,
                    retrieval_mode=self.base_settings.exemplar_retrieval_mode,
                )
            )
        return variants

    @property
    def ablation_seed(self) -> int:
        """Seed used across all variants to improve comparability."""
        if self.base_settings.seed is not None:
            return self.base_settings.seed
        return 42

    async def _run_variant(
        self, variant: AblationVariant, input_data: GenerationInput
    ) -> AblationVariantResult:
        settings = self.base_settings.model_copy(
            update={
                "exemplar_retrieval_enabled": variant.retrieval_enabled,
                "exemplar_retrieval_top_k": variant.top_k,
                "num_retrieval_examples": variant.top_k,
                "seed": self.ablation_seed,
            }
        )

        pipeline = self.pipeline_factory(settings)
        output = await pipeline.generate(input_data)
        critic_suggestion_count = _count_critic_suggestions(output)
        total_seconds, retrieval_seconds = _extract_timing(output)

        result = AblationVariantResult(
            name=variant.name,
            retrieval_enabled=variant.retrieval_enabled,
            top_k=variant.top_k,
            retrieval_mode=variant.retrieval_mode,
            run_id=str(output.metadata.get("run_id", "")),
            image_path=output.image_path,
            iteration_count=len(output.iterations),
            critic_suggestion_count=critic_suggestion_count,
            component_alignment_proxy_score=_alignment_proxy_score(critic_suggestion_count),
            total_seconds=total_seconds,
            retrieval_seconds=retrieval_seconds,
            metric_mode="proxy_only",
            component_alignment_metric="critic_suggestion_count_proxy",
        )

        if self.reference_image_path:
            judge = self.judge_factory(settings)
            scores: EvaluationScore = await judge.evaluate(
                image_path=output.image_path,
                source_context=input_data.source_context,
                caption=input_data.communicative_intent,
                reference_path=self.reference_image_path,
            )
            result.human_preference_proxy = scores_to_dict(scores)
            result.metric_mode = "judge_plus_proxy"

        return result

    @staticmethod
    def _build_summary(results: list[AblationVariantResult]) -> dict[str, str | float | int | None]:
        if not results:
            return {}

        best_alignment = max(results, key=lambda r: r.component_alignment_proxy_score)
        fastest = min(results, key=lambda r: r.total_seconds)
        fewest_iterations = min(results, key=lambda r: r.iteration_count)

        summary: dict[str, str | float | int | None] = {
            "best_alignment_variant": best_alignment.name,
            "best_alignment_score": best_alignment.component_alignment_proxy_score,
            "fastest_variant": fastest.name,
            "fastest_total_seconds": fastest.total_seconds,
            "fewest_iterations_variant": fewest_iterations.name,
            "fewest_iterations": fewest_iterations.iteration_count,
        }

        human_pref_scores: list[tuple[AblationVariantResult, float]] = []
        for result in results:
            if not result.human_preference_proxy:
                continue
            raw_score = result.human_preference_proxy.get("overall_score")
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            human_pref_scores.append((result, score))

        if human_pref_scores:
            best_human_pref, best_human_pref_score = max(
                human_pref_scores,
                key=lambda pair: pair[1],
            )
            summary["best_human_preference_variant"] = best_human_pref.name
            summary["best_human_preference_score"] = best_human_pref_score

        return summary

    async def run(self, input_data: GenerationInput, *, top_k_values: list[int]) -> AblationReport:
        if any(v < 1 for v in top_k_values):
            raise ValueError("All top-k values must be >= 1")
        if not self.base_settings.exemplar_retrieval_endpoint:
            raise ValueError("exemplar_retrieval_endpoint must be set for retrieval ablation")

        variants = self._build_variants(top_k_values)
        results: list[AblationVariantResult] = []
        for variant in variants:
            logger.info(
                "Running retrieval ablation variant",
                variant=variant.name,
                retrieval_enabled=variant.retrieval_enabled,
                top_k=variant.top_k,
            )
            result = await self._run_variant(variant, input_data)
            results.append(result)

        return AblationReport(
            created_at=datetime.datetime.now().isoformat(),
            source_context_chars=len(input_data.source_context),
            caption=input_data.communicative_intent,
            ablation_seed=self.ablation_seed,
            reference_image=self.reference_image_path,
            metric_notes={
                "component_alignment": "Proxy based on Critic suggestion count; fewer is better",
                "human_preference": (
                    "VLMJudge comparative score vs provided reference image"
                    if self.reference_image_path
                    else "Not computed (no reference image provided)"
                ),
                "iteration_count": "Number of visualizer-critic refinement iterations",
                "cost_runtime": "Wall-clock seconds from pipeline timing metadata",
            },
            variants=results,
            summary=self._build_summary(results),
        )

    @staticmethod
    def save_report(report: AblationReport, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        return output_path
