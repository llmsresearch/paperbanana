"""Tests for retrieval ablation runner."""

from __future__ import annotations

import pytest

from paperbanana.core.config import Settings
from paperbanana.core.types import (
    CritiqueResult,
    DimensionResult,
    EvaluationScore,
    GenerationInput,
    GenerationOutput,
    IterationRecord,
)
from paperbanana.evaluation.retrieval_ablation import (
    RetrievalAblationRunner,
    parse_top_k_values,
)


def test_parse_top_k_values_dedupes_and_preserves_order():
    assert parse_top_k_values("1,3,5,3,1") == [1, 3, 5]


def test_parse_top_k_values_rejects_invalid_values():
    with pytest.raises(ValueError):
        parse_top_k_values("1,abc")
    with pytest.raises(ValueError):
        parse_top_k_values("0,3")
    with pytest.raises(ValueError):
        parse_top_k_values(" , ")


class _FakePipeline:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def generate(self, input_data: GenerationInput) -> GenerationOutput:
        if self.settings.exemplar_retrieval_enabled:
            run_id = f"retrieval_k{self.settings.exemplar_retrieval_top_k}"
            suggestions = ["fix label"]
            total_seconds = 7.0 + self.settings.exemplar_retrieval_top_k
            retrieval_seconds = 2.0
        else:
            run_id = "baseline"
            suggestions = ["fix flow", "fix arrows", "fix labels"]
            total_seconds = 12.0
            retrieval_seconds = 1.0

        return GenerationOutput(
            image_path=f"/tmp/{run_id}.png",
            description="desc",
            iterations=[
                IterationRecord(
                    iteration=1,
                    description="desc",
                    image_path=f"/tmp/{run_id}.png",
                    critique=CritiqueResult(
                        critic_suggestions=suggestions,
                        revised_description=None,
                    ),
                )
            ],
            metadata={
                "run_id": run_id,
                "timing": {
                    "total_seconds": total_seconds,
                    "retrieval_seconds": retrieval_seconds,
                },
            },
        )


class _FakeJudge:
    async def evaluate(self, **kwargs):
        return EvaluationScore(
            faithfulness=DimensionResult(winner="Model", score=100.0, reasoning=""),
            conciseness=DimensionResult(winner="Model", score=100.0, reasoning=""),
            readability=DimensionResult(winner="Model", score=100.0, reasoning=""),
            aesthetics=DimensionResult(winner="Both are good", score=50.0, reasoning=""),
            overall_winner="Model",
            overall_score=100.0,
        )


@pytest.mark.asyncio
async def test_ablation_runner_runs_baseline_and_retrieval_variants():
    settings = Settings(
        output_dir="/tmp",
        reference_set_path="/tmp/refs",
        exemplar_retrieval_endpoint="https://retriever.test/query",
        exemplar_retrieval_mode="external_then_rerank",
    )
    runner = RetrievalAblationRunner(
        settings,
        pipeline_factory=lambda s: _FakePipeline(s),
    )

    report = await runner.run(
        GenerationInput(source_context="ctx", communicative_intent="cap"),
        top_k_values=[1, 3],
    )

    assert [v.name for v in report.variants] == ["baseline", "retrieval_k1", "retrieval_k3"]
    assert report.ablation_seed == 42
    assert "component_alignment" in report.metric_notes
    assert "human_preference" in report.metric_notes
    for variant in report.variants:
        assert variant.metric_mode == "proxy_only"
        assert variant.component_alignment_metric == "critic_suggestion_count_proxy"
    assert report.summary["best_alignment_variant"] in {"retrieval_k1", "retrieval_k3"}
    assert report.summary["fewest_iterations"] == 1


@pytest.mark.asyncio
async def test_ablation_runner_requires_endpoint():
    settings = Settings(output_dir="/tmp", reference_set_path="/tmp/refs")
    runner = RetrievalAblationRunner(settings, pipeline_factory=lambda s: _FakePipeline(s))

    with pytest.raises(ValueError, match="exemplar_retrieval_endpoint must be set"):
        await runner.run(
            GenerationInput(source_context="ctx", communicative_intent="cap"),
            top_k_values=[1],
        )


@pytest.mark.asyncio
async def test_ablation_runner_includes_human_preference_proxy_when_reference_provided(tmp_path):
    settings = Settings(
        output_dir="/tmp",
        reference_set_path="/tmp/refs",
        exemplar_retrieval_endpoint="https://retriever.test/query",
    )
    runner = RetrievalAblationRunner(
        settings,
        reference_image_path="/tmp/reference.png",
        pipeline_factory=lambda s: _FakePipeline(s),
        judge_factory=lambda s: _FakeJudge(),
    )

    report = await runner.run(
        GenerationInput(source_context="ctx", communicative_intent="cap"),
        top_k_values=[1],
    )

    assert report.ablation_seed == 42
    assert "VLMJudge comparative score" in report.metric_notes["human_preference"]
    for variant in report.variants:
        assert variant.human_preference_proxy is not None
        assert variant.human_preference_proxy["overall_winner"] == "Model"
        assert variant.metric_mode == "judge_plus_proxy"
        assert variant.component_alignment_metric == "critic_suggestion_count_proxy"
    assert report.summary["best_human_preference_variant"] in {"baseline", "retrieval_k1"}
    assert report.summary["best_human_preference_score"] == 100.0

    report_path = tmp_path / "retrieval_ablation_test.json"
    out = RetrievalAblationRunner.save_report(report, report_path)
    assert out == report_path
    assert out.exists()


@pytest.mark.asyncio
async def test_ablation_runner_uses_configured_seed():
    settings = Settings(
        output_dir="/tmp",
        reference_set_path="/tmp/refs",
        exemplar_retrieval_endpoint="https://retriever.test/query",
        seed=777,
    )
    seen_seeds: list[int | None] = []

    class _SeedTrackingPipeline(_FakePipeline):
        def __init__(self, settings: Settings):
            super().__init__(settings)
            seen_seeds.append(settings.seed)

    runner = RetrievalAblationRunner(
        settings,
        pipeline_factory=lambda s: _SeedTrackingPipeline(s),
    )

    report = await runner.run(
        GenerationInput(source_context="ctx", communicative_intent="cap"),
        top_k_values=[1, 2],
    )

    assert report.ablation_seed == 777
    assert seen_seeds == [777, 777, 777]
