"""Tests for the prompt A/B comparison runner."""

from __future__ import annotations

import json

import pytest

from paperbanana.core.config import Settings
from paperbanana.core.types import (
    CritiqueResult,
    DimensionResult,
    EvaluationScore,
    GenerationInput,
    GenerationOutput,
    IterationRecord,
    ReferenceExample,
)
from paperbanana.evaluation.prompt_ablation import (
    PromptAblationRunner,
    PromptComparisonEntry,
    PromptVariantResult,
    build_summary,
    compute_dimension_deltas,
    validate_prompt_dir,
)

# ── Helpers ──────────────────────────────────────────────────────


def _make_eval_dict(overall_score: float) -> dict:
    """Build a minimal evaluation dict matching scores_to_dict output."""
    d = {"overall_winner": "Model", "overall_score": overall_score}
    for dim in ["faithfulness", "conciseness", "readability", "aesthetics"]:
        d[f"{dim}_winner"] = "Model"
        d[f"{dim}_score"] = overall_score
        d[f"{dim}_reasoning"] = ""
    return d


class _FakePipeline:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def generate(self, input_data: GenerationInput) -> GenerationOutput:
        prompt_dir = self.settings.prompt_dir or "default"
        return GenerationOutput(
            image_path="/tmp/fake.png",
            description="desc",
            iterations=[
                IterationRecord(
                    iteration=1,
                    description="desc",
                    image_path="/tmp/fake.png",
                    critique=CritiqueResult(
                        critic_suggestions=[],
                        revised_description=None,
                    ),
                )
            ],
            metadata={"run_id": f"run_{prompt_dir}", "timing": {"total_seconds": 3.0}},
        )


class _FakeJudge:
    def __init__(self, score: float = 75.0):
        self._score = score

    async def evaluate(self, **kwargs) -> EvaluationScore:
        return EvaluationScore(
            faithfulness=DimensionResult(winner="Model", score=self._score, reasoning=""),
            conciseness=DimensionResult(winner="Model", score=self._score, reasoning=""),
            readability=DimensionResult(winner="Model", score=self._score, reasoning=""),
            aesthetics=DimensionResult(winner="Model", score=self._score, reasoning=""),
            overall_winner="Model",
            overall_score=self._score,
        )


def _make_prompt_dir(tmp_path, name: str = "prompts") -> str:
    """Create a minimal prompt directory structure for validation."""
    d = tmp_path / name / "diagram"
    d.mkdir(parents=True)
    (d / "planner.txt").write_text("test prompt")
    return str(tmp_path / name)


# ── compute_dimension_deltas ─────────────────────────────────────


def test_compute_deltas_positive():
    deltas, overall = compute_dimension_deltas(_make_eval_dict(60.0), _make_eval_dict(80.0))
    assert overall == 20.0
    assert deltas["faithfulness"] == 20.0


def test_compute_deltas_negative():
    deltas, overall = compute_dimension_deltas(_make_eval_dict(90.0), _make_eval_dict(70.0))
    assert overall == -20.0


def test_compute_deltas_zero():
    same = _make_eval_dict(50.0)
    deltas, overall = compute_dimension_deltas(same, same)
    assert overall == 0.0
    assert all(v == 0.0 for v in deltas.values())


# ── build_summary ────────────────────────────────────────────────


def _make_comparison(entry_id: str, baseline_score: float, variant_score: float):
    b = PromptVariantResult(
        entry_id=entry_id,
        variant_name="baseline",
        evaluation=_make_eval_dict(baseline_score),
    )
    v = PromptVariantResult(
        entry_id=entry_id,
        variant_name="variant",
        evaluation=_make_eval_dict(variant_score),
    )
    deltas, overall_delta = compute_dimension_deltas(b.evaluation, v.evaluation)
    winner = "variant" if overall_delta > 0 else ("baseline" if overall_delta < 0 else "tie")
    return PromptComparisonEntry(
        entry_id=entry_id,
        baseline=b,
        variant=v,
        dimension_deltas=deltas,
        overall_delta=overall_delta,
        winner=winner,
    )


def test_build_summary_basic():
    entries = [
        _make_comparison("a", 60.0, 80.0),
        _make_comparison("b", 90.0, 70.0),
        _make_comparison("c", 50.0, 50.0),
    ]
    summary = build_summary(entries)
    assert summary["scored"] == 3
    assert summary["variant_wins"] == 1
    assert summary["baseline_wins"] == 1
    assert summary["ties"] == 1
    assert summary["variant_win_rate"] == pytest.approx(33.3, abs=0.1)


def test_build_summary_empty():
    assert build_summary([]) == {}


def test_build_summary_no_evaluations():
    entry = PromptComparisonEntry(
        entry_id="x",
        baseline=PromptVariantResult(entry_id="x", variant_name="baseline"),
        variant=PromptVariantResult(entry_id="x", variant_name="variant"),
    )
    assert build_summary([entry]) == {}


# ── validate_prompt_dir ──────────────────────────────────────────


def test_validate_prompt_dir_valid(tmp_path):
    validate_prompt_dir(_make_prompt_dir(tmp_path))


def test_validate_prompt_dir_missing():
    with pytest.raises(ValueError, match="does not exist"):
        validate_prompt_dir("/nonexistent/prompts_v99")


def test_validate_prompt_dir_no_subdirectory(tmp_path):
    empty_dir = tmp_path / "empty_prompts"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="missing expected subdirectory"):
        validate_prompt_dir(str(empty_dir))


# ── PromptAblationRunner ────────────────────────────────────────


def _setup_runner(tmp_path, pipeline_factory=None, judge_factory=None, **settings_kw):
    """Shared setup: creates prompt dirs, reference image, entries, and runner."""
    baseline_dir = _make_prompt_dir(tmp_path, "baseline_prompts")
    variant_dir = _make_prompt_dir(tmp_path, "variant_prompts")

    from PIL import Image

    ref_img = tmp_path / "ref.png"
    Image.new("RGB", (64, 64)).save(ref_img)

    entries = [
        ReferenceExample(
            id="test_001",
            source_context="ctx",
            caption="cap",
            image_path=str(ref_img),
            category="vision",
        ),
    ]

    settings = Settings(
        output_dir=str(tmp_path / "outputs"),
        reference_set_path=str(tmp_path / "refs"),
        **settings_kw,
    )

    runner = PromptAblationRunner(
        settings,
        baseline_prompt_dir=baseline_dir,
        variant_prompt_dir=variant_dir,
        pipeline_factory=pipeline_factory or (lambda s: _FakePipeline(s)),
        judge_factory=judge_factory or (lambda s: _FakeJudge()),
    )
    return runner, entries, baseline_dir, variant_dir


@pytest.mark.asyncio
async def test_runner_compares_two_variants(tmp_path):
    seen_prompt_dirs: list[str] = []

    class _TrackingPipeline(_FakePipeline):
        def __init__(self, settings):
            super().__init__(settings)
            seen_prompt_dirs.append(settings.prompt_dir or "default")

    runner, entries, baseline_dir, variant_dir = _setup_runner(
        tmp_path,
        pipeline_factory=lambda s: _TrackingPipeline(s),
    )
    report = await runner.run(entries)

    assert len(seen_prompt_dirs) == 2
    assert baseline_dir in seen_prompt_dirs
    assert variant_dir in seen_prompt_dirs
    assert report.total_entries == 1
    assert report.compared == 1
    assert report.entries[0].winner == "tie"


@pytest.mark.asyncio
async def test_runner_detects_variant_winning(tmp_path):
    class _DifferentialJudge:
        def __init__(self):
            self._call_count = 0

        async def evaluate(self, **kwargs) -> EvaluationScore:
            self._call_count += 1
            score = 90.0 if self._call_count == 2 else 60.0
            return EvaluationScore(
                faithfulness=DimensionResult(winner="Model", score=score, reasoning=""),
                conciseness=DimensionResult(winner="Model", score=score, reasoning=""),
                readability=DimensionResult(winner="Model", score=score, reasoning=""),
                aesthetics=DimensionResult(winner="Model", score=score, reasoning=""),
                overall_winner="Model",
                overall_score=score,
            )

    judge = _DifferentialJudge()
    runner, entries, _, _ = _setup_runner(
        tmp_path,
        judge_factory=lambda s: judge,
    )
    report = await runner.run(entries)

    assert report.entries[0].winner == "variant"
    assert report.entries[0].overall_delta == 30.0
    assert report.summary["variant_wins"] == 1


@pytest.mark.asyncio
async def test_runner_uses_configured_seed(tmp_path):
    seen_seeds: list = []

    class _SeedTracker(_FakePipeline):
        def __init__(self, settings):
            super().__init__(settings)
            seen_seeds.append(settings.seed)

    runner, entries, _, _ = _setup_runner(
        tmp_path,
        pipeline_factory=lambda s: _SeedTracker(s),
        seed=42,
    )
    report = await runner.run(entries)

    assert report.seed == 42
    assert seen_seeds == [42, 42]


@pytest.mark.asyncio
async def test_runner_handles_generation_failure(tmp_path):
    class _FailingPipeline:
        def __init__(self, settings):
            pass

        async def generate(self, input_data):
            raise RuntimeError("API error")

    runner, entries, _, _ = _setup_runner(
        tmp_path,
        pipeline_factory=lambda s: _FailingPipeline(s),
    )
    report = await runner.run(entries)

    assert report.failed == 1
    assert report.entries[0].baseline.error == "API error"
    assert report.entries[0].winner == ""


@pytest.mark.asyncio
async def test_runner_rejects_invalid_prompt_dir(tmp_path):
    baseline_dir = _make_prompt_dir(tmp_path, "baseline_prompts")
    settings = Settings(output_dir=str(tmp_path), reference_set_path=str(tmp_path))
    entries = [
        ReferenceExample(id="x", source_context="ctx", caption="cap", image_path="/tmp/x.png"),
    ]

    runner = PromptAblationRunner(
        settings,
        baseline_prompt_dir=baseline_dir,
        variant_prompt_dir="/nonexistent/prompts",
        pipeline_factory=lambda s: _FakePipeline(s),
    )
    with pytest.raises(ValueError, match="does not exist"):
        await runner.run(entries)


@pytest.mark.asyncio
async def test_save_report(tmp_path):
    runner, entries, _, _ = _setup_runner(tmp_path)
    report = await runner.run(entries)

    report_path = tmp_path / "report.json"
    saved = PromptAblationRunner.save_report(report, report_path)

    assert saved.exists()
    data = json.loads(saved.read_text())
    assert data["baseline"]["name"] == "baseline"
    assert data["variant"]["name"] == "variant"
    assert data["total_entries"] == 1


# ── Settings integration ─────────────────────────────────────────


def test_benchmark_concurrency_field():
    """benchmark_concurrency should be accepted by Settings (not silently dropped)."""
    settings = Settings(benchmark_concurrency=4)
    assert settings.benchmark_concurrency == 4


def test_benchmark_concurrency_default():
    assert Settings().benchmark_concurrency == 1


def test_prompt_dir_field():
    assert Settings(prompt_dir="/custom/prompts").prompt_dir == "/custom/prompts"


def test_prompt_dir_default_is_none():
    assert Settings().prompt_dir is None
