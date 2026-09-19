"""Tests for the benchmark harness."""

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
    ReferenceExample,
)
from paperbanana.evaluation.benchmark import (
    BenchmarkEntryResult,
    BenchmarkRunner,
    aggregate_results,
    filter_examples,
)

# ── filter_examples ──────────────────────────────────────────────


def _make_examples() -> list[ReferenceExample]:
    return [
        ReferenceExample(
            id="a1",
            source_context="ctx1",
            caption="cap1",
            image_path="/img/a1.jpg",
            category="vision",
        ),
        ReferenceExample(
            id="a2",
            source_context="ctx2",
            caption="cap2",
            image_path="/img/a2.jpg",
            category="reasoning",
        ),
        ReferenceExample(
            id="a3",
            source_context="ctx3",
            caption="cap3",
            image_path="/img/a3.jpg",
            category="vision",
        ),
    ]


def test_filter_by_category():
    examples = _make_examples()
    result = filter_examples(examples, category="vision")
    assert len(result) == 2
    assert all(e.category == "vision" for e in result)


def test_filter_by_ids():
    examples = _make_examples()
    result = filter_examples(examples, ids=["a2", "a3"])
    assert [e.id for e in result] == ["a2", "a3"]


def test_filter_by_limit():
    examples = _make_examples()
    result = filter_examples(examples, limit=1)
    assert len(result) == 1
    assert result[0].id == "a1"


def test_filter_combined():
    examples = _make_examples()
    result = filter_examples(examples, category="vision", limit=1)
    assert len(result) == 1
    assert result[0].id == "a1"


def test_filter_no_match():
    examples = _make_examples()
    result = filter_examples(examples, ids=["nonexistent"])
    assert result == []


# ── aggregate_results ────────────────────────────────────────────


def _make_entry(
    entry_id: str, winner: str, score: float, category: str = ""
) -> BenchmarkEntryResult:
    return BenchmarkEntryResult(
        id=entry_id,
        category=category,
        evaluation={
            "overall_winner": winner,
            "overall_score": score,
            "faithfulness_winner": winner,
            "faithfulness_score": score,
            "faithfulness_reasoning": "",
            "conciseness_winner": winner,
            "conciseness_score": score,
            "conciseness_reasoning": "",
            "readability_winner": winner,
            "readability_score": score,
            "readability_reasoning": "",
            "aesthetics_winner": winner,
            "aesthetics_score": score,
            "aesthetics_reasoning": "",
        },
        generation_seconds=5.0,
    )


def test_aggregate_results_basic():
    entries = [
        _make_entry("a", "Model", 100.0, "vision"),
        _make_entry("b", "Human", 0.0, "vision"),
        _make_entry("c", "Both are good", 50.0, "reasoning"),
    ]
    summary = aggregate_results(entries)

    assert summary["evaluated"] == 3
    assert summary["model_wins"] == 1
    assert summary["human_wins"] == 1
    assert summary["ties"] == 1
    assert summary["model_win_rate"] == pytest.approx(33.3, abs=0.1)
    assert summary["mean_overall_score"] == 50.0
    assert "vision" in summary["category_breakdown"]
    assert "reasoning" in summary["category_breakdown"]
    assert summary["category_breakdown"]["vision"]["count"] == 2


def test_aggregate_results_empty():
    assert aggregate_results([]) == {}


def test_aggregate_results_no_evaluated_entries():
    entries = [BenchmarkEntryResult(id="x", error="failed")]
    assert aggregate_results(entries) == {}


def test_aggregate_dimension_means():
    entries = [
        _make_entry("a", "Model", 100.0),
        _make_entry("b", "Model", 80.0),
    ]
    summary = aggregate_results(entries)
    assert summary["dimension_means"]["faithfulness"] == 90.0


# ── BenchmarkRunner ──────────────────────────────────────────────


class _FakePipeline:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def generate(self, input_data: GenerationInput) -> GenerationOutput:
        return GenerationOutput(
            image_path="/tmp/fake_output.png",
            description="desc",
            iterations=[
                IterationRecord(
                    iteration=1,
                    description="desc",
                    image_path="/tmp/fake_output.png",
                    critique=CritiqueResult(
                        critic_suggestions=[],
                        revised_description=None,
                    ),
                )
            ],
            metadata={"run_id": "run_test", "timing": {"total_seconds": 5.0}},
        )


class _FakeJudge:
    async def evaluate(self, **kwargs):
        return EvaluationScore(
            faithfulness=DimensionResult(winner="Model", score=100.0, reasoning=""),
            conciseness=DimensionResult(winner="Model", score=100.0, reasoning=""),
            readability=DimensionResult(winner="Both are good", score=50.0, reasoning=""),
            aesthetics=DimensionResult(winner="Both are good", score=50.0, reasoning=""),
            overall_winner="Model",
            overall_score=100.0,
        )


@pytest.mark.asyncio
async def test_benchmark_runner_processes_entries(tmp_path):
    # Create a fake reference image so the runner doesn't skip the entry
    ref_img = tmp_path / "ref.jpg"
    from PIL import Image

    Image.new("RGB", (64, 64)).save(ref_img)

    entries = [
        ReferenceExample(
            id="test_001",
            source_context="Our method uses attention",
            caption="Architecture overview",
            image_path=str(ref_img),
            category="vision",
        ),
    ]

    settings = Settings(
        output_dir=str(tmp_path / "outputs"),
        reference_set_path=str(tmp_path / "refs"),
        save_iterations=False,
    )

    runner = BenchmarkRunner(
        settings,
        pipeline_factory=lambda s: _FakePipeline(s),
        judge_factory=lambda s: _FakeJudge(),
    )

    report = await runner.run(entries, output_dir=tmp_path / "bench_out")

    assert report.total_entries == 1
    assert report.completed == 1
    assert report.failed == 0
    assert len(report.entries) == 1
    assert report.entries[0].evaluation is not None
    assert report.entries[0].evaluation["overall_winner"] == "Model"
    assert report.summary["evaluated"] == 1
    assert (tmp_path / "bench_out" / "benchmark_report.json").exists()
    assert (tmp_path / "bench_out" / "partial_results.json").exists()


@pytest.mark.asyncio
async def test_benchmark_runner_honors_concurrency(tmp_path, monkeypatch):
    """BenchmarkRunner should respect the benchmark_concurrency setting."""
    # Create multiple fake reference images
    from PIL import Image

    ref_paths = []
    for i in range(4):
        ref_img = tmp_path / f"ref_{i}.jpg"
        Image.new("RGB", (64, 64)).save(ref_img)
        ref_paths.append(ref_img)

    entries = [
        ReferenceExample(
            id=f"test_{i}",
            source_context="ctx",
            caption="cap",
            image_path=str(ref_paths[i]),
            category="vision",
        )
        for i in range(4)
    ]

    # Settings for the runner (concurrency will be set directly on the runner)
    settings = Settings(
        output_dir=str(tmp_path / "outputs"),
        reference_set_path=str(tmp_path / "refs"),
        save_iterations=False,
    )

    runner = BenchmarkRunner(
        settings,
        pipeline_factory=lambda s: _FakePipeline(s),
        judge_factory=lambda s: _FakeJudge(),
    )

    # Override concurrency on the runner instance
    runner.concurrency = 2

    # Spy on _process_entry to ensure it is invoked once per entry
    calls = []

    async def _spy_process_entry(entry, **kwargs):
        calls.append(entry.id)
        return await BenchmarkRunner._process_entry(runner, entry, **kwargs)

    monkeypatch.setattr(runner, "_process_entry", _spy_process_entry)

    report = await runner.run(entries, output_dir=tmp_path / "bench_out")

    assert report.total_entries == 4
    assert len(report.entries) == 4
    assert sorted(calls) == sorted(e.id for e in entries)


@pytest.mark.asyncio
async def test_benchmark_runner_skips_missing_reference(tmp_path):
    entries = [
        ReferenceExample(
            id="no_ref",
            source_context="ctx",
            caption="cap",
            image_path="/nonexistent/ref.jpg",
        ),
    ]

    settings = Settings(
        output_dir=str(tmp_path / "outputs"),
        reference_set_path=str(tmp_path / "refs"),
    )

    runner = BenchmarkRunner(
        settings,
        pipeline_factory=lambda s: _FakePipeline(s),
        judge_factory=lambda s: _FakeJudge(),
    )

    report = await runner.run(entries, output_dir=tmp_path / "bench_out")

    assert report.total_entries == 1
    assert report.completed == 0
    assert report.entries[0].error == "reference image not found"


@pytest.mark.asyncio
async def test_benchmark_runner_handles_generation_failure(tmp_path):
    ref_img = tmp_path / "ref.jpg"
    from PIL import Image

    Image.new("RGB", (64, 64)).save(ref_img)

    class _FailingPipeline:
        def __init__(self, settings):
            pass

        async def generate(self, input_data):
            raise RuntimeError("API error")

    entries = [
        ReferenceExample(
            id="fail_001",
            source_context="ctx",
            caption="cap",
            image_path=str(ref_img),
        ),
    ]

    settings = Settings(
        output_dir=str(tmp_path / "outputs"),
        reference_set_path=str(tmp_path / "refs"),
    )

    runner = BenchmarkRunner(
        settings,
        pipeline_factory=lambda s: _FailingPipeline(s),
        judge_factory=lambda s: _FakeJudge(),
    )

    report = await runner.run(entries, output_dir=tmp_path / "bench_out")

    assert report.failed == 1
    assert "API error" in report.entries[0].error


@pytest.mark.asyncio
async def test_benchmark_runner_eval_only_mode(tmp_path):
    # Set up reference and generated images
    ref_img = tmp_path / "ref.jpg"
    from PIL import Image

    Image.new("RGB", (64, 64)).save(ref_img)

    gen_dir = tmp_path / "generated"
    entry_dir = gen_dir / "test_001"
    entry_dir.mkdir(parents=True)
    Image.new("RGB", (64, 64)).save(entry_dir / "final_output.png")

    entries = [
        ReferenceExample(
            id="test_001",
            source_context="ctx",
            caption="cap",
            image_path=str(ref_img),
        ),
    ]

    settings = Settings(
        output_dir=str(tmp_path / "outputs"),
        reference_set_path=str(tmp_path / "refs"),
    )

    runner = BenchmarkRunner(
        settings,
        pipeline_factory=lambda s: _FakePipeline(s),
        judge_factory=lambda s: _FakeJudge(),
    )

    report = await runner.run(
        entries, output_dir=tmp_path / "bench_out", eval_only_dir=str(gen_dir)
    )

    assert report.completed == 1
    # Generation was skipped, so generation_seconds should be 0
    assert report.entries[0].generation_seconds == 0.0
    assert report.entries[0].evaluation is not None


@pytest.mark.asyncio
async def test_benchmark_runner_eval_only_rejects_path_traversal(tmp_path):
    """Entry ids with '..' or path separators are rejected in eval-only mode."""
    ref_img = tmp_path / "ref.jpg"
    from PIL import Image

    Image.new("RGB", (64, 64)).save(ref_img)

    gen_dir = tmp_path / "generated"
    gen_dir.mkdir()

    entries = [
        ReferenceExample(
            id="..",
            source_context="ctx",
            caption="cap",
            image_path=str(ref_img),
        ),
    ]

    settings = Settings(
        output_dir=str(tmp_path / "outputs"),
        reference_set_path=str(tmp_path / "refs"),
    )
    runner = BenchmarkRunner(
        settings,
        pipeline_factory=lambda s: _FakePipeline(s),
        judge_factory=lambda s: _FakeJudge(),
    )

    report = await runner.run(
        entries, output_dir=tmp_path / "bench_out", eval_only_dir=str(gen_dir)
    )

    assert report.completed == 0
    assert report.entries[0].error is not None
    assert "invalid entry id" in report.entries[0].error.lower()


# ── Official test split (issue #235) ─────────────────────────────


def _make_test_cases() -> list:
    from paperbanana.core.types import TestCase

    return [
        TestCase(
            id="t1",
            task="diagram",
            category="vision",
            source_context="methodology text",
            caption="Overview of the framework",
            gt_image_path="/gt/t1.jpg",
            aspect_ratio="16:9",
        ),
        TestCase(
            id="t2",
            task="plot",
            source_context='{"x": [1, 2]}',
            caption="Bar chart",
            gt_image_path="/gt/t2.jpg",
            aspect_ratio="3:2",
            raw_data={"x": [1, 2]},
            gt_code="plt.bar([1], [2])",
            difficulty="easy",
        ),
    ]


def test_testcase_filtering_works_like_reference():
    cases = _make_test_cases()
    assert len(filter_examples(cases, category="vision")) == 1
    assert filter_examples(cases, ids=["t2"])[0].id == "t2"
    assert len(filter_examples(cases, limit=1)) == 1


def test_run_rejects_unknown_mode():
    import asyncio

    runner = BenchmarkRunner(Settings(), judge_factory=lambda s: object())
    with pytest.raises(ValueError, match="mode must be"):
        asyncio.run(runner.run([], mode="bogus"))


class _MockVisualizer:
    """Captures vanilla-mode calls and writes a fake output image."""

    def __init__(self):
        self.calls = []

    def set_output_dir(self, path):
        self.out_dir = path

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        from pathlib import Path

        out = Path(kwargs["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"img")
        return str(out)


class _MockJudge:
    async def evaluate(self, **kwargs):
        dim = DimensionResult(score=8, rationale="ok", winner="Both are good")
        return EvaluationScore(
            faithfulness=dim,
            conciseness=dim,
            readability=dim,
            aesthetics=dim,
            overall_winner="Both are good",
            overall_score=8.0,
        )


async def test_vanilla_mode_single_visualizer_call(tmp_path):
    """Vanilla mode makes exactly one direct visualizer call per entry,

    threading the official aspect ratio, and never builds the pipeline."""
    visualizer = _MockVisualizer()

    def _fail_pipeline(settings):
        raise AssertionError("pipeline must not be constructed in vanilla mode")

    runner = BenchmarkRunner(
        Settings(output_dir=str(tmp_path)),
        pipeline_factory=_fail_pipeline,
        judge_factory=lambda s: _MockJudge(),
        visualizer_factory=lambda s: visualizer,
    )
    gt = tmp_path / "gt.jpg"
    gt.write_bytes(b"gt-image")
    diagram_case = _make_test_cases()[0].model_copy(update={"gt_image_path": str(gt)})
    report = await runner.run(
        [diagram_case], output_dir=tmp_path / "run", mode="vanilla", split="test", task="diagram"
    )

    assert len(visualizer.calls) == 1
    call = visualizer.calls[0]
    assert call["aspect_ratio"] == "16:9"
    assert call["iteration"] == 1
    entry_result = report.entries[0]
    assert entry_result.error is None
    assert entry_result.iteration_count == 1
    assert report.mode == "vanilla"
    assert report.split == "test"


# ── judge provider/model separation ──────────────────────────────


def _atlas_settings(**overrides) -> Settings:
    kwargs = {
        "vlm_provider": "atlas",
        "atlascloud_api_key": "test-atlas-key",
        "atlascloud_vlm_model": "qwen/qwen3-vl-30b-a3b-instruct",
        "judge_vlm_provider": None,
        "judge_vlm_model": None,
    }
    kwargs.update(overrides)
    return Settings(**kwargs)


def test_judge_factory_defaults_to_pipeline_vlm():
    settings = _atlas_settings()
    runner = BenchmarkRunner(settings)
    judge = runner._default_judge_factory(settings)

    assert judge.vlm.name == "atlas"
    assert judge.vlm.model_name == "qwen/qwen3-vl-30b-a3b-instruct"


def test_judge_factory_uses_judge_model_override():
    settings = _atlas_settings(judge_vlm_model="Qwen/Qwen3-VL-235B-A22B-Instruct")
    runner = BenchmarkRunner(settings)
    judge = runner._default_judge_factory(settings)

    assert judge.vlm.name == "atlas"
    assert judge.vlm.model_name == "Qwen/Qwen3-VL-235B-A22B-Instruct"
    # Pipeline settings are untouched.
    assert settings.atlascloud_vlm_model == "qwen/qwen3-vl-30b-a3b-instruct"


def test_judge_factory_uses_judge_provider_override():
    settings = _atlas_settings(
        judge_vlm_provider="openai",
        judge_vlm_model="gpt-5.2",
        openai_api_key="test-openai-key",
        openai_vlm_model=None,
    )
    runner = BenchmarkRunner(settings)
    judge = runner._default_judge_factory(settings)

    assert judge.vlm.name == "openai"
    assert judge.vlm.model_name == "gpt-5.2"
