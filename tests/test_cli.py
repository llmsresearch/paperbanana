"""Tests for PaperBanana CLI."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from typer.testing import CliRunner

from paperbanana.cli import app

runner = CliRunner()


def test_generate_dry_run_valid_inputs():
    """paperbanana generate --input file.txt --caption 'test' --dry-run works."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Sample methodology text for testing.")
        input_path = f.name

    try:
        result = runner.invoke(
            app,
            ["generate", "--input", input_path, "--caption", "test", "--dry-run"],
        )
        assert result.exit_code == 0
        assert "Dry Run" in result.output
        assert "Input:" in result.output
        assert "test" in result.output
        assert "VLM:" in result.output
        assert "Output:" in result.output
        assert "Done!" not in result.output
    finally:
        Path(input_path).unlink(missing_ok=True)


def test_generate_dry_run_invalid_input():
    """Dry run with missing input file exits with error."""
    result = runner.invoke(
        app,
        ["generate", "--input", "/nonexistent/path.txt", "--caption", "test", "--dry-run"],
    )
    assert result.exit_code == 1
    assert "not found" in result.output.lower() or "Error" in result.output


def test_generate_accepts_progress_json_flag():
    """paperbanana generate accepts --progress-json flag in dry-run mode."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Sample methodology text for testing.")
        input_path = f.name

    try:
        result = runner.invoke(
            app,
            [
                "generate",
                "--input",
                input_path,
                "--caption",
                "test",
                "--dry-run",
                "--progress-json",
            ],
        )
        # Dry run doesn't emit progress events, but the flag should be accepted.
        assert result.exit_code == 0
    finally:
        Path(input_path).unlink(missing_ok=True)


def test_orchestrate_dry_run_writes_plan(tmp_path):
    """orchestrate --dry-run should emit a package plan on disk."""
    paper_path = tmp_path / "paper.txt"
    paper_path.write_text(
        "\n".join(
            [
                "A Very Good Paper Title",
                "",
                "1 Introduction",
                "We introduce a new system.",
                "",
                "2 Method",
                "Our method has encoder, retriever, and critic modules.",
                "",
                "3 Experiments",
                "We compare against strong baselines.",
            ]
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "orchestrate",
            "--paper",
            str(paper_path),
            "--output-dir",
            str(tmp_path),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "Dry run complete" in result.output

    plans = list(tmp_path.glob("orchestrate_*/orchestration_plan.json"))
    assert len(plans) == 1
    payload = json.loads(plans[0].read_text(encoding="utf-8"))
    assert payload["paper_title"] == "A Very Good Paper Title"
    assert len(payload["methodology_items"]) >= 1


def test_orchestrate_generates_package_with_mocked_pipeline(tmp_path, monkeypatch):
    """orchestrate writes package manifest + latex artifacts in execution mode."""
    paper_path = tmp_path / "paper.txt"
    paper_path.write_text(
        "\n".join(
            [
                "Paper Title",
                "",
                "1 Method Overview",
                "Our framework has three blocks.",
                "",
                "2 Training Pipeline",
                "We optimize with curriculum and regularization.",
            ]
        ),
        encoding="utf-8",
    )

    call_state = {"n": 0}

    class _FakePipeline:
        def __init__(self, settings=None, **kwargs):
            self.settings = settings
            self.run_id = "run_fake"

        async def generate(self, gen_input):
            from paperbanana.core.types import GenerationOutput

            call_state["n"] += 1
            out = tmp_path / f"fake_{call_state['n']}.png"
            out.write_bytes(b"fake-png")
            return GenerationOutput(
                image_path=str(out),
                description="desc",
                iterations=[],
                metadata={"run_id": f"run_{call_state['n']}"},
            )

    monkeypatch.setattr("paperbanana.core.pipeline.PaperBananaPipeline", _FakePipeline)

    result = runner.invoke(
        app,
        [
            "orchestrate",
            "--paper",
            str(paper_path),
            "--output-dir",
            str(tmp_path),
            "--max-method-figures",
            "2",
        ],
    )
    assert result.exit_code == 0
    assert "Orchestration complete" in result.output

    packages = list(tmp_path.glob("orchestrate_*/figure_package.json"))
    assert len(packages) == 1
    package = json.loads(packages[0].read_text(encoding="utf-8"))
    assert package["planned_methodology_items"] >= 1
    assert len(package["generated_items"]) >= 1

    package_dir = packages[0].parent
    assert (package_dir / "figures.tex").exists()
    assert (package_dir / "captions.md").exists()


def test_orchestrate_resume_retry_failed_item(tmp_path, monkeypatch):
    """resume-orchestrate retries failed tasks and updates package report."""
    paper_path = tmp_path / "paper.txt"
    paper_path.write_text(
        "\n".join(
            [
                "Paper Title",
                "",
                "1 Method",
                "A single section for one method figure.",
            ]
        ),
        encoding="utf-8",
    )

    call_state = {"n": 0}

    class _FlakyPipeline:
        def __init__(self, settings=None, **kwargs):
            self.settings = settings
            self.run_id = "run_flaky"

        async def generate(self, gen_input):
            from paperbanana.core.types import GenerationOutput

            call_state["n"] += 1
            if call_state["n"] == 1:
                raise RuntimeError("simulated generation failure")
            out = tmp_path / "flaky_success.png"
            out.write_bytes(b"ok")
            return GenerationOutput(
                image_path=str(out),
                description="desc",
                iterations=[],
                metadata={"run_id": "run_success"},
            )

    monkeypatch.setattr("paperbanana.core.pipeline.PaperBananaPipeline", _FlakyPipeline)

    first = runner.invoke(
        app,
        [
            "orchestrate",
            "--paper",
            str(paper_path),
            "--output-dir",
            str(tmp_path),
            "--max-method-figures",
            "1",
            "--max-plot-figures",
            "0",
        ],
    )
    assert first.exit_code == 1
    assert "failed" in first.output.lower()

    runs = list(tmp_path.glob("orchestrate_*"))
    assert len(runs) == 1
    orchestrate_dir = runs[0]
    checkpoint_path = orchestrate_dir / "orchestration_checkpoint.json"
    assert checkpoint_path.exists()
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    statuses = [x["status"] for x in checkpoint["items"].values()]
    assert statuses == ["failed"]

    second = runner.invoke(
        app,
        [
            "orchestrate",
            "--resume-orchestrate",
            str(orchestrate_dir),
            "--retry-failed",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert second.exit_code == 0
    assert "Orchestration complete" in second.output

    package = json.loads((orchestrate_dir / "figure_package.json").read_text(encoding="utf-8"))
    assert len(package["generated_items"]) == 1
    assert package["failures"] == []


def test_sweep_dry_run_writes_report(tmp_path):
    """sweep --dry-run plans variants and writes sweep_report.json."""
    input_path = tmp_path / "input.txt"
    input_path.write_text("Method details", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "sweep",
            "--input",
            str(input_path),
            "--caption",
            "Sweep caption",
            "--vlm-providers",
            "gemini,openai",
            "--iterations",
            "2,3",
            "--optimize-modes",
            "on,off",
            "--dry-run",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0
    assert "Dry run complete" in result.output

    reports = list(tmp_path.glob("sweep_*/sweep_report.json"))
    assert len(reports) == 1
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    assert payload["status"] == "dry_run"
    assert payload["total_variants"] == 8


def test_sweep_rejects_invalid_bool_axis(tmp_path):
    """sweep rejects invalid boolean tokens in mode axes."""
    input_path = tmp_path / "input.txt"
    input_path.write_text("Method details", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "sweep",
            "--input",
            str(input_path),
            "--caption",
            "Sweep caption",
            "--optimize-modes",
            "maybe",
            "--dry-run",
        ],
    )
    assert result.exit_code == 1
    assert "booleans" in result.output


def test_sweep_pdf_pages_rejected_for_text_input(tmp_path):
    """--pdf-pages is only valid for PDF inputs."""
    input_path = tmp_path / "input.txt"
    input_path.write_text("Method details", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "sweep",
            "--input",
            str(input_path),
            "--caption",
            "c",
            "--pdf-pages",
            "1-2",
            "--dry-run",
        ],
    )
    assert result.exit_code == 1
    assert "pdf" in result.output.lower()


def test_sweep_writes_report_with_mocked_pipeline(tmp_path, monkeypatch):
    """Non-dry sweep writes sweep_report.json with timing, ranking, and completed status."""
    input_path = tmp_path / "input.txt"
    input_path.write_text("Method details", encoding="utf-8")

    call_state = {"n": 0}

    class _FakePipeline:
        def __init__(self, settings=None, **kwargs):
            self.settings = settings

        async def generate(self, gen_input):
            call_state["n"] += 1
            from paperbanana.core.types import (
                CritiqueResult,
                GenerationOutput,
                IterationRecord,
            )

            n_suggestions = 0 if call_state["n"] == 1 else 2
            suggestions = [f"issue-{i}" for i in range(n_suggestions)]
            img = str(tmp_path / f"iter_{call_state['n']}.png")
            return GenerationOutput(
                image_path=img,
                description="d",
                iterations=[
                    IterationRecord(
                        iteration=1,
                        description="d",
                        image_path=img,
                        critique=CritiqueResult(critic_suggestions=suggestions),
                    )
                ],
                metadata={"run_id": f"run_{call_state['n']}"},
            )

    monkeypatch.setattr("paperbanana.core.pipeline.PaperBananaPipeline", _FakePipeline)

    result = runner.invoke(
        app,
        [
            "sweep",
            "--input",
            str(input_path),
            "--caption",
            "Sweep caption",
            "--vlm-providers",
            "gemini,openai",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0
    assert "Sweep Complete" in result.output

    reports = list(tmp_path.glob("sweep_*/sweep_report.json"))
    assert len(reports) == 1
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert "total_seconds" in payload
    assert isinstance(payload["total_seconds"], (int, float))
    assert payload["total_seconds"] >= 0
    assert "quality_proxy_note" in payload
    assert payload["summary"]["completed"] == 2
    assert payload["summary"]["failed"] == 0
    assert len(payload["results"]) == 2
    assert all(r.get("status") == "success" for r in payload["results"])
    ranked = payload["ranked_results"]
    assert len(ranked) == 2
    assert ranked[0]["variant_id"] == "variant_001"
    assert ranked[0]["quality_proxy_score"] > ranked[1]["quality_proxy_score"]


def test_generate_accepts_vector_flag():
    """--vector flag is accepted by the CLI in dry-run mode."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Sample methodology text for testing.")
        input_path = f.name

    try:
        result = runner.invoke(
            app,
            ["generate", "--input", input_path, "--caption", "test", "--dry-run", "--vector"],
        )
        assert result.exit_code == 0
    finally:
        Path(input_path).unlink(missing_ok=True)


def test_generate_no_vector_flag_accepted():
    """--no-vector flag (explicit opt-out) is accepted by the CLI in dry-run mode."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Sample methodology text for testing.")
        input_path = f.name

    try:
        result = runner.invoke(
            app,
            ["generate", "--input", input_path, "--caption", "test", "--dry-run", "--no-vector"],
        )
        assert result.exit_code == 0
    finally:
        Path(input_path).unlink(missing_ok=True)


def test_ablate_retrieval_writes_report(monkeypatch):
    """ablate-retrieval writes a JSON report and exits cleanly."""
    from paperbanana.evaluation.retrieval_ablation import AblationReport, AblationVariantResult

    captured: dict[str, object] = {}

    class _FakeRunner:
        def __init__(self, settings, reference_image_path=None):
            captured["settings"] = settings
            captured["reference_image_path"] = reference_image_path

        async def run(self, input_data, *, top_k_values):
            captured["top_k_values"] = top_k_values
            captured["caption"] = input_data.communicative_intent
            return AblationReport(
                created_at="2026-03-03T00:00:00",
                source_context_chars=len(input_data.source_context),
                caption=input_data.communicative_intent,
                ablation_seed=123,
                reference_image=None,
                metric_notes={
                    "component_alignment": "proxy",
                    "human_preference": "none",
                    "iteration_count": "count",
                    "cost_runtime": "seconds",
                },
                variants=[
                    AblationVariantResult(
                        name="baseline",
                        retrieval_enabled=False,
                        top_k=10,
                        retrieval_mode="disabled",
                        run_id="baseline",
                        image_path="/tmp/baseline.png",
                        iteration_count=1,
                        critic_suggestion_count=3,
                        component_alignment_proxy_score=70.0,
                        total_seconds=10.0,
                        retrieval_seconds=1.0,
                        metric_mode="proxy_only",
                        component_alignment_metric="critic_suggestion_count_proxy",
                    )
                ],
                summary={
                    "best_alignment_variant": "baseline",
                    "best_alignment_score": 70.0,
                    "fastest_variant": "baseline",
                    "fastest_total_seconds": 10.0,
                    "fewest_iterations_variant": "baseline",
                    "fewest_iterations": 1,
                },
            )

        @staticmethod
        def save_report(report, path):
            output_path = Path(path)
            output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
            return output_path

    monkeypatch.setattr(
        "paperbanana.evaluation.retrieval_ablation.RetrievalAblationRunner",
        _FakeRunner,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as input_file:
        input_file.write("Method details")
        input_path = input_file.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as report_file:
        report_path = report_file.name

    try:
        result = runner.invoke(
            app,
            [
                "ablate-retrieval",
                "--input",
                input_path,
                "--caption",
                "Ablation caption",
                "--exemplar-endpoint",
                "https://retriever.test/query",
                "--top-k",
                "1,3",
                "--seed",
                "123",
                "--exemplar-retries",
                "4",
                "--output-report",
                report_path,
            ],
        )

        assert result.exit_code == 0
        assert "Ablation Summary" in result.output
        assert captured["top_k_values"] == [1, 3]
        assert captured["caption"] == "Ablation caption"
        assert captured["settings"].seed == 123
        assert captured["settings"].exemplar_retrieval_max_retries == 4
        assert captured["settings"].exemplar_retrieval_enabled is True
        assert Path(report_path).exists()

        payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
        assert payload["ablation_seed"] == 123
        assert payload["summary"]["best_alignment_variant"] == "baseline"
    finally:
        Path(input_path).unlink(missing_ok=True)
        Path(report_path).unlink(missing_ok=True)


def test_setup_official_api_flow_writes_key_and_clears_base_url(monkeypatch):
    """Official setup flow writes key and resets GOOGLE_BASE_URL to default."""
    answers = iter(
        [
            "y",
            "n",
            "test-gemini-key",
        ]
    )
    monkeypatch.setattr("paperbanana.cli.Prompt.ask", lambda *args, **kwargs: next(answers))

    with runner.isolated_filesystem():
        result = runner.invoke(app, ["setup"])
        assert result.exit_code == 0
        env_text = Path(".env").read_text(encoding="utf-8")
        assert "GOOGLE_API_KEY=test-gemini-key" in env_text
        assert "GOOGLE_BASE_URL=" in env_text


def test_setup_updates_existing_env_without_overwrite(monkeypatch):
    """setup updates target keys while preserving unrelated existing env vars."""
    answers = iter(
        [
            "y",
            "n",
            "new-gemini-key",
        ]
    )
    monkeypatch.setattr("paperbanana.cli.Prompt.ask", lambda *args, **kwargs: next(answers))

    with runner.isolated_filesystem():
        Path(".env").write_text(
            "OPENAI_API_KEY=existing-openai-key\nGOOGLE_API_KEY=old-key\n",
            encoding="utf-8",
        )
        result = runner.invoke(app, ["setup"])
        assert result.exit_code == 0
        env_text = Path(".env").read_text(encoding="utf-8")
        assert "OPENAI_API_KEY=existing-openai-key" in env_text
        assert "GOOGLE_API_KEY=new-gemini-key" in env_text
        assert "GOOGLE_BASE_URL=" in env_text


def test_setup_custom_endpoint_flow_writes_url_and_key(monkeypatch):
    """Custom endpoint setup flow writes both GOOGLE_BASE_URL and GOOGLE_API_KEY."""
    answers = iter(
        [
            "n",
            "https://gemini-proxy.example.com",
            "key-custom",
        ]
    )
    monkeypatch.setattr("paperbanana.cli.Prompt.ask", lambda *args, **kwargs: next(answers))

    with runner.isolated_filesystem():
        result = runner.invoke(app, ["setup"])
        assert result.exit_code == 0
        env_text = Path(".env").read_text(encoding="utf-8")
        assert "GOOGLE_API_KEY=key-custom" in env_text
        assert "GOOGLE_BASE_URL=https://gemini-proxy.example.com" in env_text


def test_setup_custom_endpoint_requires_non_empty_url(monkeypatch):
    """Custom endpoint flow re-prompts when URL is empty."""
    answers = iter(
        [
            "n",
            "",
            "https://gemini-proxy.example.com",
            "key-custom",
        ]
    )
    monkeypatch.setattr("paperbanana.cli.Prompt.ask", lambda *args, **kwargs: next(answers))

    with runner.isolated_filesystem():
        result = runner.invoke(app, ["setup"])
        assert result.exit_code == 0
        assert "URL cannot be empty" in result.output
        env_text = Path(".env").read_text(encoding="utf-8")
        assert "GOOGLE_BASE_URL=https://gemini-proxy.example.com" in env_text


def test_batch_resume_retry_failed(tmp_path, monkeypatch):
    """batch supports checkpoint resume with --retry-failed."""
    input_a = tmp_path / "a.txt"
    input_b = tmp_path / "b.txt"
    input_a.write_text("A", encoding="utf-8")
    input_b.write_text("B", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "items": [
                    {"id": "ok", "input": str(input_a), "caption": "always ok"},
                    {"id": "flaky", "input": str(input_b), "caption": "fails once"},
                ]
            }
        ),
        encoding="utf-8",
    )

    call_state = {"flaky_calls": 0}

    class _FakePipeline:
        def __init__(self, settings=None, **kwargs):
            self.settings = settings

        async def generate(self, gen_input):
            from paperbanana.core.types import GenerationOutput, IterationRecord

            if "fails once" in gen_input.communicative_intent:
                call_state["flaky_calls"] += 1
                if call_state["flaky_calls"] == 1:
                    raise RuntimeError("transient boom")
            image_path = str(tmp_path / f"{gen_input.communicative_intent.replace(' ', '_')}.png")
            return GenerationOutput(
                image_path=image_path,
                description="d",
                iterations=[IterationRecord(iteration=1, description="d", image_path=image_path)],
                metadata={"run_id": f"run_{gen_input.communicative_intent.replace(' ', '_')}"},
            )

    monkeypatch.setattr("paperbanana.core.pipeline.PaperBananaPipeline", _FakePipeline)

    first = runner.invoke(
        app,
        ["batch", "--manifest", str(manifest), "--output-dir", str(tmp_path)],
    )
    assert first.exit_code == 1  # flaky failed → non-zero exit
    batches = sorted(tmp_path.glob("batch_*/batch_report.json"))
    assert len(batches) == 1
    batch_dir = batches[0].parent
    first_report = json.loads(batches[0].read_text(encoding="utf-8"))
    statuses = {item["id"]: item.get("status") for item in first_report["items"]}
    assert statuses["ok"] == "success"
    assert statuses["flaky"] == "failed"

    second = runner.invoke(
        app,
        [
            "batch",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path),
            "--resume-batch",
            str(batch_dir),
            "--retry-failed",
        ],
    )
    assert second.exit_code == 0
    resumed_report = json.loads((batch_dir / "batch_report.json").read_text(encoding="utf-8"))
    statuses = {item["id"]: item.get("status") for item in resumed_report["items"]}
    assert statuses["ok"] == "success"
    assert statuses["flaky"] == "success"


def test_plot_batch_supports_concurrency_and_retries(tmp_path, monkeypatch):
    """plot-batch writes attempts/status with retries."""
    data_path = tmp_path / "data.csv"
    data_path.write_text("x,y\n1,2\n2,3\n", encoding="utf-8")
    manifest = tmp_path / "plot_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "items": [
                    {"id": "p1", "data": str(data_path), "intent": "ok plot"},
                    {"id": "p2", "data": str(data_path), "intent": "flaky plot"},
                ]
            }
        ),
        encoding="utf-8",
    )

    state = {"flaky_calls": 0}

    class _FakePipeline:
        def __init__(self, settings=None, **kwargs):
            self.settings = settings

        async def generate(self, gen_input):
            from paperbanana.core.types import GenerationOutput, IterationRecord

            if "flaky" in gen_input.communicative_intent:
                state["flaky_calls"] += 1
                if state["flaky_calls"] == 1:
                    raise RuntimeError("temporary")
            img = str(tmp_path / f"{gen_input.communicative_intent.replace(' ', '_')}.png")
            return GenerationOutput(
                image_path=img,
                description="d",
                iterations=[IterationRecord(iteration=1, description="d", image_path=img)],
                metadata={"run_id": "run_plot"},
            )

    monkeypatch.setattr("paperbanana.core.pipeline.PaperBananaPipeline", _FakePipeline)

    result = runner.invoke(
        app,
        [
            "plot-batch",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path),
            "--concurrency",
            "2",
            "--max-retries",
            "1",
        ],
    )
    assert result.exit_code == 0
    reports = sorted(tmp_path.glob("batch_*/batch_report.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert all(item.get("status") == "success" for item in report["items"])
    flaky = next(item for item in report["items"] if item["id"] == "p2")
    assert flaky.get("attempts", 0) >= 2


def test_batch_prints_status_table_on_partial_failure(tmp_path, monkeypatch):
    """batch prints per-item table, correct counts, and exits 1 when any item fails."""
    from paperbanana.core.types import CritiqueResult, GenerationOutput, IterationRecord

    txt = tmp_path / "input.txt"
    txt.write_text("methodology text", encoding="utf-8")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        f"items:\n  - input: {txt.name}\n    caption: 'fig1'\n    id: item_ok\n"
        f"  - input: {txt.name}\n    caption: 'fig2'\n    id: item_fail\n",
        encoding="utf-8",
    )
    call_state = {"n": 0}

    class _FakePipeline:
        def __init__(self, settings=None, **kwargs):
            pass

        async def generate(self, gen_input):
            call_state["n"] += 1
            if call_state["n"] == 2:
                raise RuntimeError("critic parse error")
            img = str(tmp_path / "out.png")
            return GenerationOutput(
                image_path=img,
                description="d",
                iterations=[
                    IterationRecord(
                        iteration=1,
                        description="d",
                        image_path=img,
                        critique=CritiqueResult(critic_suggestions=[]),
                    )
                ],
                metadata={"run_id": "r1"},
            )

    monkeypatch.setattr("paperbanana.core.pipeline.PaperBananaPipeline", _FakePipeline)
    result = runner.invoke(
        app, ["batch", "--manifest", str(manifest), "--output-dir", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "1 succeeded" in result.output
    assert "1 failed" in result.output
    assert "✓" in result.output
    assert "✗" in result.output


def test_evaluate_plot_rejects_missing_data_file(tmp_path):
    """evaluate-plot fails early when the data file does not exist."""
    generated = tmp_path / "generated.png"
    reference = tmp_path / "reference.png"
    generated.write_bytes(b"fake-image")
    reference.write_bytes(b"fake-image")

    result = runner.invoke(
        app,
        [
            "evaluate-plot",
            "--generated",
            str(generated),
            "--reference",
            str(reference),
            "--data",
            str(tmp_path / "missing.csv"),
            "--intent",
            "Compare method variants",
        ],
    )
    assert result.exit_code == 1
    assert "Data file not found" in result.output


# ── References subcommand tests ──────────────────────────────────


def _build_reference_store(tmp_path, examples=None):
    """Create a minimal reference store for testing."""
    store_dir = tmp_path / "ref_store"
    store_dir.mkdir()
    (store_dir / "images").mkdir()

    if examples is None:
        examples = [
            {
                "id": "2404.00001v1",
                "source_context": "Methodology section text.",
                "caption": "Overview of our model architecture.",
                "image_path": "images/2404.00001v1.jpg",
                "category": "nlp_language",
                "aspect_ratio": 1.5,
            },
            {
                "id": "2404.00002v1",
                "source_context": "Another methodology section.",
                "caption": "Training pipeline for the vision model.",
                "image_path": "images/2404.00002v1.jpg",
                "category": "vision_perception",
                "aspect_ratio": 2.0,
            },
            {
                "id": "2404.00003v1",
                "source_context": "Third methodology section.",
                "caption": "Agent reasoning framework.",
                "image_path": "images/2404.00003v1.jpg",
                "category": "nlp_language",
                "aspect_ratio": 1.0,
            },
        ]

    data = {
        "metadata": {
            "name": "test",
            "version": "1.0.0",
            "categories": list({e["category"] for e in examples}),
            "total_examples": len(examples),
        },
        "examples": examples,
    }
    (store_dir / "index.json").write_text(json.dumps(data), encoding="utf-8")
    return store_dir


def test_references_list(tmp_path, monkeypatch):
    """references list prints a table with all examples."""
    store_dir = _build_reference_store(tmp_path)
    monkeypatch.setenv("REFERENCE_SET_PATH", str(store_dir))

    result = runner.invoke(app, ["references", "list"])
    assert result.exit_code == 0
    assert "2404.00001v1" in result.output
    assert "2404.00002v1" in result.output
    assert "2404.00003v1" in result.output
    assert "3" in result.output  # count in title


def test_references_list_filter_by_category(tmp_path, monkeypatch):
    """references list --category filters correctly."""
    store_dir = _build_reference_store(tmp_path)
    monkeypatch.setenv("REFERENCE_SET_PATH", str(store_dir))

    result = runner.invoke(app, ["references", "list", "--category", "vision_perception"])
    assert result.exit_code == 0
    assert "2404.00002v1" in result.output
    assert "2404.00001v1" not in result.output


def test_references_list_json(tmp_path, monkeypatch):
    """references list --json emits valid JSON."""
    store_dir = _build_reference_store(tmp_path)
    monkeypatch.setenv("REFERENCE_SET_PATH", str(store_dir))

    result = runner.invoke(app, ["references", "list", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data) == 3
    assert all("id" in item for item in data)


def test_references_show(tmp_path, monkeypatch):
    """references show displays details for a specific example."""
    store_dir = _build_reference_store(tmp_path)
    monkeypatch.setenv("REFERENCE_SET_PATH", str(store_dir))

    result = runner.invoke(app, ["references", "show", "2404.00001v1"])
    assert result.exit_code == 0
    assert "2404.00001v1" in result.output
    assert "nlp_language" in result.output
    assert "Overview of our model architecture" in result.output


def test_references_show_not_found(tmp_path, monkeypatch):
    """references show exits 1 for unknown ID."""
    store_dir = _build_reference_store(tmp_path)
    monkeypatch.setenv("REFERENCE_SET_PATH", str(store_dir))

    result = runner.invoke(app, ["references", "show", "nonexistent"])
    assert result.exit_code == 1
    assert "No reference found" in result.output


def test_references_show_json(tmp_path, monkeypatch):
    """references show --json emits valid JSON."""
    store_dir = _build_reference_store(tmp_path)
    monkeypatch.setenv("REFERENCE_SET_PATH", str(store_dir))

    result = runner.invoke(app, ["references", "show", "2404.00001v1", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["id"] == "2404.00001v1"
    assert data["category"] == "nlp_language"


def test_references_categories(tmp_path, monkeypatch):
    """references categories shows category counts."""
    store_dir = _build_reference_store(tmp_path)
    monkeypatch.setenv("REFERENCE_SET_PATH", str(store_dir))

    result = runner.invoke(app, ["references", "categories"])
    assert result.exit_code == 0
    assert "nlp_language" in result.output
    assert "vision_perception" in result.output
    assert "3" in result.output  # total


def test_references_categories_json(tmp_path, monkeypatch):
    """references categories --json emits valid JSON with counts."""
    store_dir = _build_reference_store(tmp_path)
    monkeypatch.setenv("REFERENCE_SET_PATH", str(store_dir))

    result = runner.invoke(app, ["references", "categories", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["nlp_language"] == 2
    assert data["vision_perception"] == 1
