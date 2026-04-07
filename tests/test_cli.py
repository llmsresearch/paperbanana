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
