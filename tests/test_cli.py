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
