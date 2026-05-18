"""Tests for analytics loading, aggregation, and CLI exposure."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from paperbanana.analytics import load_analytics_records, summarize_records
from paperbanana.analytics.reporting import summary_to_dict
from paperbanana.cli import app

runner = CliRunner()


def test_analytics_loader_and_summary(tmp_path):
    output_root = tmp_path / "outputs"
    run_dir = output_root / "run_001"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": "run_001",
                "vlm_provider": "gemini",
                "image_provider": "google_imagen",
                "timing": {"total_seconds": 5.0},
                "total_cost_usd": 0.12,
            }
        ),
        encoding="utf-8",
    )

    batch_dir = output_root / "batch_001"
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "batch_report.json").write_text(
        json.dumps(
            {
                "batch_id": "batch_001",
                "total_seconds": 12.0,
                "items": [
                    {"id": "a", "status": "success"},
                    {"id": "b", "status": "failed"},
                ],
            }
        ),
        encoding="utf-8",
    )

    orchestrate_dir = output_root / "orchestrate_001"
    orchestrate_dir.mkdir(parents=True, exist_ok=True)
    (orchestrate_dir / "figure_package.json").write_text(
        json.dumps(
            {
                "orchestration_id": "orchestrate_001",
                "total_seconds": 8.0,
                "generated_items": [{"id": "m1"}],
                "failures": [{"id": "m2", "error": "boom"}],
            }
        ),
        encoding="utf-8",
    )

    records = load_analytics_records(output_root)
    assert len(records) == 5

    summary = summarize_records(records)
    payload = summary_to_dict(summary)
    assert payload["total_records"] == 5
    assert payload["success_records"] == 3
    assert payload["failed_records"] == 2
    assert payload["total_seconds"] == 25.0
    assert payload["total_cost_usd"] == 0.12
    assert payload["source_type_counts"]["run"] == 1
    assert payload["source_type_counts"]["batch_item"] == 2
    assert payload["source_type_counts"]["orchestration_item"] == 2


def test_cli_analytics_json_output(tmp_path):
    output_root = tmp_path / "outputs"
    run_dir = output_root / "run_001"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": "run_001",
                "timing": {"total_seconds": 3.0},
                "vlm_provider": "openai",
                "image_provider": "openai_imagen",
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["analytics", "--path", str(output_root), "--format", "json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["total_records"] == 1
    assert payload["success_records"] == 1
    assert payload["vlm_provider_counts"]["openai"] == 1
