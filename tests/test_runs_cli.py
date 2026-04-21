"""Tests for the `paperbanana runs` CLI subcommands."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from paperbanana.cli import app

runner = CliRunner()


def _write_run(output_dir: Path, run_id: str, *, caption: str = "c") -> Path:
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_input.json").write_text(
        json.dumps(
            {
                "source_context": "ctx",
                "communicative_intent": caption,
                "diagram_type": "methodology",
            }
        ),
        encoding="utf-8",
    )
    # Create a "final output" sentinel.
    (run_dir / "final_output.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (run_dir / "metadata.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    return run_dir


def _write_batch(output_dir: Path, batch_id: str) -> Path:
    batch_dir = output_dir / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "batch_report.json").write_text(
        json.dumps(
            {
                "batch_kind": "methodology",
                "items": [
                    {"id": "a", "status": "success", "output_path": "x.png"},
                    {"id": "b", "status": "failed"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return batch_dir


def test_runs_list_show_delete_run(tmp_path: Path):
    out = tmp_path / "outputs"
    out.mkdir()
    _write_run(out, "run_20260101_000000_abcd01", caption="My caption")

    listed = runner.invoke(app, ["runs", "list", "--plain", "--output-dir", str(out)])
    assert listed.exit_code == 0
    assert "run_20260101_000000_abcd01" in listed.output
    assert "My caption" in listed.output

    shown = runner.invoke(
        app, ["runs", "show", "run_20260101_000000_abcd01", "-o", str(out), "--plain"]
    )
    assert shown.exit_code == 0
    assert "final_output.png" in shown.output
    assert "run_input.json" in shown.output

    refused = runner.invoke(app, ["runs", "delete", "run_20260101_000000_abcd01", "-o", str(out)])
    assert refused.exit_code == 1
    assert "--yes" in refused.output

    deleted = runner.invoke(
        app, ["runs", "delete", "run_20260101_000000_abcd01", "-o", str(out), "--yes"]
    )
    assert deleted.exit_code == 0
    assert not (out / "run_20260101_000000_abcd01").exists()


def test_runs_list_show_delete_batch(tmp_path: Path):
    out = tmp_path / "outputs"
    out.mkdir()
    _write_batch(out, "batch_20260101_000000_beef00")

    listed = runner.invoke(
        app, ["runs", "list", "--plain", "--kind", "batch", "--output-dir", str(out)]
    )
    assert listed.exit_code == 0
    assert "batch_20260101_000000_beef00" in listed.output

    shown = runner.invoke(
        app, ["runs", "show", "batch_20260101_000000_beef00", "-o", str(out), "--plain"]
    )
    assert shown.exit_code == 0
    assert "batch_report.json" in shown.output
    assert "methodology" in shown.output

    deleted = runner.invoke(
        app, ["runs", "delete", "batch_20260101_000000_beef00", "-o", str(out), "--yes"]
    )
    assert deleted.exit_code == 0
    assert not (out / "batch_20260101_000000_beef00").exists()
