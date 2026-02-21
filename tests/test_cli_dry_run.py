"""Tests for the --dry-run CLI flag."""

from __future__ import annotations

import tempfile
from pathlib import Path

from typer.testing import CliRunner

from paperbanana.cli import app

runner = CliRunner()


def test_dry_run_with_valid_input():
    """Dry run with valid input file should succeed."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("This is a methodology description for testing purposes.")
        f.flush()

        result = runner.invoke(
            app,
            ["generate", "--input", f.name, "--caption", "Test caption", "--dry-run"],
        )

    assert result.exit_code == 0
    assert "Dry Run" in result.output
    assert "All checks passed" in result.output


def test_dry_run_missing_input_file():
    """Dry run with nonexistent input file should report the issue."""
    result = runner.invoke(
        app,
        [
            "generate",
            "--input",
            "/nonexistent/path.txt",
            "--caption",
            "Test",
            "--dry-run",
        ],
    )

    assert "not found" in result.output


def test_dry_run_no_input():
    """Dry run without --input should still show config and warn."""
    result = runner.invoke(app, ["generate", "--dry-run"])

    assert result.exit_code == 0
    assert "Dry Run" in result.output
    assert "No --input provided" in result.output


def test_dry_run_shows_config():
    """Dry run should display provider and model configuration."""
    result = runner.invoke(app, ["generate", "--dry-run"])

    assert "VLM provider" in result.output
    assert "VLM model" in result.output
    assert "Image provider" in result.output
