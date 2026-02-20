"""Tests for PaperBanana CLI."""

from __future__ import annotations

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
