"""Tests for pre-flight PyMuPDF dependency checks (issue #131)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from paperbanana.cli import _check_pdf_dep, _require_pdf_dep, app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


def test_require_pdf_dep_passes_when_fitz_available():
    """_require_pdf_dep() does not raise when PyMuPDF is installed."""
    pytest.importorskip("fitz")
    _require_pdf_dep()  # should not raise


def test_require_pdf_dep_exits_when_fitz_missing(monkeypatch):
    """_require_pdf_dep() prints install hint and exits 1 when fitz is absent."""
    import builtins

    import click

    real_import = builtins.__import__

    def _block_fitz(name, *args, **kwargs):
        if name == "fitz":
            raise ImportError("No module named 'fitz'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_fitz)
    with pytest.raises(click.exceptions.Exit) as exc_info:
        _require_pdf_dep()
    assert exc_info.value.exit_code == 1


def test_check_pdf_dep_ignores_non_pdf():
    """_check_pdf_dep() is a no-op for non-PDF paths even without fitz."""
    _check_pdf_dep(Path("input.txt"))
    _check_pdf_dep(Path("input.png"))
    _check_pdf_dep(Path("input"))


def test_check_pdf_dep_delegates_for_pdf(monkeypatch):
    """_check_pdf_dep() calls _require_pdf_dep() for .pdf paths."""
    called = []
    monkeypatch.setattr("paperbanana.cli._require_pdf_dep", lambda: called.append(True))
    _check_pdf_dep(Path("paper.pdf"))
    assert called


def test_check_pdf_dep_case_insensitive(monkeypatch):
    """_check_pdf_dep() treats .PDF and .Pdf the same as .pdf."""
    called = []
    monkeypatch.setattr("paperbanana.cli._require_pdf_dep", lambda: called.append(True))
    _check_pdf_dep(Path("paper.PDF"))
    _check_pdf_dep(Path("paper.Pdf"))
    assert len(called) == 2


# ---------------------------------------------------------------------------
# CLI integration tests — generate command
# ---------------------------------------------------------------------------


def test_generate_pdf_input_missing_fitz(tmp_path, monkeypatch):
    """generate with .pdf input exits 1 with install hint when PyMuPDF is absent."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    import builtins

    real_import = builtins.__import__

    def _block_fitz(name, *args, **kwargs):
        if name == "fitz":
            raise ImportError("No module named 'fitz'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_fitz)

    result = runner.invoke(
        app,
        ["generate", "--input", str(pdf), "--caption", "test", "--dry-run"],
    )
    assert result.exit_code == 1
    assert "PyMuPDF" in result.output
    assert "paperbanana[pdf]" in result.output


def test_generate_pdf_input_error_before_pipeline(tmp_path, monkeypatch):
    """generate PDF error fires before any pipeline initialization."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    check_called = []
    monkeypatch.setattr("paperbanana.cli._check_pdf_dep", lambda p: check_called.append(p))

    runner.invoke(app, ["generate", "--input", str(pdf), "--caption", "test"])
    assert check_called
    assert check_called[0] == pdf


# ---------------------------------------------------------------------------
# CLI integration tests — sweep command
# ---------------------------------------------------------------------------


def test_sweep_pdf_input_missing_fitz(tmp_path, monkeypatch):
    """sweep with .pdf input exits 1 with install hint when PyMuPDF is absent."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    import builtins

    real_import = builtins.__import__

    def _block_fitz(name, *args, **kwargs):
        if name == "fitz":
            raise ImportError("No module named 'fitz'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_fitz)

    result = runner.invoke(
        app,
        ["sweep", "--input", str(pdf), "--caption", "test", "--dry-run"],
    )
    assert result.exit_code == 1
    assert "PyMuPDF" in result.output
    assert "paperbanana[pdf]" in result.output


# ---------------------------------------------------------------------------
# CLI integration tests — batch command
# ---------------------------------------------------------------------------


def test_batch_preflight_catches_pdf_without_fitz(tmp_path, monkeypatch):
    """batch exits 1 before starting any work when manifest has PDF items and fitz is absent."""
    txt = tmp_path / "input.pdf"
    txt.write_bytes(b"%PDF-1.4 fake")

    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        f"items:\n  - input: {txt}\n    caption: test\n",
        encoding="utf-8",
    )

    import builtins

    real_import = builtins.__import__

    def _block_fitz(name, *args, **kwargs):
        if name == "fitz":
            raise ImportError("No module named 'fitz'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_fitz)

    result = runner.invoke(
        app,
        ["batch", "--manifest", str(manifest), "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 1
    assert "PyMuPDF" in result.output
    assert "paperbanana[pdf]" in result.output


def test_batch_preflight_skips_check_for_txt_only_manifest(tmp_path, monkeypatch):
    """batch does not check for fitz when no PDF items are in the manifest."""
    txt = tmp_path / "input.txt"
    txt.write_text("method text", encoding="utf-8")

    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        f"items:\n  - input: {txt}\n    caption: test\n",
        encoding="utf-8",
    )

    fitz_checked = []

    import builtins

    real_import = builtins.__import__

    def _track_fitz(name, *args, **kwargs):
        if name == "fitz":
            fitz_checked.append(True)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _track_fitz)

    # Invoke but expect early exit for unrelated reasons (no pipeline mock needed)
    runner.invoke(
        app,
        ["batch", "--manifest", str(manifest), "--output-dir", str(tmp_path)],
    )
    assert not fitz_checked


# ---------------------------------------------------------------------------
# CLI integration tests — studio command
# ---------------------------------------------------------------------------


def test_studio_missing_gradio_exits_with_hint(monkeypatch):
    """studio command exits 1 with Gradio install hint when gradio is absent.

    Gradio is a lazy import inside build_studio_app(), so the module-level import
    of paperbanana.studio.app succeeds — the error only fires at runtime.
    _require_studio_dep() catches this early via a pre-flight check.
    """
    import builtins

    real_import = builtins.__import__

    def _block_gradio(name, *args, **kwargs):
        if name == "gradio":
            raise ImportError("No module named 'gradio'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_gradio)

    result = runner.invoke(app, ["studio"])
    assert result.exit_code == 1
    assert "Gradio" in result.output
    assert "paperbanana[studio]" in result.output
