"""Tests for pre-flight PyMuPDF/Gradio dependency checks (issue #131)."""

from __future__ import annotations

import builtins
from pathlib import Path

import click
import pytest
from typer.testing import CliRunner

from paperbanana.cli import _check_pdf_dep, _require_pdf_dep, app

runner = CliRunner()
_real_import = builtins.__import__


@pytest.fixture()
def block_fitz(monkeypatch):
    def _import(name, *args, **kwargs):
        if name == "fitz":
            raise ImportError("No module named 'fitz'")
        return _real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)


@pytest.fixture()
def block_gradio(monkeypatch):
    def _import(name, *args, **kwargs):
        if name == "gradio":
            raise ImportError("No module named 'gradio'")
        return _real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)


# ── unit tests ────────────────────────────────────────────────────────────────


def test_require_pdf_dep_exits_when_fitz_missing(block_fitz):
    with pytest.raises(click.exceptions.Exit) as exc_info:
        _require_pdf_dep()
    assert exc_info.value.exit_code == 1


def test_check_pdf_dep_ignores_non_pdf():
    _check_pdf_dep(Path("input.txt"))
    _check_pdf_dep(Path("input.png"))


def test_check_pdf_dep_delegates_for_pdf(monkeypatch):
    called = []
    monkeypatch.setattr("paperbanana.cli._require_pdf_dep", lambda: called.append(True))
    _check_pdf_dep(Path("paper.pdf"))
    assert called


# ── CLI integration ───────────────────────────────────────────────────────────


def test_generate_exits_with_hint_when_fitz_missing(tmp_path, block_fitz):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    result = runner.invoke(app, ["generate", "--input", str(pdf), "--caption", "test", "--dry-run"])
    assert result.exit_code == 1
    assert "PyMuPDF" in result.output
    assert "paperbanana[pdf]" in result.output


def test_batch_exits_with_hint_when_manifest_has_pdf(tmp_path, block_fitz):
    pdf = tmp_path / "input.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(f"items:\n  - input: {pdf}\n    caption: test\n", encoding="utf-8")
    result = runner.invoke(
        app, ["batch", "--manifest", str(manifest), "--output-dir", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "PyMuPDF" in result.output


def test_studio_exits_with_hint_when_gradio_missing(block_gradio):
    result = runner.invoke(app, ["studio"])
    assert result.exit_code == 1
    assert "Gradio" in result.output
    assert "paperbanana[studio]" in result.output
