"""Tests for paperbanana doctor health-check command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from paperbanana.cli import app
from paperbanana.doctor import (
    CheckResult,
    check_aws_credentials,
    check_env_key,
    check_expanded_refs,
    check_optional_package,
    run_doctor,
)

runner = CliRunner()


# ── Optional package checks ───────────────────────────────────────────────────


def test_check_optional_package_missing(monkeypatch):
    from importlib import metadata

    def _raise(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", _raise)
    r = check_optional_package("FakePkg", "fakepkg", "fake")
    assert not r.ok
    assert r.detail == "not installed"
    assert "paperbanana[fake]" in r.hint


# ── API key checks ────────────────────────────────────────────────────────────


def test_check_env_key_missing(monkeypatch):
    monkeypatch.delenv("TEST_KEY_XYZ", raising=False)
    r = check_env_key("TEST_KEY_XYZ")
    assert not r.ok
    assert r.detail == "not set"


def test_check_env_key_empty_string(monkeypatch):
    monkeypatch.setenv("TEST_KEY_XYZ", "   ")
    r = check_env_key("TEST_KEY_XYZ")
    assert not r.ok


# ── AWS credentials check ─────────────────────────────────────────────────────


def test_check_aws_credentials_via_env(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    with patch.object(Path, "exists", return_value=False):
        r = check_aws_credentials()
    assert r.ok
    assert r.detail == "configured"


def test_check_aws_credentials_missing(monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    with patch.object(Path, "exists", return_value=False):
        r = check_aws_credentials()
    assert not r.ok
    assert r.hint is not None


# ── Reference data checks ─────────────────────────────────────────────────────


def test_check_expanded_refs_not_downloaded(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERBANANA_CACHE_DIR", str(tmp_path))
    r = check_expanded_refs()
    assert not r.ok
    assert "not downloaded" in r.detail
    assert "paperbanana data download" in r.hint


def test_check_expanded_refs_downloaded(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERBANANA_CACHE_DIR", str(tmp_path))
    ref_dir = tmp_path / "reference_sets"
    ref_dir.mkdir()
    (ref_dir / "index.json").write_text(
        json.dumps({"examples": [{"id": "e1"}, {"id": "e2"}]}), encoding="utf-8"
    )
    (ref_dir / "dataset_info.json").write_text(
        json.dumps(
            {
                "datasets": ["curated"],
                "example_count": 2,
                "dataset_meta": {"curated": {"version": "1.0.0", "source": "test"}},
            }
        ),
        encoding="utf-8",
    )
    r = check_expanded_refs()
    assert r.ok
    assert "2 diagrams" in r.detail


# ── run_doctor ────────────────────────────────────────────────────────────────


def test_run_doctor_exit_0_when_all_pass():
    ok = CheckResult("x", True, "ok")
    with (
        patch("paperbanana.doctor.check_python", return_value=ok),
        patch("paperbanana.doctor.check_paperbanana", return_value=ok),
        patch("paperbanana.doctor.check_optional_package", return_value=ok),
        patch("paperbanana.doctor.check_env_key", return_value=ok),
        patch("paperbanana.doctor.check_aws_credentials", return_value=ok),
        patch("paperbanana.doctor.check_builtin_refs", return_value=ok),
        patch("paperbanana.doctor.check_expanded_refs", return_value=ok),
    ):
        assert run_doctor() == 0


def test_run_doctor_exit_1_when_any_fails():
    ok = CheckResult("x", True, "ok")
    fail = CheckResult("y", False, "missing", "fix hint")
    with (
        patch("paperbanana.doctor.check_python", return_value=ok),
        patch("paperbanana.doctor.check_paperbanana", return_value=fail),
        patch("paperbanana.doctor.check_optional_package", return_value=ok),
        patch("paperbanana.doctor.check_env_key", return_value=ok),
        patch("paperbanana.doctor.check_aws_credentials", return_value=ok),
        patch("paperbanana.doctor.check_builtin_refs", return_value=ok),
        patch("paperbanana.doctor.check_expanded_refs", return_value=ok),
    ):
        assert run_doctor() == 1


# ── CLI integration ───────────────────────────────────────────────────────────


def test_doctor_command_shows_all_sections():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code in (0, 1)
    assert "Runtime" in result.output
    assert "Optional features" in result.output
    assert "API keys" in result.output
    assert "Reference data" in result.output
