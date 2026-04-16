"""Tests for paperbanana doctor health-check command."""

from __future__ import annotations

import json
from contextlib import ExitStack
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
    check_paperbanana,
    run_doctor,
)

runner = CliRunner()


# ── Helpers ──────────────────────────────────────────────────────────────────


def _patch_all_checks(**overrides):
    """Return a context manager that patches all check functions with ok results."""
    ok = CheckResult("x", True, "ok")
    defaults = {
        "check_python": ok,
        "check_paperbanana": ok,
        "check_optional_package": ok,
        "check_env_key": ok,
        "check_aws_credentials": ok,
        "check_builtin_refs": CheckResult("Built-in set", True, "13 diagrams", critical=True),
        "check_expanded_refs": ok,
    }
    defaults.update(overrides)
    stack = ExitStack()
    for name, rv in defaults.items():
        stack.enter_context(patch(f"paperbanana.doctor.{name}", return_value=rv))
    return stack


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
    assert not r.critical


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


# ── Critical vs optional severity ────────────────────────────────────────────


def test_check_paperbanana_is_critical():
    r = check_paperbanana()
    assert r.critical


def test_optional_package_is_not_critical():
    from importlib.metadata import PackageNotFoundError

    with patch("paperbanana.doctor.pkg_version", side_effect=PackageNotFoundError("fakepkg")):
        r = check_optional_package("FakePkg", "fakepkg", "fake")
    assert not r.critical


# ── run_doctor ────────────────────────────────────────────────────────────────


def test_run_doctor_exit_0_when_all_pass():
    with _patch_all_checks():
        assert run_doctor() == 0


def test_run_doctor_exit_1_when_critical_fails():
    fail = CheckResult("paperbanana", False, "not found", critical=True)
    with _patch_all_checks(check_paperbanana=fail):
        assert run_doctor() == 1


def test_run_doctor_exit_0_when_only_optional_fails():
    """Missing optional packages should NOT cause exit code 1."""
    fail = CheckResult("OpenAI", False, "not installed", "pip install 'paperbanana[openai]'")
    with _patch_all_checks(check_optional_package=fail):
        assert run_doctor() == 0


# ── JSON output ───────────────────────────────────────────────────────────────


def test_run_doctor_json_returns_0_when_healthy():
    with _patch_all_checks():
        assert run_doctor(output_json=True) == 0


def test_run_doctor_json_returns_1_when_critical_fails():
    fail = CheckResult("paperbanana", False, "not found", critical=True)
    with _patch_all_checks(check_paperbanana=fail):
        assert run_doctor(output_json=True) == 1


# ── CLI integration ───────────────────────────────────────────────────────────


def test_doctor_command_shows_all_sections():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code in (0, 1)
    assert "Runtime" in result.output
    assert "Optional features" in result.output
    assert "API keys" in result.output
    assert "Reference data" in result.output


def test_doctor_command_json_flag():
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code in (0, 1)
    output = result.output.strip()
    parsed = json.loads(output)
    assert "version" in parsed
    assert "ok" in parsed
    assert "checks" in parsed
    assert isinstance(parsed["checks"], list)
