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
    check_builtin_refs,
    check_env_key,
    check_expanded_refs,
    check_optional_package,
    check_paperbanana,
    check_python,
    run_doctor,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# CheckResult dataclass
# ---------------------------------------------------------------------------


def test_check_result_ok_defaults():
    r = CheckResult("Test", True, "1.0")
    assert r.ok
    assert r.hint is None
    assert r.critical is False


def test_check_result_failed_carries_hint():
    r = CheckResult("Test", False, "not installed", "pip install foo")
    assert not r.ok
    assert r.hint == "pip install foo"


def test_check_result_critical_flag():
    r = CheckResult("Runtime", True, "ok", critical=True)
    assert r.critical


# ---------------------------------------------------------------------------
# Runtime checks
# ---------------------------------------------------------------------------


def test_check_python_always_passes():
    r = check_python()
    assert r.ok
    assert r.critical
    import sys

    assert str(sys.version_info.major) in r.detail


def test_check_paperbanana_passes_when_installed():
    r = check_paperbanana()
    assert r.ok
    assert r.critical
    assert r.detail  # version string is non-empty


def test_check_paperbanana_fails_when_missing():
    from importlib.metadata import PackageNotFoundError

    with patch(
        "paperbanana.doctor.pkg_version", side_effect=PackageNotFoundError("paperbanana")
    ):
        r = check_paperbanana()
    assert not r.ok
    assert r.critical


# ---------------------------------------------------------------------------
# Optional package checks
# ---------------------------------------------------------------------------


def test_check_optional_package_installed():
    # pydantic is always installed as a core dep
    r = check_optional_package("Pydantic", "pydantic", "core")
    assert r.ok
    assert r.hint is None


def test_check_optional_package_missing():
    from importlib.metadata import PackageNotFoundError

    with patch(
        "paperbanana.doctor.pkg_version", side_effect=PackageNotFoundError("fakepkg")
    ):
        r = check_optional_package("FakePkg", "fakepkg", "fake")
    assert not r.ok
    assert r.detail == "not installed"
    assert "paperbanana[fake]" in r.hint
    assert not r.critical  # optional packages are not critical


# ---------------------------------------------------------------------------
# API key checks
# ---------------------------------------------------------------------------


def test_check_env_key_set(monkeypatch):
    monkeypatch.setenv("TEST_KEY_XYZ", "abc123")
    r = check_env_key("TEST_KEY_XYZ")
    assert r.ok
    assert r.detail == "set"


def test_check_env_key_missing(monkeypatch):
    monkeypatch.delenv("TEST_KEY_XYZ", raising=False)
    r = check_env_key("TEST_KEY_XYZ")
    assert not r.ok
    assert r.detail == "not set"


def test_check_env_key_empty_string(monkeypatch):
    monkeypatch.setenv("TEST_KEY_XYZ", "   ")
    r = check_env_key("TEST_KEY_XYZ")
    assert not r.ok


# ---------------------------------------------------------------------------
# AWS credentials check
# ---------------------------------------------------------------------------


def test_check_aws_credentials_via_env(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    with patch.object(Path, "exists", return_value=False):
        r = check_aws_credentials()
    assert r.ok
    assert r.detail == "configured"


def test_check_aws_credentials_via_profile(monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.setenv("AWS_PROFILE", "default")
    with patch.object(Path, "exists", return_value=False):
        r = check_aws_credentials()
    assert r.ok


def test_check_aws_credentials_via_file(monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    with patch.object(Path, "exists", return_value=True):
        r = check_aws_credentials()
    assert r.ok


def test_check_aws_credentials_missing(monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    with patch.object(Path, "exists", return_value=False):
        r = check_aws_credentials()
    assert not r.ok
    assert r.hint is not None


# ---------------------------------------------------------------------------
# Reference data checks
# ---------------------------------------------------------------------------


def test_check_builtin_refs_with_valid_index(tmp_path, monkeypatch):
    index = tmp_path / "data" / "reference_sets" / "index.json"
    index.parent.mkdir(parents=True)
    index.write_text(json.dumps({"examples": [{"id": "a"}, {"id": "b"}]}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    r = check_builtin_refs()
    assert r.ok
    assert r.critical
    assert "2 diagrams" in r.detail


def test_check_builtin_refs_missing_index(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = check_builtin_refs()
    assert not r.ok
    assert r.critical


def test_check_expanded_refs_not_downloaded():
    from paperbanana.data.manager import DatasetManager

    with patch.object(DatasetManager, "is_downloaded", return_value=False):
        r = check_expanded_refs()
    assert not r.ok
    assert "not downloaded" in r.detail
    assert "paperbanana data download" in r.hint


def test_check_expanded_refs_downloaded():
    from paperbanana.data.manager import DatasetManager

    with (
        patch.object(DatasetManager, "is_downloaded", return_value=True),
        patch.object(
            DatasetManager,
            "get_info",
            return_value={"example_count": 42, "datasets": ["curated"]},
        ),
    ):
        r = check_expanded_refs()
    assert r.ok
    assert "42 diagrams" in r.detail
    assert "curated" in r.detail


# ---------------------------------------------------------------------------
# run_doctor orchestration
# ---------------------------------------------------------------------------


def test_run_doctor_returns_int():
    result = run_doctor()
    assert isinstance(result, int)
    assert result in (0, 1)


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
    from contextlib import ExitStack

    stack = ExitStack()
    for name, rv in defaults.items():
        stack.enter_context(patch(f"paperbanana.doctor.{name}", return_value=rv))
    return stack


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


def test_run_doctor_json_output():
    with _patch_all_checks():
        result = run_doctor(output_json=True)
    assert result == 0


def test_run_doctor_json_output_with_critical_failure():
    fail = CheckResult("paperbanana", False, "not found", critical=True)
    with _patch_all_checks(check_paperbanana=fail):
        result = run_doctor(output_json=True)
    assert result == 1


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_doctor_command_runs():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code in (0, 1)
    assert "PaperBanana" in result.output
    assert "System Check" in result.output


def test_doctor_command_shows_all_sections():
    result = runner.invoke(app, ["doctor"])
    assert "Runtime" in result.output
    assert "Optional features" in result.output
    assert "API keys" in result.output
    assert "Reference data" in result.output


def test_doctor_command_json_flag():
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code in (0, 1)
    # Output should be valid JSON
    output = result.output.strip()
    parsed = json.loads(output)
    assert "version" in parsed
    assert "ok" in parsed
    assert "checks" in parsed


def test_doctor_command_exit_1_when_critical_fails():
    fail = CheckResult("paperbanana", False, "not found", critical=True)
    with patch("paperbanana.doctor.check_paperbanana", return_value=fail):
        result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
