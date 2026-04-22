"""Tests for paperbanana show-config command."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from paperbanana.cli import app

runner = CliRunner()


def test_show_config_table_output():
    result = runner.invoke(app, ["show-config"])
    assert result.exit_code == 0
    assert "Resolved PaperBanana Settings" in result.output
    assert "vlm_provider" in result.output
    assert "image_provider" in result.output


def test_show_config_json_output():
    result = runner.invoke(app, ["show-config", "--json"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert "vlm_provider" in parsed
    assert "image_provider" in parsed
    assert "_effective_vlm_model" in parsed
    assert "_effective_image_model" in parsed


def test_show_config_masks_api_keys(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "sk-test-secret-key-value")
    result = runner.invoke(app, ["show-config", "--json"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["google_api_key"] == "sk-t****alue"
    assert "sk-test-secret-key-value" not in result.output


def test_show_config_masks_short_api_keys(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "short")
    result = runner.invoke(app, ["show-config", "--json"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["google_api_key"] == "****"
    assert "short" not in result.output


def test_show_config_with_yaml_config(tmp_path):
    cfg = tmp_path / "test.yaml"
    cfg.write_text("vlm:\n  provider: openai\n  model: gpt-4o\n")
    result = runner.invoke(app, ["show-config", "--config", str(cfg), "--json"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["vlm_provider"] == "openai"
    assert parsed["vlm_model"] == "gpt-4o"
