"""Tests for configuration validation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from paperbanana.core.config import Settings


def test_output_format_default_is_png():
    """Default output_format remains png."""
    settings = Settings()
    assert settings.output_format == "png"


def test_output_format_valid_jpeg():
    """output_format accepts jpeg."""
    settings = Settings(output_format="jpeg")
    assert settings.output_format == "jpeg"


def test_output_format_valid_webp():
    """output_format accepts webp."""
    settings = Settings(output_format="webp")
    assert settings.output_format == "webp"


def test_output_format_case_insensitive():
    """output_format normalizes to lowercase."""
    settings = Settings(output_format="JPEG")
    assert settings.output_format == "jpeg"


def test_output_format_invalid_rejected():
    """Invalid output_format is rejected with clear error."""
    with pytest.raises(ValidationError, match="output_format must be png, jpeg, or webp"):
        Settings(output_format="gif")


def test_output_format_from_yaml():
    """output_format from YAML config is validated."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump({"output": {"format": "webp"}}, f)
        path = f.name

    try:
        settings = Settings.from_yaml(path)
        assert settings.output_format == "webp"
    finally:
        Path(path).unlink(missing_ok=True)


def test_output_format_from_yaml_invalid():
    """Invalid output_format in YAML is rejected."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump({"output": {"format": "svg"}}, f)
        path = f.name

    try:
        with pytest.raises(ValidationError, match="output_format must be png, jpeg, or webp"):
            Settings.from_yaml(path)
    finally:
        Path(path).unlink(missing_ok=True)
