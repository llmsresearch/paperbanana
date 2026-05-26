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


def test_image_generation_options_from_yaml():
    """Image resolution and quality load from YAML config."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump({"pipeline": {"output_resolution": "4k"}, "image": {"quality": "high"}}, f)
        path = f.name

    try:
        settings = Settings.from_yaml(path)
        assert settings.output_resolution == "4k"
        assert settings.image_quality == "high"
    finally:
        Path(path).unlink(missing_ok=True)


def test_output_resolution_invalid_rejected():
    """Invalid output_resolution is rejected with clear error."""
    with pytest.raises(ValidationError, match="output_resolution must be 1k, 2k, or 4k"):
        Settings(output_resolution="8k")


def test_image_quality_invalid_rejected():
    """Invalid image_quality is rejected with clear error."""
    with pytest.raises(ValidationError, match="image_quality must be low, medium, high, or auto"):
        Settings(image_quality="ultra")


def test_exemplar_retrieval_top_k_must_be_positive():
    """exemplar_retrieval_top_k must be >= 1."""
    with pytest.raises(ValidationError, match="exemplar_retrieval_top_k must be >= 1"):
        Settings(exemplar_retrieval_top_k=0)


def test_exemplar_retrieval_timeout_must_be_positive():
    """exemplar_retrieval_timeout_seconds must be > 0."""
    with pytest.raises(ValidationError, match="exemplar_retrieval_timeout_seconds must be > 0"):
        Settings(exemplar_retrieval_timeout_seconds=0)


def test_exemplar_retrieval_mode_literal_validation():
    """exemplar_retrieval_mode accepts only supported modes."""
    with pytest.raises(ValidationError):
        Settings(exemplar_retrieval_mode="bad_mode")


def test_exemplar_retrieval_max_retries_must_be_non_negative():
    """exemplar_retrieval_max_retries must be >= 0."""
    with pytest.raises(ValidationError, match="exemplar_retrieval_max_retries must be >= 0"):
        Settings(exemplar_retrieval_max_retries=-1)


def test_seed_accepts_integer():
    """seed is accepted and stored."""
    settings = Settings(seed=1234)
    assert settings.seed == 1234


def test_effective_vlm_model_gemini_override():
    """Gemini VLM model override is used when provider is gemini."""
    settings = Settings(
        vlm_provider="gemini",
        vlm_model="gemini-2.0-flash",
        google_vlm_model="gemini-2.5-flash",
    )
    assert settings.effective_vlm_model == "gemini-2.5-flash"


def test_effective_vlm_model_atlas_override():
    """Atlas VLM model override is used when provider is atlas."""
    settings = Settings(
        vlm_provider="atlas",
        vlm_model="deepseek-ai/DeepSeek-V3-0324",
        atlascloud_vlm_model="qwen-turbo",
    )
    assert settings.effective_vlm_model == "qwen-turbo"


def test_effective_image_model_gemini_override():
    """Gemini image model override is used when provider is google_imagen."""
    settings = Settings(
        image_provider="google_imagen",
        image_model="gemini-3-pro-image-preview",
        google_image_model="gemini-2.5-flash-image-preview",
    )
    assert settings.effective_image_model == "gemini-2.5-flash-image-preview"


def test_effective_image_model_atlas_override():
    """Atlas image model override is used when provider is atlas_image."""
    settings = Settings(
        image_provider="atlas_image",
        image_model="openai/gpt-image-2/text-to-image",
        atlascloud_image_model="google/imagen4-fast",
    )
    assert settings.effective_image_model == "google/imagen4-fast"
