"""Tests for the provider registry."""

from __future__ import annotations

import pytest

from paperbanana.core.config import Settings
from paperbanana.providers.registry import ProviderRegistry


def test_create_gemini_vlm():
    """Test creating a Gemini VLM provider."""
    settings = Settings(
        vlm_provider="gemini",
        vlm_model="gemini-2.5-flash",
        google_api_key="test-key",
    )
    vlm = ProviderRegistry.create_vlm(settings)
    assert vlm.name == "gemini"
    assert vlm.model_name == "gemini-2.5-flash"


def test_create_google_imagen_gen():
    """Test creating a Google Imagen image gen provider."""
    settings = Settings(
        image_provider="google_imagen",
        google_api_key="test-key",
    )
    gen = ProviderRegistry.create_image_gen(settings)
    assert gen.name == "google_imagen"


def test_missing_google_api_key_raises_helpful_error():
    """Test that missing GOOGLE_API_KEY raises a helpful error with setup instructions."""
    settings = Settings(vlm_provider="gemini", google_api_key=None)
    with pytest.raises(ValueError, match="GOOGLE_API_KEY not found") as exc_info:
        ProviderRegistry.create_vlm(settings)
    error_msg = str(exc_info.value)
    assert "makersuite.google.com" in error_msg
    assert "paperbanana setup" in error_msg
    assert "export GOOGLE_API_KEY" in error_msg


def test_missing_openrouter_api_key_raises_helpful_error():
    """Test that missing OPENROUTER_API_KEY raises a helpful error with setup instructions."""
    settings = Settings(vlm_provider="openrouter", openrouter_api_key=None)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY not found") as exc_info:
        ProviderRegistry.create_vlm(settings)
    error_msg = str(exc_info.value)
    assert "openrouter.ai/keys" in error_msg
    assert "export OPENROUTER_API_KEY" in error_msg


def test_missing_google_api_key_for_image_gen_raises_helpful_error():
    """Test that missing GOOGLE_API_KEY for image gen raises a helpful error."""
    settings = Settings(image_provider="google_imagen", google_api_key=None)
    with pytest.raises(ValueError, match="GOOGLE_API_KEY not found"):
        ProviderRegistry.create_image_gen(settings)


def test_missing_openrouter_api_key_for_image_gen_raises_helpful_error():
    """Test that missing OPENROUTER_API_KEY for image gen raises a helpful error."""
    settings = Settings(image_provider="openrouter_imagen", openrouter_api_key=None)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY not found"):
        ProviderRegistry.create_image_gen(settings)


def test_empty_api_key_raises_helpful_error():
    """Test that empty or whitespace-only API key raises a helpful error."""
    settings = Settings(vlm_provider="gemini", google_api_key="   ")
    with pytest.raises(ValueError, match="GOOGLE_API_KEY not found") as exc_info:
        ProviderRegistry.create_vlm(settings)
    error_msg = str(exc_info.value)
    assert "makersuite.google.com" in error_msg
    assert "paperbanana setup" in error_msg
    assert "export GOOGLE_API_KEY" in error_msg


def test_unknown_vlm_provider_raises():
    """Test that unknown VLM provider raises ValueError."""
    settings = Settings(vlm_provider="nonexistent")
    with pytest.raises(ValueError, match="Unknown VLM provider"):
        ProviderRegistry.create_vlm(settings)


def test_unknown_image_provider_raises():
    """Test that unknown image provider raises ValueError."""
    settings = Settings(image_provider="nonexistent")
    with pytest.raises(ValueError, match="Unknown image provider"):
        ProviderRegistry.create_image_gen(settings)
