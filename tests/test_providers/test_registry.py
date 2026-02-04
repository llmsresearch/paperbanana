"""Tests for the provider registry."""

from __future__ import annotations

import pytest

from paperbanana.core.config import Settings
from paperbanana.providers.registry import ProviderRegistry


def test_create_gemini_vlm():
    """Test creating a Gemini VLM provider."""
    settings = Settings(
        vlm_provider="gemini",
        vlm_model="gemini-3-flash",
        google_api_key="test-key",
    )
    vlm = ProviderRegistry.create_vlm(settings)
    assert vlm.name == "gemini"
    assert vlm.model_name == "gemini-3-flash"


def test_create_google_imagen_gen():
    """Test creating a Google Imagen image gen provider."""
    settings = Settings(
        image_provider="google_imagen",
        google_api_key="test-key",
    )
    gen = ProviderRegistry.create_image_gen(settings)
    assert gen.name == "google_imagen"


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
