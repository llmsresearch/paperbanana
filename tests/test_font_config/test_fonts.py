"""Tests for font configuration."""

from __future__ import annotations

import pytest

from paperbanana.config.fonts import (
    FontConfig,
    get_default_font_config,
    get_legacy_font_config,
)


def test_font_config_get_font_string() -> None:
    config = FontConfig(primary_fonts=["Tahoma"], fallback_fonts=["Helvetica", "Arial"])
    assert config.get_font_string() == "Tahoma,Helvetica,Arial"


def test_font_config_get_font_string_custom_separator() -> None:
    config = FontConfig(primary_fonts=["Tahoma"], fallback_fonts=["Helvetica"])
    assert config.get_font_string(separator=" ") == "Tahoma Helvetica"


def test_font_config_get_first_available_font() -> None:
    config = FontConfig(primary_fonts=["Tahoma"], fallback_fonts=["Helvetica"])
    assert config.get_first_available_font() == "Tahoma"


def test_font_config_default_fallbacks() -> None:
    config = FontConfig(primary_fonts=["Tahoma"])
    assert config.fallback_fonts == ["Helvetica", "Arial", "sans-serif"]


def test_font_config_empty_primary_raises() -> None:
    with pytest.raises(ValueError, match="primary_fonts must not be empty"):
        FontConfig(primary_fonts=[])


def test_get_default_font_config() -> None:
    config = get_default_font_config()
    assert config.primary_fonts == ["Tahoma"]
    assert config.fallback_fonts == ["Helvetica", "Arial", "sans-serif"]
    assert config.get_first_available_font() == "Tahoma"


def test_get_legacy_font_config() -> None:
    config = get_legacy_font_config()
    assert config.primary_fonts == ["Helvetica"]
    assert config.fallback_fonts == ["Arial", "sans-serif"]
    assert config.get_first_available_font() == "Helvetica"


def test_font_config_multiple_primary_fonts() -> None:
    config = FontConfig(primary_fonts=["Tahoma", "Arial"], fallback_fonts=["Helvetica"])
    assert config.get_font_string() == "Tahoma,Arial,Helvetica"
    assert config.get_first_available_font() == "Tahoma"
