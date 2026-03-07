"""Tests for slide style presets."""

from __future__ import annotations

import pytest

from paperbanana.guidelines.slide_styles import (
    get_style_prompt,
    get_style_info,
    list_styles,
    match_style,
)


def test_list_styles_returns_23_presets():
    styles = list_styles()
    assert len(styles) == 23
    assert "blueprint" in styles
    assert "tech-keynote" in styles
    assert "scientific" in styles


def test_get_style_prompt_returns_string():
    prompt = get_style_prompt("blueprint")
    assert "Blueprint" in prompt
    assert "Color Palette" in prompt


def test_get_style_prompt_case_insensitive():
    assert get_style_prompt("Blueprint") == get_style_prompt("blueprint")


def test_get_style_prompt_unknown_raises():
    with pytest.raises(KeyError, match="Unknown slide style"):
        get_style_prompt("nonexistent-style")


def test_get_style_info_has_required_keys():
    info = get_style_info("scientific")
    assert "source" in info
    assert "feel" in info
    assert "prompt" in info
    assert info["source"] == "baoyu-slide-deck"


def test_match_style_finds_scientific():
    result = match_style("biology and medical imaging research")
    assert result == "scientific"


def test_match_style_finds_ml_ai():
    result = match_style("deep learning transformer neural network")
    assert result == "ml-ai"


def test_match_style_returns_none_for_garbage():
    result = match_style("xyzzy foobar")
    assert result is None
