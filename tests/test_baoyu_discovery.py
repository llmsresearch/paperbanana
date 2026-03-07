"""Tests for baoyu path discovery."""

from __future__ import annotations

from unittest.mock import patch
from pathlib import Path

from paperbanana.utils.baoyu_discovery import discover_baoyu_slide_deck


def test_discover_returns_none_when_no_cache():
    """Returns None when baoyu plugin cache doesn't exist."""
    with patch("paperbanana.utils.baoyu_discovery.BAOYU_CACHE_GLOB", "/nonexistent/*/skills/baoyu-slide-deck"):
        result = discover_baoyu_slide_deck()
    assert result is None


def test_discover_returns_path_with_skill_md():
    """Returns a path that contains SKILL.md when cache exists."""
    result = discover_baoyu_slide_deck()
    if result is not None:
        assert (result / "SKILL.md").exists()
        assert (result / "references" / "styles").is_dir()


def test_discover_styles_dir():
    """Discovered path has references/styles/ with .md files."""
    result = discover_baoyu_slide_deck()
    if result is not None:
        styles = list((result / "references" / "styles").glob("*.md"))
        assert len(styles) >= 14
