"""Slide style picker component using the 23 built-in presets."""

from __future__ import annotations

from paperbanana.guidelines.slide_styles import SLIDE_STYLE_PRESETS, get_style_prompt, list_styles


def get_style_choices() -> list[str]:
    """Return style names for a Gradio Radio/Dropdown."""
    return list_styles()


def get_style_info(style_name: str) -> str:
    """Return a human-readable description for a style."""
    if not style_name or style_name not in SLIDE_STYLE_PRESETS:
        return ""
    preset = SLIDE_STYLE_PRESETS[style_name]
    feel = preset.get("feel", "")
    source = preset.get("source", "")
    return f"**{style_name}** ({source})\n\n{feel}"


def get_style_prompt_text(style_name: str) -> str:
    """Return the full style prompt for preview."""
    if not style_name:
        return ""
    try:
        return get_style_prompt(style_name)
    except KeyError:
        return f"Unknown style: {style_name}"
