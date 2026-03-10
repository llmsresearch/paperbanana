"""Utilities for bridging Gradio UI with PaperBanana async pipeline."""

from __future__ import annotations

from typing import Any

from paperbanana.app.components.config_panel import (
    IMAGE_PROVIDER_MAP,
    VLM_PROVIDER_MAP,
)
from paperbanana.core.config import Settings


def settings_from_ui(
    vlm_provider_label: str,
    vlm_model: str,
    image_provider_label: str,
    image_model: str,
    api_key: str,
    api_key_2: str,
    iterations: int,
    resolution: str,
    auto_refine: bool,
    optimize_inputs: bool,
    output_dir: str = "outputs",
) -> Settings:
    """Build a Settings object from Gradio UI state.

    Supports all provider combinations with independent VLM and image gen keys.
    """
    vlm_provider = VLM_PROVIDER_MAP.get(vlm_provider_label, "gemini")
    image_provider = IMAGE_PROVIDER_MAP.get(image_provider_label, "google_imagen")

    # Route API keys to the correct Settings fields
    api_key_kwargs: dict[str, Any] = {}
    _assign_key(api_key_kwargs, vlm_provider, api_key)

    # If second key exists and is for a different provider family, assign it too
    if api_key_2 and api_key_2.strip():
        _assign_key_for_image(api_key_kwargs, image_provider, api_key_2)

    return Settings(
        vlm_provider=vlm_provider,
        vlm_model=vlm_model,
        image_provider=image_provider,
        image_model=image_model,
        refinement_iterations=iterations,
        output_resolution=resolution,
        auto_refine=auto_refine,
        optimize_inputs=optimize_inputs,
        output_dir=output_dir,
        **api_key_kwargs,
    )


def _assign_key(kwargs: dict, vlm_provider: str, key: str) -> None:
    """Assign the primary API key to the correct Settings field."""
    if not key or not key.strip():
        return
    mapping = {
        "gemini": "google_api_key",
        "openrouter": "openrouter_api_key",
        "openai": "openai_api_key",
        "anthropic": "anthropic_api_key",
    }
    field = mapping.get(vlm_provider)
    if field:
        kwargs[field] = key


def _assign_key_for_image(kwargs: dict, image_provider: str, key: str) -> None:
    """Assign the secondary API key for the image gen provider."""
    if not key or not key.strip():
        return
    mapping = {
        "google_imagen": "google_api_key",
        "openrouter_imagen": "openrouter_api_key",
        "openai_imagen": "openai_api_key",
    }
    field = mapping.get(image_provider)
    if field and field not in kwargs:
        kwargs[field] = key
