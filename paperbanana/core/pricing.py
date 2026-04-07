"""Pricing tables for VLM and image generation providers.

Prices are in USD. VLM prices are per 1K tokens. Image prices are per image.
Last updated: 2026-03-18.
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger()

# (provider, model_prefix) -> {"input_per_1k": USD, "output_per_1k": USD}
VLM_PRICING: dict[tuple[str, str], dict[str, float]] = {
    # Google Gemini — free tier
    ("gemini", "gemini-2.0-flash"): {"input_per_1k": 0.0, "output_per_1k": 0.0},
    ("gemini", "gemini-2.5-flash"): {"input_per_1k": 0.00015, "output_per_1k": 0.0006},
    ("gemini", "gemini-2.5-pro"): {"input_per_1k": 0.00125, "output_per_1k": 0.01},
    ("gemini", "gemini-3-pro"): {"input_per_1k": 0.00125, "output_per_1k": 0.005},
    # OpenAI
    ("openai", "gpt-5.2"): {"input_per_1k": 0.0025, "output_per_1k": 0.01},
    ("openai", "gpt-5.1"): {"input_per_1k": 0.002, "output_per_1k": 0.008},
    ("openai", "gpt-4o"): {"input_per_1k": 0.0025, "output_per_1k": 0.01},
    ("openai", "gpt-4o-mini"): {"input_per_1k": 0.00015, "output_per_1k": 0.0006},
    # Anthropic
    ("anthropic", "claude-sonnet-4"): {"input_per_1k": 0.003, "output_per_1k": 0.015},
    ("anthropic", "claude-3-5-sonnet"): {"input_per_1k": 0.003, "output_per_1k": 0.015},
    ("anthropic", "claude-3-5-haiku"): {"input_per_1k": 0.0008, "output_per_1k": 0.004},
    ("anthropic", "claude-opus-4"): {"input_per_1k": 0.015, "output_per_1k": 0.075},
    # Bedrock (approximate — varies by region)
    ("bedrock", "us.amazon.nova-pro"): {"input_per_1k": 0.0008, "output_per_1k": 0.0032},
    ("bedrock", "us.amazon.nova-lite"): {"input_per_1k": 0.00006, "output_per_1k": 0.00024},
    ("bedrock", "anthropic.claude-3-5-sonnet"): {"input_per_1k": 0.003, "output_per_1k": 0.015},
    # OpenRouter — passthrough pricing depends on underlying model;
    # use a reasonable default for popular models
    ("openrouter", "google/gemini-3-flash-preview"): {"input_per_1k": 0.0, "output_per_1k": 0.0},
    ("openrouter", "google/gemini-2.0-flash"): {"input_per_1k": 0.0, "output_per_1k": 0.0},
    ("openrouter", "openai/gpt-4o"): {"input_per_1k": 0.0025, "output_per_1k": 0.01},
}

# (provider, model_prefix) -> USD per image
IMAGE_GEN_PRICING: dict[tuple[str, str], float] = {
    # Google Imagen — free tier
    ("google_imagen", "gemini-3-pro-image-preview"): 0.0,
    # OpenAI
    ("openai_imagen", "gpt-image-1.5"): 0.02,
    ("openai_imagen", "gpt-image-1"): 0.04,
    ("openai_imagen", "dall-e-3"): 0.04,
    # Bedrock Nova Canvas
    ("bedrock_imagen", "amazon.nova-canvas"): 0.04,
    # OpenRouter — depends on underlying model
    ("openrouter_imagen", "google/gemini-3-pro-image-preview"): 0.0,
}


def lookup_vlm_price(provider: str, model: str) -> dict[str, float] | None:
    """Look up VLM pricing by provider and model (prefix match).

    Returns {"input_per_1k": float, "output_per_1k": float} or None if unknown.
    """
    # Exact match first
    key = (provider, model)
    if key in VLM_PRICING:
        return VLM_PRICING[key]

    # Prefix match: find the longest prefix that matches
    best_match = None
    best_len = 0
    for (p, m), pricing in VLM_PRICING.items():
        if p == provider and model.startswith(m) and len(m) > best_len:
            best_match = pricing
            best_len = len(m)

    if best_match is None:
        logger.warning("Unknown VLM pricing", provider=provider, model=model)
    return best_match


def lookup_image_price(provider: str, model: str) -> float | None:
    """Look up image generation price per image (prefix match).

    Returns USD per image or None if unknown.
    """
    key = (provider, model)
    if key in IMAGE_GEN_PRICING:
        return IMAGE_GEN_PRICING[key]

    best_match = None
    best_len = 0
    for (p, m), price in IMAGE_GEN_PRICING.items():
        if p == provider and model.startswith(m) and len(m) > best_len:
            best_match = price
            best_len = len(m)

    if best_match is None:
        logger.warning("Unknown image gen pricing", provider=provider, model=model)
    return best_match
