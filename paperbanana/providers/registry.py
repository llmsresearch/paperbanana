"""Provider registry and factory for PaperBanana."""

from __future__ import annotations

import structlog

from paperbanana.core.config import Settings
from paperbanana.providers.base import ImageGenProvider, VLMProvider

logger = structlog.get_logger()


_API_KEY_HINTS = {
    "GOOGLE_API_KEY": (
        "GOOGLE_API_KEY not found.\n\n"
        "To fix this:\n"
        "  1. Get a free API key at: https://makersuite.google.com/app/apikey\n"
        "  2. Run: paperbanana setup\n\n"
        "Or set it manually:\n"
        "  export GOOGLE_API_KEY=your-key-here"
    ),
    "OPENROUTER_API_KEY": (
        "OPENROUTER_API_KEY not found.\n\n"
        "To fix this:\n"
        "  1. Get an API key at: https://openrouter.ai/keys\n"
        "  2. Set the environment variable:\n\n"
        "  export OPENROUTER_API_KEY=your-key-here"
    ),
    "OPENAI_API_KEY": (
        "OPENAI_API_KEY not found.\n\n"
        "To fix this:\n"
        "  1. Get an API key at: https://platform.openai.com/api-keys\n"
        "  2. Set the environment variable:\n\n"
        "  export OPENAI_API_KEY=your-key-here"
    ),
}


def _validate_api_key(key_value: str | None, env_var_name: str) -> None:
    """Raise a helpful error if the required API key is missing."""
    if key_value is None or not key_value.strip():
        hint = _API_KEY_HINTS.get(env_var_name, f"{env_var_name} is not set.")
        raise ValueError(hint)


class ProviderRegistry:
    """Factory for creating VLM and image generation providers from config."""

    @staticmethod
    def create_vlm(settings: Settings) -> VLMProvider:
        """Create a VLM provider based on settings."""
        provider = settings.vlm_provider.lower()
        logger.info("Creating VLM provider", provider=provider, model=settings.vlm_model)

        if provider == "gemini":
            _validate_api_key(settings.google_api_key, "GOOGLE_API_KEY")
            from paperbanana.providers.vlm.gemini import GeminiVLM

            return GeminiVLM(
                api_key=settings.google_api_key,
                model=settings.vlm_model,
            )
        elif provider == "openrouter":
            _validate_api_key(settings.openrouter_api_key, "OPENROUTER_API_KEY")
            from paperbanana.providers.vlm.openrouter import OpenRouterVLM

            return OpenRouterVLM(
                api_key=settings.openrouter_api_key,
                model=settings.vlm_model,
            )
        elif provider == "openai":
            _validate_api_key(settings.openai_api_key, "OPENAI_API_KEY")
            from paperbanana.providers.vlm.openai import OpenAIVLM

            return OpenAIVLM(
                api_key=settings.openai_api_key,
                model=settings.openai_vlm_model or settings.vlm_model,
                base_url=settings.openai_base_url,
            )
        else:
            raise ValueError(
                f"Unknown VLM provider: {provider}. Available: gemini, openrouter, openai"
            )

    @staticmethod
    def create_image_gen(settings: Settings) -> ImageGenProvider:
        """Create an image generation provider based on settings."""
        provider = settings.image_provider.lower()
        logger.info("Creating image gen provider", provider=provider, model=settings.image_model)

        if provider == "google_imagen":
            _validate_api_key(settings.google_api_key, "GOOGLE_API_KEY")
            from paperbanana.providers.image_gen.google_imagen import GoogleImagenGen

            return GoogleImagenGen(
                api_key=settings.google_api_key,
                model=settings.image_model,
            )
        elif provider == "openrouter_imagen":
            _validate_api_key(settings.openrouter_api_key, "OPENROUTER_API_KEY")
            from paperbanana.providers.image_gen.openrouter_imagen import (
                OpenRouterImageGen,
            )

            return OpenRouterImageGen(
                api_key=settings.openrouter_api_key,
                model=settings.image_model,
            )
        elif provider == "openai_imagen":
            _validate_api_key(settings.openai_api_key, "OPENAI_API_KEY")
            from paperbanana.providers.image_gen.openai_imagen import OpenAIImageGen

            return OpenAIImageGen(
                api_key=settings.openai_api_key,
                model=settings.openai_image_model or settings.image_model,
                base_url=settings.openai_base_url,
            )
        else:
            raise ValueError(
                f"Unknown image provider: {provider}. "
                f"Available: google_imagen, openrouter_imagen, openai_imagen"
            )
