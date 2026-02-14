"""Provider registry and factory for PaperBanana."""

from __future__ import annotations

import structlog

from paperbanana.core.config import Settings
from paperbanana.providers.base import ImageGenProvider, VLMProvider

logger = structlog.get_logger()


class ProviderRegistry:
    """Factory for creating VLM and image generation providers from config."""

    @staticmethod
    def create_vlm(settings: Settings) -> VLMProvider:
        """Create a VLM provider based on settings."""
        provider = settings.vlm_provider.lower()
        logger.info("Creating VLM provider", provider=provider, model=settings.vlm_model)

        if provider == "gemini":
            from paperbanana.providers.vlm.gemini import GeminiVLM

            return GeminiVLM(
                api_key=settings.google_api_key,
                model=settings.vlm_model,
            )
        elif provider == "openrouter":
            from paperbanana.providers.vlm.openrouter import OpenRouterVLM

            return OpenRouterVLM(
                api_key=settings.openrouter_api_key,
                model=settings.vlm_model,
            )
        else:
            raise ValueError(f"Unknown VLM provider: {provider}. Available: gemini, openrouter")

    @staticmethod
    def create_image_gen(settings: Settings) -> ImageGenProvider:
        """Create an image generation provider based on settings."""
        provider = settings.image_provider.lower()
        logger.info("Creating image gen provider", provider=provider, model=settings.image_model)

        if provider == "google_imagen":
            from paperbanana.providers.image_gen.google_imagen import GoogleImagenGen

            return GoogleImagenGen(
                api_key=settings.google_api_key,
                model=settings.image_model,
            )
        elif provider == "openrouter_imagen":
            from paperbanana.providers.image_gen.openrouter_imagen import (
                OpenRouterImageGen,
            )

            return OpenRouterImageGen(
                api_key=settings.openrouter_api_key,
                model=settings.image_model,
            )
        else:
            raise ValueError(
                f"Unknown image provider: {provider}. Available: google_imagen, openrouter_imagen"
            )
