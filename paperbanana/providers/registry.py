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
    "ANTHROPIC_API_KEY": (
        "ANTHROPIC_API_KEY not found.\n\n"
        "To fix this:\n"
        "  1. Get an API key at: https://console.anthropic.com/settings/keys\n"
        "  2. Set the environment variable:\n\n"
        "  export ANTHROPIC_API_KEY=your-key-here"
    ),
    "AWS_CREDENTIALS": (
        "AWS credentials not found for Bedrock.\n\n"
        "To fix this, configure one of:\n"
        "  1. Environment variables:\n"
        "     export AWS_ACCESS_KEY_ID=your-key\n"
        "     export AWS_SECRET_ACCESS_KEY=your-secret\n\n"
        "  2. AWS credentials file (~/.aws/credentials):\n"
        "     aws configure\n\n"
        "  3. IAM role (for EC2/ECS/Lambda)"
    ),
}


def _validate_api_key(key_value: str | None, env_var_name: str) -> None:
    """Raise a helpful error if the required API key is missing."""
    if key_value is None or not key_value.strip():
        hint = _API_KEY_HINTS.get(env_var_name, f"{env_var_name} is not set.")
        raise ValueError(hint)


def _validate_bedrock_auth(region: str, profile: str | None) -> None:
    """Raise a helpful error if AWS credentials are not available."""
    try:
        import boto3
    except ImportError:
        raise ImportError(
            "boto3 is required for the Bedrock provider. "
            "Install with: pip install 'paperbanana[bedrock]'"
        )
    session = boto3.Session(region_name=region, profile_name=profile)
    credentials = session.get_credentials()
    if credentials is None:
        raise ValueError(_API_KEY_HINTS["AWS_CREDENTIALS"])


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
                model=settings.google_vlm_model or settings.vlm_model,
                base_url=settings.google_base_url,
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
        elif provider == "bedrock":
            _validate_bedrock_auth(settings.aws_region, settings.aws_profile)
            from paperbanana.providers.vlm.bedrock import BedrockVLM

            return BedrockVLM(
                model=settings.bedrock_vlm_model or settings.vlm_model,
                region=settings.aws_region,
                profile=settings.aws_profile,
            )
        elif provider == "anthropic":
            _validate_api_key(settings.anthropic_api_key, "ANTHROPIC_API_KEY")
            from paperbanana.providers.vlm.anthropic import AnthropicVLM

            return AnthropicVLM(
                api_key=settings.anthropic_api_key,
                model=settings.vlm_model,
            )
        elif provider == "ollama":
            from paperbanana.providers.vlm.ollama import OllamaVLM

            return OllamaVLM(
                model=settings.ollama_model or settings.vlm_model,
                base_url=settings.ollama_base_url,
                json_mode=settings.ollama_json_mode,
            )
        elif provider == "openai_local":
            from paperbanana.providers.vlm.openai import OpenAIVLM

            return OpenAIVLM(
                api_key=settings.openai_api_key or "not-needed",
                model=settings.openai_vlm_model or settings.vlm_model,
                base_url=settings.openai_local_base_url,
                json_mode=settings.openai_local_json_mode,
                provider_name="openai_local",
            )
        elif provider == "claude_code":
            from paperbanana.providers.vlm.claude_code import ClaudeCodeVLM

            vlm = ClaudeCodeVLM(model=settings.vlm_model)
            if not vlm.is_available():
                raise ValueError(
                    "claude CLI not found in PATH.\n\n"
                    "Install Claude Code and sign in, then"
                    " ensure `claude` is available on PATH."
                )
            return vlm
        else:
            raise ValueError(
                "Unknown VLM provider: "
                f"{provider}. Available: gemini, openrouter, openai, openai_local, "
                f"bedrock, anthropic, ollama, claude_code"
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
                model=settings.google_image_model or settings.image_model,
                base_url=settings.google_base_url,
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
        elif provider == "bedrock_imagen":
            _validate_bedrock_auth(settings.aws_region, settings.aws_profile)
            from paperbanana.providers.image_gen.bedrock_imagen import BedrockImageGen

            return BedrockImageGen(
                model=settings.bedrock_image_model or settings.image_model,
                region=settings.aws_region,
                profile=settings.aws_profile,
            )
        else:
            raise ValueError(
                f"Unknown image provider: {provider}. "
                f"Available: google_imagen, openrouter_imagen, openai_imagen, bedrock_imagen"
            )
