"""Anthropic Claude VLM provider."""

from __future__ import annotations

from typing import Optional

import structlog
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from paperbanana.core.utils import image_to_base64
from paperbanana.providers.base import VLMProvider

logger = structlog.get_logger()


class AnthropicVLM(VLMProvider):
    """VLM provider using the Anthropic Python SDK (async).

    Works with Claude 3.x / 4.x models via the Messages API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20251023",
    ):
        self._api_key = api_key
        self._model = model
        self._client = None

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self._model

    def _get_client(self):
        """Lazy-init an AsyncAnthropic client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic

                self._client = AsyncAnthropic(api_key=self._api_key)
            except ImportError:
                raise ImportError(
                    "anthropic is required for the Anthropic provider. "
                    "Install with: pip install 'paperbanana[anthropic]'"
                )
        return self._client

    def is_available(self) -> bool:
        return self._api_key is not None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    async def generate(
        self,
        prompt: str,
        images: Optional[list[Image.Image]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        response_format: Optional[str] = None,
    ) -> str:
        client = self._get_client()

        content: list[dict] = []
        if images:
            for img in images:
                b64 = image_to_base64(img)
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    }
                )

        content.append({"type": "text", "text": prompt})

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        params: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }

        if system_prompt:
            params["system"] = system_prompt

        if response_format == "json":
            # Use structured outputs with a permissive JSON schema so callers
            # receive machine-parseable JSON while retaining flexibility.
            params["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "additionalProperties": True,
                    },
                }
            }

        response = await client.messages.create(**params)

        # Anthropic returns a list of content blocks; we concatenate all text blocks.
        parts: list[str] = []
        for block in getattr(response, "content", []):
            # Support both SDK objects and plain dicts in tests.
            block_type = getattr(block, "type", None)
            if isinstance(block, dict):
                block_type = block.get("type")
            if block_type == "text":
                text_value = getattr(block, "text", None)
                if isinstance(block, dict):
                    text_value = block.get("text", text_value)
                if text_value:
                    parts.append(text_value)

        text = "".join(parts)

        logger.debug(
            "Anthropic response",
            model=self._model,
            usage=getattr(response, "usage", None),
        )
        return text
