"""MiniMax VLM provider (Anthropic-compatible API)."""

from __future__ import annotations

from typing import Optional

import structlog
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from paperbanana.core.utils import image_to_base64
from paperbanana.providers.base import VLMProvider

logger = structlog.get_logger()

_MINIMAX_MODELS = [
    "MiniMax-M2.7",
    "MiniMax-M2.7-highspeed",
]

_DEFAULT_BASE_URL = "https://api.minimax.io/anthropic"


class MiniMaxVLM(VLMProvider):
    """VLM provider using MiniMax's Anthropic-compatible API.

    Supports MiniMax-M2.7 and MiniMax-M2.7-highspeed models.
    Requires ``MINIMAX_API_KEY`` environment variable.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "MiniMax-M2.7",
        base_url: str = _DEFAULT_BASE_URL,
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._client = None

    @property
    def name(self) -> str:
        return "minimax"

    @property
    def model_name(self) -> str:
        return self._model

    def _get_client(self):
        """Lazy-init an AsyncAnthropic client pointed at MiniMax's endpoint."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic

                self._client = AsyncAnthropic(
                    api_key=self._api_key,
                    base_url=self._base_url,
                )
            except ImportError:
                raise ImportError(
                    "anthropic is required for the MiniMax provider. "
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

        # MiniMax requires temperature in (0.0, 1.0]; clamp to 1.0 if zero.
        safe_temperature = temperature if temperature > 0.0 else 1.0

        params: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": safe_temperature,
        }

        if system_prompt:
            # MiniMax Anthropic-compatible API accepts a single string system message.
            params["system"] = system_prompt

        # MiniMax does not support response_format / output_config — skip it.

        response = await client.messages.create(**params)

        parts: list[str] = []
        for block in getattr(response, "content", []):
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

        usage = getattr(response, "usage", None)
        logger.debug("MiniMax response", model=self._model, usage=usage)

        if self.cost_tracker is not None and usage is not None:
            self.cost_tracker.record_vlm_call(
                provider=self.name,
                model=self._model,
                input_tokens=getattr(usage, "input_tokens", 0),
                output_tokens=getattr(usage, "output_tokens", 0),
            )
        return text
