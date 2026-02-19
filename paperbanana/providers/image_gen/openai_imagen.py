"""OpenAI image generation provider — works with both OpenAI and Azure OpenAI endpoints."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Optional

import structlog
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from paperbanana.providers.base import ImageGenProvider

logger = structlog.get_logger()


class OpenAIImageGen(ImageGenProvider):
    """Image generation using the OpenAI Python SDK (async).

    Supports GPT-Image-1.5, GPT-Image-1, DALL-E 3, and other OpenAI image models.
    Compatible with both OpenAI and Azure OpenAI / Foundry endpoints.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-image-1.5",
        base_url: str = "https://api.openai.com/v1",
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._client = None

    @property
    def name(self) -> str:
        return "openai_imagen"

    @property
    def model_name(self) -> str:
        return self._model

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(
                    api_key=self._api_key,
                    base_url=self._base_url,
                )
            except ImportError:
                raise ImportError(
                    "openai is required for the OpenAI provider. "
                    "Install with: pip install 'paperbanana[openai]'"
                )
        return self._client

    def is_available(self) -> bool:
        return self._api_key is not None

    def _size_string(self, width: int, height: int) -> str:
        """Map pixel dimensions to an OpenAI-supported size string."""
        ratio = width / height
        if ratio > 1.2:
            return "1536x1024"
        if ratio < 0.83:
            return "1024x1536"
        return "1024x1024"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    async def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
    ) -> Image.Image:
        client = self._get_client()

        full_prompt = prompt
        if negative_prompt:
            full_prompt += f"\n\nAvoid: {negative_prompt}"

        result = await client.images.generate(
            model=self._model,
            prompt=full_prompt,
            n=1,
            size=self._size_string(width, height),
        )

        b64_data = result.data[0].b64_json
        image_bytes = base64.b64decode(b64_data)
        return Image.open(BytesIO(image_bytes))
