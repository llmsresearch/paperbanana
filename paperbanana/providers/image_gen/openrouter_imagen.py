"""OpenRouter image generation provider — uses any image model via the OpenAI-compatible API."""

from __future__ import annotations

import base64
import re
from io import BytesIO
from typing import Optional

import structlog
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from paperbanana.providers.base import ImageGenProvider

logger = structlog.get_logger()


class OpenRouterImageGen(ImageGenProvider):
    """Image generation routed through OpenRouter.

    Talks to models that support ``modalities: ["image", "text"]``
    (e.g. google/gemini-3-pro-image-preview) and returns a PIL Image
    decoded from the base64 response.

    Get an API key at https://openrouter.ai/keys
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "google/gemini-3-pro-image-preview",
    ):
        self._api_key = api_key
        self._model = model
        self._client = None

    @property
    def name(self) -> str:
        return "openrouter_imagen"

    @property
    def model_name(self) -> str:
        return self._model

    def _get_client(self):
        """Lazy-init an async httpx client pointed at the OpenRouter API."""
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                base_url="https://openrouter.ai/api/v1",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "HTTP-Referer": "https://github.com/llmsresearch/paperbanana",
                    "X-Title": "PaperBanana",
                },
                # Image generation can take a while
                timeout=180.0,
            )
        return self._client

    def is_available(self) -> bool:
        return self._api_key is not None

    @property
    def supported_ratios(self) -> list[str]:
        # Prompt-based hints — any ratio is conceptually supported
        return ["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"]

    def _aspect_ratio_hint(self, width: int, height: int) -> str:
        """Turn pixel dimensions into a human-readable aspect ratio hint for the prompt."""
        ratio = width / height
        if ratio > 1.5:
            return "wide landscape format (16:9)"
        if ratio > 1.2:
            return "landscape format (3:2)"
        if ratio < 0.67:
            return "tall portrait format (9:16)"
        if ratio < 0.83:
            return "portrait format (2:3)"
        return "square format (1:1)"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    async def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        aspect_ratio: Optional[str] = None,
    ) -> Image.Image:
        client = self._get_client()

        # OpenRouter doesn't have native aspect-ratio params like the Google SDK,
        # so we bake the desired format into the prompt itself.
        if aspect_ratio:
            aspect_hint = f"{aspect_ratio} format"
        else:
            aspect_hint = self._aspect_ratio_hint(width, height)
        full_prompt = f"{prompt}\n\nGenerate this as a {aspect_hint} image."
        if negative_prompt:
            full_prompt += f"\n\nAvoid: {negative_prompt}"

        payload = {
            "model": self._model,
            "messages": [
                {"role": "user", "content": full_prompt},
            ],
            # This tells OpenRouter we want an image back, not just text
            "modalities": ["image", "text"],
        }

        if seed is not None:
            payload["seed"] = seed

        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

        message = data["choices"][0]["message"]

        # Primary path: images come as base64 data-URLs in the "images" array
        images = message.get("images", [])
        if images:
            for img_entry in images:
                url = img_entry.get("image_url", {}).get("url", "")
                if url.startswith("data:image/"):
                    b64_data = url.split(",", 1)[1]
                    image_bytes = base64.b64decode(b64_data)
                    return Image.open(BytesIO(image_bytes))

        # Fallback: some models inline the base64 data directly in the text content
        content = message.get("content", "")
        if "data:image/" in content:
            match = re.search(r"data:image/[^;]+;base64,([A-Za-z0-9+/=]+)", content)
            if match:
                image_bytes = base64.b64decode(match.group(1))
                return Image.open(BytesIO(image_bytes))

        logger.error("No image data in OpenRouter response", model=self._model)
        raise ValueError(
            f"OpenRouter response for {self._model} did not contain image data. "
            f"Content preview: {content[:200]}"
        )
