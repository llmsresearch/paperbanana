"""Google Gemini 3 Pro image generation provider."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Optional

import structlog
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from paperbanana.providers.base import ImageGenProvider

logger = structlog.get_logger()


class GoogleImagenGen(ImageGenProvider):
    """Google Gemini 3 Pro Image generation via google-genai SDK.

    Uses the gemini-3-pro-image-preview model with response_modalities=["IMAGE"].
    Requires a Google API key (free tier available).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-3-pro-image-preview",
        base_url: Optional[str] = None,
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._client = None

    @property
    def name(self) -> str:
        return "google_imagen"

    @property
    def model_name(self) -> str:
        return self._model

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai

                client_kwargs = {"api_key": self._api_key}
                if self._base_url:
                    client_kwargs["http_options"] = {"base_url": self._base_url}
                self._client = genai.Client(**client_kwargs)
            except ImportError:
                raise ImportError(
                    "google-genai is required for Google Imagen provider. "
                    "Install with: pip install 'paperbanana[google]'"
                )
        return self._client

    def is_available(self) -> bool:
        return self._api_key is not None

    @property
    def supported_ratios(self) -> list[str]:
        return ["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"]

    # All aspect ratios supported by Google Imagen API
    _SUPPORTED_RATIOS = {"1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"}

    def _aspect_ratio(self, width: int, height: int) -> str:
        """Infer aspect ratio from pixel dimensions."""
        ratio = width / height
        if ratio > 2.0:
            return "21:9"
        if ratio > 1.5:
            return "16:9"
        if ratio > 1.2:
            return "4:3"
        if ratio > 1.05:
            return "3:2"  # not a standard ratio but close to 4:3
        if ratio < 0.5:
            return "9:16"
        if ratio < 0.67:
            return "2:3"  # not a standard ratio but close to 3:4
        if ratio < 0.83:
            return "3:4"
        return "1:1"

    def _image_size(self, width: int, height: int) -> str:
        max_dim = max(width, height)
        if max_dim <= 1024:
            return "1K"
        if max_dim <= 2048:
            return "2K"
        return "4K"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        aspect_ratio: Optional[str] = None,
    ) -> Image.Image:
        from google.genai import types

        self._get_client()

        if negative_prompt:
            prompt = f"{prompt}\n\nAvoid: {negative_prompt}"

        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio or self._aspect_ratio(width, height),
                image_size=self._image_size(width, height),
            ),
        )

        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=config,
        )

        parts = None
        if getattr(response, "candidates", None):
            parts = response.candidates[0].content.parts
        else:
            parts = getattr(response, "parts", None)

        if not parts:
            raise ValueError("Gemini image response had no content parts.")

        for part in parts:
            if hasattr(part, "as_image"):
                try:
                    img = part.as_image()
                    if self.cost_tracker is not None:
                        self.cost_tracker.record_image_call(provider=self.name, model=self._model)
                    return img
                except Exception:
                    pass
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                data = inline.data
                image_bytes = base64.b64decode(data) if isinstance(data, str) else data
                if self.cost_tracker is not None:
                    self.cost_tracker.record_image_call(provider=self.name, model=self._model)
                return Image.open(BytesIO(image_bytes))

        logger.error("No image data in Gemini response", model=self._model)
        raise ValueError("Gemini image response did not contain image data.")
