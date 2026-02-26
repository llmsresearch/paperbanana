"""Google image generation provider — supports both Gemini and Imagen models."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Optional

import structlog
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from paperbanana.providers.base import ImageGenProvider

logger = structlog.get_logger()


def _is_imagen_model(model: str) -> bool:
    """Check if the model uses the Imagen API (generate_images) vs Gemini (generate_content)."""
    return model.startswith("imagen-")


class GoogleImagenGen(ImageGenProvider):
    """Google image generation via google-genai SDK.

    Supports two API paths:
    - Gemini models (gemini-3-pro-image-preview): generate_content with response_modalities=["IMAGE"]
    - Imagen models (imagen-4.0-*): generate_images dedicated endpoint
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "imagen-4.0-generate-001",
    ):
        self._api_key = api_key
        self._model = model
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

                self._client = genai.Client(api_key=self._api_key)
            except ImportError:
                raise ImportError(
                    "google-genai is required for Google Imagen provider. "
                    "Install with: pip install 'paperbanana[google]'"
                )
        return self._client

    def is_available(self) -> bool:
        return self._api_key is not None

    def _aspect_ratio(self, width: int, height: int) -> str:
        ratio = width / height
        if ratio > 1.5:
            return "16:9"
        if ratio > 1.2:
            return "3:2"
        if ratio < 0.67:
            return "9:16"
        if ratio < 0.83:
            return "2:3"
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
    ) -> Image.Image:
        self._get_client()

        if _is_imagen_model(self._model):
            return await self._generate_imagen(prompt, negative_prompt, width, height)
        else:
            return await self._generate_gemini(prompt, negative_prompt, width, height)

    async def _generate_imagen(
        self,
        prompt: str,
        negative_prompt: Optional[str],
        width: int,
        height: int,
    ) -> Image.Image:
        """Generate image using Imagen API (generate_images)."""
        from google.genai import types

        if negative_prompt:
            prompt = f"{prompt}\n\nAvoid: {negative_prompt}"

        config = types.GenerateImagesConfig(
            number_of_images=1,
            aspect_ratio=self._aspect_ratio(width, height),
        )

        logger.info("Calling Imagen API", model=self._model)

        response = self._client.models.generate_images(
            model=self._model,
            prompt=prompt,
            config=config,
        )

        if not response.generated_images:
            raise ValueError(f"Imagen API returned no images (model={self._model})")

        # generated_images[0].image is a google.genai.types.Image with .image_bytes
        gen_image = response.generated_images[0].image

        # The SDK's Image object has .image_bytes property
        if hasattr(gen_image, "image_bytes") and gen_image.image_bytes:
            return Image.open(BytesIO(gen_image.image_bytes))

        # Fallback: try ._pil_image or show()
        if hasattr(gen_image, "_pil_image") and gen_image._pil_image:
            return gen_image._pil_image

        raise ValueError("Imagen response did not contain extractable image data.")

    async def _generate_gemini(
        self,
        prompt: str,
        negative_prompt: Optional[str],
        width: int,
        height: int,
    ) -> Image.Image:
        """Generate image using Gemini API (generate_content with IMAGE modality)."""
        from google.genai import types

        if negative_prompt:
            prompt = f"{prompt}\n\nAvoid: {negative_prompt}"

        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=self._aspect_ratio(width, height),
                image_size=self._image_size(width, height),
            ),
        )

        logger.info("Calling Gemini image API", model=self._model)

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
                    return part.as_image()
                except Exception:
                    pass
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                data = inline.data
                image_bytes = base64.b64decode(data) if isinstance(data, str) else data
                return Image.open(BytesIO(image_bytes))

        logger.error("No image data in Gemini response", model=self._model)
        raise ValueError("Gemini image response did not contain image data.")
