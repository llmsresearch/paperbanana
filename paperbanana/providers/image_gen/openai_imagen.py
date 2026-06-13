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


def _is_gpt_image_2(model: str) -> bool:
    return model.lower() == "gpt-image-2"


#: gpt-image-2 total-pixel budget: exactly 4K UHD. 3840x2160 is accepted,
#: 3840x2176 is rejected (probed June 2026 on Azure).
GPT_IMAGE_2_PIXEL_BUDGET = 3840 * 2160


def legal_gpt_image_2_dims(width: int, height: int) -> tuple[int, int]:
    """Clamp arbitrary pixel dims to a size gpt-image-2 accepts.

    API constraints (discovered empirically, June 2026): width and height
    divisible by 16, longest edge <= 3840, aspect ratio <= 3:1, and total
    pixels <= 3840*2160 (the 4K-UHD budget).
    """
    import math

    w, h = float(max(width, 1)), float(max(height, 1))
    if w / h > 3.0:
        h = w / 3.0
    elif h / w > 3.0:
        w = h / 3.0
    scale = min(1.0, 3840.0 / max(w, h))
    if w * h * scale * scale > GPT_IMAGE_2_PIXEL_BUDGET:
        scale = (GPT_IMAGE_2_PIXEL_BUDGET / (w * h)) ** 0.5
    w *= scale
    h *= scale
    # Snap DOWN to /16 so the snap itself can never re-exceed the budget.
    wi = min(3840, max(256, math.floor(w / 16) * 16))
    hi = min(3840, max(256, math.floor(h / 16) * 16))
    # Flooring can push the ratio back over 3:1; grow the short side to fix
    # (only reachable at extreme aspects, far below the pixel budget).
    if wi / hi > 3.0:
        hi = math.ceil(wi / 3.0 / 16) * 16
    elif hi / wi > 3.0:
        wi = math.ceil(hi / 3.0 / 16) * 16
    return wi, hi


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

    @property
    def supported_ratios(self) -> list[str]:
        if _is_gpt_image_2(self._model):
            return ["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"]
        # Earlier GPT Image models only have 3 native sizes.
        return ["1:1", "3:2", "2:3"]

    def _size_string(self, width: int, height: int) -> str:
        """Map pixel dimensions to an OpenAI-supported size string."""
        if _is_gpt_image_2(self._model):
            w, h = legal_gpt_image_2_dims(width, height)
            return f"{w}x{h}"
        ratio = width / height
        if ratio > 1.2:
            return "1536x1024"
        if ratio < 0.83:
            return "1024x1536"
        return "1024x1024"

    # OpenAI only supports 1024x1024, 1536x1024, 1024x1536.
    # Map all aspect ratios to the closest supported size.
    _RATIO_TO_SIZE = {
        "21:9": "1536x1024",
        "16:9": "1536x1024",
        "4:3": "1536x1024",
        "3:2": "1536x1024",
        "1:1": "1024x1024",
        "2:3": "1024x1536",
        "3:4": "1024x1536",
        "9:16": "1024x1536",
    }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    async def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        aspect_ratio: Optional[str] = None,
        quality: Optional[str] = None,
        images: Optional[list[Image.Image]] = None,
    ) -> Image.Image:
        """Generate an image; with ``images`` set, performs a guided edit.

        Guided edits route to the OpenAI ``images.edit`` endpoint
        (supported by GPT-Image models on both OpenAI and Azure), making
        this provider usable for image-conditioned generation — e.g.
        poster figure re-authoring.
        """
        client = self._get_client()

        full_prompt = prompt
        if negative_prompt:
            full_prompt += f"\n\nAvoid: {negative_prompt}"

        if _is_gpt_image_2(self._model):
            size = self._size_string(width, height)
        else:
            size = self._RATIO_TO_SIZE.get(aspect_ratio, self._size_string(width, height))

        kwargs = {
            "model": self._model,
            "prompt": full_prompt,
            "n": 1,
            "size": size,
        }
        if quality:
            kwargs["quality"] = quality

        if images:
            files = []
            for i, img in enumerate(images):
                buf = BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)
                files.append((f"image_{i}.png", buf, "image/png"))
            result = await client.images.edit(image=files if len(files) > 1 else files[0], **kwargs)
        else:
            result = await client.images.generate(**kwargs)

        b64_data = result.data[0].b64_json
        image_bytes = base64.b64decode(b64_data)

        if self.cost_tracker is not None:
            self.cost_tracker.record_image_call(provider=self.name, model=self._model)
        return Image.open(BytesIO(image_bytes))
