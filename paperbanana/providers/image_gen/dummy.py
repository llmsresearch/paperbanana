"""Dummy image generation provider for when no image generation is needed."""

from __future__ import annotations

from typing import Optional

from PIL import Image

from paperbanana.providers.base import ImageGenProvider


class DummyImageGen(ImageGenProvider):
    """Dummy image generation provider that doesn't actually generate images.

    Used for statistical plots where the visualization is done via matplotlib
    code generation rather than image models.
    """

    @property
    def name(self) -> str:
        return "none"

    @property
    def model_name(self) -> str:
        return "dummy"

    async def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        aspect_ratio: Optional[str] = None,
    ) -> Image.Image:
        raise RuntimeError(
            "DummyImageGen should never be called. "
            "This provider is only for statistical plot generation."
        )