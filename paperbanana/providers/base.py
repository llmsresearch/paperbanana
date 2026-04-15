"""Abstract base classes for all providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from PIL import Image

if TYPE_CHECKING:
    from paperbanana.core.cost_tracker import CostTracker


class VLMProvider(ABC):
    """Abstract interface for Vision-Language Model providers.

    All VLM providers (used by Retriever, Planner, Stylist, Critic agents)
    must implement this interface.
    """

    cost_tracker: CostTracker | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging and config."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier being used."""
        ...

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        images: Optional[list[Image.Image]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        response_format: Optional[str] = None,
    ) -> str:
        """Generate text from a prompt, optionally with images.

        Args:
            prompt: The user prompt text.
            images: Optional list of images for vision tasks.
            system_prompt: Optional system-level instructions.
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens in the response.
            response_format: Optional format hint ("json" for JSON mode).

        Returns:
            Generated text response.
        """
        ...

    @property
    def supports_json_mode(self) -> bool:
        """Whether this provider reliably handles response_format='json'."""
        return True

    def is_available(self) -> bool:
        """Check if this provider is configured and available."""
        return True


class ImageGenProvider(ABC):
    """Abstract interface for image generation providers.

    Used by the Visualizer agent to generate methodology diagrams
    and other academic illustrations.
    """

    cost_tracker: CostTracker | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging and config."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier being used."""
        ...

    @property
    def supported_ratios(self) -> list[str]:
        """Aspect ratios this provider supports. Override in subclasses."""
        return ["1:1", "16:9"]  # conservative default

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        aspect_ratio: Optional[str] = None,
    ) -> Image.Image:
        """Generate an image from a text prompt.

        Args:
            prompt: Text description of the desired image.
            negative_prompt: What to avoid in the image.
            width: Output image width in pixels.
            height: Output image height in pixels.
            seed: Random seed for reproducibility.
            aspect_ratio: Target aspect ratio (1:1, 2:3, 3:2, 3:4, 4:3, 9:16, 16:9, 21:9).
                takes precedence over width/height for providers that support it.

        Returns:
            Generated PIL Image.
        """
        ...

    def is_available(self) -> bool:
        """Check if this provider is configured and available."""
        return True
