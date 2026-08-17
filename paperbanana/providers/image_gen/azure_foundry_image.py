"""Microsoft Foundry image generation through an OpenAI-compatible deployment."""

from __future__ import annotations

from typing import Optional

from paperbanana.providers.image_gen.openai_imagen import OpenAIImageGen


class AzureFoundryImageGen(OpenAIImageGen):
    """Image generation and guided edits through Microsoft Foundry."""

    def __init__(
        self,
        api_key: Optional[str],
        deployment: str,
        base_url: str,
    ):
        super().__init__(api_key=api_key, model=deployment, base_url=base_url)

    @property
    def name(self) -> str:
        return "azure_foundry_image"
