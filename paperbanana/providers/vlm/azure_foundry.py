"""Microsoft Foundry VLM provider using an OpenAI-compatible deployment."""

from __future__ import annotations

from typing import Optional

from paperbanana.providers.vlm.openai import OpenAIVLM


class AzureFoundryVLM(OpenAIVLM):
    """Chat and vision through a Microsoft Foundry model deployment."""

    def __init__(
        self,
        api_key: Optional[str],
        deployment: str,
        base_url: str,
        json_mode: bool = True,
    ):
        super().__init__(
            api_key=api_key,
            model=deployment,
            base_url=base_url,
            json_mode=json_mode,
            provider_name="azure_foundry",
        )
