"""Atlas Cloud VLM provider built on the OpenAI-compatible chat API."""

from __future__ import annotations

from typing import Optional

from paperbanana.providers.vlm.openai import OpenAIVLM


class AtlasVLM(OpenAIVLM):
    """Atlas Cloud chat provider via the OpenAI SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-ai/DeepSeek-V3-0324",
        base_url: str = "https://api.atlascloud.ai/v1",
        json_mode: bool = True,
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            json_mode=json_mode,
            provider_name="atlas",
        )
