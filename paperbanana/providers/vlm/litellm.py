"""LiteLLM VLM provider — unified access to 100+ LLM providers."""

from __future__ import annotations

from typing import Optional

import structlog
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from paperbanana.core.utils import image_to_base64
from paperbanana.providers.base import VLMProvider

logger = structlog.get_logger()


class LiteLLMVLM(VLMProvider):
    """VLM provider using the LiteLLM SDK (async).

    Supports any LiteLLM model string, e.g. ``openai/gpt-4o``,
    ``anthropic/claude-sonnet-4-6``, ``groq/llama-3.3-70b-versatile``.
    Provider API keys are read from environment variables automatically.
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        self._model = model
        self._api_key = api_key
        self._api_base = api_base

    @property
    def name(self) -> str:
        return "litellm"

    @property
    def model_name(self) -> str:
        return self._model

    def is_available(self) -> bool:
        try:
            import litellm  # noqa: F401

            return True
        except ImportError:
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    async def generate(
        self,
        prompt: str,
        images: Optional[list[Image.Image]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        response_format: Optional[str] = None,
    ) -> str:
        import litellm

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        content = []
        if images:
            for img in images:
                b64 = image_to_base64(img)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    }
                )
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})

        kwargs = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "drop_params": True,
        }

        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base

        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        response = await litellm.acompletion(**kwargs)
        text = response.choices[0].message.content

        usage = getattr(response, "usage", None)
        logger.debug("LiteLLM response", model=self._model, usage=usage)

        if self.cost_tracker is not None and usage is not None:
            self.cost_tracker.record_vlm_call(
                provider=self.name,
                model=self._model,
                input_tokens=getattr(usage, "prompt_tokens", 0),
                output_tokens=getattr(usage, "completion_tokens", 0),
            )
        return text
