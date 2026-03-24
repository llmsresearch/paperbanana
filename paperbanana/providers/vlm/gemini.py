"""Google Gemini VLM provider (FREE tier)."""

from __future__ import annotations

import re
from typing import Optional

import structlog
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from paperbanana.core.utils import image_to_base64
from paperbanana.providers.base import VLMProvider

logger = structlog.get_logger()

# Gemini 2.5+ models use "thinking" tokens counted within max_output_tokens.
_THINKING_MODEL_RE = re.compile(r"gemini-(?:[3-9]|[1-9]\d|2\.[5-9]|2\.\d{2,})")
_DEFAULT_THINKING_BUDGET = 8192


class GeminiVLM(VLMProvider):
    """Google Gemini VLM using the google-genai SDK.

    Free tier: https://makersuite.google.com/app/apikey
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
        base_url: Optional[str] = None,
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._client = None

    @property
    def name(self) -> str:
        return "gemini"

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
                    "google-genai is required for Gemini provider. "
                    "Install with: pip install 'paperbanana[google]'"
                )
        return self._client

    def _is_thinking_model(self) -> bool:
        """Return True if the model uses thinking tokens (Gemini 2.5+)."""
        return bool(_THINKING_MODEL_RE.search(self._model.lower()))

    def is_available(self) -> bool:
        return self._api_key is not None

    @retry(stop=stop_after_attempt(8), wait=wait_exponential(min=2, max=120))
    async def generate(
        self,
        prompt: str,
        images: Optional[list[Image.Image]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        response_format: Optional[str] = None,
    ) -> str:
        from google.genai import types

        client = self._get_client()

        contents = []
        if images:
            for img in images:
                b64 = image_to_base64(img)
                contents.append(
                    types.Part.from_bytes(
                        data=__import__("base64").b64decode(b64),
                        mime_type="image/png",
                    )
                )
        contents.append(prompt)

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if system_prompt:
            config.system_instruction = system_prompt
        if response_format == "json":
            config.response_mime_type = "application/json"

        # Thinking models (Gemini 2.5+): thinking tokens share the
        # max_output_tokens budget, starving actual content.  Cap thinking
        # and scale the total so callers get the full max_tokens for content.
        if self._is_thinking_model():
            thinking_budget = _DEFAULT_THINKING_BUDGET
            config.thinking_config = types.ThinkingConfig(
                thinking_budget=thinking_budget,
            )
            config.max_output_tokens = max_tokens + thinking_budget
            logger.debug(
                "Thinking model detected, adjusted token budget",
                model=self._model,
                thinking_budget=thinking_budget,
                max_output_tokens=config.max_output_tokens,
            )
        response = client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )

        usage = getattr(response, "usage_metadata", None)
        logger.debug("Gemini response", model=self._model, usage=usage)

        if self.cost_tracker is not None and usage is not None:
            self.cost_tracker.record_vlm_call(
                provider=self.name,
                model=self._model,
                input_tokens=getattr(usage, "prompt_token_count", 0),
                output_tokens=getattr(usage, "candidates_token_count", 0),
            )
        return response.text
