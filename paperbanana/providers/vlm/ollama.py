"""Ollama VLM provider — local open-weight models via OpenAI-compatible API."""

from __future__ import annotations

from typing import Optional

import httpx
import structlog
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from paperbanana.core.utils import image_to_base64
from paperbanana.providers.base import VLMProvider

logger = structlog.get_logger()


class OllamaVLM(VLMProvider):
    """VLM provider for locally-hosted models via Ollama."""

    def __init__(
        self,
        model: str = "qwen2.5-vl",
        base_url: str = "http://localhost:11434/v1",
        json_mode: bool = False,
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._json_mode = json_mode
        self._client: httpx.AsyncClient | None = None

    # ── Provider identity ────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supports_json_mode(self) -> bool:
        return self._json_mode

    # ── HTTP client ──────────────────────────────────────────────────

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            # 300 s gives CPU-only machines a fair chance before we retry.
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=300.0,
            )
        return self._client

    async def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def is_available(self) -> bool:
        """Ollama needs no API key — just confirm the server is reachable."""
        try:
            # Strip /v1 to hit the Ollama root health endpoint rather than
            # the OpenAI-compat prefix, which isn't always a valid GET target.
            root_url = self._base_url
            if root_url.endswith("/v1"):
                root_url = root_url[:-3]
            resp = httpx.get(root_url, timeout=3.0)
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    # ── Generation ───────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=15))
    async def generate(
        self,
        prompt: str,
        images: Optional[list[Image.Image]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        response_format: Optional[str] = None,
    ) -> str:
        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Build the multimodal content block — same wire format as OpenAI.
        content: list[dict] = []
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

        payload: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format == "json" and self._json_mode:
            payload["response_format"] = {"type": "json_object"}

        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()

        data = response.json()
        text = data["choices"][0]["message"]["content"]

        # usage is a plain dict from the JSON body — .get() not getattr().
        usage = data.get("usage")
        logger.debug("Ollama response", model=self._model, usage=usage)

        if self.cost_tracker is not None and usage is not None:
            self.cost_tracker.record_vlm_call(
                provider=self.name,
                model=self._model,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            )

        return text
