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

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supports_json_mode(self) -> bool:
        return self._json_mode

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._base_url, timeout=300.0)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def is_available(self) -> bool:
        try:
            root = self._base_url[:-3] if self._base_url.endswith("/v1") else self._base_url
            return httpx.get(root, timeout=3.0).status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

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
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        content: list[dict] = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_to_base64(img)}"},
            }
            for img in (images or [])
        ]
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
        resp = await self._get_client().post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        logger.debug("Ollama response", model=self._model, usage=data.get("usage"))
        return data["choices"][0]["message"]["content"]
