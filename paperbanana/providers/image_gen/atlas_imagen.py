"""Atlas Cloud image generation provider."""

from __future__ import annotations

import asyncio
from io import BytesIO
from typing import Any, Optional

import structlog
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from paperbanana.providers.base import ImageGenProvider

logger = structlog.get_logger()


class AtlasImageGen(ImageGenProvider):
    """Image generation via Atlas Cloud's async prediction API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-image-2/text-to-image",
        base_url: str = "https://api.atlascloud.ai/api/v1",
        poll_interval_seconds: float = 2.0,
        max_poll_attempts: int = 60,
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._poll_interval_seconds = poll_interval_seconds
        self._max_poll_attempts = max_poll_attempts
        self._client = None

    @property
    def name(self) -> str:
        return "atlas_imagen"

    @property
    def model_name(self) -> str:
        return self._model

    def _get_client(self):
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=180.0,
            )
        return self._client

    def is_available(self) -> bool:
        return self._api_key is not None

    @property
    def supported_ratios(self) -> list[str]:
        return ["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"]

    def _aspect_ratio_hint(self, width: int, height: int) -> str:
        ratio = width / height
        if ratio > 1.8:
            return "ultra-wide landscape format (21:9)"
        if ratio > 1.4:
            return "wide landscape format (16:9)"
        if ratio > 1.1:
            return "landscape format (3:2)"
        if ratio < 0.55:
            return "tall portrait format (9:16)"
        if ratio < 0.8:
            return "portrait format (2:3)"
        return "square format (1:1)"

    def _size_string(self, width: int, height: int, aspect_ratio: Optional[str] = None) -> str:
        ratio = width / height
        if aspect_ratio and ":" in aspect_ratio:
            try:
                w_part, h_part = aspect_ratio.split(":", 1)
                ratio = float(w_part) / float(h_part)
            except (ValueError, ZeroDivisionError):
                ratio = width / height
        if 0.85 <= ratio <= 1.15:
            return "1024x1024"
        if ratio > 1.0:
            return "1536x1024"
        return "1024x1536"

    #: Aspect ratios accepted by the Google async image models (nano-banana / imagen).
    _NATIVE_RATIOS = {
        "1:1": 1.0, "5:4": 1.25, "4:3": 4 / 3, "3:2": 1.5, "16:9": 16 / 9, "21:9": 21 / 9,
        "4:5": 0.8, "3:4": 0.75, "2:3": 2 / 3, "9:16": 9 / 16,
    }

    def _uses_native_aspect(self) -> bool:
        """Google's async image models (nano-banana, imagen) take aspect_ratio +
        resolution instead of gpt-image-2's fixed `size` strings, and support up
        to 4k — so a poster base can be rendered at full print detail."""
        m = self._model.lower()
        return any(tag in m for tag in ("nano-banana", "imagen", "/gemini"))

    def _nearest_native_ratio(self, width: int, height: int) -> str:
        target = width / max(height, 1)
        return min(self._NATIVE_RATIOS, key=lambda r: abs(self._NATIVE_RATIOS[r] - target))

    def _build_prompt(
        self,
        prompt: str,
        negative_prompt: Optional[str],
        width: int,
        height: int,
        aspect_ratio: Optional[str],
    ) -> str:
        aspect_hint = (
            f"{aspect_ratio} format" if aspect_ratio else self._aspect_ratio_hint(width, height)
        )
        parts = [prompt, f"Generate this as a {aspect_hint} image."]
        if negative_prompt:
            parts.append(f"Avoid: {negative_prompt}")
        return "\n\n".join(parts)

    def _extract_prediction_id(self, payload: dict[str, Any]) -> str:
        data = payload.get("data", {})
        prediction_id = data.get("id") or payload.get("id")
        if not prediction_id:
            raise ValueError(f"Atlas image generation did not return a prediction id: {payload}")
        return str(prediction_id)

    def _extract_output_url(self, payload: dict[str, Any]) -> str | None:
        data = payload.get("data", {})
        outputs = data.get("outputs") or payload.get("outputs") or []
        if not outputs:
            return None
        first = outputs[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            return first.get("url") or first.get("image_url") or first.get("output")
        return None

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
    ) -> Image.Image:
        client = self._get_client()

        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": self._build_prompt(prompt, negative_prompt, width, height, aspect_ratio),
            "enable_base64_output": False,
            "enable_sync_mode": False,
        }
        if self._uses_native_aspect():
            # nano-banana / imagen: native aspect_ratio + up-to-4k resolution, so
            # the poster base is print-detailed (not capped at gpt-image-2's 1536px).
            payload["aspect_ratio"] = aspect_ratio or self._nearest_native_ratio(width, height)
            payload["resolution"] = "4k"
            payload["output_format"] = "png"
            payload["media_resolution"] = "high"
        else:
            payload["size"] = self._size_string(width, height, aspect_ratio)
            if quality:
                payload["quality"] = quality
        if seed is not None:
            payload["seed"] = seed

        response = await client.post("/model/generateImage", json=payload)
        response.raise_for_status()
        prediction_id = self._extract_prediction_id(response.json())

        output_url = None
        for _ in range(self._max_poll_attempts):
            poll_response = await client.get(f"/model/prediction/{prediction_id}")
            poll_response.raise_for_status()
            poll_payload = poll_response.json()
            data = poll_payload.get("data", {})
            status = str(data.get("status", "")).lower()

            if status == "completed":
                output_url = self._extract_output_url(poll_payload)
                if output_url:
                    break
                raise ValueError(
                    "Atlas image prediction "
                    f"{prediction_id} completed without an output URL: {poll_payload}"
                )
            if status == "failed":
                raise ValueError(
                    f"Atlas image prediction {prediction_id} failed: "
                    f"{data.get('error') or poll_payload}"
                )

            await asyncio.sleep(self._poll_interval_seconds)

        if not output_url:
            raise TimeoutError(
                f"Timed out waiting for Atlas image prediction {prediction_id} after "
                f"{self._max_poll_attempts} polls."
            )

        image_response = await client.get(output_url)
        image_response.raise_for_status()

        if self.cost_tracker is not None:
            self.cost_tracker.record_image_call(provider=self.name, model=self._model)
        return Image.open(BytesIO(image_response.content))
