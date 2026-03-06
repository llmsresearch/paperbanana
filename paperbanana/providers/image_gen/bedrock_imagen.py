"""AWS Bedrock image generation provider using Nova Canvas."""

from __future__ import annotations

import asyncio
import base64
import json
from io import BytesIO
from typing import Optional

import structlog
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from paperbanana.providers.base import ImageGenProvider

logger = structlog.get_logger()


class BedrockImageGen(ImageGenProvider):
    """Image generation using AWS Bedrock Nova Canvas.

    Authenticates via the standard boto3 credential chain
    (env vars, ~/.aws/credentials, IAM role).
    """

    # Nova Canvas requires explicit pixel dimensions for each ratio.
    _RATIO_TO_DIMENSIONS: dict[str, tuple[int, int]] = {
        "1:1": (1024, 1024),
        "2:3": (768, 1152),
        "3:2": (1152, 768),
        "3:4": (768, 1024),
        "4:3": (1024, 768),
        "9:16": (720, 1280),
        "16:9": (1280, 720),
    }

    def __init__(
        self,
        model: str = "amazon.nova-canvas-v1:0",
        region: str = "us-east-1",
        profile: Optional[str] = None,
    ):
        self._model = model
        self._region = region
        self._profile = profile
        self._client = None

    @property
    def name(self) -> str:
        return "bedrock_imagen"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supported_ratios(self) -> list[str]:
        return list(self._RATIO_TO_DIMENSIONS.keys())

    def _get_client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError:
                raise ImportError(
                    "boto3 is required for the Bedrock provider. "
                    "Install with: pip install 'paperbanana[bedrock]'"
                )
            session = boto3.Session(
                region_name=self._region,
                profile_name=self._profile,
            )
            self._client = session.client("bedrock-runtime")
        return self._client

    def is_available(self) -> bool:
        try:
            import boto3
        except ImportError:
            return False
        session = boto3.Session(
            region_name=self._region,
            profile_name=self._profile,
        )
        credentials = session.get_credentials()
        return credentials is not None

    def _resolve_dimensions(
        self, width: int, height: int, aspect_ratio: Optional[str] = None
    ) -> tuple[int, int]:
        """Return pixel dimensions for the closest supported ratio."""
        if aspect_ratio and aspect_ratio in self._RATIO_TO_DIMENSIONS:
            return self._RATIO_TO_DIMENSIONS[aspect_ratio]

        # Snap width/height to the closest supported ratio.
        ratio = width / height
        best_key = "1:1"
        best_diff = float("inf")
        for key, (w, h) in self._RATIO_TO_DIMENSIONS.items():
            diff = abs(ratio - w / h)
            if diff < best_diff:
                best_diff = diff
                best_key = key
        return self._RATIO_TO_DIMENSIONS[best_key]

    @staticmethod
    def _build_nova_canvas_payload(
        prompt: str,
        width: int,
        height: int,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> dict:
        """Build the invoke_model JSON body for Nova Canvas TEXT_IMAGE task."""
        params: dict = {
            "text": prompt,
        }
        if negative_prompt:
            params["negativeText"] = negative_prompt

        image_config: dict = {
            "width": width,
            "height": height,
            "numberOfImages": 1,
        }
        if seed is not None:
            image_config["seed"] = seed

        return {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": params,
            "imageGenerationConfig": image_config,
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    async def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        aspect_ratio: Optional[str] = None,
    ) -> Image.Image:
        client = self._get_client()

        w, h = self._resolve_dimensions(width, height, aspect_ratio)
        body = self._build_nova_canvas_payload(
            prompt=prompt,
            width=w,
            height=h,
            negative_prompt=negative_prompt,
            seed=seed,
        )

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.invoke_model(
                modelId=self._model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            ),
        )

        result = json.loads(response["body"].read())
        b64_data = result["images"][0]
        image_bytes = base64.b64decode(b64_data)
        return Image.open(BytesIO(image_bytes))
