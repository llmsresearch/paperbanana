"""Tests for AWS Bedrock VLM and image generation providers."""

from __future__ import annotations

import base64
import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from paperbanana.core.config import Settings
from paperbanana.providers.registry import ProviderRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_boto3_session(has_credentials: bool = True):
    """Return a mock boto3.Session class."""
    mock_session_cls = MagicMock()
    mock_session_instance = MagicMock()
    mock_session_cls.return_value = mock_session_instance

    if has_credentials:
        mock_creds = MagicMock()
        mock_session_instance.get_credentials.return_value = mock_creds
    else:
        mock_session_instance.get_credentials.return_value = None

    mock_client = MagicMock()
    mock_session_instance.client.return_value = mock_client
    return mock_session_cls, mock_session_instance, mock_client


def _make_small_image() -> Image.Image:
    return Image.new("RGB", (64, 64), color="red")


def _b64_png_image() -> str:
    """Return a small PNG image as base64."""
    img = _make_small_image()
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Registry creation tests
# ---------------------------------------------------------------------------


class TestRegistryCreation:
    def test_create_bedrock_vlm(self):
        mock_session_cls, _, _ = _mock_boto3_session()
        with patch.dict("sys.modules", {"boto3": MagicMock(Session=mock_session_cls)}):
            settings = Settings(
                vlm_provider="bedrock",
                vlm_model="us.amazon.nova-pro-v1:0",
            )
            vlm = ProviderRegistry.create_vlm(settings)
            assert vlm.name == "bedrock"
            assert vlm.model_name == "us.amazon.nova-pro-v1:0"

    def test_create_bedrock_imagen(self):
        mock_session_cls, _, _ = _mock_boto3_session()
        with patch.dict("sys.modules", {"boto3": MagicMock(Session=mock_session_cls)}):
            settings = Settings(
                image_provider="bedrock_imagen",
                image_model="amazon.nova-canvas-v1:0",
            )
            gen = ProviderRegistry.create_image_gen(settings)
            assert gen.name == "bedrock_imagen"
            assert gen.model_name == "amazon.nova-canvas-v1:0"

    def test_bedrock_vlm_model_override(self):
        mock_session_cls, _, _ = _mock_boto3_session()
        with patch.dict("sys.modules", {"boto3": MagicMock(Session=mock_session_cls)}):
            settings = Settings(
                vlm_provider="bedrock",
                vlm_model="gemini-2.0-flash",
                bedrock_vlm_model="anthropic.claude-sonnet-4-20250514-v1:0",
            )
            vlm = ProviderRegistry.create_vlm(settings)
            assert vlm.model_name == "anthropic.claude-sonnet-4-20250514-v1:0"

    def test_bedrock_image_model_override(self):
        mock_session_cls, _, _ = _mock_boto3_session()
        with patch.dict("sys.modules", {"boto3": MagicMock(Session=mock_session_cls)}):
            settings = Settings(
                image_provider="bedrock_imagen",
                image_model="gemini-3-pro-image-preview",
                bedrock_image_model="amazon.nova-canvas-v1:0",
            )
            gen = ProviderRegistry.create_image_gen(settings)
            assert gen.model_name == "amazon.nova-canvas-v1:0"


# ---------------------------------------------------------------------------
# Missing credentials tests
# ---------------------------------------------------------------------------


class TestMissingCredentials:
    def test_missing_aws_credentials_vlm(self):
        mock_session_cls, _, _ = _mock_boto3_session(has_credentials=False)
        with patch.dict("sys.modules", {"boto3": MagicMock(Session=mock_session_cls)}):
            settings = Settings(vlm_provider="bedrock")
            with pytest.raises(ValueError, match="AWS credentials not found"):
                ProviderRegistry.create_vlm(settings)

    def test_missing_aws_credentials_imagen(self):
        mock_session_cls, _, _ = _mock_boto3_session(has_credentials=False)
        with patch.dict("sys.modules", {"boto3": MagicMock(Session=mock_session_cls)}):
            settings = Settings(image_provider="bedrock_imagen")
            with pytest.raises(ValueError, match="AWS credentials not found"):
                ProviderRegistry.create_image_gen(settings)

    def test_missing_boto3_raises_import_error(self):
        with patch.dict("sys.modules", {"boto3": None}):
            settings = Settings(vlm_provider="bedrock")
            with pytest.raises(ImportError, match="boto3 is required"):
                ProviderRegistry.create_vlm(settings)

    def test_missing_boto3_imagen_raises_import_error(self):
        with patch.dict("sys.modules", {"boto3": None}):
            settings = Settings(image_provider="bedrock_imagen")
            with pytest.raises(ImportError, match="boto3 is required"):
                ProviderRegistry.create_image_gen(settings)


# ---------------------------------------------------------------------------
# is_available tests
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def test_vlm_available_with_credentials(self):
        mock_session_cls, _, _ = _mock_boto3_session(has_credentials=True)
        with patch.dict("sys.modules", {"boto3": MagicMock(Session=mock_session_cls)}):
            from paperbanana.providers.vlm.bedrock import BedrockVLM

            vlm = BedrockVLM()
            assert vlm.is_available() is True

    def test_vlm_unavailable_without_credentials(self):
        mock_session_cls, _, _ = _mock_boto3_session(has_credentials=False)
        with patch.dict("sys.modules", {"boto3": MagicMock(Session=mock_session_cls)}):
            from paperbanana.providers.vlm.bedrock import BedrockVLM

            vlm = BedrockVLM()
            assert vlm.is_available() is False

    def test_imagen_available_with_credentials(self):
        mock_session_cls, _, _ = _mock_boto3_session(has_credentials=True)
        with patch.dict("sys.modules", {"boto3": MagicMock(Session=mock_session_cls)}):
            from paperbanana.providers.image_gen.bedrock_imagen import BedrockImageGen

            gen = BedrockImageGen()
            assert gen.is_available() is True

    def test_imagen_unavailable_without_credentials(self):
        mock_session_cls, _, _ = _mock_boto3_session(has_credentials=False)
        with patch.dict("sys.modules", {"boto3": MagicMock(Session=mock_session_cls)}):
            from paperbanana.providers.image_gen.bedrock_imagen import BedrockImageGen

            gen = BedrockImageGen()
            assert gen.is_available() is False

    def test_vlm_unavailable_without_boto3(self):
        with patch.dict("sys.modules", {"boto3": None}):
            from paperbanana.providers.vlm.bedrock import BedrockVLM

            vlm = BedrockVLM()
            assert vlm.is_available() is False

    def test_imagen_unavailable_without_boto3(self):
        with patch.dict("sys.modules", {"boto3": None}):
            from paperbanana.providers.image_gen.bedrock_imagen import BedrockImageGen

            gen = BedrockImageGen()
            assert gen.is_available() is False


# ---------------------------------------------------------------------------
# VLM generate tests
# ---------------------------------------------------------------------------


class TestBedrockVLMGenerate:
    async def test_generate_text_only(self):
        mock_session_cls, _, mock_client = _mock_boto3_session()
        mock_client.converse.return_value = {
            "output": {
                "message": {
                    "content": [{"text": "Hello from Bedrock"}],
                }
            },
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }

        with patch.dict("sys.modules", {"boto3": MagicMock(Session=mock_session_cls)}):
            from paperbanana.providers.vlm.bedrock import BedrockVLM

            vlm = BedrockVLM(model="us.amazon.nova-pro-v1:0")
            result = await vlm.generate("Hello")

        assert result == "Hello from Bedrock"
        call_kwargs = mock_client.converse.call_args[1]
        assert call_kwargs["modelId"] == "us.amazon.nova-pro-v1:0"
        assert call_kwargs["messages"][0]["content"][-1] == {"text": "Hello"}
        assert "system" not in call_kwargs

    async def test_generate_with_system_prompt(self):
        mock_session_cls, _, mock_client = _mock_boto3_session()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "OK"}]}},
        }

        with patch.dict("sys.modules", {"boto3": MagicMock(Session=mock_session_cls)}):
            from paperbanana.providers.vlm.bedrock import BedrockVLM

            vlm = BedrockVLM()
            await vlm.generate("Hello", system_prompt="Be helpful")

        call_kwargs = mock_client.converse.call_args[1]
        assert call_kwargs["system"] == [{"text": "Be helpful"}]

    async def test_generate_with_images(self):
        mock_session_cls, _, mock_client = _mock_boto3_session()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "I see an image"}]}},
        }

        img = _make_small_image()
        with patch.dict("sys.modules", {"boto3": MagicMock(Session=mock_session_cls)}):
            from paperbanana.providers.vlm.bedrock import BedrockVLM

            vlm = BedrockVLM()
            result = await vlm.generate("Describe this", images=[img])

        assert result == "I see an image"
        call_kwargs = mock_client.converse.call_args[1]
        content = call_kwargs["messages"][0]["content"]
        # First content block should be the image
        assert "image" in content[0]
        assert content[0]["image"]["format"] == "png"
        assert isinstance(content[0]["image"]["source"]["bytes"], bytes)
        # Last content block should be the text
        assert content[-1] == {"text": "Describe this"}

    async def test_generate_passes_temperature_and_max_tokens(self):
        mock_session_cls, _, mock_client = _mock_boto3_session()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "OK"}]}},
        }

        with patch.dict("sys.modules", {"boto3": MagicMock(Session=mock_session_cls)}):
            from paperbanana.providers.vlm.bedrock import BedrockVLM

            vlm = BedrockVLM()
            await vlm.generate("Hello", temperature=0.5, max_tokens=2048)

        call_kwargs = mock_client.converse.call_args[1]
        assert call_kwargs["inferenceConfig"]["temperature"] == 0.5
        assert call_kwargs["inferenceConfig"]["maxTokens"] == 2048


# ---------------------------------------------------------------------------
# ImageGen generate tests
# ---------------------------------------------------------------------------


class TestBedrockImageGenGenerate:
    async def test_generate_basic(self):
        mock_session_cls, _, mock_client = _mock_boto3_session()

        b64_img = _b64_png_image()
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps({"images": [b64_img]}).encode()
        mock_client.invoke_model.return_value = {"body": mock_body}

        with patch.dict("sys.modules", {"boto3": MagicMock(Session=mock_session_cls)}):
            from paperbanana.providers.image_gen.bedrock_imagen import BedrockImageGen

            gen = BedrockImageGen(model="amazon.nova-canvas-v1:0")
            result = await gen.generate("A methodology diagram")

        assert isinstance(result, Image.Image)
        call_kwargs = mock_client.invoke_model.call_args[1]
        assert call_kwargs["modelId"] == "amazon.nova-canvas-v1:0"
        body = json.loads(call_kwargs["body"])
        assert body["taskType"] == "TEXT_IMAGE"
        assert body["textToImageParams"]["text"] == "A methodology diagram"

    async def test_generate_with_negative_prompt(self):
        mock_session_cls, _, mock_client = _mock_boto3_session()

        b64_img = _b64_png_image()
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps({"images": [b64_img]}).encode()
        mock_client.invoke_model.return_value = {"body": mock_body}

        with patch.dict("sys.modules", {"boto3": MagicMock(Session=mock_session_cls)}):
            from paperbanana.providers.image_gen.bedrock_imagen import BedrockImageGen

            gen = BedrockImageGen()
            await gen.generate("A diagram", negative_prompt="blurry, low quality")

        call_kwargs = mock_client.invoke_model.call_args[1]
        body = json.loads(call_kwargs["body"])
        assert body["textToImageParams"]["negativeText"] == "blurry, low quality"

    async def test_generate_with_aspect_ratio(self):
        mock_session_cls, _, mock_client = _mock_boto3_session()

        b64_img = _b64_png_image()
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps({"images": [b64_img]}).encode()
        mock_client.invoke_model.return_value = {"body": mock_body}

        with patch.dict("sys.modules", {"boto3": MagicMock(Session=mock_session_cls)}):
            from paperbanana.providers.image_gen.bedrock_imagen import BedrockImageGen

            gen = BedrockImageGen()
            await gen.generate("A diagram", aspect_ratio="16:9")

        call_kwargs = mock_client.invoke_model.call_args[1]
        body = json.loads(call_kwargs["body"])
        assert body["imageGenerationConfig"]["width"] == 1280
        assert body["imageGenerationConfig"]["height"] == 720

    async def test_generate_with_seed(self):
        mock_session_cls, _, mock_client = _mock_boto3_session()

        b64_img = _b64_png_image()
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps({"images": [b64_img]}).encode()
        mock_client.invoke_model.return_value = {"body": mock_body}

        with patch.dict("sys.modules", {"boto3": MagicMock(Session=mock_session_cls)}):
            from paperbanana.providers.image_gen.bedrock_imagen import BedrockImageGen

            gen = BedrockImageGen()
            await gen.generate("A diagram", seed=42)

        call_kwargs = mock_client.invoke_model.call_args[1]
        body = json.loads(call_kwargs["body"])
        assert body["imageGenerationConfig"]["seed"] == 42


# ---------------------------------------------------------------------------
# Dimension resolution tests
# ---------------------------------------------------------------------------


class TestDimensionResolution:
    def test_known_ratio(self):
        from paperbanana.providers.image_gen.bedrock_imagen import BedrockImageGen

        gen = BedrockImageGen()
        assert gen._resolve_dimensions(1024, 1024, "16:9") == (1280, 720)
        assert gen._resolve_dimensions(1024, 1024, "1:1") == (1024, 1024)
        assert gen._resolve_dimensions(1024, 1024, "9:16") == (720, 1280)

    def test_snap_to_closest(self):
        from paperbanana.providers.image_gen.bedrock_imagen import BedrockImageGen

        gen = BedrockImageGen()
        # 1920x1080 is 16:9 ratio → should snap to (1280, 720)
        assert gen._resolve_dimensions(1920, 1080) == (1280, 720)
        # Square-ish → snaps to 1:1
        assert gen._resolve_dimensions(1000, 1000) == (1024, 1024)
        # Portrait → snaps to closest portrait ratio
        w, h = gen._resolve_dimensions(768, 1024)
        assert h > w  # should be portrait

    def test_unknown_ratio_falls_back(self):
        from paperbanana.providers.image_gen.bedrock_imagen import BedrockImageGen

        gen = BedrockImageGen()
        # Unknown string ratio falls back to dimension snapping
        w, h = gen._resolve_dimensions(1024, 1024, "5:7")
        assert (w, h) == (1024, 1024)  # closest is 1:1

    def test_supported_ratios_matches_dimension_map(self):
        from paperbanana.providers.image_gen.bedrock_imagen import BedrockImageGen

        gen = BedrockImageGen()
        assert gen.supported_ratios == list(gen._RATIO_TO_DIMENSIONS.keys())


# ---------------------------------------------------------------------------
# Error message tests
# ---------------------------------------------------------------------------


class TestErrorMessages:
    def test_unknown_vlm_provider_lists_bedrock(self):
        settings = Settings(vlm_provider="nonexistent")
        with pytest.raises(ValueError, match="bedrock"):
            ProviderRegistry.create_vlm(settings)

    def test_unknown_image_provider_lists_bedrock_imagen(self):
        settings = Settings(image_provider="nonexistent")
        with pytest.raises(ValueError, match="bedrock_imagen"):
            ProviderRegistry.create_image_gen(settings)


# ---------------------------------------------------------------------------
# Config effective model tests
# ---------------------------------------------------------------------------


class TestEffectiveModels:
    def test_effective_vlm_model_bedrock(self):
        settings = Settings(
            vlm_provider="bedrock",
            vlm_model="gemini-2.0-flash",
            bedrock_vlm_model="us.amazon.nova-pro-v1:0",
        )
        assert settings.effective_vlm_model == "us.amazon.nova-pro-v1:0"

    def test_effective_vlm_model_bedrock_no_override(self):
        settings = Settings(vlm_provider="bedrock", vlm_model="gemini-2.0-flash")
        assert settings.effective_vlm_model == "gemini-2.0-flash"

    def test_effective_image_model_bedrock(self):
        settings = Settings(
            image_provider="bedrock_imagen",
            image_model="gemini-3-pro-image-preview",
            bedrock_image_model="amazon.nova-canvas-v1:0",
        )
        assert settings.effective_image_model == "amazon.nova-canvas-v1:0"

    def test_effective_image_model_bedrock_no_override(self):
        settings = Settings(
            image_provider="bedrock_imagen", image_model="gemini-3-pro-image-preview"
        )
        assert settings.effective_image_model == "gemini-3-pro-image-preview"
