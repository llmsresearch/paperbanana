"""Tests for the LiteLLM VLM provider."""

from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from paperbanana.providers.vlm.litellm import LiteLLMVLM


def _install_litellm_stub(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Install a fake litellm module and return the mock acompletion."""
    fake = types.ModuleType("litellm")
    fake.acompletion = AsyncMock(name="litellm.acompletion")

    fake.RateLimitError = type("RateLimitError", (Exception,), {})
    fake.AuthenticationError = type("AuthenticationError", (Exception,), {})
    fake.NotFoundError = type("NotFoundError", (Exception,), {})
    fake.APIConnectionError = type("APIConnectionError", (Exception,), {})
    fake.Timeout = type("Timeout", (Exception,), {})
    fake.APIError = type("APIError", (Exception,), {})

    monkeypatch.setitem(sys.modules, "litellm", fake)
    return fake.acompletion


def _mock_response(content: str = "Hello", prompt_tokens: int = 10, completion_tokens: int = 5):
    usage = types.SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    message = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=message)
    return types.SimpleNamespace(choices=[choice], usage=usage)


# ── Init + config ──────────────────────────────────────────────


class TestLiteLLMVLMInit:
    def test_default_model(self) -> None:
        vlm = LiteLLMVLM()
        assert vlm.model_name == "openai/gpt-4o-mini"
        assert vlm.name == "litellm"

    def test_custom_model(self) -> None:
        vlm = LiteLLMVLM(model="anthropic/claude-sonnet-4-6")
        assert vlm.model_name == "anthropic/claude-sonnet-4-6"

    def test_is_available_with_litellm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_litellm_stub(monkeypatch)
        vlm = LiteLLMVLM()
        assert vlm.is_available() is True

    def test_is_available_without_litellm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delitem(sys.modules, "litellm", raising=False)
        monkeypatch.setattr(
            "builtins.__import__",
            _make_import_raiser("litellm", monkeypatch),
        )
        vlm = LiteLLMVLM()
        assert vlm.is_available() is False


def _make_import_raiser(blocked_name: str, monkeypatch: pytest.MonkeyPatch):
    """Return an __import__ replacement that raises ImportError for a specific module."""
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _raiser(name, *args, **kwargs):
        if name == blocked_name:
            raise ImportError(f"No module named '{blocked_name}'")
        return real_import(name, *args, **kwargs)

    return _raiser


# ── Generate (text-only) ──────────────────────────────────────


class TestLiteLLMVLMGenerate:
    @pytest.mark.asyncio
    async def test_generate_text_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_acompletion = _install_litellm_stub(monkeypatch)
        mock_acompletion.return_value = _mock_response("The answer is 4.")

        vlm = LiteLLMVLM(model="openai/gpt-4o")
        text = await vlm.generate("What is 2+2?")

        assert text == "The answer is 4."
        call_kwargs = mock_acompletion.call_args.kwargs
        assert call_kwargs["model"] == "openai/gpt-4o"
        assert call_kwargs["drop_params"] is True
        assert call_kwargs["temperature"] == 1.0
        assert call_kwargs["max_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_acompletion = _install_litellm_stub(monkeypatch)
        mock_acompletion.return_value = _mock_response("Done")

        vlm = LiteLLMVLM()
        await vlm.generate("test", system_prompt="You are a helpful assistant")

        messages = mock_acompletion.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant"

    @pytest.mark.asyncio
    async def test_generate_with_json_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_acompletion = _install_litellm_stub(monkeypatch)
        mock_acompletion.return_value = _mock_response('{"key": "value"}')

        vlm = LiteLLMVLM()
        await vlm.generate("test", response_format="json")

        call_kwargs = mock_acompletion.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_generate_passes_temp_and_max_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_acompletion = _install_litellm_stub(monkeypatch)
        mock_acompletion.return_value = _mock_response("ok")

        vlm = LiteLLMVLM()
        await vlm.generate("test", temperature=0.5, max_tokens=1024)

        call_kwargs = mock_acompletion.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 1024

    @pytest.mark.asyncio
    async def test_api_key_forwarded_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_acompletion = _install_litellm_stub(monkeypatch)
        mock_acompletion.return_value = _mock_response("ok")

        vlm = LiteLLMVLM(api_key="sk-test-123")
        await vlm.generate("test")

        assert mock_acompletion.call_args.kwargs["api_key"] == "sk-test-123"

    @pytest.mark.asyncio
    async def test_api_key_omitted_when_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_acompletion = _install_litellm_stub(monkeypatch)
        mock_acompletion.return_value = _mock_response("ok")

        vlm = LiteLLMVLM()
        await vlm.generate("test")

        assert "api_key" not in mock_acompletion.call_args.kwargs

    @pytest.mark.asyncio
    async def test_api_base_forwarded_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_acompletion = _install_litellm_stub(monkeypatch)
        mock_acompletion.return_value = _mock_response("ok")

        vlm = LiteLLMVLM(api_base="http://localhost:4000")
        await vlm.generate("test")

        assert mock_acompletion.call_args.kwargs["api_base"] == "http://localhost:4000"


# ── Generate (with images / vision) ───────────────────────────


class TestLiteLLMVLMVision:
    @pytest.mark.asyncio
    async def test_generate_with_images(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_acompletion = _install_litellm_stub(monkeypatch)
        mock_acompletion.return_value = _mock_response("I see a red square")

        monkeypatch.setattr(
            "paperbanana.providers.vlm.litellm.image_to_base64",
            lambda _: "fake-b64-data",
        )

        vlm = LiteLLMVLM()
        img = Image.new("RGB", (4, 4), color="red")
        text = await vlm.generate("What do you see?", images=[img])

        assert text == "I see a red square"
        messages = mock_acompletion.call_args.kwargs["messages"]
        user_content = messages[-1]["content"]
        assert user_content[0]["type"] == "image_url"
        assert "fake-b64-data" in user_content[0]["image_url"]["url"]
        assert user_content[-1]["type"] == "text"


# ── Error handling ────────────────────────────────────────────


class TestLiteLLMVLMErrors:
    @pytest.mark.asyncio
    async def test_auth_error_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_acompletion = _install_litellm_stub(monkeypatch)
        litellm_mod = sys.modules["litellm"]
        mock_acompletion.side_effect = litellm_mod.AuthenticationError("Invalid API key")

        vlm = LiteLLMVLM()
        with pytest.raises(Exception, match="Invalid API key"):
            await vlm.generate.__wrapped__(vlm, "test")

    @pytest.mark.asyncio
    async def test_not_found_error_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_acompletion = _install_litellm_stub(monkeypatch)
        litellm_mod = sys.modules["litellm"]
        mock_acompletion.side_effect = litellm_mod.NotFoundError("Model not found: bad/model")

        vlm = LiteLLMVLM(model="bad/model")
        with pytest.raises(Exception, match="not found"):
            await vlm.generate.__wrapped__(vlm, "test")

    @pytest.mark.asyncio
    async def test_timeout_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_acompletion = _install_litellm_stub(monkeypatch)
        litellm_mod = sys.modules["litellm"]
        mock_acompletion.side_effect = litellm_mod.Timeout("Request timed out")

        vlm = LiteLLMVLM()
        with pytest.raises(Exception, match="timed out"):
            await vlm.generate.__wrapped__(vlm, "test")

    @pytest.mark.asyncio
    async def test_null_content_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_acompletion = _install_litellm_stub(monkeypatch)
        mock_acompletion.return_value = _mock_response(content=None)

        vlm = LiteLLMVLM()
        result = await vlm.generate("test")
        assert result is None


# ── Cost tracking ─────────────────────────────────────────────


class TestLiteLLMVLMCostTracking:
    @pytest.mark.asyncio
    async def test_cost_tracker_called_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_acompletion = _install_litellm_stub(monkeypatch)
        mock_acompletion.return_value = _mock_response(
            "ok", prompt_tokens=100, completion_tokens=50
        )

        tracker = MagicMock()
        vlm = LiteLLMVLM(model="anthropic/claude-sonnet-4-6")
        vlm.cost_tracker = tracker

        await vlm.generate("test")

        tracker.record_vlm_call.assert_called_once_with(
            provider="litellm",
            model="anthropic/claude-sonnet-4-6",
            input_tokens=100,
            output_tokens=50,
        )

    @pytest.mark.asyncio
    async def test_no_cost_tracker_no_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_acompletion = _install_litellm_stub(monkeypatch)
        mock_acompletion.return_value = _mock_response("ok")

        vlm = LiteLLMVLM()
        vlm.cost_tracker = None
        text = await vlm.generate("test")
        assert text == "ok"

    @pytest.mark.asyncio
    async def test_no_usage_in_response_skips_tracking(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_acompletion = _install_litellm_stub(monkeypatch)
        message = types.SimpleNamespace(content="ok")
        choice = types.SimpleNamespace(message=message)
        mock_acompletion.return_value = types.SimpleNamespace(choices=[choice], usage=None)

        tracker = MagicMock()
        vlm = LiteLLMVLM()
        vlm.cost_tracker = tracker

        await vlm.generate("test")
        tracker.record_vlm_call.assert_not_called()
