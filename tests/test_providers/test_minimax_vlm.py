"""Tests for the MiniMax VLM provider."""

from __future__ import annotations

import types
from typing import Any

import pytest
from PIL import Image

from paperbanana.providers.vlm.minimax import MiniMaxVLM


@pytest.mark.asyncio
async def test_generate_text_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """MiniMaxVLM.generate should send a basic text-only request and return text."""
    captured: dict[str, Any] = {}

    class _FakeMessages:
        async def create(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            block = types.SimpleNamespace(type="text", text="hello minimax")
            resp = types.SimpleNamespace(content=[block], usage=None)
            return resp

    class _FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.messages = _FakeMessages()

    fake_mod = types.ModuleType("anthropic")
    fake_mod.AsyncAnthropic = _FakeClient  # type: ignore[attr-defined]

    import sys

    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)

    vlm = MiniMaxVLM(api_key="test-key", model="MiniMax-M2.7")
    text = await vlm.generate("Hello MiniMax")

    assert text == "hello minimax"
    assert captured["model"] == "MiniMax-M2.7"
    assert captured["max_tokens"] == 4096
    assert isinstance(captured["messages"], list)
    assert captured["messages"][0]["role"] == "user"


@pytest.mark.asyncio
async def test_generate_with_images(monkeypatch: pytest.MonkeyPatch) -> None:
    """MiniMaxVLM.generate should inline images as base64 in the request."""
    captured: dict[str, Any] = {}

    class _FakeMessages:
        async def create(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            block = types.SimpleNamespace(type="text", text="analysis result")
            resp = types.SimpleNamespace(content=[block], usage=None)
            return resp

    class _FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.messages = _FakeMessages()

    fake_mod = types.ModuleType("anthropic")
    fake_mod.AsyncAnthropic = _FakeClient  # type: ignore[attr-defined]

    import sys

    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)

    monkeypatch.setattr(
        "paperbanana.providers.vlm.minimax.image_to_base64",
        lambda _img: "fake-base64-data",
    )

    vlm = MiniMaxVLM(api_key="test-key", model="MiniMax-M2.7")
    img = Image.new("RGB", (4, 4))
    result = await vlm.generate("Analyze this image", images=[img])

    assert result == "analysis result"
    msg = captured["messages"][0]
    content = msg["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["data"] == "fake-base64-data"
    assert content[-1]["type"] == "text"
    assert content[-1]["text"] == "Analyze this image"


@pytest.mark.asyncio
async def test_response_format_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """MiniMaxVLM should not pass response_format / output_config to the API."""
    captured: dict[str, Any] = {}

    class _FakeMessages:
        async def create(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            block = types.SimpleNamespace(type="text", text="{}")
            resp = types.SimpleNamespace(content=[block], usage=None)
            return resp

    class _FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.messages = _FakeMessages()

    fake_mod = types.ModuleType("anthropic")
    fake_mod.AsyncAnthropic = _FakeClient  # type: ignore[attr-defined]

    import sys

    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)

    vlm = MiniMaxVLM(api_key="test-key", model="MiniMax-M2.7")
    await vlm.generate("Return JSON", response_format="json")

    assert "output_config" not in captured
    assert "response_format" not in captured


@pytest.mark.asyncio
async def test_temperature_clamped_above_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """MiniMaxVLM must not pass temperature=0 (MiniMax requires temperature > 0)."""
    captured: dict[str, Any] = {}

    class _FakeMessages:
        async def create(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            block = types.SimpleNamespace(type="text", text="ok")
            resp = types.SimpleNamespace(content=[block], usage=None)
            return resp

    class _FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.messages = _FakeMessages()

    fake_mod = types.ModuleType("anthropic")
    fake_mod.AsyncAnthropic = _FakeClient  # type: ignore[attr-defined]

    import sys

    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)

    vlm = MiniMaxVLM(api_key="test-key", model="MiniMax-M2.7")
    await vlm.generate("Hello", temperature=0.0)

    assert captured["temperature"] == 1.0


@pytest.mark.asyncio
async def test_system_prompt_passed_as_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """MiniMaxVLM should pass system_prompt as a plain string to the API."""
    captured: dict[str, Any] = {}

    class _FakeMessages:
        async def create(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            block = types.SimpleNamespace(type="text", text="ok")
            resp = types.SimpleNamespace(content=[block], usage=None)
            return resp

    class _FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.messages = _FakeMessages()

    fake_mod = types.ModuleType("anthropic")
    fake_mod.AsyncAnthropic = _FakeClient  # type: ignore[attr-defined]

    import sys

    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)

    vlm = MiniMaxVLM(api_key="test-key", model="MiniMax-M2.7")
    await vlm.generate("Hello", system_prompt="You are a helpful assistant.")

    assert captured["system"] == "You are a helpful assistant."
    assert isinstance(captured["system"], str)


def test_provider_metadata() -> None:
    """MiniMaxVLM should report name='minimax' and the correct model_name."""
    vlm = MiniMaxVLM(api_key="test-key", model="MiniMax-M2.7-highspeed")
    assert vlm.name == "minimax"
    assert vlm.model_name == "MiniMax-M2.7-highspeed"


def test_is_available_with_key() -> None:
    """is_available should return True when api_key is set."""
    vlm = MiniMaxVLM(api_key="some-key")
    assert vlm.is_available() is True


def test_is_available_without_key() -> None:
    """is_available should return False when api_key is None."""
    vlm = MiniMaxVLM(api_key=None)
    assert vlm.is_available() is False


def test_default_base_url() -> None:
    """Default base_url should point to MiniMax Anthropic-compatible endpoint."""
    vlm = MiniMaxVLM(api_key="k")
    assert vlm._base_url == "https://api.minimax.io/anthropic"


def test_custom_base_url() -> None:
    """Custom base_url should be preserved."""
    vlm = MiniMaxVLM(api_key="k", base_url="https://custom.example.com/anthropic")
    assert vlm._base_url == "https://custom.example.com/anthropic"
