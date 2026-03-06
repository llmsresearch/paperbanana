"""Tests for the Anthropic VLM provider."""

from __future__ import annotations

import types
from typing import Any

import pytest
from PIL import Image

from paperbanana.providers.vlm.anthropic import AnthropicVLM


@pytest.mark.asyncio
async def test_generate_text_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """AnthropicVLM.generate should send a basic text-only request and return text."""
    captured: dict[str, Any] = {}

    class _FakeMessages:
        async def create(self, **kwargs: Any) -> Any:  # type: ignore[override]
            captured.update(kwargs)
            block = types.SimpleNamespace(type="text", text="hello world")
            resp = types.SimpleNamespace(content=[block], usage=None)
            return resp

    class _FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
            self.messages = _FakeMessages()

    # Patch anthropic.AsyncAnthropic before the provider imports it.
    fake_mod = types.ModuleType("anthropic")
    fake_mod.AsyncAnthropic = _FakeClient  # type: ignore[attr-defined]

    import sys

    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)

    vlm = AnthropicVLM(api_key="test-key", model="claude-3-5-sonnet-20251023")
    text = await vlm.generate("Hi Claude")

    assert text == "hello world"
    assert captured["model"] == vlm.model_name
    assert captured["max_tokens"] == 4096
    assert isinstance(captured["messages"], list)
    assert captured["messages"][0]["role"] == "user"


@pytest.mark.asyncio
async def test_generate_with_images_and_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """AnthropicVLM.generate should inline images and enable JSON mode when requested."""
    captured: dict[str, Any] = {}

    class _FakeMessages:
        async def create(self, **kwargs: Any) -> Any:  # type: ignore[override]
            captured.update(kwargs)
            block = types.SimpleNamespace(type="text", text="{}")
            resp = types.SimpleNamespace(content=[block], usage=None)
            return resp

    class _FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
            self.messages = _FakeMessages()

    fake_mod = types.ModuleType("anthropic")
    fake_mod.AsyncAnthropic = _FakeClient  # type: ignore[attr-defined]

    import sys

    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)

    # Avoid depending on real base64 implementation details.
    def _fake_image_to_base64(_img: Image.Image) -> str:
        return "base64-image-data"

    monkeypatch.setattr(
        "paperbanana.providers.vlm.anthropic.image_to_base64",
        _fake_image_to_base64,
    )

    vlm = AnthropicVLM(api_key="test-key", model="claude-3-5-sonnet-20251023")
    img = Image.new("RGB", (4, 4))

    await vlm.generate("Hi with image", images=[img], response_format="json")

    assert captured["model"] == vlm.model_name
    msg = captured["messages"][0]
    assert msg["role"] == "user"
    content = msg["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["data"] == "base64-image-data"
    assert content[-1]["type"] == "text"
    assert content[-1]["text"] == "Hi with image"

    output_config = captured["output_config"]
    fmt = output_config["format"]
    assert fmt["type"] == "json_schema"
    assert isinstance(fmt["schema"], dict)
