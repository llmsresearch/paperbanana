"""Tests for the Ollama VLM provider."""

from __future__ import annotations

import pytest
from PIL import Image

from paperbanana.providers.vlm.ollama import OllamaVLM


class _FakeResponse:
    """Minimal httpx.Response stand-in."""

    status_code = 200

    def __init__(self, text: str = "hello"):
        self._text = text

    def raise_for_status(self):
        pass

    def json(self):
        return {
            "choices": [{"message": {"content": self._text}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }


class _FakeClient:
    """Captures POST payloads for test inspection."""

    def __init__(self, text: str = "hello"):
        self.captured: dict = {}
        self._text = text
        self.closed = False

    async def post(self, url, json=None, **kw):
        self.captured = {"url": url, "json": json}
        return _FakeResponse(self._text)

    async def aclose(self):
        self.closed = True


@pytest.fixture
def vlm():
    return OllamaVLM(model="qwen2.5-vl")


def test_properties(vlm: OllamaVLM):
    assert vlm.name == "ollama"
    assert vlm.model_name == "qwen2.5-vl"
    assert vlm.supports_json_mode is False
    assert OllamaVLM(model="x", json_mode=True).supports_json_mode is True


@pytest.mark.asyncio
async def test_generate_text_only(vlm: OllamaVLM):
    client = _FakeClient("output")
    vlm._client = client
    result = await vlm.generate("Hello")
    assert result == "output"
    payload = client.captured["json"]
    assert payload["model"] == "qwen2.5-vl"
    user_content = payload["messages"][-1]["content"]
    assert any(c["type"] == "text" and c["text"] == "Hello" for c in user_content)
    assert "response_format" not in payload


@pytest.mark.asyncio
async def test_generate_with_image(vlm: OllamaVLM, monkeypatch):
    monkeypatch.setattr("paperbanana.providers.vlm.ollama.image_to_base64", lambda _: "b64data")
    client = _FakeClient("described")
    vlm._client = client
    result = await vlm.generate("Describe", images=[Image.new("RGB", (4, 4))])
    assert result == "described"
    content = client.captured["json"]["messages"][-1]["content"]
    assert any(c["type"] == "image_url" and "b64data" in c["image_url"]["url"] for c in content)


@pytest.mark.asyncio
@pytest.mark.parametrize("json_mode,expect_key", [(False, False), (True, True)])
async def test_json_mode_toggle(json_mode, expect_key):
    vlm = OllamaVLM(model="test", json_mode=json_mode)
    vlm._client = _FakeClient('{"k":"v"}')
    await vlm.generate("Return JSON", response_format="json")
    assert ("response_format" in vlm._client.captured["json"]) is expect_key


@pytest.mark.asyncio
async def test_close(vlm: OllamaVLM):
    client = _FakeClient()
    vlm._client = client
    await vlm.close()
    assert client.closed and vlm._client is None
