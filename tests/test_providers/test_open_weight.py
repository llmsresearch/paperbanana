"""Tests for open-weight VLM support: extract_json, registry, agent integration."""

from __future__ import annotations

import json

import pytest

from paperbanana.core.config import Settings
from paperbanana.core.types import ReferenceExample
from paperbanana.core.utils import extract_json
from paperbanana.providers.registry import ProviderRegistry


class TestExtractJson:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ('{"a": 1}', {"a": 1}),
            ("[1, 2]", [1, 2]),
            ('Here:\n```json\n{"id": "x"}\n```\nDone.', {"id": "x"}),
            ('Sure:\n```\n{"w": "M"}\n```', {"w": "M"}),
            ('Answer is {"s": 42} ok.', {"s": 42}),
            ('IDs: ["a", "b"] done.', ["a", "b"]),
        ],
    )
    def test_parses(self, text, expected):
        assert extract_json(text) == expected

    def test_nested(self):
        obj = {"outer": {"inner": [1, 2]}, "k": "v"}
        assert extract_json(f"Result: {json.dumps(obj)}") == obj

    def test_strings_with_braces(self):
        obj = {"text": "use {curly} braces"}
        assert extract_json(json.dumps(obj)) == obj

    @pytest.mark.parametrize("text", ["Plain text.", "", '{"key": "val", "incomplete'])
    def test_returns_none(self, text):
        assert extract_json(text) is None


class TestRegistryLocalProviders:
    def test_ollama(self):
        vlm = ProviderRegistry.create_vlm(Settings(vlm_provider="ollama", vlm_model="llava"))
        assert vlm.name == "ollama" and vlm.model_name == "llava"
        assert vlm.supports_json_mode is False

    def test_ollama_model_override(self):
        vlm = ProviderRegistry.create_vlm(
            Settings(vlm_provider="ollama", vlm_model="default", ollama_model="qwen2.5-vl:72b")
        )
        assert vlm.model_name == "qwen2.5-vl:72b"

    def test_openai_local(self):
        vlm = ProviderRegistry.create_vlm(
            Settings(vlm_provider="openai_local", vlm_model="Qwen/Qwen2.5-VL-7B")
        )
        assert vlm.name == "openai_local" and vlm.supports_json_mode is False

    def test_unknown_provider_mentions_new(self):
        with pytest.raises(ValueError, match="ollama"):
            ProviderRegistry.create_vlm(Settings(vlm_provider="nonexistent"))


class _MockVLM:
    """Mock VLM with configurable json_mode support."""

    name = "mock"
    model_name = "mock"

    def __init__(self, response: str, json_mode: bool = True):
        self._response = response
        self.supports_json_mode = json_mode
        self.last_response_format = "NOT_CALLED"

    async def generate(
        self,
        prompt,
        images=None,
        system_prompt=None,
        temperature=1.0,
        max_tokens=4096,
        response_format=None,
    ):
        self.last_response_format = response_format
        return self._response


class TestAgentJsonMode:
    @pytest.mark.asyncio
    async def test_retriever_skips_json(self):
        from paperbanana.agents.retriever import RetrieverAgent

        vlm = _MockVLM('```json\n{"selected_ids": ["ref_001"]}\n```', json_mode=False)
        agent = RetrieverAgent(vlm)
        candidates = [
            ReferenceExample(
                id=f"ref_{i:03d}",
                source_context=f"C{i}",
                caption=f"Cap{i}",
                image_path=f"img/{i}.png",
            )
            for i in range(5)
        ]
        result = await agent.run(
            source_context="t",
            caption="t",
            candidates=candidates,
            num_examples=2,
        )
        assert vlm.last_response_format is None
        assert len(result) == 1 and result[0].id == "ref_001"

    @pytest.mark.asyncio
    async def test_critic_skips_json(self, tmp_path):
        from PIL import Image

        from paperbanana.agents.critic import CriticAgent

        vlm = _MockVLM('```json\n{"critic_suggestions": ["fix"]}\n```', json_mode=False)
        agent = CriticAgent(vlm)
        img_path = tmp_path / "test.png"
        Image.new("RGB", (4, 4)).save(img_path)
        (tmp_path / "diagram").mkdir()
        (tmp_path / "diagram" / "critic.txt").write_text(
            "Eval: {source_context}\n{caption}\n{description}"
        )
        agent.prompt_dir = tmp_path
        result = await agent.run(
            image_path=str(img_path),
            description="d",
            source_context="c",
            caption="cap",
        )
        assert vlm.last_response_format is None
        assert result.needs_revision and "fix" in result.critic_suggestions
