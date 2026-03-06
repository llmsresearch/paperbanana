"""Tests for the Retriever agent."""

from __future__ import annotations

import json

import pytest

from paperbanana.agents.retriever import RetrieverAgent
from paperbanana.core.types import ReferenceExample


class MockVLM:
    """Mock VLM provider for testing."""

    name = "mock"
    model_name = "mock-model"

    def __init__(self, response: str = ""):
        self._response = response

    async def generate(
        self,
        prompt,
        images=None,
        system_prompt=None,
        temperature=1.0,
        max_tokens=4096,
        response_format=None,
    ):
        return self._response

    def is_available(self):
        return True


def _make_examples(n: int) -> list[ReferenceExample]:
    return [
        ReferenceExample(
            id=f"ref_{i:03d}",
            source_context=f"Example source context {i}",
            caption=f"Example caption {i}",
            image_path=f"images/{i}.png",
            category="test",
        )
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_retriever_returns_all_when_few_candidates():
    """When candidates <= num_examples, return all."""
    vlm = MockVLM()
    agent = RetrieverAgent(vlm)
    candidates = _make_examples(3)

    result = await agent.run(
        source_context="test",
        caption="test",
        candidates=candidates,
        num_examples=5,
    )

    assert len(result) == 3


@pytest.mark.asyncio
async def test_retriever_empty_candidates():
    """When no candidates, return empty list."""
    vlm = MockVLM()
    agent = RetrieverAgent(vlm)

    result = await agent.run(
        source_context="test",
        caption="test",
        candidates=[],
        num_examples=5,
    )

    assert result == []


@pytest.mark.asyncio
async def test_retriever_parses_vlm_response():
    """Test that retriever correctly parses VLM JSON response."""
    response = json.dumps(
        {
            "selected_ids": ["ref_001", "ref_003"],
            "reasoning": {
                "ref_001": "Relevant because...",
                "ref_003": "Relevant because...",
            },
        }
    )

    vlm = MockVLM(response=response)
    agent = RetrieverAgent(vlm)
    candidates = _make_examples(5)

    result = await agent.run(
        source_context="test",
        caption="test",
        candidates=candidates,
        num_examples=2,
    )

    assert len(result) == 2
    assert result[0].id == "ref_001"
    assert result[1].id == "ref_003"


@pytest.mark.asyncio
async def test_retriever_handles_malformed_json():
    """Test fallback when VLM returns invalid JSON."""
    vlm = MockVLM(response="this is not json")
    agent = RetrieverAgent(vlm)
    candidates = _make_examples(5)

    result = await agent.run(
        source_context="test",
        caption="test",
        candidates=candidates,
        num_examples=3,
    )

    # Should fall back to candidates, truncated to num_examples
    assert len(result) == 3
