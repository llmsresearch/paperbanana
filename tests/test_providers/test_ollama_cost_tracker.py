"""Tests for OllamaVLM cost tracker integration and Ollama pricing."""

from __future__ import annotations

import types

import pytest

from paperbanana.core.cost_tracker import CostTracker
from paperbanana.core.pricing import lookup_vlm_price
from paperbanana.providers.vlm.ollama import OllamaVLM


def _fake_response(
    prompt_tokens: int = 50, completion_tokens: int = 20, include_usage: bool = True
):
    payload: dict = {"choices": [{"message": {"content": "hello"}}]}
    if include_usage:
        payload["usage"] = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
    return types.SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)


class _FakeClient:
    def __init__(self, response) -> None:
        self._response = response

    async def post(self, path, json=None, **kw):
        return self._response

    async def aclose(self) -> None:
        pass


# ── Pricing ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("provider", ["ollama", "openai_local"])
def test_local_providers_return_zero_cost(provider):
    # Any model name must return $0 with pricing_known=True — never "unknown".
    result = lookup_vlm_price(provider, "any-model-name")
    assert result == {"input_per_1k": 0.0, "output_per_1k": 0.0}


def test_tracker_marks_ollama_pricing_as_known():
    # pricing_known must be True, not False — $0.00 is correct, not unknown.
    tracker = CostTracker()
    tracker.record_vlm_call(
        provider="ollama", model="qwen2.5-vl", input_tokens=100, output_tokens=50
    )
    assert tracker.entries[0].pricing_known is True
    assert tracker.total_cost == 0.0


# ── Cost tracker integration ─────────────────────────────────────────────────


async def test_tracker_receives_correct_token_counts():
    vlm = OllamaVLM()
    vlm._client = _FakeClient(_fake_response(prompt_tokens=80, completion_tokens=30))
    tracker = CostTracker()
    vlm.cost_tracker = tracker

    await vlm.generate("describe this")

    assert len(tracker.entries) == 1
    entry = tracker.entries[0]
    assert entry.provider == "ollama"
    assert entry.input_tokens == 80
    assert entry.output_tokens == 30


async def test_no_tracker_attached_does_not_crash():
    vlm = OllamaVLM()
    vlm._client = _FakeClient(_fake_response())
    # cost_tracker is None by default — must not raise
    assert await vlm.generate("hello") == "hello"


async def test_missing_usage_in_response_skips_tracker():
    # Some Ollama builds omit usage — tracker must stay silent, not record zeros.
    vlm = OllamaVLM()
    vlm._client = _FakeClient(_fake_response(include_usage=False))
    tracker = CostTracker()
    vlm.cost_tracker = tracker

    await vlm.generate("hello")

    assert len(tracker.entries) == 0
