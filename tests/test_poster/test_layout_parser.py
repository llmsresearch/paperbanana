"""Layout parser tests: skeleton-nesting drift and bounded retries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from paperbanana.poster.agents.layout_parser import (
    LayoutParserAgent,
    _normalize_parser_payload,
)

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"


class _ScriptedVLM:
    name = "mock"
    model_name = "mock-model"
    cost_tracker = None

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    async def generate(self, prompt, images=None, **kwargs):
        self.calls += 1
        if not self._responses:
            raise AssertionError("scripted VLM exhausted")
        return self._responses.pop(0)


def _nested() -> dict:
    return {
        "skeleton": {
            "bands": [
                {"kind": "header", "columns": 1, "height_frac": 0.12},
                {"kind": "body", "columns": 3, "height_frac": 0.88},
            ],
            "panels": [
                {
                    "band_index": 1,
                    "column": 0,
                    "col_span": 1,
                    "height_frac": 0.5,
                    "has_figure": False,
                    "emphasis": "normal",
                }
            ],
        },
        "n_figures": 4,
        "visual_share": 0.35,
        "venue": "cvpr",
    }


def _flat() -> dict:
    nested = _nested()
    return {**nested.pop("skeleton"), **nested}


def test_normalize_lifts_flat_bands_into_skeleton():
    out = _normalize_parser_payload(_flat())
    assert "skeleton" in out
    assert len(out["skeleton"]["bands"]) == 2
    assert len(out["skeleton"]["panels"]) == 1
    assert out["n_figures"] == 4


def test_normalize_leaves_nested_payload_untouched():
    payload = _nested()
    assert _normalize_parser_payload(payload) == payload


async def test_parser_accepts_flat_shape():
    vlm = _ScriptedVLM([json.dumps(_flat())])
    agent = LayoutParserAgent(vlm, prompt_dir=str(PROMPT_DIR))
    parsed = await agent.run(Image.new("RGB", (1600, 900), "white"), source_name="t")
    assert len(parsed.skeleton.bands) == 2
    assert parsed.visual_share == 0.35
    assert vlm.calls == 1


async def test_parser_retries_then_succeeds():
    vlm = _ScriptedVLM(["not json at all", json.dumps(_nested())])
    agent = LayoutParserAgent(vlm, prompt_dir=str(PROMPT_DIR))
    parsed = await agent.run(Image.new("RGB", (1600, 900), "white"), source_name="t")
    assert parsed.n_figures == 4
    assert vlm.calls == 2


async def test_parser_exhausts_to_error():
    vlm = _ScriptedVLM(["nope", "still nope", "nope again"])
    agent = LayoutParserAgent(vlm, prompt_dir=str(PROMPT_DIR))
    with pytest.raises(ValueError, match="after 3 attempts"):
        await agent.run(Image.new("RGB", (1600, 900), "white"), source_name="t")
    assert vlm.calls == 3
