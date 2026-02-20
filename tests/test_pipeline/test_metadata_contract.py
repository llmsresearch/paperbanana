"""Tests for pipeline metadata contract and timing shape."""

from __future__ import annotations

import json
from numbers import Real

import pytest
from PIL import Image

from paperbanana.core.config import Settings
from paperbanana.core.pipeline import PaperBananaPipeline
from paperbanana.core.types import DiagramType, GenerationInput


class _MockVLM:
    name = "mock-vlm"
    model_name = "mock-model"

    def __init__(self, responses: list[str]):
        self._responses = responses
        self._idx = 0

    async def generate(self, *args, **kwargs):
        idx = min(self._idx, len(self._responses) - 1)
        self._idx += 1
        return self._responses[idx]


class _MockImageGen:
    name = "mock-image-gen"
    model_name = "mock-image-model"

    async def generate(self, *args, **kwargs):
        return Image.new("RGB", (128, 128), color=(255, 255, 255))


@pytest.mark.asyncio
async def test_generation_output_metadata_contains_required_keys_and_timing(tmp_path):
    settings = Settings(
        output_dir=str(tmp_path / "outputs"),
        reference_set_path=str(tmp_path / "empty_refs"),
        refinement_iterations=2,
        save_iterations=False,
    )

    # With an empty reference store, retriever returns [] without calling VLM.
    # VLM call order becomes: planner -> stylist -> critic
    vlm = _MockVLM(
        responses=[
            "Initial plan description",
            "Styled final description",
            json.dumps({"critic_suggestions": [], "revised_description": None}),
        ]
    )
    image_gen = _MockImageGen()
    pipeline = PaperBananaPipeline(settings=settings, vlm_client=vlm, image_gen_fn=image_gen)

    result = await pipeline.generate(
        GenerationInput(
            source_context="A minimal source context",
            communicative_intent="A minimal caption",
            diagram_type=DiagramType.METHODOLOGY,
        )
    )

    metadata = result.metadata
    for key in (
        "run_id",
        "timestamp",
        "vlm_provider",
        "image_provider",
        "refinement_iterations",
        "config_snapshot",
        "timing",
    ):
        assert key in metadata

    timing = metadata["timing"]
    for key in (
        "total_seconds",
        "retrieval_seconds",
        "planning_seconds",
        "styling_seconds",
        "iterations",
    ):
        assert key in timing

    for scalar_key in ("total_seconds", "retrieval_seconds", "planning_seconds", "styling_seconds"):
        assert isinstance(timing[scalar_key], Real)
        assert timing[scalar_key] >= 0

    assert isinstance(timing["iterations"], list)
    assert len(timing["iterations"]) >= 1

    iter_item = timing["iterations"][0]
    for key in ("iteration", "visualizer_seconds", "critic_seconds"):
        assert key in iter_item

    assert isinstance(iter_item["iteration"], int)
    assert iter_item["iteration"] >= 1
    assert isinstance(iter_item["visualizer_seconds"], Real)
    assert isinstance(iter_item["critic_seconds"], Real)
    assert iter_item["visualizer_seconds"] >= 0
    assert iter_item["critic_seconds"] >= 0
