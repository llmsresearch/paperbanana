"""Integration tests for exemplar retrieval pipeline behavior."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from paperbanana.core.config import Settings
from paperbanana.core.pipeline import PaperBananaPipeline
from paperbanana.core.types import DiagramType, GenerationInput
from paperbanana.reference.exemplar_retrieval import ExemplarHit, ExemplarRetrievalError


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


class _FakeExternalRetriever:
    def __init__(self, hits: list[ExemplarHit]):
        self._hits = hits

    async def retrieve(self, *args, **kwargs):
        return self._hits


class _FailingExternalRetriever:
    async def retrieve(self, *args, **kwargs):
        raise ExemplarRetrievalError("simulated failure")


@pytest.mark.asyncio
async def test_external_only_mode_skips_internal_retriever(tmp_path):
    settings = Settings(
        output_dir=str(tmp_path / "outputs"),
        reference_set_path=str(tmp_path / "empty_refs"),
        save_iterations=False,
        refinement_iterations=1,
        exemplar_retrieval_enabled=True,
        exemplar_retrieval_endpoint="https://retriever.test/query",
        exemplar_retrieval_mode="external_only",
        num_retrieval_examples=2,
    )
    vlm = _MockVLM(
        responses=[
            "planner description",
            "styled description",
            json.dumps({"critic_suggestions": [], "revised_description": None}),
        ]
    )
    pipeline = PaperBananaPipeline(settings=settings, vlm_client=vlm, image_gen_fn=_MockImageGen())
    pipeline._external_exemplar_retriever = _FakeExternalRetriever(
        [
            ExemplarHit(id="ext_1", caption="c1", source_context="s1", image_path=""),
            ExemplarHit(id="ext_2", caption="c2", source_context="s2", image_path=""),
        ]
    )
    pipeline.retriever.run = AsyncMock(return_value=[])

    result = await pipeline.generate(
        GenerationInput(
            source_context="source context",
            communicative_intent="caption",
            diagram_type=DiagramType.METHODOLOGY,
        )
    )

    pipeline.retriever.run.assert_not_awaited()
    assert result.metadata["retrieval"]["mode"] == "external_only"
    assert result.metadata["retrieval"]["external_candidate_ids"] == ["ext_1", "ext_2"]


@pytest.mark.asyncio
async def test_external_failure_falls_back_to_internal_retriever(tmp_path):
    settings = Settings(
        output_dir=str(tmp_path / "outputs"),
        reference_set_path=str(tmp_path / "empty_refs"),
        save_iterations=False,
        refinement_iterations=1,
        exemplar_retrieval_enabled=True,
        exemplar_retrieval_endpoint="https://retriever.test/query",
        exemplar_retrieval_mode="external_then_rerank",
    )
    vlm = _MockVLM(
        responses=[
            "planner description",
            "styled description",
            json.dumps({"critic_suggestions": [], "revised_description": None}),
        ]
    )
    pipeline = PaperBananaPipeline(settings=settings, vlm_client=vlm, image_gen_fn=_MockImageGen())
    pipeline._external_exemplar_retriever = _FailingExternalRetriever()
    pipeline.retriever.run = AsyncMock(return_value=[])

    result = await pipeline.generate(
        GenerationInput(
            source_context="source context",
            communicative_intent="caption",
            diagram_type=DiagramType.METHODOLOGY,
        )
    )

    pipeline.retriever.run.assert_awaited_once()
    assert result.metadata["retrieval"]["mode"] == "fallback_error"
    assert result.metadata["retrieval"]["external_candidate_ids"] == []
