"""Tests for graceful agent failure handling and retry in the pipeline (issue #135)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("PIL", reason="PIL/Pillow required for pipeline image mock")
from pathlib import Path

from PIL import Image

from paperbanana.core import pipeline as pipeline_mod
from paperbanana.core.config import Settings
from paperbanana.core.pipeline import PaperBananaPipeline
from paperbanana.core.types import (
    CritiqueResult,
    DiagramType,
    GenerationInput,
)

# ── Shared mocks ─────────────────────────────────────────────────


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


def _make_settings(tmp_path, **overrides):
    defaults = dict(
        output_dir=str(tmp_path / "outputs"),
        reference_set_path=str(tmp_path / "empty_refs"),
        refinement_iterations=1,
        save_iterations=False,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _default_input():
    return GenerationInput(
        source_context="Test methodology text",
        communicative_intent="Test caption",
        diagram_type=DiagramType.METHODOLOGY,
    )


# ── Fixture: fast retries (no waiting) ───────────────────────────


@pytest.fixture(autouse=True)
def _fast_retry(monkeypatch):
    """Replace _call_with_retry with a zero-wait version for fast tests."""

    async def _fast_call_with_retry(label, fn, *args, max_attempts=3, **kwargs):
        last_exc = None
        for attempt_num in range(1, max_attempts + 1):
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt_num >= max_attempts:
                    raise
        raise last_exc  # unreachable but satisfies type checker

    monkeypatch.setattr(pipeline_mod, "_call_with_retry", _fast_call_with_retry)


# ── Test: Optimizer failure falls back to original input ─────────


@pytest.mark.asyncio
async def test_optimizer_failure_falls_back_to_original_input(tmp_path):
    """When optimizer raises, pipeline continues with the original input."""
    settings = _make_settings(tmp_path, optimize_inputs=True)
    # VLM responses: planner, stylist, critic (optimizer is patched to fail)
    vlm = _MockVLM(
        responses=[
            "Plan description",
            "Styled description",
            json.dumps({"critic_suggestions": [], "revised_description": None}),
        ]
    )
    image_gen = _MockImageGen()
    pipeline = PaperBananaPipeline(settings=settings, vlm_client=vlm, image_gen_fn=image_gen)

    # Patch the optimizer to raise
    pipeline.optimizer.run = AsyncMock(side_effect=RuntimeError("API timeout"))

    result = await pipeline.generate(_default_input())

    assert result.image_path
    assert Path(result.image_path).exists()
    # Pipeline completed successfully despite optimizer failure
    assert len(result.iterations) >= 1


# ── Test: Stylist failure falls back to planner output ───────────


@pytest.mark.asyncio
async def test_stylist_failure_falls_back_to_planner_output(tmp_path):
    """When stylist fails after retries, pipeline uses the planner's raw output."""
    settings = _make_settings(tmp_path)
    # VLM responses: planner, critic (stylist is patched to fail)
    vlm = _MockVLM(
        responses=[
            "Planner raw description",
            json.dumps({"critic_suggestions": [], "revised_description": None}),
        ]
    )
    image_gen = _MockImageGen()
    pipeline = PaperBananaPipeline(settings=settings, vlm_client=vlm, image_gen_fn=image_gen)

    pipeline.stylist.run = AsyncMock(side_effect=RuntimeError("Stylist API error"))

    result = await pipeline.generate(_default_input())

    assert result.image_path
    assert Path(result.image_path).exists()
    # Description should contain the planner output (stylist was bypassed)
    assert "Planner raw description" in result.description


# ── Test: Critic failure accepts current image ───────────────────


@pytest.mark.asyncio
async def test_critic_failure_accepts_current_image(tmp_path):
    """When critic fails, pipeline accepts the current image and stops iterating."""
    settings = _make_settings(tmp_path, refinement_iterations=2)
    # VLM responses: planner, stylist (critic is patched to fail)
    vlm = _MockVLM(
        responses=[
            "Plan description",
            "Styled description",
        ]
    )
    image_gen = _MockImageGen()
    pipeline = PaperBananaPipeline(settings=settings, vlm_client=vlm, image_gen_fn=image_gen)

    pipeline.critic.run = AsyncMock(side_effect=RuntimeError("Critic API error"))

    result = await pipeline.generate(_default_input())

    assert result.image_path
    assert Path(result.image_path).exists()
    # Should have one iteration (critic failed but image was accepted)
    assert len(result.iterations) == 1
    # The fallback CritiqueResult should have no suggestions
    assert not result.iterations[0].critique.needs_revision


# ── Test: Retry succeeds after transient failures ────────────────


@pytest.mark.asyncio
async def test_agent_retries_on_transient_failure(tmp_path):
    """A required agent that fails transiently succeeds after retry."""
    settings = _make_settings(tmp_path)
    vlm = _MockVLM(responses=[])
    image_gen = _MockImageGen()
    pipeline = PaperBananaPipeline(settings=settings, vlm_client=vlm, image_gen_fn=image_gen)

    # Planner fails twice, succeeds on third attempt
    call_count = 0

    async def flaky_planner(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("transient error")
        return ("Planned description", None)

    pipeline.planner.run = flaky_planner
    pipeline.stylist.run = AsyncMock(return_value="Styled description")
    pipeline.critic.run = AsyncMock(
        return_value=CritiqueResult(critic_suggestions=[], revised_description=None)
    )

    result = await pipeline.generate(_default_input())

    assert result.image_path
    assert call_count == 3  # failed twice, succeeded on third


# ── Test: Required phase raises after all retries exhausted ──────


@pytest.mark.asyncio
async def test_required_agent_raises_after_retries_exhausted(tmp_path):
    """A required agent that always fails re-raises after all retry attempts."""
    settings = _make_settings(tmp_path)
    vlm = _MockVLM(responses=[])
    image_gen = _MockImageGen()
    pipeline = PaperBananaPipeline(settings=settings, vlm_client=vlm, image_gen_fn=image_gen)

    pipeline.planner.run = AsyncMock(side_effect=RuntimeError("persistent error"))

    with pytest.raises(RuntimeError, match="persistent error"):
        await pipeline.generate(_default_input())


# ── Test: Visualizer retries on transient failure ────────────────


@pytest.mark.asyncio
async def test_visualizer_retries_on_transient_failure(tmp_path):
    """Visualizer succeeds after a transient failure is retried."""
    settings = _make_settings(tmp_path)
    vlm = _MockVLM(
        responses=[
            "Plan description",
            "Styled description",
            json.dumps({"critic_suggestions": [], "revised_description": None}),
        ]
    )
    image_gen = _MockImageGen()
    pipeline = PaperBananaPipeline(settings=settings, vlm_client=vlm, image_gen_fn=image_gen)

    call_count = 0
    original_visualizer_run = pipeline.visualizer.run

    async def flaky_visualizer(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise RuntimeError("image gen transient error")
        return await original_visualizer_run(**kwargs)

    pipeline.visualizer.run = flaky_visualizer

    result = await pipeline.generate(_default_input())

    assert result.image_path
    assert call_count == 2  # failed once, succeeded on second
