"""Tests for prompt recording into run artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from paperbanana.core.config import Settings
from paperbanana.core.pipeline import PaperBananaPipeline
from paperbanana.core.types import DiagramType, GenerationInput


class _SeqVLM:
    name = "seq-vlm"
    model_name = "seq-model"

    def __init__(self, responses: list[str]):
        self._responses = responses
        self._idx = 0

    async def generate(self, *args, **kwargs):
        idx = min(self._idx, len(self._responses) - 1)
        self._idx += 1
        return self._responses[idx]


class _ImageGen:
    name = "fake-image-gen"
    model_name = "fake-image-model"

    async def generate(self, *args, **kwargs):
        return Image.new("RGB", (64, 64), color=(255, 255, 255))


def _make_reference_set(tmp_path: Path) -> Path:
    ref_dir = tmp_path / "refs"
    (ref_dir / "images").mkdir(parents=True, exist_ok=True)

    examples = []
    for i in range(3):
        img_path = ref_dir / "images" / f"ref_{i}.png"
        Image.new("RGB", (8, 8), color=(i * 40, 0, 0)).save(img_path, format="PNG")
        examples.append(
            {
                "id": f"ref_{i}",
                "source_context": f"context {i}",
                "caption": f"caption {i}",
                "image_path": f"images/ref_{i}.png",
                "category": "test",
            }
        )

    (ref_dir / "index.json").write_text(json.dumps({"examples": examples}), encoding="utf-8")
    return ref_dir


def _get_single_run_dir(output_dir: Path) -> Path:
    runs = [p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    assert len(runs) == 1
    return runs[0]


@pytest.mark.asyncio
async def test_prompts_saved_when_enabled(tmp_path: Path):
    ref_dir = _make_reference_set(tmp_path)
    out_dir = tmp_path / "out"

    vlm = _SeqVLM(
        responses=[
            json.dumps({"selected_ids": ["ref_2"]}),  # retriever
            "Plan text\nRECOMMENDED_RATIO: 16:9",  # planner
            "Styled description",  # stylist
            json.dumps({"critic_suggestions": [], "revised_description": None}),  # critic
        ]
    )

    settings = Settings(
        output_dir=str(out_dir),
        reference_set_path=str(ref_dir),
        num_retrieval_examples=1,
        refinement_iterations=1,
        save_iterations=False,
        save_prompts=True,
    )

    pipeline = PaperBananaPipeline(settings=settings, vlm_client=vlm, image_gen_fn=_ImageGen())
    await pipeline.generate(
        GenerationInput(
            source_context="Methodology text",
            communicative_intent="Caption",
            diagram_type=DiagramType.METHODOLOGY,
        )
    )

    run_dir = _get_single_run_dir(out_dir)
    prompts_dir = run_dir / "prompts"
    assert prompts_dir.exists()

    for expected in (
        "retriever.txt",
        "planner.txt",
        "stylist.txt",
        "visualizer_diagram_iter_1.txt",
        "critic_iter_1.txt",
    ):
        assert (prompts_dir / expected).exists(), f"Missing {expected}"


@pytest.mark.asyncio
async def test_prompts_not_saved_when_disabled(tmp_path: Path):
    ref_dir = _make_reference_set(tmp_path)
    out_dir = tmp_path / "out"

    vlm = _SeqVLM(
        responses=[
            json.dumps({"selected_ids": ["ref_1"]}),
            "Plan text",
            "Styled description",
            json.dumps({"critic_suggestions": [], "revised_description": None}),
        ]
    )

    settings = Settings(
        output_dir=str(out_dir),
        reference_set_path=str(ref_dir),
        num_retrieval_examples=1,
        refinement_iterations=1,
        save_iterations=False,
        save_prompts=False,
    )

    pipeline = PaperBananaPipeline(settings=settings, vlm_client=vlm, image_gen_fn=_ImageGen())
    await pipeline.generate(
        GenerationInput(
            source_context="Methodology text",
            communicative_intent="Caption",
            diagram_type=DiagramType.METHODOLOGY,
        )
    )

    run_dir = _get_single_run_dir(out_dir)
    assert not (run_dir / "prompts").exists()

