"""Integration and smoke tests for lock-aware regeneration from DiagramIR."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("PIL", reason="PIL/Pillow required for pipeline image mock")

from paperbanana.core.config import Settings
from paperbanana.core.pipeline import PaperBananaPipeline
from paperbanana.core.types import (
    CritiqueResult,
    DiagramIR,
    DiagramIREdge,
    DiagramIRLocks,
    DiagramIRNode,
)


class _MockVLM:
    name = "mock-vlm"
    model_name = "mock-model"

    async def generate(self, *args, **kwargs):
        return "unused"


class _MockImageGen:
    name = "mock-image-gen"
    model_name = "mock-image-model"

    async def generate(self, *args, **kwargs):
        return Image.new("RGB", (128, 128), color=(255, 255, 255))


def _make_locked_ir() -> DiagramIR:
    return DiagramIR(
        title="Locked diagram",
        nodes=[
            DiagramIRNode(id="n1", label="Locked Input"),
            DiagramIRNode(id="n2", label="Output"),
        ],
        edges=[DiagramIREdge(source="n1", target="n2", label="flow")],
        locks=DiagramIRLocks(
            locked_node_ids=["n1"],
            locked_edge_refs=["n1->n2", "n1->n2:flow"],
        ),
    )


@pytest.mark.asyncio
async def test_regenerate_from_ir_writes_metadata_and_artifacts(tmp_path):
    """Pipeline regeneration writes run inputs/artifacts and lock metadata."""
    settings = Settings(
        output_dir=str(tmp_path / "outputs"),
        reference_set_path=str(tmp_path / "empty_refs"),
        refinement_iterations=1,
        save_iterations=True,
    )
    pipeline = PaperBananaPipeline(
        settings=settings,
        vlm_client=_MockVLM(),
        image_gen_fn=_MockImageGen(),
    )
    ir = _make_locked_ir()

    result = await pipeline.regenerate_from_ir(
        diagram_ir=ir,
        source_context="Method context",
        caption="Figure caption",
        aspect_ratio="4:3",
    )

    assert Path(result.image_path).exists()
    assert result.metadata["regeneration"]["mode"] == "diagram_ir_locked"
    assert result.metadata["regeneration"]["locked_nodes"] == 1
    assert result.metadata["regeneration"]["locked_edges"] == 2
    assert result.description.count("[LOCKED]") >= 1

    run_dir = Path(settings.output_dir) / result.metadata["run_id"]
    assert (run_dir / "run_input.json").exists()
    assert (run_dir / "diagram_ir_input.json").exists()
    run_input = json.loads((run_dir / "run_input.json").read_text(encoding="utf-8"))
    assert run_input["regeneration_mode"] == "diagram_ir_locked"
    assert run_input["locked_nodes"] == ["n1"]


@pytest.mark.asyncio
async def test_regenerate_smoke_reapplies_locks_when_critic_suggests_locked_edit(
    tmp_path, monkeypatch
):
    """Locked constraints remain in prompts after critic proposes conflicting edits."""
    settings = Settings(
        output_dir=str(tmp_path / "outputs"),
        reference_set_path=str(tmp_path / "empty_refs"),
        refinement_iterations=2,
        save_iterations=False,
    )
    pipeline = PaperBananaPipeline(
        settings=settings,
        vlm_client=_MockVLM(),
        image_gen_fn=_MockImageGen(),
    )
    ir = _make_locked_ir()
    seen_descriptions: list[str] = []

    async def _fake_visualizer_run(
        *,
        description,
        diagram_type,
        raw_data,
        iteration,
        seed,
        aspect_ratio,
        vector_formats,
    ):
        seen_descriptions.append(description)
        out = Path(settings.output_dir) / pipeline.run_id / f"diagram_iter_{iteration}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), color=(120, 120, 120)).save(out)
        return str(out)

    critique_calls = {"n": 0}

    async def _fake_critic_run(**kwargs):
        critique_calls["n"] += 1
        if critique_calls["n"] == 1:
            return CritiqueResult(
                critic_suggestions=["Change locked node n1 label to Hacked"],
                revised_description="Please rename n1 to Hacked.",
            )
        return CritiqueResult(critic_suggestions=[], revised_description=None)

    monkeypatch.setattr(pipeline.visualizer, "run", _fake_visualizer_run)
    monkeypatch.setattr(pipeline.critic, "run", _fake_critic_run)

    result = await pipeline.regenerate_from_ir(
        diagram_ir=ir,
        source_context="Method context",
        caption="Figure caption",
    )

    assert Path(result.image_path).exists()
    assert len(result.iterations) == 2
    assert len(seen_descriptions) == 2
    assert "n1: Locked Input [LOCKED]" in seen_descriptions[0]
    assert "Please rename n1 to Hacked." in seen_descriptions[1]
    assert "n1: Locked Input [LOCKED]" in seen_descriptions[1]
