"""Tests for the `paperbanana regenerate` CLI command."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from paperbanana.cli import app
from paperbanana.core.types import (
    DiagramIR,
    DiagramIREdge,
    DiagramIRLocks,
    DiagramIRNode,
    GenerationOutput,
)

runner = CliRunner()


def _write_valid_ir(path: Path) -> None:
    ir = DiagramIR(
        title="Locked figure",
        nodes=[
            DiagramIRNode(id="n1", label="Input"),
            DiagramIRNode(id="n2", label="Output"),
        ],
        edges=[DiagramIREdge(source="n1", target="n2", label="flow")],
        locks=DiagramIRLocks(
            locked_node_ids=["n1"],
            locked_edge_refs=["n1->n2", "n1->n2:flow"],
        ),
    )
    path.write_text(ir.model_dump_json(indent=2), encoding="utf-8")


def test_regenerate_rejects_missing_ir_file(tmp_path):
    """`regenerate` exits with an error when --diagram-ir is missing."""
    input_path = tmp_path / "input.txt"
    input_path.write_text("Method text", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "regenerate",
            "--diagram-ir",
            str(tmp_path / "missing.json"),
            "--input",
            str(input_path),
            "--caption",
            "Locked caption",
        ],
    )

    assert result.exit_code == 1
    assert "Diagram IR not found" in result.output


def test_regenerate_runs_pipeline_and_passes_locks(tmp_path, monkeypatch):
    """`regenerate` parses DiagramIR locks and passes them to pipeline."""
    input_path = tmp_path / "input.txt"
    input_path.write_text("Method text", encoding="utf-8")
    ir_path = tmp_path / "diagram_ir.json"
    _write_valid_ir(ir_path)

    captured: dict[str, object] = {}

    class _FakePipeline:
        def __init__(self, settings=None, progress_callback=None, **kwargs):
            captured["settings"] = settings
            captured["progress_callback"] = progress_callback

        async def regenerate_from_ir(
            self,
            *,
            diagram_ir,
            source_context,
            caption,
            aspect_ratio,
            progress_callback=None,
        ):
            captured["diagram_ir"] = diagram_ir
            captured["source_context"] = source_context
            captured["caption"] = caption
            captured["aspect_ratio"] = aspect_ratio
            captured["regenerate_progress_callback"] = progress_callback

            out_path = tmp_path / "regen.png"
            out_path.write_bytes(b"fake")
            return GenerationOutput(
                image_path=str(out_path),
                description="done",
                iterations=[],
                metadata={"run_id": "run_regen"},
            )

    monkeypatch.setattr("paperbanana.cli.PaperBananaPipeline", _FakePipeline)
    monkeypatch.setattr(
        "paperbanana.core.source_loader.load_methodology_source",
        lambda *args, **kwargs: "Loaded source context",
    )

    result = runner.invoke(
        app,
        [
            "regenerate",
            "--diagram-ir",
            str(ir_path),
            "--input",
            str(input_path),
            "--caption",
            "Locked caption",
            "--aspect-ratio",
            "16:9",
        ],
    )

    assert result.exit_code == 0
    assert "Lock-aware IR Regeneration" in result.output
    assert "run_regen" in result.output

    parsed_ir = captured["diagram_ir"]
    assert isinstance(parsed_ir, DiagramIR)
    assert parsed_ir.locks.locked_node_ids == ["n1"]
    assert "n1->n2" in parsed_ir.locks.locked_edge_refs
    assert captured["source_context"] == "Loaded source context"
    assert captured["caption"] == "Locked caption"
    assert captured["aspect_ratio"] == "16:9"
