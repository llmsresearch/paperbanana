"""Tests for diagram IR and Graphviz rendering helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paperbanana.core.diagram_ir import DiagramIR, DiagramNode, build_dot_id_map
from paperbanana.vector.graphviz_render import (
    diagram_ir_to_dot,
    find_dot_executable,
    render_dot_to_file,
)


def test_diagram_ir_validates_edges_reference_nodes() -> None:
    with pytest.raises(ValueError, match="unknown source"):
        DiagramIR(
            nodes=[DiagramNode(id="a", label="A")],
            edges=[{"source": "x", "target": "a"}],
        )


def test_diagram_ir_to_dot_rankdir_and_edges() -> None:
    ir = DiagramIR(
        layout_direction="TB",
        nodes=[
            DiagramNode(id="enc", label="Encoder"),
            DiagramNode(id="dec", label="Decoder"),
        ],
        edges=[{"source": "enc", "target": "dec", "label": "latent"}],
    )
    dot = diagram_ir_to_dot(ir)
    assert "rankdir=TB" in dot
    assert "enc" in dot and "dec" in dot
    assert "latent" in dot


def test_build_dot_id_map_sanitizes_ids() -> None:
    nodes = [
        DiagramNode(id="a-b", label="X"),
        DiagramNode(id="c.d", label="Y"),
    ]
    m = build_dot_id_map(nodes)
    assert len(m) == 2
    assert len(set(m.values())) == 2


def test_render_dot_to_file_when_graphviz_available(tmp_path: Path) -> None:
    if not find_dot_executable():
        pytest.skip("Graphviz `dot` not on PATH")
    ir = DiagramIR(
        nodes=[
            DiagramNode(id="n1", label="One"),
            DiagramNode(id="n2", label="Two"),
        ],
        edges=[{"source": "n1", "target": "n2"}],
    )
    dot = diagram_ir_to_dot(ir)
    out = tmp_path / "out.svg"
    assert render_dot_to_file(dot, str(out), "svg") is True
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "<svg" in text


def test_diagram_ir_json_roundtrip() -> None:
    ir = DiagramIR(
        nodes=[DiagramNode(id="x", label="X")],
        edges=[],
    )
    data = json.loads(ir.model_dump_json())
    ir2 = DiagramIR.model_validate(data)
    assert ir2.nodes[0].id == "x"
