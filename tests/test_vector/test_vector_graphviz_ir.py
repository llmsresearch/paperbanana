"""Tests for diagram IR and Graphviz rendering helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paperbanana.config.fonts import FontConfig, get_legacy_font_config
from paperbanana.core.types import DiagramIR, DiagramIREdge, DiagramIRNode
from paperbanana.vector.graphviz_render import (
    diagram_ir_to_dot,
    find_dot_executable,
    render_dot_to_file,
)


def test_diagram_ir_to_dot_default_rankdir_and_edges() -> None:
    ir = DiagramIR(
        title="Demo",
        nodes=[
            DiagramIRNode(id="enc", label="Encoder"),
            DiagramIRNode(id="dec", label="Decoder"),
        ],
        edges=[DiagramIREdge(source="enc", target="dec", label="latent")],
    )
    dot = diagram_ir_to_dot(ir)
    assert "rankdir=LR" in dot
    assert "enc" in dot and "dec" in dot
    assert "latent" in dot


def test_diagram_ir_to_dot_sanitizes_node_ids() -> None:
    ir = DiagramIR(
        title="Sanitize IDs",
        nodes=[
            DiagramIRNode(id="a-b", label="X"),
            DiagramIRNode(id="c.d", label="Y"),
        ],
        edges=[DiagramIREdge(source="a-b", target="c.d")],
    )
    dot = diagram_ir_to_dot(ir)
    assert "a_b" in dot
    assert "c_d" in dot


def test_render_dot_to_file_when_graphviz_available(tmp_path: Path) -> None:
    if not find_dot_executable():
        pytest.skip("Graphviz `dot` not on PATH")
    ir = DiagramIR(
        title="Render",
        nodes=[
            DiagramIRNode(id="n1", label="One"),
            DiagramIRNode(id="n2", label="Two"),
        ],
        edges=[DiagramIREdge(source="n1", target="n2")],
    )
    dot = diagram_ir_to_dot(ir)
    out = tmp_path / "out.svg"
    assert render_dot_to_file(dot, str(out), "svg") is True
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "<svg" in text


def test_diagram_ir_json_roundtrip() -> None:
    ir = DiagramIR(
        title="Roundtrip",
        nodes=[DiagramIRNode(id="x", label="X")],
        edges=[],
    )
    data = json.loads(ir.model_dump_json())
    ir2 = DiagramIR.model_validate(data)
    assert ir2.nodes[0].id == "x"


def test_diagram_ir_to_dot_uses_default_tahoma_font() -> None:
    ir = DiagramIR(
        title="Font Test",
        nodes=[DiagramIRNode(id="n1", label="Node")],
        edges=[],
    )
    dot = diagram_ir_to_dot(ir)
    assert "Tahoma" in dot
    assert "Helvetica" in dot
    assert "Arial" in dot


def test_diagram_ir_to_dot_with_custom_font_config() -> None:
    ir = DiagramIR(
        title="Font Test",
        nodes=[DiagramIRNode(id="n1", label="Node")],
        edges=[],
    )
    config = get_legacy_font_config()
    dot = diagram_ir_to_dot(ir, font_config=config)
    assert "Helvetica,Arial" in dot
    assert "Tahoma" not in dot


def test_diagram_ir_to_dot_font_config_in_graph_node_and_edge() -> None:
    ir = DiagramIR(
        title="Font Test",
        nodes=[DiagramIRNode(id="n1", label="Node")],
        edges=[],
    )
    config = FontConfig(primary_fonts=["CustomFont"])
    dot = diagram_ir_to_dot(ir, font_config=config)
    assert 'fontname="CustomFont,Helvetica,Arial,sans-serif"' in dot
    lines = dot.split("\n")
    fontname_lines = [line for line in lines if "fontname=" in line]
    assert len(fontname_lines) >= 3
