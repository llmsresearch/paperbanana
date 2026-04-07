"""Tests for diagram IR SVG rendering helpers."""

from __future__ import annotations

from paperbanana.core.diagram_ir import _balanced_port, _select_ports, save_svg_from_ir
from paperbanana.core.types import DiagramIR, DiagramIREdge, DiagramIRGroup, DiagramIRNode


def test_save_svg_from_ir_renders_lanes_and_labels(tmp_path):
    ir = DiagramIR(
        title="Test Diagram",
        nodes=[
            DiagramIRNode(id="n1", label="Input", lane="Phase 1"),
            DiagramIRNode(id="n2", label="Encoder", lane="Phase 1"),
            DiagramIRNode(id="n3", label="Decoder", lane="Phase 2"),
        ],
        edges=[
            DiagramIREdge(source="n1", target="n2", label="tokens"),
            DiagramIREdge(source="n2", target="n3", label="features"),
        ],
        groups=[
            DiagramIRGroup(id="g1", label="Phase 1", node_ids=["n1", "n2"]),
            DiagramIRGroup(id="g2", label="Phase 2", node_ids=["n3"]),
        ],
    )
    out = tmp_path / "out.svg"
    save_svg_from_ir(ir, out)
    text = out.read_text(encoding="utf-8")
    assert "Phase 1" in text
    assert "Phase 2" in text
    assert "tokens" in text
    assert "features" in text


def test_save_svg_from_ir_uses_orthogonal_path_edges(tmp_path):
    ir = DiagramIR(
        title="Routing",
        nodes=[
            DiagramIRNode(id="n1", label="A", lane="L1"),
            DiagramIRNode(id="n2", label="B", lane="L1"),
            DiagramIRNode(id="n3", label="C", lane="L2"),
        ],
        edges=[
            DiagramIREdge(source="n1", target="n2", label="same-lane"),
            DiagramIREdge(source="n2", target="n3", label="cross-lane"),
        ],
    )
    out = tmp_path / "route.svg"
    save_svg_from_ir(ir, out)
    text = out.read_text(encoding="utf-8")
    # Orthogonal routing is rendered as path segments (M ... L ... L ... L ...).
    assert '<path d="M ' in text
    assert "same-lane" in text
    assert "cross-lane" in text


def test_select_ports_prefers_horizontal_for_same_lane():
    src_port, dst_port = _select_ports((100, 200), (500, 200), 240, 90, same_lane=True)
    # Right side of src, left side of dst.
    assert src_port[0] == 340
    assert dst_port[0] == 500


def test_select_ports_supports_vertical_for_cross_lane():
    src_port, dst_port = _select_ports((300, 120), (320, 420), 240, 90, same_lane=False)
    # Bottom of src to top of dst expected for downward relation.
    assert src_port[1] == 210
    assert dst_port[1] == 420


def test_balanced_port_distributes_along_side():
    p1 = _balanced_port((100, 100), "right", 240, 90, slot_idx=0, slot_count=3)
    p2 = _balanced_port((100, 100), "right", 240, 90, slot_idx=1, slot_count=3)
    p3 = _balanced_port((100, 100), "right", 240, 90, slot_idx=2, slot_count=3)
    assert p1[0] == 340 and p2[0] == 340 and p3[0] == 340
    assert p1[1] < p2[1] < p3[1]
