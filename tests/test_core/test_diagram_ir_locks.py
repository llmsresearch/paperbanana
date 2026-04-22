"""Tests for DiagramIR lock metadata and lock-aware formatting."""

from __future__ import annotations

import pytest

from paperbanana.core.diagram_ir import format_diagram_ir_for_regeneration
from paperbanana.core.types import DiagramIR, DiagramIREdge, DiagramIRLocks, DiagramIRNode


def test_diagram_ir_locks_validate_known_references() -> None:
    ir = DiagramIR(
        title="Locked",
        nodes=[
            DiagramIRNode(id="n1", label="Input"),
            DiagramIRNode(id="n2", label="Output"),
        ],
        edges=[DiagramIREdge(id="e1", source="n1", target="n2", label="flow")],
        locks=DiagramIRLocks(
            locked_node_ids=["n1"],
            locked_edge_refs=["e1", "n1->n2", "n1->n2:flow"],
        ),
    )
    assert ir.locks.locked_node_ids == ["n1"]


def test_diagram_ir_locks_reject_unknown_references() -> None:
    with pytest.raises(ValueError):
        DiagramIR(
            title="Bad lock",
            nodes=[DiagramIRNode(id="n1", label="Only")],
            edges=[],
            locks=DiagramIRLocks(locked_node_ids=["missing"]),
        )


def test_format_diagram_ir_for_regeneration_includes_lock_hints() -> None:
    ir = DiagramIR(
        title="Format Test",
        nodes=[
            DiagramIRNode(id="n1", label="Encode"),
            DiagramIRNode(id="n2", label="Decode"),
        ],
        edges=[DiagramIREdge(source="n1", target="n2")],
        locks=DiagramIRLocks(locked_node_ids=["n1"], locked_edge_refs=["n1->n2"]),
    )

    text = format_diagram_ir_for_regeneration(ir)
    assert "n1: Encode [LOCKED]" in text
    assert "Hard constraints:" in text
