"""Tests for IRPlannerAgent JSON parsing."""

from __future__ import annotations

from paperbanana.agents.ir_planner import IRPlannerAgent


def test_ir_planner_parse_basic_json():
    raw = """
{
  "title": "Overview",
  "nodes": [
    {"id": "n1", "label": "Input"},
    {"id": "n2", "label": "Encoder"}
  ],
  "edges": [
    {"source": "n1", "target": "n2", "label": "features"}
  ]
}
"""
    ir = IRPlannerAgent._parse_ir(raw, caption="c")
    assert ir.title == "Overview"
    assert len(ir.nodes) == 2
    assert ir.edges[0].source == "n1"
    assert ir.edges[0].target == "n2"


def test_ir_planner_parse_fenced_json_and_invalid_edges():
    raw = """```json
{"title":"T","nodes":[{"id":"n1","label":"A"},{"id":"n2","label":"B"}],"edges":[{"source":"n1","target":"bad"}]}
```"""
    ir = IRPlannerAgent._parse_ir(raw, caption="c")
    # invalid edge target is filtered, then fallback linear edge is added
    assert len(ir.nodes) == 2
    assert len(ir.edges) == 1
    assert ir.edges[0].source == "n1"
    assert ir.edges[0].target == "n2"


def test_ir_planner_parse_groups_and_infer_lane():
    raw = """
{
  "title": "Grouped",
  "nodes": [
    {"id": "n1", "label": "Input"},
    {"id": "n2", "label": "Model"}
  ],
  "edges": [{"source":"n1","target":"n2"}],
  "groups": [{"id":"g1","label":"Phase A","node_ids":["n1","n2"]}]
}
"""
    ir = IRPlannerAgent._parse_ir(raw, caption="c")
    assert len(ir.groups) == 1
    assert ir.groups[0].label == "Phase A"
    assert ir.nodes[0].lane == "Phase A"
    assert ir.nodes[1].lane == "Phase A"
