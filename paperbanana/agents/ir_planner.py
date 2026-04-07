"""IR Planner Agent: produces structured diagram IR JSON."""

from __future__ import annotations

import json
import re

import structlog

from paperbanana.agents.base import BaseAgent
from paperbanana.core.types import DiagramIR, DiagramIREdge, DiagramIRGroup, DiagramIRNode
from paperbanana.providers.base import VLMProvider

logger = structlog.get_logger()


class IRPlannerAgent(BaseAgent):
    """Generate editable diagram IR from context + caption + description."""

    def __init__(
        self, vlm_provider: VLMProvider, prompt_dir: str = "prompts", prompt_recorder=None
    ):
        super().__init__(vlm_provider, prompt_dir, prompt_recorder=prompt_recorder)

    @property
    def agent_name(self) -> str:
        return "ir_planner"

    async def run(
        self,
        source_context: str,
        caption: str,
        styled_description: str,
    ) -> DiagramIR:
        template = self.load_prompt("diagram")
        prompt = self.format_prompt(
            template,
            prompt_label="ir_planner",
            source_context=source_context,
            caption=caption,
            styled_description=styled_description,
        )
        raw = await self.vlm.generate(
            prompt=prompt,
            temperature=0.2,
            max_tokens=3072,
            response_format="json",
        )
        return self._parse_ir(raw, caption=caption)

    @classmethod
    def _parse_ir(cls, raw: str, caption: str) -> DiagramIR:
        data = cls._extract_json(raw)
        title = str(data.get("title") or caption or "Methodology Diagram").strip()
        nodes_raw = data.get("nodes") or []
        edges_raw = data.get("edges") or []
        groups_raw = data.get("groups") or []

        nodes: list[DiagramIRNode] = []
        seen_ids: set[str] = set()
        for i, n in enumerate(nodes_raw):
            if not isinstance(n, dict):
                continue
            nid = str(n.get("id") or f"n{i + 1}").strip() or f"n{i + 1}"
            if nid in seen_ids:
                nid = f"{nid}_{i + 1}"
            seen_ids.add(nid)
            label = str(n.get("label") or "").strip()
            if not label:
                continue
            lane = n.get("lane")
            lane_str = str(lane).strip() if lane is not None else None
            nodes.append(DiagramIRNode(id=nid, label=label, lane=lane_str or None))

        node_ids_set = {n.id for n in nodes}
        if not nodes:
            nodes = [
                DiagramIRNode(id="n1", label="Input"),
                DiagramIRNode(id="n2", label="Core method"),
                DiagramIRNode(id="n3", label="Output"),
            ]
            seen_ids = {"n1", "n2", "n3"}
            node_ids_set = {"n1", "n2", "n3"}

        edges: list[DiagramIREdge] = []
        for e in edges_raw:
            if not isinstance(e, dict):
                continue
            src = str(e.get("source") or "").strip()
            dst = str(e.get("target") or "").strip()
            if src in node_ids_set and dst in node_ids_set:
                label = e.get("label")
                edges.append(
                    DiagramIREdge(
                        source=src,
                        target=dst,
                        label=str(label).strip() if label is not None else None,
                    )
                )

        if not edges and len(nodes) > 1:
            edges = [
                DiagramIREdge(source=nodes[i].id, target=nodes[i + 1].id)
                for i in range(len(nodes) - 1)
            ]

        groups: list[DiagramIRGroup] = []
        valid_group_ids: set[str] = set()
        for i, g in enumerate(groups_raw):
            if not isinstance(g, dict):
                continue
            gid = str(g.get("id") or f"g{i + 1}").strip() or f"g{i + 1}"
            if gid in valid_group_ids:
                gid = f"{gid}_{i + 1}"
            label = str(g.get("label") or "").strip()
            if not label:
                continue
            node_ids = [str(nid).strip() for nid in (g.get("node_ids") or [])]
            node_ids = [nid for nid in node_ids if nid in node_ids_set]
            valid_group_ids.add(gid)
            groups.append(DiagramIRGroup(id=gid, label=label, node_ids=node_ids))

        # If groups include node membership and nodes are missing lane labels,
        # infer lane from the first containing group.
        node_to_group: dict[str, str] = {}
        for g in groups:
            for nid in g.node_ids:
                node_to_group.setdefault(nid, g.label)
        for n in nodes:
            if not n.lane and n.id in node_to_group:
                n.lane = node_to_group[n.id]

        return DiagramIR(title=title, nodes=nodes, edges=edges, groups=groups)

    @staticmethod
    def _extract_json(raw: str) -> dict:
        text = raw.strip()
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if fenced:
            snippet = fenced.group(1)
            try:
                data = json.loads(snippet)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass

        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            snippet = text[brace_start : brace_end + 1]
            data = json.loads(snippet)
            if isinstance(data, dict):
                return data

        raise ValueError("IR planner did not return valid JSON object")
