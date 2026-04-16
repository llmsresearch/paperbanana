"""Render Diagram IR to SVG/PDF via the Graphviz `dot` binary."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import structlog

from paperbanana.core.types import DiagramIR

logger = structlog.get_logger()

_SHAPE_MAP = {
    "box": "box",
    "rounded": "box",
    "ellipse": "ellipse",
    "cylinder": "cylinder",
    "plain": "plaintext",
}


def _sanitize_dot_name(raw: str, used: set[str]) -> str:
    """Map a node id to a valid, unique Graphviz identifier."""
    s = raw.strip()
    base = re.sub(r"[^a-zA-Z0-9_]", "_", s)
    if not base:
        base = "node"
    if base[0].isdigit():
        base = "n_" + base
    name = base
    i = 0
    while name in used:
        i += 1
        name = f"{base}_{i}"
    used.add(name)
    return name


def _build_dot_id_map(ir: DiagramIR) -> dict[str, str]:
    used: set[str] = set()
    return {n.id: _sanitize_dot_name(n.id, used) for n in ir.nodes}


def find_dot_executable() -> str | None:
    """Return path to `dot` if available on PATH."""
    return shutil.which("dot")


def _escape_dot_label(text: str) -> str:
    s = text.replace("\\", "\\\\").replace('"', '\\"')
    s = re.sub(r"[\r\n]+", " ", s)
    return s


def diagram_ir_to_dot(ir: DiagramIR) -> str:
    """Convert Diagram IR to a Graphviz digraph (UTF-8)."""
    id_map = _build_dot_id_map(ir)
    rankdir = getattr(ir, "layout_direction", "LR")

    lines: list[str] = [
        "digraph G {",
        f'  graph [rankdir={rankdir}, bgcolor="white", fontname="Helvetica"];',
        '  node [fontname="Helvetica", fontsize=10];',
        '  edge [fontname="Helvetica", fontsize=9];',
    ]

    # Nodes not assigned to any group
    grouped: set[str] = set()
    for g in ir.groups:
        grouped.update(g.node_ids)

    for n in ir.nodes:
        if n.id in grouped:
            continue
        gid = id_map[n.id]
        node_shape = getattr(n, "shape", "rounded")
        shape = _SHAPE_MAP.get(node_shape, "box")
        style = "rounded,filled" if node_shape == "rounded" else "filled"
        fill = "#f8f9fa"
        lbl = _escape_dot_label(n.label)
        lines.append(
            f'  {gid} [label="{lbl}", shape={shape}, style="{style}", fillcolor="{fill}"];'
        )

    for g in ir.groups:
        if not g.node_ids:
            continue
        safe_g = re.sub(r"[^a-zA-Z0-9_]", "_", g.id.strip())
        if not safe_g:
            safe_g = "group"
        if safe_g[0].isdigit():
            safe_g = "g_" + safe_g
        if not safe_g.startswith("cluster"):
            safe_g = "cluster_" + safe_g
        lines.append(f"  subgraph {safe_g} {{")
        glabel = _escape_dot_label(g.label or g.id)
        lines.append(f'    label="{glabel}";')
        lines.append('    style="rounded";')
        lines.append('    color="#cccccc";')
        lines.append('    bgcolor="#fafafa";')
        for n in ir.nodes:
            if n.id not in g.node_ids:
                continue
            gid = id_map[n.id]
            node_shape = getattr(n, "shape", "rounded")
            shape = _SHAPE_MAP.get(node_shape, "box")
            style = "rounded,filled" if node_shape == "rounded" else "filled"
            fill = "#eef2ff"
            lbl = _escape_dot_label(n.label)
            lines.append(
                f'    {gid} [label="{lbl}", shape={shape}, style="{style}", fillcolor="{fill}"];'
            )
        lines.append("  }")

    for e in ir.edges:
        a = id_map[e.source]
        b = id_map[e.target]
        if e.label:
            lab = _escape_dot_label(e.label)
            lines.append(f'  {a} -> {b} [label="{lab}"];')
        else:
            lines.append(f"  {a} -> {b};")

    lines.append("}")
    return "\n".join(lines) + "\n"


def render_dot_to_file(dot_source: str, output_path: str | Path, fmt: str) -> bool:
    """Run `dot -T{fmt}` to write output_path. fmt is 'svg' or 'pdf'."""
    dot_bin = find_dot_executable()
    if not dot_bin:
        logger.warning("Graphviz `dot` not found on PATH; skipping vector render")
        return False
    fmt = fmt.lower()
    if fmt not in ("svg", "pdf"):
        raise ValueError("fmt must be svg or pdf")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".dot",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(dot_source)
            tmp_path = tmp.name
        assert tmp_path is not None
        result = subprocess.run(
            [dot_bin, f"-T{fmt}", "-o", str(out), tmp_path],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if result.returncode != 0:
            logger.warning(
                "dot failed",
                returncode=result.returncode,
                stderr=(result.stderr or "")[:500],
            )
            return False
        return out.exists() and out.stat().st_size > 0
    except OSError as e:
        logger.warning("dot execution failed", error=str(e))
        return False
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
