"""Diagram IR extraction and SVG export helpers."""

from __future__ import annotations

import base64
import html
import re
from pathlib import Path

from paperbanana.core.types import DiagramIR, DiagramIREdge, DiagramIRNode


def _select_ports(
    src_xy: tuple[int, int],
    dst_xy: tuple[int, int],
    node_w: int,
    node_h: int,
    same_lane: bool,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Select source/target connection ports (left/right/top/bottom)."""
    sx, sy = src_xy
    dx, dy = dst_xy
    src_cx = sx + node_w // 2
    src_cy = sy + node_h // 2
    dst_cx = dx + node_w // 2
    dst_cy = dy + node_h // 2
    dx_c = dst_cx - src_cx
    dy_c = dst_cy - src_cy

    if same_lane:
        if dx_c >= 0:
            return (sx + node_w, src_cy), (dx, dst_cy)
        return (sx, src_cy), (dx + node_w, dst_cy)

    # Cross-lane: prefer vertical ports if primarily vertical relation,
    # else horizontal ports when the nodes are strongly side-separated.
    if abs(dy_c) > abs(dx_c):
        if dy_c >= 0:
            return (src_cx, sy + node_h), (dst_cx, dy)
        return (src_cx, sy), (dst_cx, dy + node_h)

    if dx_c >= 0:
        return (sx + node_w, src_cy), (dx, dst_cy)
    return (sx, src_cy), (dx + node_w, dst_cy)


def _port_side(node_xy: tuple[int, int], port_xy: tuple[int, int], node_w: int, node_h: int) -> str:
    """Return side name for a selected port relative to node rect."""
    x, y = node_xy
    px, py = port_xy
    if px == x:
        return "left"
    if px == x + node_w:
        return "right"
    if py == y:
        return "top"
    if py == y + node_h:
        return "bottom"
    return "right"


def _balanced_port(
    node_xy: tuple[int, int],
    side: str,
    node_w: int,
    node_h: int,
    slot_idx: int,
    slot_count: int,
) -> tuple[int, int]:
    """Compute a balanced anchor point on a node side."""
    x, y = node_xy
    # Keep anchors away from corners.
    if slot_count < 1:
        slot_count = 1
    frac = (slot_idx + 1) / (slot_count + 1)
    if side == "left":
        return (x, y + int(node_h * frac))
    if side == "right":
        return (x + node_w, y + int(node_h * frac))
    if side == "top":
        return (x + int(node_w * frac), y)
    if side == "bottom":
        return (x + int(node_w * frac), y + node_h)
    return (x + node_w, y + node_h // 2)


def extract_diagram_ir(description: str, title: str = "PaperBanana Diagram") -> DiagramIR:
    """Build a simple ordered IR from a textual diagram description.

    This heuristic parser favors predictable editability over perfect semantic parsing:
    - numbered lines / bullets become nodes
    - edges connect nodes in order
    """
    lines = [ln.strip() for ln in description.splitlines()]
    candidates: list[str] = []
    for ln in lines:
        if not ln:
            continue
        cleaned = re.sub(r"^(\d+[\).\s-]+|[-*]\s+)", "", ln).strip()
        if len(cleaned) < 3:
            continue
        if cleaned.lower().startswith(("note:", "legend:", "style:", "color:")):
            continue
        candidates.append(cleaned)

    # De-duplicate while preserving order.
    seen: set[str] = set()
    labels: list[str] = []
    for c in candidates:
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        labels.append(c)
        if len(labels) >= 12:
            break

    if not labels:
        labels = ["Input", "Core method", "Output"]

    nodes = [
        DiagramIRNode(
            id=f"n{i + 1}",
            label=(label if len(label) <= 72 else (label[:69] + "...")),
        )
        for i, label in enumerate(labels)
    ]
    edges = [
        DiagramIREdge(source=nodes[i].id, target=nodes[i + 1].id)
        for i in range(max(0, len(nodes) - 1))
    ]
    return DiagramIR(title=title, nodes=nodes, edges=edges)


def save_svg_from_ir(diagram_ir: DiagramIR, output_path: str | Path) -> Path:
    """Render an editable SVG from DiagramIR."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width = 1600
    height = 900
    margin_x = 80
    margin_y = 120
    node_w = 240
    node_h = 90
    lane_gap = 28
    lane_h = 140
    top_y = margin_y + 56
    left_gutter = 170
    canvas_w = width - margin_x - left_gutter - 40

    parts: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">',
        '<path d="M0,0 L0,6 L9,3 z" fill="#4b5563" />',
        "</marker>",
        "</defs>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_x}" y="70" font-family="Arial, sans-serif" font-size="34" '
        'font-weight="700" fill="#111827">'
        f"{html.escape(diagram_ir.title)}</text>",
    ]

    # Determine lane order from groups, then node lane labels, then fallback.
    lane_order: list[str] = []
    for g in diagram_ir.groups:
        if g.label not in lane_order:
            lane_order.append(g.label)
    for n in diagram_ir.nodes:
        ln = (n.lane or "").strip()
        if ln and ln not in lane_order:
            lane_order.append(ln)
    if not lane_order:
        lane_order = ["Main"]

    lane_colors = [
        "#eff6ff",
        "#f5f3ff",
        "#ecfeff",
        "#f0fdf4",
        "#fff7ed",
    ]
    lane_y: dict[str, int] = {}
    for i, lane in enumerate(lane_order):
        y = top_y + i * (lane_h + lane_gap)
        lane_y[lane] = y
        bg = lane_colors[i % len(lane_colors)]
        parts.append(
            f'<rect x="{margin_x}" y="{y}" width="{width - 2 * margin_x}" '
            f'height="{lane_h}" rx="14" '
            f'fill="{bg}" stroke="#cbd5e1" stroke-width="1.5"/>'
        )
        parts.append(
            f'<text x="{margin_x + 16}" y="{y + 34}" '
            'font-family="Arial, sans-serif" font-size="20" '
            'font-weight="700" fill="#334155">'
            f"{html.escape(lane)}</text>"
        )

    # Place nodes in columns within each lane.
    lane_nodes: dict[str, list[DiagramIRNode]] = {k: [] for k in lane_order}
    for node in diagram_ir.nodes:
        lane = (node.lane or "").strip() or lane_order[0]
        if lane not in lane_nodes:
            lane_nodes[lane] = []
            lane_y[lane] = top_y + len(lane_y) * (lane_h + lane_gap)
        lane_nodes[lane].append(node)

    node_pos: dict[str, tuple[int, int]] = {}
    max_cols = max((len(v) for v in lane_nodes.values()), default=1)
    step_x = max(260, (canvas_w - node_w) // max(1, max_cols - 1))
    for lane in lane_order:
        y = lane_y[lane] + 38
        for i, node in enumerate(lane_nodes.get(lane, [])):
            x = margin_x + left_gutter + i * step_x
            node_pos[node.id] = (x, y)
            parts.append(
                f'<rect x="{x}" y="{y}" width="{node_w}" height="{node_h}" rx="12" '
                'fill="#f8fafc" stroke="#94a3b8" stroke-width="2"/>'
            )
            parts.append(
                f'<text x="{x + 16}" y="{y + 34}" font-family="Arial, sans-serif" font-size="18" '
                'fill="#111827">'
                f"{html.escape(node.label)}</text>"
            )

    edge_label_offsets: dict[tuple[int, int], int] = {}
    route_channel_counts: dict[tuple[int, int], int] = {}
    lane_index = {lane: i for i, lane in enumerate(lane_order)}
    lane_channel_y = {
        lane: lane_y[lane] + 22 for lane in lane_order
    }  # header band, clear of node boxes
    bus_base_x = margin_x + left_gutter - 56
    bus_step = 18
    lane_pair_bus: dict[tuple[str, str], int] = {}

    # Pre-compute occupancy counts for each node side.
    node_lookup = {n.id: n for n in diagram_ir.nodes}
    node_side_total: dict[tuple[str, str], int] = {}
    node_side_used: dict[tuple[str, str], int] = {}
    edge_ports: dict[tuple[str, str], tuple[tuple[int, int], tuple[int, int], str, str]] = {}
    for edge in diagram_ir.edges:
        src = node_pos.get(edge.source)
        dst = node_pos.get(edge.target)
        src_node = node_lookup.get(edge.source)
        dst_node = node_lookup.get(edge.target)
        if not src or not dst or not src_node or not dst_node:
            continue
        src_lane = (src_node.lane or "").strip()
        dst_lane = (dst_node.lane or "").strip()
        same_lane = (not src_lane and not dst_lane) or src_lane == dst_lane
        src_port, dst_port = _select_ports(src, dst, node_w, node_h, same_lane=same_lane)
        src_side = _port_side(src, src_port, node_w, node_h)
        dst_side = _port_side(dst, dst_port, node_w, node_h)
        node_side_total[(edge.source, src_side)] = (
            node_side_total.get((edge.source, src_side), 0) + 1
        )
        node_side_total[(edge.target, dst_side)] = (
            node_side_total.get((edge.target, dst_side), 0) + 1
        )
        edge_ports[(edge.source, edge.target)] = (src_port, dst_port, src_side, dst_side)

    for edge in diagram_ir.edges:
        src = node_pos.get(edge.source)
        dst = node_pos.get(edge.target)
        if not src or not dst:
            continue
        src_node = node_lookup.get(edge.source)
        dst_node = node_lookup.get(edge.target)
        if not src_node or not dst_node:
            continue
        src_lane = (src_node.lane or "").strip() if src_node else ""
        dst_lane = (dst_node.lane or "").strip() if dst_node else ""
        same_lane = (not src_lane and not dst_lane) or src_lane == dst_lane
        _raw_src, _raw_dst, src_side, dst_side = edge_ports.get(
            (edge.source, edge.target),
            ((0, 0), (0, 0), "right", "left"),
        )

        src_slot = node_side_used.get((edge.source, src_side), 0)
        src_total = node_side_total.get((edge.source, src_side), 1)
        node_side_used[(edge.source, src_side)] = src_slot + 1
        dst_slot = node_side_used.get((edge.target, dst_side), 0)
        dst_total = node_side_total.get((edge.target, dst_side), 1)
        node_side_used[(edge.target, dst_side)] = dst_slot + 1

        x1, y1 = _balanced_port(src, src_side, node_w, node_h, src_slot, src_total)
        x2, y2 = _balanced_port(dst, dst_side, node_w, node_h, dst_slot, dst_total)

        # Obstacle-aware orthogonal routing:
        # - Same lane: route via lane header channel (y) to avoid node boxes.
        # - Cross lane: route via a left-side vertical bus channel (x) to avoid node boxes.
        if src_lane and dst_lane and src_lane != dst_lane:
            key = (src_lane, dst_lane)
            if key not in lane_pair_bus:
                src_i = lane_index.get(src_lane, 0)
                dst_i = lane_index.get(dst_lane, 0)
                spread = abs(dst_i - src_i) + len(lane_pair_bus)
                lane_pair_bus[key] = bus_base_x - (spread * bus_step)
            bus_x = lane_pair_bus[key]
            # If using vertical ports, route through top bus channel.
            if x1 != (src[0] + node_w) and x1 != src[0]:
                bus_y = top_y - 18
                d = f"M {x1} {y1} L {x1} {bus_y} L {bus_x} {bus_y} L {bus_x} {y2} L {x2} {y2}"
            else:
                exit_x = x1 + 16 if x1 <= src[0] + node_w // 2 else x1 - 16
                pre_x2 = x2 - 16 if x2 >= dst[0] + node_w // 2 else x2 + 16
                d = (
                    f"M {x1} {y1} "
                    f"L {exit_x} {y1} "
                    f"L {bus_x} {y1} "
                    f"L {bus_x} {y2} "
                    f"L {pre_x2} {y2} "
                    f"L {x2} {y2}"
                )
            label_x = bus_x
            label_y = (y1 + y2) // 2 - 8
        else:
            lane = src_lane or dst_lane or lane_order[0]
            base_ch = lane_channel_y.get(lane, y1 - 26)
            # Offset channels for dense same-lane edges.
            channel_idx = route_channel_counts.get((lane_index.get(lane, 0), base_ch), 0)
            route_channel_counts[(lane_index.get(lane, 0), base_ch)] = channel_idx + 1
            ch_y = base_ch - (channel_idx * 12)
            # Same-lane routing using selected side.
            exit_x = x1 + 16 if x1 <= src[0] + node_w // 2 else x1 - 16
            pre_x2 = x2 - 16 if x2 >= dst[0] + node_w // 2 else x2 + 16
            d = (
                f"M {x1} {y1} "
                f"L {exit_x} {y1} "
                f"L {exit_x} {ch_y} "
                f"L {pre_x2} {ch_y} "
                f"L {pre_x2} {y2} "
                f"L {x2} {y2}"
            )
            label_x = (x1 + x2) // 2
            label_y = ch_y - 6

        parts.append(
            f'<path d="{d}" fill="none" stroke="#4b5563" '
            'stroke-width="2.5" marker-end="url(#arrow)"/>'
        )
        if edge.label:
            key = (label_x, label_y)
            bump = edge_label_offsets.get(key, 0)
            edge_label_offsets[key] = bump + 1
            my = label_y - (bump * 16)
            parts.append(
                f'<text x="{label_x}" y="{my}" font-family="Arial, sans-serif" font-size="14" '
                'text-anchor="middle" fill="#334155">'
                f"{html.escape(edge.label)}</text>"
            )

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")
    return output_path


def save_raster_wrapped_svg(image_path: str | Path, output_path: str | Path) -> Path:
    """Wrap a raster image in an SVG container to support svg output for all modes."""
    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ext = image_path.suffix.lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext or 'png'}"
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    content = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" '
        'viewBox="0 0 1600 900">\n'
        f'  <image href="data:{mime};base64,{data}" x="0" y="0" width="1600" height="900"/>\n'
        "</svg>\n"
    )
    output_path.write_text(content, encoding="utf-8")
    return output_path
