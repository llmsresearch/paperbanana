"""Deterministic poster layout solver.

The content agent decides *what* goes on the poster and in which reading
order; this module decides *where*. Geometry is never produced by a
model, so layout invariants (no overlap, page bounds, margins) hold by
construction.

Algorithm: the ``header`` panel spans the full content width at the top;
remaining panels fill ``columns`` columns in reading order, each column
receiving a contiguous run of panels whose cumulative weight approaches
the per-column average. Within a column, panel heights are proportional
to their weights.
"""

from __future__ import annotations

from paperbanana.poster.types import BBox, Panel, PosterIR

#: Fraction of page height reserved for the header band by default.
DEFAULT_HEADER_FRAC = 0.13


class LayoutError(ValueError):
    """Raised when the requested layout is geometrically impossible."""


def body_column_height_mm(ir: PosterIR, header_frac: float = DEFAULT_HEADER_FRAC) -> float:
    """Vertical space available to each body column, in mm."""
    content_h = ir.size.height_mm - 2 * ir.margin_mm
    header_h = content_h * header_frac
    body_top = ir.margin_mm + header_h + ir.gutter_mm
    return ir.size.height_mm - ir.margin_mm - body_top


def assign_bboxes(
    ir: PosterIR,
    header_frac: float = DEFAULT_HEADER_FRAC,
    column_capacity_mm: float | None = None,
) -> PosterIR:
    """Return a copy of the IR with every panel placed.

    Args:
        ir: Poster IR; panel ``order`` and ``weight`` drive placement.
        header_frac: Vertical fraction of the page given to the header band.
        column_capacity_mm: When set, panel weights are treated as physical
            heights (mm) and columns are packed first-fit against this
            capacity instead of weight-balanced — used by the content
            fitter, where weights are measured content heights.

    Raises:
        LayoutError: If there is no header panel, no body panels, or the
            page/margins/columns leave no positive content area.
    """
    if not 0.0 < header_frac < 0.5:
        raise LayoutError(f"header_frac must be in (0, 0.5), got {header_frac}")

    panels = sorted((p.model_copy(deep=True) for p in ir.panels), key=lambda p: p.order)
    headers = [p for p in panels if p.role == "header"]
    if len(headers) != 1:
        raise LayoutError(f"layout requires exactly one header panel, found {len(headers)}")
    header = headers[0]
    body = [p for p in panels if p.role != "header"]
    if not body:
        raise LayoutError("layout requires at least one non-header panel")

    content_w = ir.size.width_mm - 2 * ir.margin_mm
    content_h = ir.size.height_mm - 2 * ir.margin_mm
    if content_w <= 0 or content_h <= 0:
        raise LayoutError(
            f"margins {ir.margin_mm}mm leave no content area on a "
            f"{ir.size.width_mm}x{ir.size.height_mm}mm page"
        )

    header_h = content_h * header_frac
    header.bbox = BBox(x_mm=ir.margin_mm, y_mm=ir.margin_mm, w_mm=content_w, h_mm=header_h)
    header.column = None

    body_top = ir.margin_mm + header_h + ir.gutter_mm
    body_h = ir.size.height_mm - ir.margin_mm - body_top
    col_w = (content_w - (ir.columns - 1) * ir.gutter_mm) / ir.columns
    if col_w <= 0 or body_h <= 0:
        raise LayoutError(
            f"{ir.columns} columns with {ir.gutter_mm}mm gutters do not fit in "
            f"{content_w:.0f}x{body_h:.0f}mm of body space"
        )

    columns = _split_into_columns(
        body, ir.columns, capacity=column_capacity_mm, gutter_mm=ir.gutter_mm
    )
    for col_index, col_panels in enumerate(columns):
        x = ir.margin_mm + col_index * (col_w + ir.gutter_mm)
        total_weight = sum(p.weight for p in col_panels)
        gutters = ir.gutter_mm * (len(col_panels) - 1)
        usable_h = body_h - gutters
        if usable_h <= 0:
            raise LayoutError(
                f"column {col_index} cannot fit {len(col_panels)} panels with "
                f"{ir.gutter_mm}mm gutters in {body_h:.0f}mm"
            )
        y = body_top
        for i, panel in enumerate(col_panels):
            h = usable_h * (panel.weight / total_weight)
            if i == len(col_panels) - 1:
                # Absorb floating-point drift so the column ends exactly at
                # the bottom margin.
                h = ir.size.height_mm - ir.margin_mm - y
            panel.bbox = BBox(x_mm=x, y_mm=y, w_mm=col_w, h_mm=h)
            panel.column = col_index
            y += h + ir.gutter_mm

    placed = {p.id: p for p in [header, *[p for col in columns for p in col]]}
    return ir.model_copy(update={"panels": [placed[p.id] for p in panels]})


def _split_into_columns(
    body: list[Panel],
    n_columns: int,
    capacity: float | None = None,
    gutter_mm: float = 0.0,
) -> list[list[Panel]]:
    """Split panels (already in reading order) into contiguous column runs.

    Without ``capacity``: greedy weight balancing — a column closes once
    its cumulative weight reaches the per-column average.

    With ``capacity``: weights are physical heights (mm); a column closes
    when adding the *next* panel (plus its gutter) would exceed the
    capacity. Forced closes (to leave one panel per remaining column) can
    still overshoot — the content fitter detects that as overflow.
    """
    if len(body) < n_columns:
        raise LayoutError(
            f"{len(body)} body panels cannot fill {n_columns} columns; "
            "reduce columns or merge panels"
        )
    total = sum(p.weight for p in body)
    target = total / n_columns
    columns: list[list[Panel]] = []
    current: list[Panel] = []
    acc = 0.0
    for i, panel in enumerate(body):
        current.append(panel)
        # Gutters count against physical capacity but not weight balance.
        acc += panel.weight + (gutter_mm if capacity is not None and len(current) > 1 else 0.0)
        remaining = len(body) - i - 1
        cols_after_close = n_columns - len(columns) - 1
        if cols_after_close == 0:
            continue  # final column takes everything left
        # Always close when the remaining panels are only just enough to
        # give each remaining column one panel.
        must_close = remaining == cols_after_close
        if capacity is not None:
            next_burst = remaining > 0 and acc + gutter_mm + body[i + 1].weight > capacity
            should_close = next_burst
        else:
            should_close = acc >= target
        if remaining > 0 and (must_close or should_close):
            columns.append(current)
            current, acc = [], 0.0
    if current:
        columns.append(current)
    if len(columns) != n_columns:
        raise LayoutError(
            f"internal layout error: produced {len(columns)} columns for {n_columns} requested"
        )
    return columns
