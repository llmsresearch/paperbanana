"""Deterministic skyline placement for proposed band/column structures.

v4 division of labor: the *structure* of a poster (bands, columns per
band, panel membership, spans, emphasis) is LEARNED — proposed by the
LayoutProposerAgent grounded in real-poster exemplars. This module is
pure mechanism: it turns a structure plus measured content heights into
exact millimetre geometry. It makes no design decisions; the packing
heuristics of v1-v3 (balanced column splitting, escalation) are gone.

Placement model:
- Bands stack top-to-bottom in band order.
- header/banner/footer bands: one full-width panel, height = measured.
- body bands: equal-width columns with per-column y-cursors (a skyline);
  panels in reading order; a panel spanning columns [c, c+s) sits at
  ``max(cursor[c:c+s])`` and advances all spanned cursors. Heights are
  measured content + padding + bounded growth.
- Residual page slack stretches body bands proportionally to their
  loads; inside panels, the renderer's justification distributes it.

Overflow (the stack exceeding the page) raises :class:`BandOverflowError`
with per-column loads — feedback for text shortening or a structural
re-proposal, never silently absorbed.
"""

from __future__ import annotations

from pydantic import BaseModel

from paperbanana.poster.types import BBox, Panel, PosterIR

#: Banner bands are emphasis, not content: cap them at this page fraction.
MAX_BANNER_FRAC = 0.08


class LayoutError(ValueError):
    """Raised when the requested layout is geometrically impossible."""


class ColumnLoad(BaseModel):
    """Measured load of one column in one band, for overflow feedback."""

    band_id: str
    column: int
    required_mm: float
    available_mm: float


class BandOverflowError(LayoutError):
    """The proposed structure cannot fit the measured content on the page."""

    def __init__(self, overflows: list[ColumnLoad], page_deficit_mm: float):
        self.overflows = overflows
        self.page_deficit_mm = page_deficit_mm
        details = "; ".join(
            f"band '{o.band_id}' col {o.column}: needs {o.required_mm:.0f}mm "
            f"of {o.available_mm:.0f}mm"
            for o in overflows
        )
        super().__init__(
            f"proposed layout overflows the page by {page_deficit_mm:.0f}mm "
            f"({details or 'total band stack too tall'})"
        )


def place_bands(ir: PosterIR, measured_mm: dict[str, float]) -> PosterIR:
    """Return a copy of the IR with every band sized and every panel placed.

    Args:
        ir: Poster IR with band structure set (bboxes may be stale/None).
        measured_mm: Panel id -> measured content height at that panel's
            placed width (excluding padding), from the renderer's font
            metrics.

    Raises:
        LayoutError: Structural impossibility (no content area, missing
            measurements).
        BandOverflowError: Content cannot fit; carries per-column loads.
    """
    panels = [p.model_copy(deep=True) for p in ir.panels]
    for panel in panels:
        if panel.id not in measured_mm:
            raise LayoutError(f"no measurement for panel '{panel.id}'")

    content_w = ir.size.width_mm - 2 * ir.margin_mm
    page_h = ir.size.height_mm
    if content_w <= 0 or page_h - 2 * ir.margin_mm <= 0:
        raise LayoutError(
            f"margins {ir.margin_mm}mm leave no content area on a "
            f"{ir.size.width_mm}x{ir.size.height_mm}mm page"
        )

    from paperbanana.poster.renderer import PANEL_PADDING_MM

    bands = sorted((b.model_copy(deep=True) for b in ir.bands), key=lambda b: b.order)
    by_band: dict[str, list[Panel]] = {
        b.id: sorted([p for p in panels if p.band_id == b.id], key=lambda p: p.order) for b in bands
    }

    # Pass 1 — natural band heights from measured content.
    band_heights: dict[str, float] = {}
    band_column_loads: dict[str, list[float]] = {}
    for band in bands:
        members = by_band[band.id]
        if band.kind in ("header", "banner", "footer"):
            height = measured_mm[members[0].id] + 2 * PANEL_PADDING_MM
            if band.kind == "banner":
                height = min(height, page_h * MAX_BANNER_FRAC)
            band_heights[band.id] = height
            band_column_loads[band.id] = [height]
        else:
            cursors = [0.0] * band.columns
            for panel in members:
                start, end = panel.column, panel.column + panel.col_span
                y = max(cursors[start:end])
                if y > 0:
                    y += ir.gutter_mm
                h = measured_mm[panel.id] + 2 * PANEL_PADDING_MM
                panel.weight = max(0.1, round(measured_mm[panel.id], 1))
                bottom = y + h
                for c in range(start, end):
                    cursors[c] = bottom
            band_heights[band.id] = max(cursors)
            band_column_loads[band.id] = cursors

    total_gutters = ir.gutter_mm * (len(bands) - 1)
    natural_total = sum(band_heights.values()) + total_gutters
    available_total = page_h - 2 * ir.margin_mm
    if natural_total > available_total + 0.1:
        deficit = natural_total - available_total
        overflows = [
            ColumnLoad(
                band_id=band.id,
                column=c,
                required_mm=load,
                available_mm=max(0.0, band_heights[band.id] - deficit),
            )
            for band in bands
            for c, load in enumerate(band_column_loads[band.id])
            if band.kind == "body"
        ]
        worst = sorted(overflows, key=lambda o: -o.required_mm)[:6]
        raise BandOverflowError(worst, deficit)

    # Pass 2 — distribute page slack to body bands proportionally to load.
    slack = available_total - natural_total
    body_ids = [b.id for b in bands if b.kind == "body"]
    body_total = sum(band_heights[b] for b in body_ids) or 1.0
    final_heights = dict(band_heights)
    for band_id in body_ids:
        final_heights[band_id] += slack * (band_heights[band_id] / body_total)

    # Pass 3 — place panels with exact geometry.
    y_band = ir.margin_mm
    for band in bands:
        members = by_band[band.id]
        band_h = final_heights[band.id]
        band.height_mm = round(band_h, 2)
        if band.kind in ("header", "banner", "footer"):
            members[0].bbox = BBox(x_mm=ir.margin_mm, y_mm=y_band, w_mm=content_w, h_mm=band_h)
            members[0].column = 0
        else:
            col_w = (content_w - (band.columns - 1) * ir.gutter_mm) / band.columns
            if col_w <= 0:
                raise LayoutError(
                    f"band '{band.id}': {band.columns} columns with {ir.gutter_mm}mm "
                    f"gutters do not fit in {content_w:.0f}mm"
                )
            # Justified fill: every column ends at the band bottom — the
            # canvas contract every real poster in the corpus satisfies.
            # HOW the column's space is shared is the learned designer's
            # call: the proposer's height_frac is the share weight, with
            # measured content as the proportional default and always the
            # hard floor (shares distribute slack, never compress content).
            natural = {p.id: measured_mm[p.id] + 2 * PANEL_PADDING_MM for p in members}
            extra = {p.id: 0.0 for p in members}
            band_bottom = y_band + band_h

            def _bottoms() -> list[float]:
                cur = [y_band] * band.columns
                for panel in members:
                    start, end = panel.column, panel.column + panel.col_span
                    y = max(cur[start:end])
                    if y > y_band:
                        y += ir.gutter_mm
                    bottom = y + natural[panel.id] + extra[panel.id]
                    for c in range(start, end):
                        cur[c] = bottom
                return cur

            for _ in range(3):  # spans couple columns; residue converges fast
                remaining = [band_bottom - b for b in _bottoms()]
                if max(remaining, default=0.0) < 0.5:
                    break
                for c in range(band.columns):
                    col_members = [p for p in members if p.column <= c < p.column + p.col_span]
                    if not col_members or remaining[c] <= 0.5:
                        continue
                    weights = [
                        max(p.height_frac or natural[p.id] / band_h, 0.01) for p in col_members
                    ]
                    total_w = sum(weights)
                    avail = remaining[c]
                    for p, w in zip(col_members, weights):
                        span_cols = range(p.column, p.column + p.col_span)
                        grant = min(avail * (w / total_w), min(remaining[cc] for cc in span_cols))
                        if grant <= 0:
                            continue
                        extra[p.id] += grant
                        for cc in span_cols:
                            remaining[cc] -= grant

            cursors = [y_band] * band.columns
            for panel in members:
                start, end = panel.column, panel.column + panel.col_span
                y = max(cursors[start:end])
                if y > y_band:
                    y += ir.gutter_mm
                h = natural[panel.id] + extra[panel.id]
                # Numeric residue from coupled spans must never leak past
                # the band edge.
                h = min(h, band_bottom - y)
                width = col_w * panel.col_span + ir.gutter_mm * (panel.col_span - 1)
                panel.bbox = BBox(
                    x_mm=ir.margin_mm + start * (col_w + ir.gutter_mm),
                    y_mm=y,
                    w_mm=width,
                    h_mm=h,
                )
                bottom = y + h
                for c in range(start, end):
                    cursors[c] = bottom
        y_band += band_h + ir.gutter_mm

    placed = {p.id: p for plist in by_band.values() for p in plist}
    return ir.model_copy(
        update={
            "panels": [placed[p.id] for p in ir.panels],
            "bands": bands,
        }
    )


def panel_width_mm(ir: PosterIR, panel: Panel) -> float:
    """Physical width a panel will occupy, derivable before placement."""
    band = ir.band(panel.band_id)
    content_w = ir.size.width_mm - 2 * ir.margin_mm
    if band.kind != "body":
        return content_w
    col_w = (content_w - (band.columns - 1) * ir.gutter_mm) / band.columns
    return col_w * panel.col_span + ir.gutter_mm * (panel.col_span - 1)
