"""Closed vocabulary of IR edit operations the critic may request.

The critic returns structured ops, never freeform IR JSON: anything the
schema cannot express is reported in the critique summary instead of
being improvised. Applying ops is deterministic IR surgery; geometry-
affecting ops clear panel bboxes so the layout solver re-places them.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter

from paperbanana.poster.types import PosterIR


class EditOpError(ValueError):
    """An edit op references unknown targets or violates hard minima."""


class RewriteText(BaseModel):
    """Replace the content of one text element (e.g. to fix overflow)."""

    op: Literal["rewrite_text"] = "rewrite_text"
    panel_id: str
    element_index: int = Field(ge=0, description="Index within the panel's elements")
    content: str = Field(min_length=1)


class SetTypeScale(BaseModel):
    """Change the point size of one text level."""

    op: Literal["set_type_scale"] = "set_type_scale"
    level: str
    pt: float = Field(gt=0)


class SetPanelWeight(BaseModel):
    """Change a panel's vertical share; triggers re-layout."""

    op: Literal["set_panel_weight"] = "set_panel_weight"
    panel_id: str
    weight: float = Field(gt=0)


class MovePanelOrder(BaseModel):
    """Move a panel to a new reading-order position; triggers re-layout."""

    op: Literal["move_panel_order"] = "move_panel_order"
    panel_id: str
    new_position: int = Field(ge=0, description="Target index in reading order")


class RecolorToken(BaseModel):
    """Change one palette token."""

    op: Literal["recolor_token"] = "recolor_token"
    token: str
    hex: str = Field(pattern=r"^#?[0-9a-fA-F]{6}$")


class RecurateFigure(BaseModel):
    """Request re-curation of a figure (handled by the pipeline, not here)."""

    op: Literal["recurate_figure"] = "recurate_figure"
    asset_id: str
    note: str


class SetEmphasis(BaseModel):
    """Set a panel's visual emphasis (accent fill)."""

    op: Literal["set_emphasis"] = "set_emphasis"
    panel_id: str
    emphasis: Literal["normal", "accent"]


class MovePanelToBand(BaseModel):
    """Move a panel into a band/column position; triggers re-layout."""

    op: Literal["move_panel_to_band"] = "move_panel_to_band"
    panel_id: str
    band_id: str
    column: int = Field(default=0, ge=0)
    col_span: int = Field(default=1, ge=1)


class SetColSpan(BaseModel):
    """Change a panel's column span within its band; triggers re-layout."""

    op: Literal["set_col_span"] = "set_col_span"
    panel_id: str
    col_span: int = Field(ge=1)


class RewriteBanner(BaseModel):
    """Replace the takeaway banner's text."""

    op: Literal["rewrite_banner"] = "rewrite_banner"
    content: str = Field(min_length=1)


class RemoveCallout(BaseModel):
    """Remove a big-number callout from a panel."""

    op: Literal["remove_callout"] = "remove_callout"
    panel_id: str


PosterEditOp = Annotated[
    Union[
        RewriteText,
        SetTypeScale,
        SetPanelWeight,
        MovePanelOrder,
        RecolorToken,
        RecurateFigure,
        SetEmphasis,
        MovePanelToBand,
        SetColSpan,
        RewriteBanner,
        RemoveCallout,
    ],
    Field(discriminator="op"),
]

_OPS_ADAPTER: TypeAdapter[list[PosterEditOp]] = TypeAdapter(list[PosterEditOp])


def parse_edit_ops(payloads: list[dict]) -> list[PosterEditOp]:
    """Validate raw critic payloads into typed ops.

    Raises:
        EditOpError: If any payload is not a known, well-formed op.
    """
    try:
        return _OPS_ADAPTER.validate_python(payloads)
    except Exception as exc:
        raise EditOpError(f"invalid edit ops from critic: {exc}") from exc


def apply_edit_ops(
    ir: PosterIR,
    ops: list[PosterEditOp],
    min_pt: dict[str, float] | None = None,
) -> tuple[PosterIR, list[RecurateFigure]]:
    """Apply ops to a copy of the IR.

    Args:
        ir: Current poster IR.
        ops: Ops to apply, in order.
        min_pt: Hard per-level font minima (venue rule merged with the
            legibility floor); a ``set_type_scale`` below the floor raises.

    Returns:
        Tuple of (new IR, deferred figure re-curation requests). The new
        IR has bboxes cleared if any op changed geometry inputs.

    Raises:
        EditOpError: On unknown panels/levels/tokens or scale below minima.
    """
    data = ir.model_dump()
    panels = data["panels"]
    by_id = {p["id"]: p for p in panels}
    deferred: list[RecurateFigure] = []
    needs_relayout = False
    minima = min_pt or {}

    for op in ops:
        if isinstance(op, RewriteText):
            panel = by_id.get(op.panel_id)
            if panel is None:
                raise EditOpError(f"rewrite_text: unknown panel '{op.panel_id}'")
            if op.element_index >= len(panel["elements"]):
                raise EditOpError(
                    f"rewrite_text: panel '{op.panel_id}' has no element {op.element_index}"
                )
            element = panel["elements"][op.element_index]
            if element.get("kind") != "text":
                raise EditOpError(
                    f"rewrite_text: element {op.element_index} of '{op.panel_id}' is "
                    f"a {element.get('kind')!r} element, not text"
                )
            element["content"] = op.content
        elif isinstance(op, SetTypeScale):
            if op.level not in data["style"]["type_scale_pt"]:
                raise EditOpError(f"set_type_scale: unknown text level '{op.level}'")
            floor = minima.get(op.level, 0)
            if op.pt < floor:
                raise EditOpError(
                    f"set_type_scale: {op.pt}pt for '{op.level}' is below the hard "
                    f"minimum of {floor}pt; shorten content instead of shrinking text"
                )
            data["style"]["type_scale_pt"][op.level] = op.pt
        elif isinstance(op, SetPanelWeight):
            panel = by_id.get(op.panel_id)
            if panel is None:
                raise EditOpError(f"set_panel_weight: unknown panel '{op.panel_id}'")
            panel["weight"] = op.weight
            needs_relayout = True
        elif isinstance(op, MovePanelOrder):
            panel = by_id.get(op.panel_id)
            if panel is None:
                raise EditOpError(f"move_panel_order: unknown panel '{op.panel_id}'")
            ordered = sorted(panels, key=lambda p: p["order"])
            ordered.remove(panel)
            position = min(op.new_position, len(ordered))
            ordered.insert(position, panel)
            for i, p in enumerate(ordered):
                p["order"] = i
            needs_relayout = True
        elif isinstance(op, RecolorToken):
            if op.token not in data["style"]["palette"]:
                raise EditOpError(f"recolor_token: unknown palette token '{op.token}'")
            value = op.hex if op.hex.startswith("#") else f"#{op.hex}"
            data["style"]["palette"][op.token] = value
        elif isinstance(op, RecurateFigure):
            if op.asset_id not in data["assets"]:
                raise EditOpError(f"recurate_figure: unknown asset '{op.asset_id}'")
            deferred.append(op)
        elif isinstance(op, SetEmphasis):
            panel = by_id.get(op.panel_id)
            if panel is None:
                raise EditOpError(f"set_emphasis: unknown panel '{op.panel_id}'")
            panel["emphasis"] = op.emphasis
        elif isinstance(op, MovePanelToBand):
            panel = by_id.get(op.panel_id)
            if panel is None:
                raise EditOpError(f"move_panel_to_band: unknown panel '{op.panel_id}'")
            band = next((b for b in data["bands"] if b["id"] == op.band_id), None)
            if band is None:
                raise EditOpError(f"move_panel_to_band: unknown band '{op.band_id}'")
            if band["kind"] != "body":
                raise EditOpError(
                    f"move_panel_to_band: band '{op.band_id}' is {band['kind']}, not body"
                )
            if op.column + op.col_span > band["columns"]:
                raise EditOpError(
                    f"move_panel_to_band: columns [{op.column}, {op.column + op.col_span}) "
                    f"exceed band '{op.band_id}' ({band['columns']} columns)"
                )
            panel["band_id"] = op.band_id
            panel["column"] = op.column
            panel["col_span"] = op.col_span
            needs_relayout = True
        elif isinstance(op, SetColSpan):
            panel = by_id.get(op.panel_id)
            if panel is None:
                raise EditOpError(f"set_col_span: unknown panel '{op.panel_id}'")
            band = next(b for b in data["bands"] if b["id"] == panel["band_id"])
            if panel["column"] + op.col_span > band["columns"]:
                raise EditOpError(
                    f"set_col_span: span {op.col_span} from column {panel['column']} "
                    f"exceeds band '{band['id']}' ({band['columns']} columns)"
                )
            panel["col_span"] = op.col_span
            needs_relayout = True
        elif isinstance(op, RewriteBanner):
            banner_elements = [
                el for p in panels for el in p["elements"] if el.get("kind") == "banner"
            ]
            if not banner_elements:
                raise EditOpError("rewrite_banner: the poster has no banner element")
            banner_elements[0]["content"] = op.content
        elif isinstance(op, RemoveCallout):
            panel = by_id.get(op.panel_id)
            if panel is None:
                raise EditOpError(f"remove_callout: unknown panel '{op.panel_id}'")
            kept = [el for el in panel["elements"] if el.get("kind") != "big_number"]
            if len(kept) == len(panel["elements"]):
                raise EditOpError(f"remove_callout: panel '{op.panel_id}' has no callout")
            if not kept:
                raise EditOpError(f"remove_callout: panel '{op.panel_id}' would become empty")
            panel["elements"] = kept
            needs_relayout = True
        else:  # pragma: no cover - closed union
            raise EditOpError(f"unhandled op type: {op!r}")

    if needs_relayout:
        for p in panels:
            p["bbox"] = None
        for b in data["bands"]:
            b["height_mm"] = None
    return PosterIR(**data), deferred
