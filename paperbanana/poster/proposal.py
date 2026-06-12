"""Layout proposal contract: what the learned proposer is allowed to say.

The proposer decides STRUCTURE — bands, columns per band, panel
placement (with spans), emphasis, banner usage, hero figure, callout
placement. It never authors content: banner text comes from the
storyboard's ``takeaway``, callout values from its verbatim
``key_stats``. Geometry in millimetres never appears here; the skyline
placer derives it from measurements.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from paperbanana.poster.types import BandKind, PanelEmphasis


class ProposedBand(BaseModel):
    id: str
    kind: BandKind
    columns: int = Field(default=1, ge=1, le=6)


class ProposedPlacement(BaseModel):
    panel_id: str
    band_id: str
    column: int = Field(default=0, ge=0)
    col_span: int = Field(default=1, ge=1)
    emphasis: PanelEmphasis = "normal"


class ProposedCallout(BaseModel):
    panel_id: str
    key_stat_id: str = Field(description="Must reference a storyboard key_stat; never invented")


class LayoutProposal(BaseModel):
    """One complete structural proposal, in reading order."""

    bands: list[ProposedBand] = Field(min_length=1, description="Top-to-bottom")
    placements: list[ProposedPlacement] = Field(min_length=1, description="Reading order")
    use_banner: bool = False
    hero_figure_id: Optional[str] = None
    callouts: list[ProposedCallout] = Field(default_factory=list, max_length=3)
    rationale: str = ""


RepairOp = Literal[
    "add_header_band",
    "add_banner_band",
    "clamp_col_span",
    "clamp_column",
    "reduce_band_columns",
    "reorder_bands",
    "rename_duplicate_band",
    "promote_full_span",
    "drop_empty_band",
]


class Violation(BaseModel):
    """A problem with a proposal; fatal ones go back to the proposer."""

    code: str
    fatal: bool
    target: str
    detail: str


class RepairAction(BaseModel):
    """A mechanical, logged transformation applied by the legalizer."""

    op: RepairOp
    target: str
    before: str
    after: str
    reason: str
