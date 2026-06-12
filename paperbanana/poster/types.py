"""Pydantic data models for the poster generation head.

The :class:`PosterIR` is the single source of truth for a poster: agents
edit it, the deterministic renderer consumes it, and preflight validates
it. All geometry is expressed in physical millimetres so legibility and
print-fidelity rules are checkable without rendering.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

MM_PER_INCH = 25.4
EMU_PER_MM = 36000

TextLevel = Literal["title", "authors", "affiliation", "heading", "body", "caption", "footnote"]

PanelRole = Literal[
    "header",
    "abstract",
    "motivation",
    "background",
    "method",
    "results",
    "analysis",
    "ablation",
    "comparison",
    "takeaway",
    "conclusion",
    "references",
    "acknowledgments",
    "qr",
    "custom",
]

#: Roles with structural meaning; everything else is presentational.
STRUCTURAL_ROLES = frozenset({"header", "qr"})

FigureDecision = Literal["reuse", "reauthor", "generate"]

Orientation = Literal["portrait", "landscape"]


def mm_to_inches(mm: float) -> float:
    """Convert millimetres to inches."""
    return mm / MM_PER_INCH


def inches_to_mm(inches: float) -> float:
    """Convert inches to millimetres."""
    return inches * MM_PER_INCH


def mm_to_emu(mm: float) -> int:
    """Convert millimetres to English Metric Units (pptx native unit)."""
    return round(mm * EMU_PER_MM)


class PhysicalSize(BaseModel):
    """Physical page size in millimetres."""

    width_mm: float = Field(gt=0)
    height_mm: float = Field(gt=0)

    @property
    def orientation(self) -> Orientation:
        return "landscape" if self.width_mm >= self.height_mm else "portrait"


class BBox(BaseModel):
    """Absolute bounding box in millimetres, origin at poster top-left."""

    x_mm: float = Field(ge=0)
    y_mm: float = Field(ge=0)
    w_mm: float = Field(gt=0)
    h_mm: float = Field(gt=0)

    @property
    def x2_mm(self) -> float:
        return self.x_mm + self.w_mm

    @property
    def y2_mm(self) -> float:
        return self.y_mm + self.h_mm

    def overlaps(self, other: "BBox", tolerance_mm: float = 0.1) -> bool:
        """True if this box overlaps another beyond the given tolerance."""
        return not (
            self.x2_mm <= other.x_mm + tolerance_mm
            or other.x2_mm <= self.x_mm + tolerance_mm
            or self.y2_mm <= other.y_mm + tolerance_mm
            or other.y2_mm <= self.y_mm + tolerance_mm
        )


class StyleTokens(BaseModel):
    """Typography and palette tokens applied by the deterministic renderer."""

    palette: dict[str, str] = Field(
        description=(
            "Named hex colors; required keys: primary, secondary, accent, "
            "background, panel_bg, text"
        )
    )
    font_heading: str = Field(description="Font family for title/headings")
    font_body: str = Field(description="Font family for body/caption text")
    type_scale_pt: dict[TextLevel, float] = Field(
        description="Point size per text level; preflight enforces venue minima"
    )

    @model_validator(mode="after")
    def validate_tokens(self) -> "StyleTokens":
        required_colors = {"primary", "secondary", "accent", "background", "panel_bg", "text"}
        missing = required_colors - set(self.palette)
        if missing:
            raise ValueError(f"palette missing required keys: {sorted(missing)}")
        for name, value in self.palette.items():
            v = value.lstrip("#")
            if len(v) != 6 or any(c not in "0123456789abcdefABCDEF" for c in v):
                raise ValueError(f"palette['{name}'] is not a 6-digit hex color: {value!r}")
        required_levels = {"title", "heading", "body", "caption"}
        missing_levels = required_levels - set(self.type_scale_pt)
        if missing_levels:
            raise ValueError(f"type_scale_pt missing required levels: {sorted(missing_levels)}")
        return self


class FigureProvenance(BaseModel):
    """Where a poster figure came from and how it was produced."""

    origin: Literal["paper", "generated"]
    paper_figure_id: Optional[str] = None
    source_page: Optional[int] = Field(default=None, ge=1, description="1-based PDF page")
    source_bbox_norm: Optional[list[float]] = Field(
        default=None,
        description="[x0, y0, x1, y1] in 0..1 page coordinates",
    )
    decision: FigureDecision
    decision_reason: str
    edit_instructions: Optional[str] = None
    caption_anchored: bool = True
    faithfulness: Literal["verified", "not_required", "failed"] = "not_required"

    @model_validator(mode="after")
    def validate_provenance(self) -> "FigureProvenance":
        if self.origin == "paper" and self.source_page is None:
            raise ValueError("paper-origin figures must record source_page")
        if self.source_bbox_norm is not None:
            if len(self.source_bbox_norm) != 4:
                raise ValueError("source_bbox_norm must have 4 values [x0, y0, x1, y1]")
            x0, y0, x1, y1 = self.source_bbox_norm
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                raise ValueError(f"source_bbox_norm out of order/range: {self.source_bbox_norm}")
        if self.decision == "reauthor" and not self.edit_instructions:
            raise ValueError("reauthor decisions must carry edit_instructions")
        return self


class FigureAsset(BaseModel):
    """A raster asset placed on the poster, with full provenance."""

    id: str
    path: str
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    provenance: FigureProvenance

    def effective_dpi(self, placed_width_mm: float) -> float:
        """Print resolution of this asset when placed at the given width."""
        return self.width_px / mm_to_inches(placed_width_mm)


class TextElement(BaseModel):
    """A text block; lines starting with '- ' render as bullets."""

    kind: Literal["text"] = "text"
    level: TextLevel
    content: str = Field(min_length=1)


class FigureElement(BaseModel):
    """A placed figure, referencing an asset by id."""

    kind: Literal["figure"] = "figure"
    asset_id: str
    caption: Optional[str] = None
    max_height_frac: float = Field(default=0.6, gt=0, le=1.0)


class QRElement(BaseModel):
    """A QR code linking to the paper or project page."""

    kind: Literal["qr"] = "qr"
    url: str = Field(min_length=1)
    label: Optional[str] = None


PosterElement = Annotated[Union[TextElement, FigureElement, QRElement], Field(discriminator="kind")]


class Panel(BaseModel):
    """One poster panel; geometry is assigned by the layout solver."""

    id: str
    role: PanelRole
    title: Optional[str] = None
    order: int = Field(ge=0, description="Reading order across the poster")
    column: Optional[int] = Field(default=None, ge=0, description="Assigned by layout solver")
    weight: float = Field(default=1.0, gt=0, description="Relative vertical share in its column")
    bbox: Optional[BBox] = None
    z: int = 0
    elements: list[PosterElement] = Field(min_length=1)


class PosterIR(BaseModel):
    """Editable intermediate representation of a poster (schema v1)."""

    schema_version: int = 1
    venue: str
    venue_spec_year: int
    size: PhysicalSize
    orientation: Orientation
    print_scale: int = Field(
        default=1,
        ge=1,
        le=4,
        description=(
            "Design-scale divisor: the pptx page is rendered at size/print_scale "
            "with fonts scaled down identically, and the print shop prints at "
            "print_scale x 100%. Required when the physical size exceeds the "
            "pptx 56-inch page cap (e.g. CVPR 84-inch posters use 2)."
        ),
    )
    target_dpi: int = Field(default=300, gt=0)
    columns: int = Field(default=3, ge=1, le=6)
    margin_mm: float = Field(default=20.0, ge=0)
    gutter_mm: float = Field(default=10.0, ge=0)
    style: StyleTokens
    panels: list[Panel] = Field(min_length=1)
    assets: dict[str, FigureAsset] = Field(default_factory=dict)
    paper_title: str
    authors: list[str] = Field(default_factory=list)
    affiliations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_ir(self) -> "PosterIR":
        if self.size.orientation != self.orientation:
            raise ValueError(
                f"orientation '{self.orientation}' contradicts size "
                f"{self.size.width_mm}x{self.size.height_mm}mm"
            )
        panel_ids = [p.id for p in self.panels]
        if len(panel_ids) != len(set(panel_ids)):
            raise ValueError(f"duplicate panel ids: {panel_ids}")
        orders = [p.order for p in self.panels]
        if len(orders) != len(set(orders)):
            raise ValueError(f"duplicate panel orders: {orders}")
        for asset_id, asset in self.assets.items():
            if asset.id != asset_id:
                raise ValueError(f"asset key '{asset_id}' != asset.id '{asset.id}'")
        for panel in self.panels:
            for element in panel.elements:
                if isinstance(element, FigureElement) and element.asset_id not in self.assets:
                    raise ValueError(
                        f"panel '{panel.id}' references unknown asset '{element.asset_id}'"
                    )
        placed = [p for p in self.panels if p.bbox is not None]
        for panel in placed:
            box = panel.bbox
            assert box is not None
            if box.x2_mm > self.size.width_mm + 0.1 or box.y2_mm > self.size.height_mm + 0.1:
                raise ValueError(
                    f"panel '{panel.id}' bbox exceeds page bounds "
                    f"({box.x2_mm:.1f}, {box.y2_mm:.1f}) > "
                    f"({self.size.width_mm}, {self.size.height_mm})"
                )
        for i, a in enumerate(placed):
            for b in placed[i + 1 :]:
                assert a.bbox is not None and b.bbox is not None
                if a.z == b.z and a.bbox.overlaps(b.bbox):
                    raise ValueError(f"panels '{a.id}' and '{b.id}' overlap at z={a.z}")
        return self

    def panels_in_order(self) -> list[Panel]:
        return sorted(self.panels, key=lambda p: p.order)

    def is_fully_placed(self) -> bool:
        return all(p.bbox is not None for p in self.panels)


class PaperSection(BaseModel):
    """A titled section of the source paper."""

    heading: str
    text: str


class PaperFigure(BaseModel):
    """A figure or table extracted from the source paper."""

    id: str
    kind: Literal["figure", "table"]
    page: int = Field(ge=1)
    bbox_norm: list[float] = Field(description="[x0, y0, x1, y1] in 0..1 page coordinates")
    caption: str
    image_path: str
    extract_dpi: int = Field(gt=0)
    caption_anchored: bool = True

    @model_validator(mode="after")
    def validate_bbox(self) -> "PaperFigure":
        if len(self.bbox_norm) != 4:
            raise ValueError("bbox_norm must have 4 values [x0, y0, x1, y1]")
        x0, y0, x1, y1 = self.bbox_norm
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError(f"bbox_norm out of order/range: {self.bbox_norm}")
        return self


class PaperAssets(BaseModel):
    """Everything ingested from the source paper PDF."""

    pdf_path: str
    page_count: int = Field(ge=1)
    title: str
    authors: list[str] = Field(default_factory=list)
    affiliations: list[str] = Field(default_factory=list)
    abstract: str = ""
    sections: list[PaperSection] = Field(default_factory=list)
    figures: list[PaperFigure] = Field(default_factory=list)


class StoryboardPanel(BaseModel):
    """Content-agent draft of one panel before styling and layout."""

    id: str
    role: PanelRole
    title: Optional[str] = None
    order: int = Field(ge=0)
    weight: float = Field(default=1.0, gt=0)
    text_blocks: list[str] = Field(default_factory=list)
    figure_ids: list[str] = Field(
        default_factory=list,
        description="PaperFigure ids placed in this panel, or 'new:<slug>' generation requests",
    )


class Storyboard(BaseModel):
    """Content agent output: panel plan plus layout hints."""

    panels: list[StoryboardPanel] = Field(min_length=1)
    columns: int = Field(default=3, ge=1, le=6)
    qr_url: Optional[str] = None
    new_figure_briefs: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Generation briefs for figures the paper lacks, keyed by slug; "
            "panels reference them as 'new:<slug>'"
        ),
    )

    @model_validator(mode="after")
    def validate_storyboard(self) -> "Storyboard":
        ids = [p.id for p in self.panels]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate storyboard panel ids: {ids}")
        for panel in self.panels:
            for fid in panel.figure_ids:
                if fid.startswith("new:") and fid[4:] not in self.new_figure_briefs:
                    raise ValueError(
                        f"panel '{panel.id}' requests '{fid}' but no brief exists for it"
                    )
        return self


class FigureDecisionResult(BaseModel):
    """Figure curator verdict for one figure."""

    figure_id: str
    decision: FigureDecision
    reason: str
    edit_instructions: Optional[str] = None
    generate_brief: Optional[str] = None

    @model_validator(mode="after")
    def validate_decision(self) -> "FigureDecisionResult":
        if self.decision == "reauthor" and not self.edit_instructions:
            raise ValueError(f"figure '{self.figure_id}': reauthor requires edit_instructions")
        if self.decision == "generate" and not self.generate_brief:
            raise ValueError(f"figure '{self.figure_id}': generate requires generate_brief")
        return self


class PosterCritique(BaseModel):
    """Critic verdict for one refinement iteration."""

    blocking: bool
    summary: str
    edit_ops: list[dict] = Field(
        default_factory=list,
        description="PosterEditOp payloads, validated by paperbanana.poster.edits",
    )


class PreflightCheck(BaseModel):
    """One deterministic preflight check result."""

    id: str
    status: Literal["pass", "fail", "warn"]
    value: str
    threshold: str
    detail: str


class PreflightReport(BaseModel):
    """Aggregate preflight outcome; ``passed`` means zero failures."""

    checks: list[PreflightCheck] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.status != "fail" for c in self.checks)

    @property
    def failures(self) -> list[PreflightCheck]:
        return [c for c in self.checks if c.status == "fail"]

    @property
    def warnings(self) -> list[PreflightCheck]:
        return [c for c in self.checks if c.status == "warn"]


class PosterOutput(BaseModel):
    """Final artifacts of a poster run."""

    run_dir: str
    pptx_path: str
    pdf_path: str
    preview_path: str
    ir_path: str
    preflight: PreflightReport
    iterations: int
    figure_decisions: list[FigureDecisionResult] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
