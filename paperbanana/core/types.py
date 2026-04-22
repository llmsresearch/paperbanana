"""Core data types for PaperBanana pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

VectorExportMode = Literal["none", "svg", "pdf", "both"]

# Supported aspect ratios for diagram/plot generation.
SUPPORTED_ASPECT_RATIOS = {
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "9:16",
    "16:9",
    "21:9",
}


class PipelineProgressStage(str, Enum):
    """Pipeline stage identifiers for progress callbacks."""

    OPTIMIZER_START = "optimizer_start"
    OPTIMIZER_END = "optimizer_end"
    RETRIEVER_START = "retriever_start"
    RETRIEVER_END = "retriever_end"
    PLANNER_START = "planner_start"
    PLANNER_END = "planner_end"
    STYLIST_START = "stylist_start"
    STYLIST_END = "stylist_end"
    STRUCTURER_START = "structurer_start"
    STRUCTURER_END = "structurer_end"
    VISUALIZER_START = "visualizer_start"
    VISUALIZER_END = "visualizer_end"
    CRITIC_START = "critic_start"
    CRITIC_END = "critic_end"
    CAPTION_START = "caption_start"
    CAPTION_END = "caption_end"


class PipelineProgressEvent(BaseModel):
    """Single progress event emitted by the pipeline for callbacks."""

    stage: PipelineProgressStage = Field(description="Pipeline stage identifier")
    message: str = Field(description="Human-readable message")
    seconds: Optional[float] = Field(default=None, description="Elapsed seconds for this step")
    iteration: Optional[int] = Field(default=None, description="Refinement iteration (1-based)")
    extra: Optional[dict[str, Any]] = Field(default=None, description="Optional extra data")


class DiagramType(str, Enum):
    """Type of academic illustration to generate."""

    METHODOLOGY = "methodology"
    STATISTICAL_PLOT = "statistical_plot"


class GenerationInput(BaseModel):
    """Input to the PaperBanana generation pipeline."""

    source_context: str = Field(description="Methodology section text or relevant paper excerpt")
    communicative_intent: str = Field(description="Figure caption describing what to communicate")
    diagram_type: DiagramType = Field(default=DiagramType.METHODOLOGY)
    raw_data: Optional[dict[str, Any]] = Field(
        default=None, description="Raw data for statistical plots (CSV path or dict)"
    )
    aspect_ratio: Optional[str] = Field(
        default=None,
        description=(
            "Target aspect ratio. "
            "Supported: 1:1, 2:3, 3:2, 3:4, 4:3, 9:16, 16:9, 21:9. "
            "If None, uses provider default."
        ),
    )
    reference_ids: Optional[list[str]] = Field(
        default=None,
        description=(
            "Explicit reference example IDs to use, bypassing automatic retrieval. "
            "When provided, the RetrieverAgent is skipped and these examples are "
            "looked up directly from the ReferenceStore."
        ),
    )
    vector_export: Optional[VectorExportMode] = Field(
        default=None,
        description="Optional vector export (svg/pdf/both); None uses Settings.vector_export",
    )

    @field_validator("aspect_ratio")
    @classmethod
    def validate_aspect_ratio(cls, v: Optional[str]) -> Optional[str]:
        """Ensure aspect_ratio, when provided, is one of the supported values."""
        if v is None:
            return v
        if v not in SUPPORTED_ASPECT_RATIOS:
            supported = ", ".join(sorted(SUPPORTED_ASPECT_RATIOS))
            raise ValueError(f"aspect_ratio must be one of: {supported}")
        return v


class ReferenceExample(BaseModel):
    """A single reference example from the curated set."""

    id: str
    source_context: str
    caption: str
    image_path: str
    category: Optional[str] = None
    aspect_ratio: Optional[float] = None
    structure_hints: Optional[dict[str, Any] | list[Any] | str] = None


class CritiqueResult(BaseModel):
    """Output from the Critic agent."""

    critic_suggestions: list[str] = Field(default_factory=list)
    revised_description: Optional[str] = Field(
        default=None, description="Revised description if revision needed"
    )

    @property
    def needs_revision(self) -> bool:
        return len(self.critic_suggestions) > 0

    @property
    def summary(self) -> str:
        if not self.critic_suggestions:
            return "No issues found. Image is publication-ready."
        return "; ".join(self.critic_suggestions[:3])


class IterationRecord(BaseModel):
    """Record of a single refinement iteration."""

    iteration: int
    description: str
    image_path: str
    critique: Optional[CritiqueResult] = None


class GenerationOutput(BaseModel):
    """Output from the PaperBanana generation pipeline."""

    image_path: str = Field(description="Path to the final generated image")
    description: str = Field(description="Final optimized description")
    iterations: list[IterationRecord] = Field(
        default_factory=list, description="History of refinement iterations"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    generated_caption: Optional[str] = Field(
        default=None,
        description=(
            "Auto-generated publication-ready figure caption. "
            "Only present when generate_caption=True was passed to the pipeline."
        ),
    )
    vector_svg_path: Optional[str] = Field(
        default=None, description="Path to exported SVG (methodology + vector export)"
    )
    vector_pdf_path: Optional[str] = Field(
        default=None, description="Path to exported PDF (methodology + vector export)"
    )


class DiagramIRNode(BaseModel):
    """A node in an editable diagram intermediate representation."""

    id: str
    label: str
    lane: Optional[str] = None


class DiagramIREdge(BaseModel):
    """A directed edge in the diagram intermediate representation."""

    source: str
    target: str
    label: Optional[str] = None


class DiagramIRGroup(BaseModel):
    """A visual lane/group container in the diagram IR."""

    id: str
    label: str
    node_ids: list[str] = Field(default_factory=list)


class DiagramIR(BaseModel):
    """Lightweight intermediate representation for editable exports."""

    title: str
    nodes: list[DiagramIRNode] = Field(default_factory=list)
    edges: list[DiagramIREdge] = Field(default_factory=list)
    groups: list[DiagramIRGroup] = Field(default_factory=list)


VALID_WINNERS = {"Model", "Human", "Both are good", "Both are bad"}

WINNER_SCORE_MAP: dict[str, float] = {
    "Model": 100.0,
    "Human": 0.0,
    "Both are good": 50.0,
    "Both are bad": 50.0,
}


class DimensionResult(BaseModel):
    """Result for a single comparative evaluation dimension."""

    winner: str = Field(description="Model | Human | Both are good | Both are bad")
    score: float = Field(
        ge=0.0,
        le=100.0,
        description="100 (Model wins), 0 (Human wins), 50 (Tie)",
    )
    reasoning: str = Field(default="", description="Comparison reasoning")


class EvaluationScore(BaseModel):
    """Comparative evaluation scores for a generated illustration.

    Uses the paper's referenced comparison approach where a VLM judge
    compares model-generated vs human-drawn diagrams on four dimensions,
    with hierarchical aggregation (Primary: Faithfulness + Readability,
    Secondary: Conciseness + Aesthetics).
    """

    faithfulness: DimensionResult
    conciseness: DimensionResult
    readability: DimensionResult
    aesthetics: DimensionResult
    overall_winner: str = Field(description="Hierarchical aggregation result")
    overall_score: float = Field(
        ge=0.0,
        le=100.0,
        description="100 (Model wins), 0 (Human wins), 50 (Tie)",
    )


class RunMetadata(BaseModel):
    """Metadata for a single pipeline run, for reproducibility."""

    run_id: str
    timestamp: str
    vlm_provider: str
    vlm_model: str
    image_provider: str
    image_model: str
    refinement_iterations: int
    seed: Optional[int] = None
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
