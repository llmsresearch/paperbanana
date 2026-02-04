"""Core data types for PaperBanana pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


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


class ReferenceExample(BaseModel):
    """A single reference example from the curated set."""

    id: str
    source_context: str
    caption: str
    image_path: str
    category: Optional[str] = None


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
        ge=0.0, le=100.0,
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
    overall_winner: str = Field(
        description="Hierarchical aggregation result"
    )
    overall_score: float = Field(
        ge=0.0, le=100.0,
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
