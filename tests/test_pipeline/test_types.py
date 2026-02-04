"""Tests for core types."""

from __future__ import annotations

import pytest

from paperbanana.core.types import (
    CritiqueResult,
    DiagramType,
    DimensionResult,
    EvaluationScore,
    GenerationInput,
    ReferenceExample,
)


def test_generation_input():
    """Test GenerationInput creation."""
    gi = GenerationInput(
        source_context="Test methodology",
        communicative_intent="Test caption",
    )
    assert gi.diagram_type == DiagramType.METHODOLOGY
    assert gi.raw_data is None


def test_generation_input_plot():
    """Test GenerationInput for statistical plots."""
    gi = GenerationInput(
        source_context="Test data",
        communicative_intent="Test plot",
        diagram_type=DiagramType.STATISTICAL_PLOT,
        raw_data={"data": [1, 2, 3]},
    )
    assert gi.diagram_type == DiagramType.STATISTICAL_PLOT


def test_reference_example():
    """Test ReferenceExample creation."""
    ref = ReferenceExample(
        id="ref_001",
        source_context="Source",
        caption="Caption",
        image_path="path/to/image.png",
        category="test",
    )
    assert ref.id == "ref_001"


def test_critique_result():
    """Test CritiqueResult creation with suggestions."""
    cr = CritiqueResult(
        critic_suggestions=["Missing component X", "Text is garbled"],
        revised_description="Updated description",
    )
    assert cr.needs_revision
    assert len(cr.critic_suggestions) == 2
    assert "Missing component X" in cr.summary


def test_critique_result_no_revision():
    """Test CritiqueResult with no issues."""
    cr = CritiqueResult(critic_suggestions=[], revised_description=None)
    assert not cr.needs_revision
    assert "publication-ready" in cr.summary


def test_dimension_result():
    """Test DimensionResult creation."""
    dr = DimensionResult(
        winner="Model",
        score=100.0,
        reasoning="Model diagram is more faithful.",
    )
    assert dr.winner == "Model"
    assert dr.score == 100.0


def test_dimension_result_score_range():
    """Test that DimensionResult score is within valid range."""
    with pytest.raises(Exception):
        DimensionResult(winner="Model", score=150.0, reasoning="")


def test_evaluation_score_comparative():
    """Test EvaluationScore with comparative results."""
    model_wins = DimensionResult(winner="Model", score=100.0)
    tie = DimensionResult(winner="Both are good", score=50.0)

    score = EvaluationScore(
        faithfulness=model_wins,
        conciseness=tie,
        readability=tie,
        aesthetics=tie,
        overall_winner="Model",
        overall_score=100.0,
    )
    assert score.overall_winner == "Model"
    assert score.faithfulness.winner == "Model"
    assert score.readability.score == 50.0


def test_evaluation_score_overall_range():
    """Test that overall_score is within valid range."""
    tie = DimensionResult(winner="Both are good", score=50.0)
    with pytest.raises(Exception):
        EvaluationScore(
            faithfulness=tie,
            conciseness=tie,
            readability=tie,
            aesthetics=tie,
            overall_winner="Both are good",
            overall_score=150.0,  # Out of range
        )
