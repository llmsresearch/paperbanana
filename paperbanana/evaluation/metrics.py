"""Evaluation metric utilities for PaperBanana."""

from __future__ import annotations

from paperbanana.core.types import EvaluationScore

DIMENSIONS = ["faithfulness", "conciseness", "readability", "aesthetics"]


def format_scores(scores: EvaluationScore) -> str:
    """Format comparative evaluation scores as a readable string."""
    lines = []
    for dim in DIMENSIONS:
        result = getattr(scores, dim)
        lines.append(f"{dim.capitalize():14s} {result.winner:16s} ({result.score:.0f})")
    lines.append(f"{'Overall':14s} {scores.overall_winner:16s} ({scores.overall_score:.0f})")
    return "\n".join(lines)


def scores_to_dict(scores: EvaluationScore) -> dict:
    """Convert evaluation scores to a flat dictionary."""
    result = {}
    for dim in DIMENSIONS:
        dim_result = getattr(scores, dim)
        result[f"{dim}_winner"] = dim_result.winner
        result[f"{dim}_score"] = dim_result.score
        result[f"{dim}_reasoning"] = dim_result.reasoning
    result["overall_winner"] = scores.overall_winner
    result["overall_score"] = scores.overall_score
    return result
