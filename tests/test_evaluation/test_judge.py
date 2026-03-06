"""Tests for VLMJudge hierarchical aggregation logic."""

from __future__ import annotations

import json

from paperbanana.core.types import DimensionResult
from paperbanana.evaluation.judge import VLMJudge


class MockVLM:
    """Mock VLM provider for testing."""

    name = "mock"
    model_name = "mock-model"

    def __init__(self, responses: dict[str, str] | None = None):
        self._responses = responses or {}
        self._call_count = 0

    async def generate(
        self,
        prompt,
        images=None,
        system_prompt=None,
        temperature=1.0,
        max_tokens=4096,
        response_format=None,
    ):
        self._call_count += 1
        # Return responses in order of dimensions
        dims = ["faithfulness", "conciseness", "readability", "aesthetics"]
        idx = self._call_count - 1
        if idx < len(dims) and dims[idx] in self._responses:
            return self._responses[dims[idx]]
        return json.dumps(
            {
                "comparison_reasoning": "Default tie.",
                "winner": "Both are good",
            }
        )

    def is_available(self):
        return True


def _make_judge(responses: dict[str, str] | None = None) -> VLMJudge:
    vlm = MockVLM(responses)
    return VLMJudge(vlm, prompt_dir="prompts")


def test_parse_result_model_wins():
    """Test parsing a Model wins response."""
    judge = _make_judge()
    result = judge._parse_result(
        json.dumps(
            {
                "comparison_reasoning": "Model is better.",
                "winner": "Model",
            }
        ),
        "faithfulness",
    )
    assert result.winner == "Model"
    assert result.score == 100.0


def test_parse_result_human_wins():
    """Test parsing a Human wins response."""
    judge = _make_judge()
    result = judge._parse_result(
        json.dumps(
            {
                "comparison_reasoning": "Human is better.",
                "winner": "Human",
            }
        ),
        "faithfulness",
    )
    assert result.winner == "Human"
    assert result.score == 0.0


def test_parse_result_tie():
    """Test parsing a tie response."""
    judge = _make_judge()
    result = judge._parse_result(
        json.dumps(
            {
                "comparison_reasoning": "Both good.",
                "winner": "Both are good",
            }
        ),
        "readability",
    )
    assert result.winner == "Both are good"
    assert result.score == 50.0


def test_parse_result_invalid_json():
    """Test fallback when response is not valid JSON."""
    judge = _make_judge()
    result = judge._parse_result("not json", "faithfulness")
    assert result.winner == "Both are good"
    assert result.score == 50.0


def test_parse_result_invalid_winner():
    """Test fallback when winner value is invalid."""
    judge = _make_judge()
    result = judge._parse_result(
        json.dumps({"winner": "InvalidValue"}),
        "faithfulness",
    )
    assert result.winner == "Both are good"


# --- Hierarchical aggregation tests ---


def _dim(winner: str) -> DimensionResult:
    from paperbanana.core.types import WINNER_SCORE_MAP

    return DimensionResult(
        winner=winner,
        score=WINNER_SCORE_MAP.get(winner, 50.0),
    )


def test_aggregate_model_wins_both_primary():
    """Model wins both primary dims -> Model overall."""
    judge = _make_judge()
    results = {
        "faithfulness": _dim("Model"),
        "readability": _dim("Model"),
        "conciseness": _dim("Human"),
        "aesthetics": _dim("Human"),
    }
    assert judge._hierarchical_aggregate(results) == "Model"


def test_aggregate_human_wins_both_primary():
    """Human wins both primary dims -> Human overall."""
    judge = _make_judge()
    results = {
        "faithfulness": _dim("Human"),
        "readability": _dim("Human"),
        "conciseness": _dim("Model"),
        "aesthetics": _dim("Model"),
    }
    assert judge._hierarchical_aggregate(results) == "Human"


def test_aggregate_model_wins_one_primary_tie_other():
    """Model wins one primary, other ties -> Model overall."""
    judge = _make_judge()
    results = {
        "faithfulness": _dim("Model"),
        "readability": _dim("Both are good"),
        "conciseness": _dim("Human"),
        "aesthetics": _dim("Human"),
    }
    assert judge._hierarchical_aggregate(results) == "Model"


def test_aggregate_primary_split_falls_to_secondary():
    """Primary split (Model + Human) -> falls to secondary."""
    judge = _make_judge()
    results = {
        "faithfulness": _dim("Model"),
        "readability": _dim("Human"),
        "conciseness": _dim("Model"),
        "aesthetics": _dim("Both are good"),
    }
    # Primary: split -> secondary: Model + Tie -> Model
    assert judge._hierarchical_aggregate(results) == "Model"


def test_aggregate_all_tie():
    """All dimensions tie -> Both are good overall."""
    judge = _make_judge()
    results = {
        "faithfulness": _dim("Both are good"),
        "readability": _dim("Both are good"),
        "conciseness": _dim("Both are good"),
        "aesthetics": _dim("Both are good"),
    }
    assert judge._hierarchical_aggregate(results) == "Both are good"


def test_aggregate_primary_tie_secondary_human():
    """Primary both tie, secondary: Human wins -> Human overall."""
    judge = _make_judge()
    results = {
        "faithfulness": _dim("Both are good"),
        "readability": _dim("Both are good"),
        "conciseness": _dim("Human"),
        "aesthetics": _dim("Human"),
    }
    assert judge._hierarchical_aggregate(results) == "Human"


def test_aggregate_complete_split():
    """Primary split + secondary split -> Both are good (complete tie)."""
    judge = _make_judge()
    results = {
        "faithfulness": _dim("Model"),
        "readability": _dim("Human"),
        "conciseness": _dim("Human"),
        "aesthetics": _dim("Model"),
    }
    assert judge._hierarchical_aggregate(results) == "Both are good"
