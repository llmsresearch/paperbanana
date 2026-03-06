"""VLM-as-Judge evaluation using referenced comparison (paper Section 4.2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import structlog

from paperbanana.core.types import (
    VALID_WINNERS,
    WINNER_SCORE_MAP,
    DimensionResult,
    EvaluationScore,
)
from paperbanana.core.utils import load_image
from paperbanana.providers.base import VLMProvider

logger = structlog.get_logger()

DIMENSIONS = ["faithfulness", "conciseness", "readability", "aesthetics"]

# Primary dimensions take precedence in hierarchical aggregation
PRIMARY_DIMENSIONS = ["faithfulness", "readability"]
SECONDARY_DIMENSIONS = ["conciseness", "aesthetics"]


class VLMJudge:
    """Evaluates generated illustrations using a VLM as judge.

    Implements the paper's referenced comparison approach:
    - Compares model-generated diagram against human-drawn reference
    - Four dimensions: Faithfulness, Conciseness, Readability, Aesthetics
    - Hierarchical aggregation: Primary (Faithfulness + Readability) then
      Secondary (Conciseness + Aesthetics)
    """

    def __init__(self, vlm_provider: VLMProvider, prompt_dir: str = "prompts"):
        self.vlm = vlm_provider
        self.prompt_dir = Path(prompt_dir)

    async def evaluate(
        self,
        image_path: str,
        source_context: str,
        caption: str,
        reference_path: str,
    ) -> EvaluationScore:
        """Evaluate a generated image by comparing against a human reference.

        Args:
            image_path: Path to the model-generated image.
            source_context: Original methodology text.
            caption: Figure caption.
            reference_path: Path to the human-drawn reference image.

        Returns:
            EvaluationScore with comparative results and hierarchical overall.
        """
        model_image = load_image(image_path)
        reference_image = load_image(reference_path)

        # Both images: [Human reference, Model generated]
        images = [reference_image, model_image]

        results: dict[str, DimensionResult] = {}

        for dim in DIMENSIONS:
            logger.info("Evaluating dimension", dimension=dim)

            prompt = self._load_eval_prompt(dim, source_context, caption)

            response = await self.vlm.generate(
                prompt=prompt,
                images=images,
                temperature=0.1,
                max_tokens=1024,
                response_format="json",
            )

            results[dim] = self._parse_result(response, dim)

        # Hierarchical aggregation
        overall_winner = self._hierarchical_aggregate(results)
        overall_score = WINNER_SCORE_MAP.get(overall_winner, 50.0)

        return EvaluationScore(
            faithfulness=results["faithfulness"],
            conciseness=results["conciseness"],
            readability=results["readability"],
            aesthetics=results["aesthetics"],
            overall_winner=overall_winner,
            overall_score=overall_score,
        )

    def _load_eval_prompt(self, dimension: str, source_context: str, caption: str) -> str:
        """Load evaluation prompt for a specific dimension."""
        prompt_path = self.prompt_dir / "evaluation" / f"{dimension}.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Evaluation prompt not found: {prompt_path}")

        template = prompt_path.read_text(encoding="utf-8")
        return template.format(source_context=source_context, caption=caption)

    def _parse_result(self, response: str, dimension: str) -> DimensionResult:
        """Parse a comparative result from VLM response."""
        try:
            data = json.loads(response)
            winner = data.get("winner", "Both are good")
            reasoning = data.get("comparison_reasoning", "")

            # Validate winner value
            if winner not in VALID_WINNERS:
                logger.warning(
                    "Invalid winner value, defaulting to tie",
                    dimension=dimension,
                    winner=winner,
                )
                winner = "Both are good"

            score = WINNER_SCORE_MAP.get(winner, 50.0)
            return DimensionResult(winner=winner, score=score, reasoning=reasoning)
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(
                "Failed to parse evaluation response",
                dimension=dimension,
                error=str(e),
            )
            return DimensionResult(
                winner="Both are good",
                score=50.0,
                reasoning="Could not parse evaluation response.",
            )

    def _hierarchical_aggregate(self, results: dict[str, DimensionResult]) -> str:
        """Apply hierarchical aggregation per paper Section 4.2.

        Primary dimensions (Faithfulness + Readability) take precedence.
        If primary dimensions yield a decisive winner, that determines
        the overall result. Otherwise, secondary dimensions (Conciseness +
        Aesthetics) break the tie using the same logic.
        """
        primary_winner = self._aggregate_pair(
            results[PRIMARY_DIMENSIONS[0]].winner,
            results[PRIMARY_DIMENSIONS[1]].winner,
        )

        if primary_winner is not None:
            return primary_winner

        # Primary dimensions tied — fall back to secondary
        secondary_winner = self._aggregate_pair(
            results[SECONDARY_DIMENSIONS[0]].winner,
            results[SECONDARY_DIMENSIONS[1]].winner,
        )

        if secondary_winner is not None:
            return secondary_winner

        # Complete tie
        return "Both are good"

    def _aggregate_pair(self, w1: str, w2: str) -> Optional[str]:
        """Aggregate two dimension winners into a decisive result.

        Returns a decisive winner if one exists, or None if tied.

        Decisive: wins both, or wins one with a tie.
        Tied: each wins one, or both tie.
        """
        s1 = self._winner_to_side(w1)
        s2 = self._winner_to_side(w2)

        # Both decisive for the same side
        if s1 == s2 and s1 in ("Model", "Human"):
            return s1

        # One decisive, one tie
        if s1 in ("Model", "Human") and s2 == "Tie":
            return s1
        if s2 in ("Model", "Human") and s1 == "Tie":
            return s2

        # Tied: both tie, or split (one Model, one Human)
        return None

    def _winner_to_side(self, winner: str) -> str:
        """Map winner string to side: 'Model', 'Human', or 'Tie'."""
        if winner == "Model":
            return "Model"
        if winner == "Human":
            return "Human"
        # "Both are good" and "Both are bad" are both ties
        return "Tie"
