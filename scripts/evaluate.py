"""Run comparative evaluation on generated diagrams.

Compares model-generated diagrams against human-drawn references using
VLM-as-Judge on four dimensions: Faithfulness, Conciseness, Readability,
Aesthetics. Uses hierarchical aggregation per paper Section 4.2.

Usage:
    python scripts/evaluate.py \
        --generated outputs/run_*/final_output.png \
        --reference data/reference_sets/images/paper.jpg \
        --context method.txt \
        --caption "Overview of the proposed framework"
"""

from __future__ import annotations

import argparse
import asyncio
import glob
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


async def evaluate_single(
    image_path: str,
    reference_path: str,
    context: str,
    caption: str,
):
    """Evaluate a single generated image against a human reference."""
    from paperbanana.core.config import Settings
    from paperbanana.evaluation.judge import VLMJudge
    from paperbanana.evaluation.metrics import format_scores
    from paperbanana.providers.registry import ProviderRegistry

    settings = Settings()
    vlm = ProviderRegistry.create_vlm(settings)
    judge = VLMJudge(vlm)

    scores = await judge.evaluate(
        image_path=image_path,
        source_context=context,
        caption=caption,
        reference_path=reference_path,
    )

    print(f"\nResults for: {image_path}")
    print(f"  Reference:     {reference_path}")
    print(format_scores(scores))

    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated diagrams via comparative VLM-as-Judge"
    )
    parser.add_argument(
        "--generated", required=True, nargs="+",
        help="Paths to generated images (supports glob)",
    )
    parser.add_argument(
        "--reference", required=True,
        help="Path to human-drawn reference image",
    )
    parser.add_argument(
        "--context", required=True,
        help="Path to source context text file",
    )
    parser.add_argument(
        "--caption", required=True,
        help="Figure caption",
    )
    args = parser.parse_args()

    context = Path(args.context).read_text(encoding="utf-8")
    reference_path = args.reference

    if not Path(reference_path).exists():
        print(f"Reference image not found: {reference_path}")
        return

    # Expand globs
    image_paths = []
    for pattern in args.generated:
        image_paths.extend(glob.glob(pattern))

    if not image_paths:
        print("No images found matching the provided paths.")
        return

    print(f"Evaluating {len(image_paths)} image(s) against reference...")

    async def run_all():
        results = []
        for path in image_paths:
            scores = await evaluate_single(path, reference_path, context, args.caption)
            results.append((path, scores))
        return results

    results = asyncio.run(run_all())

    if len(results) > 1:
        model_wins = sum(1 for _, s in results if s.overall_winner == "Model")
        human_wins = sum(1 for _, s in results if s.overall_winner == "Human")
        ties = len(results) - model_wins - human_wins
        avg_score = sum(s.overall_score for _, s in results) / len(results)

        print(f"\n{'=' * 50}")
        print(f"Summary across {len(results)} images:")
        print(f"  Model wins:   {model_wins}")
        print(f"  Human wins:   {human_wins}")
        print(f"  Ties:         {ties}")
        print(f"  Avg score:    {avg_score:.0f}/100")


if __name__ == "__main__":
    main()
