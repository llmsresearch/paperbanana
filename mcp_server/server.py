"""PaperBanana MCP Server.

Exposes PaperBanana's core functionality as MCP tools usable from
Claude Code, Cursor, or any MCP client.

Tools:
    generate_diagram — Generate a methodology diagram from text
    generate_plot    — Generate a statistical plot from JSON data
    evaluate_diagram — Evaluate a generated diagram against a reference

Usage:
    paperbanana-mcp          # stdio transport (default)
"""

from __future__ import annotations

import json

from fastmcp import FastMCP
from fastmcp.utilities.types import Image

from paperbanana.core.config import Settings
from paperbanana.core.pipeline import PaperBananaPipeline
from paperbanana.core.types import DiagramType, GenerationInput
from paperbanana.evaluation.judge import VLMJudge
from paperbanana.providers.registry import ProviderRegistry

mcp = FastMCP("PaperBanana")


@mcp.tool
async def generate_diagram(
    source_context: str,
    caption: str,
    iterations: int = 3,
) -> Image:
    """Generate a publication-quality methodology diagram from text.

    Args:
        source_context: Methodology section text or relevant paper excerpt.
        caption: Figure caption describing what the diagram should communicate.
        iterations: Number of refinement iterations (default 3).

    Returns:
        The generated diagram as a PNG image.
    """
    settings = Settings(refinement_iterations=iterations)
    pipeline = PaperBananaPipeline(settings=settings)

    gen_input = GenerationInput(
        source_context=source_context,
        communicative_intent=caption,
        diagram_type=DiagramType.METHODOLOGY,
    )

    result = await pipeline.generate(gen_input)
    return Image(path=result.image_path)


@mcp.tool
async def generate_plot(
    data_json: str,
    intent: str,
    iterations: int = 3,
) -> Image:
    """Generate a publication-quality statistical plot from JSON data.

    Args:
        data_json: JSON string containing the data to plot.
            Example: '{"x": [1,2,3], "y": [4,5,6], "labels": ["a","b","c"]}'
        intent: Description of the desired plot (e.g. "Bar chart comparing model accuracy").
        iterations: Number of refinement iterations (default 3).

    Returns:
        The generated plot as a PNG image.
    """
    raw_data = json.loads(data_json)

    settings = Settings(refinement_iterations=iterations)
    pipeline = PaperBananaPipeline(settings=settings)

    gen_input = GenerationInput(
        source_context=f"Data for plotting:\n{data_json}",
        communicative_intent=intent,
        diagram_type=DiagramType.STATISTICAL_PLOT,
        raw_data=raw_data,
    )

    result = await pipeline.generate(gen_input)
    return Image(path=result.image_path)


@mcp.tool
async def evaluate_diagram(
    generated_path: str,
    reference_path: str,
    context: str,
    caption: str,
) -> str:
    """Evaluate a generated diagram against a human reference on 4 dimensions.

    Compares the model-generated image to a human-drawn reference using
    Faithfulness, Conciseness, Readability, and Aesthetics scoring with
    hierarchical aggregation.

    Args:
        generated_path: File path to the model-generated image.
        reference_path: File path to the human-drawn reference image.
        context: Original methodology text used to generate the diagram.
        caption: Figure caption describing what the diagram communicates.

    Returns:
        Formatted evaluation scores with per-dimension results and overall winner.
    """
    settings = Settings()
    vlm = ProviderRegistry.create_vlm(settings)
    judge = VLMJudge(vlm_provider=vlm)

    scores = await judge.evaluate(
        image_path=generated_path,
        source_context=context,
        caption=caption,
        reference_path=reference_path,
    )

    lines = [
        "Evaluation Results",
        "=" * 40,
        f"Faithfulness:  {scores.faithfulness.winner} — {scores.faithfulness.reasoning}",
        f"Conciseness:   {scores.conciseness.winner} — {scores.conciseness.reasoning}",
        f"Readability:   {scores.readability.winner} — {scores.readability.reasoning}",
        f"Aesthetics:    {scores.aesthetics.winner} — {scores.aesthetics.reasoning}",
        "-" * 40,
        f"Overall Winner: {scores.overall_winner} (score: {scores.overall_score})",
    ]
    return "\n".join(lines)


def main():
    """MCP server entry point."""
    mcp.run()


if __name__ == "__main__":
    main()
