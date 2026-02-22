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
import os
from io import BytesIO
from pathlib import Path

import structlog
from fastmcp import FastMCP
from fastmcp.utilities.types import Image
from PIL import Image as PILImage

from paperbanana.core.config import Settings
from paperbanana.core.pipeline import PaperBananaPipeline
from paperbanana.core.types import DiagramType, GenerationInput
from paperbanana.core.utils import detect_image_mime_type, find_prompt_dir
from paperbanana.evaluation.judge import VLMJudge
from paperbanana.providers.registry import ProviderRegistry

logger = structlog.get_logger()

# Claude API enforces a 5 MB limit on base64-encoded images in tool results.
# Base64 inflates raw bytes by ~4/3, so we cap the raw file at 3.75 MB to
# stay safely under the wire.
_MAX_IMAGE_BYTES = int(os.environ.get("PAPERBANANA_MAX_IMAGE_BYTES", 3_750_000))


def _compress_for_api(image_path: str) -> tuple[str, str]:
    """Return *(effective_path, format)* for an image that fits the API limit.

    If the file at *image_path* already fits, returns it as-is.  Otherwise the
    image is re-saved as optimised JPEG (which is dramatically smaller for the
    photographic output typical of AI image generators) next to the original.
    """
    raw_size = Path(image_path).stat().st_size
    mime = detect_image_mime_type(image_path)
    fmt = mime.split("/")[1]  # e.g. "png", "jpeg"

    if raw_size <= _MAX_IMAGE_BYTES:
        return image_path, fmt

    logger.info(
        "Image exceeds API size limit, compressing to JPEG",
        original_bytes=raw_size,
        limit=_MAX_IMAGE_BYTES,
    )

    img = PILImage.open(image_path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")

    compressed_path = str(Path(image_path).with_suffix(".mcp.jpg"))

    # Try quality 85 first; fall back to progressively lower quality.
    for quality in (85, 70, 50):
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        if buf.tell() <= _MAX_IMAGE_BYTES:
            Path(compressed_path).write_bytes(buf.getvalue())
            logger.info(
                "Compressed image saved",
                quality=quality,
                compressed_bytes=buf.tell(),
            )
            return compressed_path, "jpeg"

    # Last resort: scale down.
    for scale in (0.75, 0.5):
        resized = img.resize(
            (int(img.width * scale), int(img.height * scale)),
            PILImage.LANCZOS,
        )
        buf = BytesIO()
        resized.save(buf, format="JPEG", quality=70, optimize=True)
        if buf.tell() <= _MAX_IMAGE_BYTES:
            Path(compressed_path).write_bytes(buf.getvalue())
            logger.info(
                "Resized and compressed image saved",
                scale=scale,
                compressed_bytes=buf.tell(),
            )
            return compressed_path, "jpeg"

    # Give up gracefully — return whatever we have.
    Path(compressed_path).write_bytes(buf.getvalue())
    return compressed_path, "jpeg"


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
    effective_path, fmt = _compress_for_api(result.image_path)
    return Image(path=effective_path, format=fmt)


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
    effective_path, fmt = _compress_for_api(result.image_path)
    return Image(path=effective_path, format=fmt)


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
    judge = VLMJudge(vlm_provider=vlm, prompt_dir=find_prompt_dir())

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
