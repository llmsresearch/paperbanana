"""PaperBanana MCP Server.

Exposes PaperBanana's core functionality as MCP tools usable from
Claude Code, Cursor, or any MCP client.

Tools:
    generate_diagram    — Generate a methodology diagram from text
    generate_plot       — Generate a statistical plot from JSON data
    evaluate_diagram    — Evaluate a generated diagram against a reference
    evaluate_plot       — Evaluate a generated plot against a reference
    download_references — Download expanded reference set (~294 examples)
    orchestrate_figures — Full-paper figure package (plan + optional generation)
    batch_diagrams      — Batch methodology diagrams from a YAML/JSON manifest
    batch_plots         — Batch statistical plots from a YAML/JSON manifest

Usage:
    paperbanana-mcp          # stdio transport (default)
"""

from __future__ import annotations

import asyncio
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
from paperbanana.core.workflow_runner import (
    run_methodology_batch,
    run_orchestration_package,
    run_plot_batch,
)
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

    Raises ``ValueError`` if the image cannot be compressed below the limit
    after all quality and resize attempts.
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
    for scale in (0.75, 0.5, 0.25):
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

    raise ValueError(
        f"Image at {image_path} ({raw_size} bytes) could not be "
        f"compressed below the {_MAX_IMAGE_BYTES} byte API limit."
    )


def _embed_caption(image_path: str, caption: str) -> None:
    """Embed a caption into a PNG image's tEXt metadata chunk.

    This preserves the MCP return type (always ``Image``) while making
    the caption accessible to any client that reads PNG metadata.
    Non-PNG files or write errors are silently ignored.
    """
    try:
        from PIL.PngImagePlugin import PngInfo

        img = PILImage.open(image_path)
        if img.format != "PNG":
            return
        meta = PngInfo()
        # Carry over existing text chunks
        existing = img.info or {}
        for k, v in existing.items():
            if isinstance(v, str):
                meta.add_text(k, v)
        meta.add_text("Caption", caption)
        img.save(image_path, pnginfo=meta)
    except Exception:
        logger.debug("Failed to embed caption in image metadata")


mcp = FastMCP("PaperBanana")


@mcp.tool
async def generate_diagram(
    source_context: str,
    caption: str,
    iterations: int = 3,
    aspect_ratio: str | None = None,
    optimize: bool = False,
    auto_refine: bool = False,
    generate_caption: bool = False,
) -> Image:
    """Generate a publication-quality methodology diagram from text.

    Args:
        source_context: Methodology section text or relevant paper excerpt.
        caption: Figure caption describing what the diagram should communicate.
        iterations: Number of refinement iterations (default 3, used when auto_refine=False).
        aspect_ratio: Target aspect ratio. Supported:
            1:1, 2:3, 3:2, 3:4, 4:3, 9:16, 16:9, 21:9. Default: landscape.
        optimize: Enrich context and sharpen caption before generation (default True).
            Set False to skip preprocessing for faster results.
        auto_refine: Let critic loop until satisfied (default True, max 30 iterations).
            Set False to use fixed iteration count for faster results.
        generate_caption: Auto-generate a publication-ready figure caption
            after generation. When True, the caption is embedded in the
            image metadata (PNG tEXt chunk, key "Caption") and logged.

    Returns:
        The generated diagram as a PNG image.
    """
    settings = Settings(
        refinement_iterations=iterations,
        optimize_inputs=optimize,
        auto_refine=auto_refine,
        generate_caption=generate_caption,
    )

    def _on_progress(event: str, payload: dict) -> None:
        logger.info(
            "mcp_progress",
            tool="generate_diagram",
            progress_event=event,
            **payload,
        )

    pipeline = PaperBananaPipeline(settings=settings, progress_callback=_on_progress)

    gen_input = GenerationInput(
        source_context=source_context,
        communicative_intent=caption,
        diagram_type=DiagramType.METHODOLOGY,
        aspect_ratio=aspect_ratio,
    )

    result = await pipeline.generate(gen_input)
    effective_path, fmt = _compress_for_api(result.image_path)

    if result.generated_caption:
        _embed_caption(effective_path, result.generated_caption)
        logger.info(
            "generated_caption",
            tool="generate_diagram",
            caption=result.generated_caption,
        )

    return Image(path=effective_path, format=fmt)


@mcp.tool
async def generate_plot(
    data_json: str,
    intent: str,
    iterations: int = 3,
    aspect_ratio: str | None = None,
    optimize: bool = False,
    auto_refine: bool = False,
    generate_caption: bool = False,
) -> Image:
    """Generate a publication-quality statistical plot from JSON data.

    Args:
        data_json: JSON string containing the data to plot.
            Example: '{"x": [1,2,3], "y": [4,5,6], "labels": ["a","b","c"]}'
        intent: Description of the desired plot (e.g. "Bar chart comparing model accuracy").
        iterations: Number of refinement iterations (default 3, used when auto_refine=False).
        aspect_ratio: Target aspect ratio. Supported:
            1:1, 2:3, 3:2, 3:4, 4:3, 9:16, 16:9, 21:9. Default: landscape.
        optimize: Enrich context and sharpen caption before generation (default True).
            Set False to skip preprocessing for faster results.
        auto_refine: Let critic loop until satisfied (default True, max 30 iterations).
            Set False to use fixed iteration count for faster results.
        generate_caption: Auto-generate a publication-ready figure caption
            after generation. When True, the caption is embedded in the
            image metadata (PNG tEXt chunk, key "Caption") and logged.

    Returns:
        The generated plot as a PNG image.
    """
    raw_data = json.loads(data_json)

    settings = Settings(
        refinement_iterations=iterations,
        optimize_inputs=optimize,
        auto_refine=auto_refine,
        generate_caption=generate_caption,
    )

    def _on_progress(event: str, payload: dict) -> None:
        logger.info(
            "mcp_progress",
            tool="generate_plot",
            progress_event=event,
            **payload,
        )

    pipeline = PaperBananaPipeline(settings=settings, progress_callback=_on_progress)

    gen_input = GenerationInput(
        source_context=f"Data for plotting:\n{data_json}",
        communicative_intent=intent,
        diagram_type=DiagramType.STATISTICAL_PLOT,
        raw_data=raw_data,
        aspect_ratio=aspect_ratio,
    )

    result = await pipeline.generate(gen_input)
    effective_path, fmt = _compress_for_api(result.image_path)

    if result.generated_caption:
        _embed_caption(effective_path, result.generated_caption)
        logger.info(
            "generated_caption",
            tool="generate_plot",
            caption=result.generated_caption,
        )

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
        task=DiagramType.METHODOLOGY,
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


@mcp.tool
async def evaluate_plot(
    generated_path: str,
    reference_path: str,
    data_json: str,
    intent: str,
) -> str:
    """Evaluate a generated statistical plot against a human reference on 4 dimensions.

    Args:
        generated_path: File path to the model-generated plot.
        reference_path: File path to the human reference plot.
        data_json: JSON string containing the source data used to generate the plot.
        intent: Communicative intent used for plot generation.

    Returns:
        Formatted evaluation scores with per-dimension results and overall winner.
    """
    settings = Settings()
    vlm = ProviderRegistry.create_vlm(settings)
    judge = VLMJudge(vlm_provider=vlm, prompt_dir=find_prompt_dir())
    source_context = f"Data for plotting:\n{data_json}"

    scores = await judge.evaluate(
        image_path=generated_path,
        source_context=source_context,
        caption=intent,
        reference_path=reference_path,
        task=DiagramType.STATISTICAL_PLOT,
    )

    lines = [
        "Plot Evaluation Results",
        "=" * 40,
        f"Faithfulness:  {scores.faithfulness.winner} — {scores.faithfulness.reasoning}",
        f"Conciseness:   {scores.conciseness.winner} — {scores.conciseness.reasoning}",
        f"Readability:   {scores.readability.winner} — {scores.readability.reasoning}",
        f"Aesthetics:    {scores.aesthetics.winner} — {scores.aesthetics.reasoning}",
        "-" * 40,
        f"Overall Winner: {scores.overall_winner} (score: {scores.overall_score})",
    ]
    return "\n".join(lines)


@mcp.tool
async def download_references(
    force: bool = False,
) -> str:
    """Download the expanded reference set from official PaperBananaBench.

    Downloads ~257MB of reference diagrams (294 examples) from HuggingFace
    and caches them locally. The Retriever agent uses these for better
    in-context learning during diagram generation.

    Only needs to be run once — subsequent calls detect the cached data
    and return immediately. Use force=True to re-download.

    Args:
        force: Re-download even if already cached.

    Returns:
        Status message with cache location and example count.
    """
    from paperbanana.data.manager import DatasetManager

    dm = DatasetManager()

    if dm.is_downloaded() and not force:
        info = dm.get_info() or {}
        return (
            f"Expanded reference set already cached.\n"
            f"Location: {dm.reference_dir}\n"
            f"Examples: {dm.get_example_count()}\n"
            f"Version: {info.get('version', 'unknown')}\n"
            f"Use force=True to re-download."
        )

    count = dm.download(force=force)
    return (
        f"Downloaded {count} reference examples.\n"
        f"Cached to: {dm.reference_dir}\n"
        f"The Retriever agent will now use these for better diagram generation."
    )


def _json_result(payload: dict) -> str:
    return json.dumps(payload, indent=2)


@mcp.tool
async def orchestrate_figures(
    paper: str | None = None,
    resume_orchestrate: str | None = None,
    output_dir: str = "outputs",
    data_dir: str | None = None,
    max_method_figures: int = 4,
    max_plot_figures: int = 4,
    pdf_pages: str | None = None,
    dry_run: bool = False,
    config: str | None = None,
    vlm_provider: str | None = None,
    vlm_model: str | None = None,
    image_provider: str | None = None,
    image_model: str | None = None,
    iterations: int | None = None,
    auto: bool = False,
    max_iterations: int | None = None,
    optimize: bool = False,
    format: str = "png",
    save_prompts: bool | None = None,
    venue: str | None = None,
    retry_failed: bool = False,
    max_retries: int = 0,
    concurrency: int = 1,
) -> str:
    """Plan and optionally generate a multi-figure publication package from a paper.

    Mirrors ``paperbanana orchestrate``. Use ``dry_run=True`` to write
    ``orchestration_plan.json`` only (no API generation). For continuation,
    pass ``resume_orchestrate`` with an orchestration id or package directory path.

    Returns:
        JSON string with orchestration_id, paths to ``figure_package.json``,
        ``figures.tex``, ``captions.md``, ``orchestration_plan.json``, counts,
        ``strict_success``, and ``failures`` when applicable.
    """

    def _run() -> dict:
        return run_orchestration_package(
            paper=paper,
            resume_orchestrate=resume_orchestrate,
            output_dir=Path(output_dir),
            data_dir=data_dir,
            max_method_figures=max_method_figures,
            max_plot_figures=max_plot_figures,
            pdf_pages=pdf_pages,
            dry_run=dry_run,
            config=config,
            vlm_provider=vlm_provider,
            vlm_model=vlm_model,
            image_provider=image_provider,
            image_model=image_model,
            iterations=iterations,
            auto=auto,
            max_iterations=max_iterations,
            optimize=optimize,
            format=format,
            save_prompts=save_prompts,
            venue=venue,
            retry_failed=retry_failed,
            max_retries=max_retries,
            concurrency=concurrency,
            progress_callback=lambda m: logger.info("mcp_orchestrate", message=m),
        )

    try:
        result = await asyncio.to_thread(_run)
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as e:
        return _json_result({"error": str(e), "strict_success": False})
    return _json_result(result)


@mcp.tool
async def batch_diagrams(
    manifest_path: str,
    output_dir: str = "outputs",
    config: str | None = None,
    vlm_provider: str | None = None,
    vlm_model: str | None = None,
    image_provider: str | None = None,
    image_model: str | None = None,
    iterations: int | None = None,
    auto: bool = False,
    max_iterations: int | None = None,
    optimize: bool = False,
    format: str = "png",
    save_prompts: bool | None = None,
    venue: str | None = None,
    auto_download_data: bool = False,
    resume_batch: str | None = None,
    retry_failed: bool = False,
    max_retries: int = 0,
    concurrency: int = 1,
) -> str:
    """Run methodology diagram batch from a manifest (YAML or JSON).

    Each manifest item needs ``input`` (text or PDF path) and ``caption``.
    Paths are resolved relative to the manifest file directory.
    Returns JSON with ``batch_dir``, ``batch_report_path``, per-item summary,
    ``composite_path`` when configured, and ``strict_success`` (false if any item failed).
    """

    def _run() -> dict:
        return run_methodology_batch(
            manifest_path=Path(manifest_path),
            output_dir=Path(output_dir),
            config=config,
            vlm_provider=vlm_provider,
            vlm_model=vlm_model,
            image_provider=image_provider,
            image_model=image_model,
            iterations=iterations,
            auto=auto,
            max_iterations=max_iterations,
            optimize=optimize,
            format=format,
            save_prompts=save_prompts,
            venue=venue,
            auto_download_data=auto_download_data,
            resume_batch=resume_batch,
            retry_failed=retry_failed,
            max_retries=max_retries,
            concurrency=concurrency,
            progress_callback=lambda m: logger.info("mcp_batch_diagrams", message=m),
        )

    try:
        result = await asyncio.to_thread(_run)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        return _json_result({"error": str(e), "strict_success": False})
    return _json_result(result)


@mcp.tool
async def batch_plots(
    manifest_path: str,
    output_dir: str = "outputs",
    config: str | None = None,
    vlm_provider: str | None = None,
    vlm_model: str | None = None,
    image_provider: str | None = None,
    image_model: str | None = None,
    iterations: int | None = None,
    auto: bool = False,
    max_iterations: int | None = None,
    optimize: bool = False,
    format: str = "png",
    save_prompts: bool | None = None,
    venue: str | None = None,
    aspect_ratio: str | None = None,
    resume_batch: str | None = None,
    retry_failed: bool = False,
    max_retries: int = 0,
    concurrency: int = 1,
) -> str:
    """Run statistical plot batch from a manifest (YAML or JSON).

    Each item needs ``data`` (CSV or JSON path) and ``intent``. When
    ``vlm_provider`` is omitted, defaults to ``gemini`` (same as CLI plot-batch).
    Returns JSON with ``batch_dir``, ``batch_report_path``, item summary, and
    ``strict_success``.
    """

    def _run() -> dict:
        return run_plot_batch(
            manifest_path=Path(manifest_path),
            output_dir=Path(output_dir),
            config=config,
            vlm_provider=vlm_provider,
            vlm_model=vlm_model,
            image_provider=image_provider,
            image_model=image_model,
            iterations=iterations,
            auto=auto,
            max_iterations=max_iterations,
            optimize=optimize,
            format=format,
            save_prompts=save_prompts,
            venue=venue,
            aspect_ratio=aspect_ratio,
            resume_batch=resume_batch,
            retry_failed=retry_failed,
            max_retries=max_retries,
            concurrency=concurrency,
            progress_callback=lambda m: logger.info("mcp_batch_plots", message=m),
        )

    try:
        result = await asyncio.to_thread(_run)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        return _json_result({"error": str(e), "strict_success": False})
    return _json_result(result)


def main():
    """MCP server entry point."""
    mcp.run()


if __name__ == "__main__":
    main()
