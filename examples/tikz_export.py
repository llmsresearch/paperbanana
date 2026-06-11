"""End-to-end TikZ export example.

Demonstrates three usage patterns:

1. Generate a methodology diagram and export TikZ in a single pipeline run.
2. Export TikZ from an already-generated image (post-hoc, via the `tikz` CLI
   command or the TikZExporterAgent API directly).
3. Export PGFPlots from a generated statistical plot.

Run any section independently — each is guarded by a ``# --- `` comment.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

# ---------------------------------------------------------------------------
# Pattern 1: Generate + export TikZ in one pipeline run
# ---------------------------------------------------------------------------
# CLI equivalent:
#   paperbanana generate \
#     --input examples/sample_inputs/transformer_method.txt \
#     --caption "Overview of our Transformer encoder architecture" \
#     --export-tikz
#
# The pipeline saves:
#   outputs/<run_id>/final_output.png   ← raster image
#   outputs/<run_id>/final_output.tex   ← TikZ source (new)


async def generate_with_tikz_export(
    input_path: str = "examples/sample_inputs/transformer_method.txt",
    caption: str = "Overview of our Transformer encoder architecture",
) -> None:
    """Run the full pipeline and export TikZ alongside the PNG."""
    from dotenv import load_dotenv

    load_dotenv()

    from paperbanana.core.config import Settings
    from paperbanana.core.pipeline import PaperBananaPipeline
    from paperbanana.core.types import DiagramType, GenerationInput

    settings = Settings(export_tikz=True)

    gen_input = GenerationInput(
        source_context=Path(input_path).read_text(encoding="utf-8"),
        communicative_intent=caption,
        diagram_type=DiagramType.METHODOLOGY,
    )

    pipeline = PaperBananaPipeline(settings=settings)
    result = await pipeline.generate(gen_input)

    print(f"Image : {result.image_path}")
    print(f"TikZ  : {result.tikz_path}")


# ---------------------------------------------------------------------------
# Pattern 2: Post-hoc export from an existing image
# ---------------------------------------------------------------------------
# CLI equivalent:
#   paperbanana tikz \
#     --input outputs/my_run/final_output.png \
#     --source-context examples/sample_inputs/transformer_method.txt \
#     --caption "Overview of our Transformer encoder architecture"


async def export_existing_image(
    image_path: str,
    source_context_path: str = "",
    caption: str = "",
    output_tex: str = "",
) -> str:
    """Convert an existing generated image to TikZ source.

    Args:
        image_path: Path to the PNG/JPEG image to convert.
        source_context_path: Optional path to the methodology text file.
        caption: Optional figure caption for context.
        output_tex: Where to write the .tex file (defaults to same dir as image).

    Returns:
        Path to the saved .tex file.
    """
    from dotenv import load_dotenv

    load_dotenv()

    from paperbanana.agents.tikz_exporter import TikZExporterAgent
    from paperbanana.core.config import Settings
    from paperbanana.core.types import DiagramType
    from paperbanana.providers.registry import ProviderRegistry

    settings = Settings()
    vlm = ProviderRegistry.create_vlm(settings)
    agent = TikZExporterAgent(vlm)

    context_text = ""
    if source_context_path:
        context_text = Path(source_context_path).read_text(encoding="utf-8")

    tikz_source = await agent.run(
        image_path=image_path,
        source_context=context_text,
        caption=caption,
        diagram_type=DiagramType.METHODOLOGY,
        venue=settings.venue,
    )

    tex_path = Path(output_tex) if output_tex else Path(image_path).with_suffix(".tex")
    tex_path.write_text(tikz_source, encoding="utf-8")
    print(f"TikZ source saved to: {tex_path}")
    return str(tex_path)


# ---------------------------------------------------------------------------
# Pattern 3: Generate a statistical plot and export PGFPlots
# ---------------------------------------------------------------------------
# CLI equivalent:
#   paperbanana plot \
#     --data examples/sample_data/benchmark_slice.csv \
#     --intent "Accuracy vs. model size across four baselines" \
#     --export-pgfplots


async def generate_plot_with_pgfplots(
    data_path: str = "examples/sample_data/benchmark_slice.csv",
    intent: str = "Accuracy vs. model size across four baselines",
) -> None:
    """Run the plot pipeline and export PGFPlots markup alongside the PNG."""
    from dotenv import load_dotenv

    load_dotenv()

    from paperbanana.core.config import Settings
    from paperbanana.core.pipeline import PaperBananaPipeline
    from paperbanana.core.plot_data import load_statistical_plot_payload
    from paperbanana.core.types import DiagramType, GenerationInput

    source_context, raw_data = load_statistical_plot_payload(Path(data_path))

    settings = Settings(export_pgfplots=True)

    gen_input = GenerationInput(
        source_context=source_context,
        communicative_intent=intent,
        diagram_type=DiagramType.STATISTICAL_PLOT,
        raw_data={"data": raw_data},
    )

    pipeline = PaperBananaPipeline(settings=settings)
    result = await pipeline.generate(gen_input)

    print(f"Plot      : {result.image_path}")
    print(f"PGFPlots  : {result.tikz_path}")


# ---------------------------------------------------------------------------
# Main — run pattern 1 as a quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(generate_with_tikz_export())
