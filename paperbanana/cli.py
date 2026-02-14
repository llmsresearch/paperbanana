"""PaperBanana CLI — Generate publication-quality academic illustrations."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt

from paperbanana.core.config import Settings
from paperbanana.core.types import DiagramType, GenerationInput

app = typer.Typer(
    name="paperbanana",
    help="Generate publication-quality academic illustrations from text.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def generate(
    input: str = typer.Option(..., "--input", "-i", help="Path to methodology text file"),
    caption: str = typer.Option(
        ..., "--caption", "-c", help="Figure caption / communicative intent"
    ),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output image path"),
    vlm_provider: Optional[str] = typer.Option(
        None, "--vlm-provider", help="VLM provider (gemini)"
    ),
    vlm_model: Optional[str] = typer.Option(None, "--vlm-model", help="VLM model name"),
    image_provider: Optional[str] = typer.Option(
        None, "--image-provider", help="Image gen provider"
    ),
    image_model: Optional[str] = typer.Option(None, "--image-model", help="Image gen model name"),
    iterations: Optional[int] = typer.Option(
        None, "--iterations", "-n", help="Refinement iterations"
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Path to config YAML file"),
):
    """Generate a methodology diagram from a text description."""
    # Load source text
    input_path = Path(input)
    if not input_path.exists():
        console.print(f"[red]Error: Input file not found: {input}[/red]")
        raise typer.Exit(1)

    source_context = input_path.read_text(encoding="utf-8")

    # Build settings — only override values explicitly passed via CLI
    overrides = {}
    if vlm_provider:
        overrides["vlm_provider"] = vlm_provider
    if vlm_model:
        overrides["vlm_model"] = vlm_model
    if image_provider:
        overrides["image_provider"] = image_provider
    if image_model:
        overrides["image_model"] = image_model
    if iterations is not None:
        overrides["refinement_iterations"] = iterations
    if output:
        overrides["output_dir"] = str(Path(output).parent)

    if config:
        settings = Settings.from_yaml(config, **overrides)
    else:
        from dotenv import load_dotenv

        load_dotenv()
        settings = Settings(**overrides)

    # Build generation input
    gen_input = GenerationInput(
        source_context=source_context,
        communicative_intent=caption,
        diagram_type=DiagramType.METHODOLOGY,
    )

    console.print(
        Panel.fit(
            f"[bold]PaperBanana[/bold] - Generating Methodology Diagram\n\n"
            f"VLM: {settings.vlm_provider} / {settings.vlm_model}\n"
            f"Image: {settings.image_provider} / {settings.image_model}\n"
            f"Iterations: {settings.refinement_iterations}",
            border_style="blue",
        )
    )

    # Run pipeline
    from paperbanana.core.pipeline import PaperBananaPipeline

    async def _run():
        pipeline = PaperBananaPipeline(settings=settings)
        return await pipeline.generate(gen_input)

    with Progress(
        SpinnerColumn(spinner_name="line"),  # ASCII-safe spinner for Windows compatibility
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Generating diagram...", total=None)
        result = asyncio.run(_run())

    console.print(f"\n[green]Done![/green] Output saved to: [bold]{result.image_path}[/bold]")
    console.print(f"Run ID: {result.metadata.get('run_id', 'unknown')}")
    console.print(f"Total iterations: {len(result.iterations)}")


@app.command()
def plot(
    data: str = typer.Option(..., "--data", "-d", help="Path to data file (CSV or JSON)"),
    intent: str = typer.Option(..., "--intent", help="Communicative intent for the plot"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output image path"),
    vlm_provider: str = typer.Option("gemini", "--vlm-provider", help="VLM provider"),
    iterations: int = typer.Option(3, "--iterations", "-n", help="Number of refinement iterations"),
):
    """Generate a statistical plot from data."""
    data_path = Path(data)
    if not data_path.exists():
        console.print(f"[red]Error: Data file not found: {data}[/red]")
        raise typer.Exit(1)

    # Load data
    import json as json_mod

    if data_path.suffix == ".csv":
        import pandas as pd

        df = pd.read_csv(data_path)
        raw_data = df.to_dict(orient="records")
        source_context = (
            f"CSV data with columns: {list(df.columns)}\n"
            f"Rows: {len(df)}\nSample:\n{df.head().to_string()}"
        )
    else:
        with open(data_path) as f:
            raw_data = json_mod.load(f)
        source_context = f"JSON data:\n{json_mod.dumps(raw_data, indent=2)[:2000]}"

    from dotenv import load_dotenv

    load_dotenv()

    settings = Settings(
        vlm_provider=vlm_provider,
        refinement_iterations=iterations,
    )

    gen_input = GenerationInput(
        source_context=source_context,
        communicative_intent=intent,
        diagram_type=DiagramType.STATISTICAL_PLOT,
        raw_data={"data": raw_data},
    )

    console.print(
        Panel.fit(
            f"[bold]PaperBanana[/bold] - Generating Statistical Plot\n\n"
            f"Data: {data_path.name}\n"
            f"Intent: {intent}",
            border_style="green",
        )
    )

    from paperbanana.core.pipeline import PaperBananaPipeline

    async def _run():
        pipeline = PaperBananaPipeline(settings=settings)
        return await pipeline.generate(gen_input)

    result = asyncio.run(_run())
    console.print(f"\n[green]Done![/green] Plot saved to: [bold]{result.image_path}[/bold]")


@app.command()
def setup():
    """Interactive setup wizard — get generating in 2 minutes with FREE APIs."""
    console.print(
        Panel.fit(
            "[bold]Welcome to PaperBanana Setup[/bold]\n\n"
            "We'll set up FREE API keys so you can start generating diagrams.",
            border_style="yellow",
        )
    )

    console.print("\n[bold]Step 1: Google Gemini API Key[/bold] (FREE, no credit card)")
    console.print("This powers the AI agents that plan and critique your diagrams.\n")

    import webbrowser

    open_browser = Prompt.ask(
        "Open browser to get a free Gemini API key?",
        choices=["y", "n"],
        default="y",
    )
    if open_browser == "y":
        webbrowser.open("https://makersuite.google.com/app/apikey")

    gemini_key = Prompt.ask("\nPaste your Gemini API key")

    # Save to .env
    env_path = Path(".env")
    lines = []
    lines.append(f"GOOGLE_API_KEY={gemini_key}")

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    console.print(f"\n[green]Setup complete![/green] Keys saved to {env_path}")
    console.print("\nTry it out:")
    console.print(
        "  [bold]paperbanana generate --input method.txt"
        " --caption 'Overview of our framework'[/bold]"
    )


@app.command()
def evaluate(
    generated: str = typer.Option(..., "--generated", "-g", help="Path to generated image"),
    context: str = typer.Option(..., "--context", help="Path to source context text file"),
    caption: str = typer.Option(..., "--caption", "-c", help="Figure caption"),
    reference: str = typer.Option(..., "--reference", "-r", help="Path to human reference image"),
    vlm_provider: str = typer.Option(
        "gemini", "--vlm-provider", help="VLM provider for evaluation"
    ),
):
    """Evaluate a generated diagram vs human reference (comparative)."""
    from paperbanana.evaluation.judge import VLMJudge

    generated_path = Path(generated)
    if not generated_path.exists():
        console.print(f"[red]Error: Generated image not found: {generated}[/red]")
        raise typer.Exit(1)

    reference_path = Path(reference)
    if not reference_path.exists():
        console.print(f"[red]Error: Reference image not found: {reference}[/red]")
        raise typer.Exit(1)

    context_text = Path(context).read_text(encoding="utf-8")

    from dotenv import load_dotenv

    load_dotenv()

    settings = Settings(vlm_provider=vlm_provider)
    from paperbanana.providers.registry import ProviderRegistry

    vlm = ProviderRegistry.create_vlm(settings)

    judge = VLMJudge(vlm)

    async def _run():
        return await judge.evaluate(
            image_path=str(generated_path),
            source_context=context_text,
            caption=caption,
            reference_path=str(reference_path),
        )

    scores = asyncio.run(_run())

    dims = ["faithfulness", "conciseness", "readability", "aesthetics"]
    dim_lines = []
    for dim in dims:
        result = getattr(scores, dim)
        dim_lines.append(f"{dim.capitalize():14s} {result.winner}")

    console.print(
        Panel.fit(
            "[bold]Evaluation Results (Comparative)[/bold]\n\n"
            + "\n".join(dim_lines)
            + f"\n[bold]{'Overall':14s} {scores.overall_winner}[/bold]",
            border_style="cyan",
        )
    )

    for dim in dims:
        result = getattr(scores, dim)
        if result.reasoning:
            console.print(f"\n[bold]{dim}[/bold]: {result.reasoning}")


@app.command()
def version():
    """Show version information."""
    import sys
    from importlib.metadata import version

    try:
        pkg_version = version("paperbanana")
    except Exception:
        pkg_version = "unknown"

    console.print(f"PaperBanana version: [bold]{pkg_version}[/bold]")
    console.print(f"Python version:      [bold]{sys.version.split()[0]}[/bold]")


if __name__ == "__main__":
    app()
