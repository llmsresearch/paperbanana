"""PaperBanana CLI — Generate publication-quality academic illustrations."""
import platform
import importlib.metadata

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from paperbanana.core.config import Settings
from paperbanana.core.logging import configure_logging
from paperbanana.core.types import DiagramType, GenerationInput

app = typer.Typer(
    name="paperbanana",
    help="Generate publication-quality academic illustrations from text.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def generate(
    input: Optional[str] = typer.Option(
        None, "--input", "-i", help="Path to methodology text file"
    ),
    caption: Optional[str] = typer.Option(
        None, "--caption", "-c", help="Figure caption / communicative intent"
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
    auto: bool = typer.Option(
        False, "--auto", help="Loop until critic is satisfied (with safety cap)"
    ),
    max_iterations: Optional[int] = typer.Option(
        None, "--max-iterations", help="Safety cap for --auto mode (default: 30)"
    ),
    optimize: bool = typer.Option(
        False, "--optimize", help="Preprocess inputs for better generation (parallel enrichment)"
    ),
    continue_last: bool = typer.Option(False, "--continue", help="Continue from the latest run"),
    continue_run: Optional[str] = typer.Option(
        None, "--continue-run", help="Continue from a specific run ID"
    ),
    feedback: Optional[str] = typer.Option(
        None, "--feedback", help="User feedback for the critic when continuing a run"
    ),
    format: str = typer.Option(
        "png",
        "--format",
        "-f",
        help="Output image format (png, jpeg, or webp)",
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Path to config YAML file"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed agent progress and timing"
    ),
):
    """Generate a methodology diagram from a text description."""
    if format not in ("png", "jpeg", "webp"):
        console.print(f"[red]Error: Format must be png, jpeg, or webp. Got: {format}[/red]")
        raise typer.Exit(1)

    if feedback and not continue_run and not continue_last:
        console.print("[red]Error: --feedback requires --continue or --continue-run[/red]")
        raise typer.Exit(1)

    configure_logging(verbose=verbose)

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
    if auto:
        overrides["auto_refine"] = True
    if max_iterations is not None:
        overrides["max_iterations"] = max_iterations
    if optimize:
        overrides["optimize_inputs"] = True
    if output:
        overrides["output_dir"] = str(Path(output).parent)
    overrides["output_format"] = format

    if config:
        settings = Settings.from_yaml(config, **overrides)
    else:
        from dotenv import load_dotenv

        load_dotenv()
        settings = Settings(**overrides)

    from paperbanana.core.pipeline import PaperBananaPipeline

    # ── Continue-run mode ─────────────────────────────────────────
    if continue_run is not None or continue_last:
        from paperbanana.core.resume import find_latest_run, load_resume_state

        if continue_run:
            run_id = continue_run
        else:
            try:
                run_id = find_latest_run(settings.output_dir)
                console.print(f"  [dim]Using latest run:[/dim] [bold]{run_id}[/bold]")
            except FileNotFoundError as e:
                console.print(f"[red]Error: {e}[/red]")
                raise typer.Exit(1)

        try:
            resume_state = load_resume_state(settings.output_dir, run_id)
        except (FileNotFoundError, ValueError) as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

        iter_label = "auto" if auto else str(iterations or settings.refinement_iterations)
        console.print(
            Panel.fit(
                f"[bold]PaperBanana[/bold] - Continuing Run\n\n"
                f"Run ID: {run_id}\n"
                f"From iteration: {resume_state.last_iteration}\n"
                f"Additional iterations: {iter_label}\n"
                + (f"User feedback: {feedback[:80]}..." if feedback else ""),
                border_style="yellow",
            )
        )

        console.print()

        async def _run_continue():
            pipeline = PaperBananaPipeline(settings=settings)

            orig_visualizer_run = pipeline.visualizer.run
            orig_critic_run = pipeline.critic.run

            async def _visualizer_run(*a, **kw):
                iteration = kw.get("iteration", "")
                console.print(f"  [dim]●[/dim] Generating image (iter {iteration})...", end="")
                t = time.perf_counter()
                result = await orig_visualizer_run(*a, **kw)
                console.print(f" [green]✓[/green] [dim]{time.perf_counter() - t:.1f}s[/dim]")
                return result

            async def _critic_run(*a, **kw):
                console.print("  [dim]●[/dim] Critic reviewing...", end="")
                t = time.perf_counter()
                result = await orig_critic_run(*a, **kw)
                elapsed = time.perf_counter() - t
                console.print(f" [green]✓[/green] [dim]{elapsed:.1f}s[/dim]")
                if result.needs_revision:
                    console.print(
                        f"    [yellow]↻[/yellow] Revision needed: [dim]{result.summary}[/dim]"
                    )
                else:
                    console.print("    [green]✓[/green] [bold green]Critic satisfied[/bold green]")
                return result

            pipeline.visualizer.run = _visualizer_run
            pipeline.critic.run = _critic_run

            return await pipeline.continue_run(
                resume_state=resume_state,
                additional_iterations=iterations,
                user_feedback=feedback,
            )

        result = asyncio.run(_run_continue())

        console.print(f"\n[green]Done![/green] Output saved to: [bold]{result.image_path}[/bold]")
        console.print(f"Run ID: {result.metadata.get('run_id', 'unknown')}")
        console.print(f"New iterations: {len(result.iterations)}")
        return

    # ── Normal generation mode ────────────────────────────────────
    if not input:
        console.print("[red]Error: --input is required for new runs[/red]")
        raise typer.Exit(1)
    if not caption:
        console.print("[red]Error: --caption is required for new runs[/red]")
        raise typer.Exit(1)

    # Load source text
    input_path = Path(input)
    if not input_path.exists():
        console.print(f"[red]Error: Input file not found: {input}[/red]")
        raise typer.Exit(1)

    source_context = input_path.read_text(encoding="utf-8")

    # Build generation input
    gen_input = GenerationInput(
        source_context=source_context,
        communicative_intent=caption,
        diagram_type=DiagramType.METHODOLOGY,
    )

    if auto:
        iter_label = f"auto (max {settings.max_iterations})"
    else:
        iter_label = str(settings.refinement_iterations)

    console.print(
        Panel.fit(
            f"[bold]PaperBanana[/bold] - Generating Methodology Diagram\n\n"
            f"VLM: {settings.vlm_provider} / {settings.effective_vlm_model}\n"
            f"Image: {settings.image_provider} / {settings.effective_image_model}\n"
            f"Iterations: {iter_label}",
            border_style="blue",
        )
    )

    # Run pipeline

    console.print()
    total_start = time.perf_counter()

    async def _run_with_progress():
        pipeline = PaperBananaPipeline(settings=settings)

        # Patch agents to print step-by-step progress with timing
        orig_optimizer_run = pipeline.optimizer.run
        orig_retriever_run = pipeline.retriever.run
        orig_planner_run = pipeline.planner.run
        orig_stylist_run = pipeline.stylist.run
        orig_visualizer_run = pipeline.visualizer.run
        orig_critic_run = pipeline.critic.run

        async def _optimizer_run(*a, **kw):
            console.print("  [dim]●[/dim] Optimizing inputs (parallel)...", end="")
            t = time.perf_counter()
            result = await orig_optimizer_run(*a, **kw)
            console.print(f" [green]✓[/green] [dim]{time.perf_counter() - t:.1f}s[/dim]")
            return result

        async def _retriever_run(*a, **kw):
            console.print("  [dim]●[/dim] Retrieving examples...", end="")
            t = time.perf_counter()
            result = await orig_retriever_run(*a, **kw)
            console.print(
                f" [green]✓[/green] [dim]{time.perf_counter() - t:.1f}s"
                f" ({len(result)} examples)[/dim]"
            )
            return result

        async def _planner_run(*a, **kw):
            console.print("  [dim]●[/dim] Planning description...", end="")
            t = time.perf_counter()
            result = await orig_planner_run(*a, **kw)
            console.print(
                f" [green]✓[/green] [dim]{time.perf_counter() - t:.1f}s ({len(result)} chars)[/dim]"
            )
            return result

        async def _stylist_run(*a, **kw):
            console.print("  [dim]●[/dim] Styling description...", end="")
            t = time.perf_counter()
            result = await orig_stylist_run(*a, **kw)
            console.print(f" [green]✓[/green] [dim]{time.perf_counter() - t:.1f}s[/dim]")
            return result

        async def _visualizer_run(*a, **kw):
            iteration = kw.get("iteration", "")
            total = (
                settings.max_iterations if settings.auto_refine else settings.refinement_iterations
            )
            label = f"{iteration}/{total}"
            if settings.auto_refine:
                label += " (auto)"
            if iteration == 1:
                console.print("[bold]Phase 2[/bold] — Iterative Refinement")
            console.print(f"  [dim]●[/dim] Generating image [{label}]...", end="")
            t = time.perf_counter()
            result = await orig_visualizer_run(*a, **kw)
            console.print(f" [green]✓[/green] [dim]{time.perf_counter() - t:.1f}s[/dim]")
            return result

        async def _critic_run(*a, **kw):
            console.print("  [dim]●[/dim] Critic reviewing...", end="")
            t = time.perf_counter()
            result = await orig_critic_run(*a, **kw)
            elapsed = time.perf_counter() - t
            console.print(f" [green]✓[/green] [dim]{elapsed:.1f}s[/dim]")
            if result.needs_revision:
                for s in result.critic_suggestions[:3]:
                    console.print(f"    [yellow]↻[/yellow] [dim]{s}[/dim]")
            else:
                console.print("    [green]✓[/green] [bold green]Critic satisfied[/bold green]")
            return result

        pipeline.optimizer.run = _optimizer_run
        pipeline.retriever.run = _retriever_run
        pipeline.planner.run = _planner_run
        pipeline.stylist.run = _stylist_run
        pipeline.visualizer.run = _visualizer_run
        pipeline.critic.run = _critic_run

        if settings.optimize_inputs:
            console.print("[bold]Phase 0[/bold] — Input Optimization")
        console.print("[bold]Phase 1[/bold] — Planning")

        return await pipeline.generate(gen_input)

    result = asyncio.run(_run_with_progress())
    total_elapsed = time.perf_counter() - total_start

    console.print(
        f"\n[green]✓ Done![/green] [dim]{total_elapsed:.1f}s total"
        f" · {len(result.iterations)} iterations[/dim]\n"
    )
    console.print(f"  Output: [bold]{result.image_path}[/bold]")
    console.print(f"  Run ID: [dim]{result.metadata.get('run_id', 'unknown')}[/dim]")


@app.command()
def plot(
    data: str = typer.Option(..., "--data", "-d", help="Path to data file (CSV or JSON)"),
    intent: str = typer.Option(..., "--intent", help="Communicative intent for the plot"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output image path"),
    vlm_provider: str = typer.Option("gemini", "--vlm-provider", help="VLM provider"),
    iterations: int = typer.Option(3, "--iterations", "-n", help="Number of refinement iterations"),
    format: str = typer.Option(
        "png",
        "--format",
        "-f",
        help="Output image format (png, jpeg, or webp)",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed agent progress and timing"
    ),
):
    """Generate a statistical plot from data."""
    if format not in ("png", "jpeg", "webp"):
        console.print(f"[red]Error: Format must be png, jpeg, or webp. Got: {format}[/red]")
        raise typer.Exit(1)

    configure_logging(verbose=verbose)
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
        output_format=format,
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
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed agent progress and timing"
    ),
):
    """Evaluate a generated diagram vs human reference (comparative)."""
    configure_logging(verbose=verbose)
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

def version_callback(value: bool):
    """
    Callback function to print the version of paperbanana and Python.
    """
    if value:
        pkg_version = importlib.metadata.version("paperbanana")
        py_version = platform.python_version()
        print(f"paperbanana version: {pkg_version}")
        print(f"Python version: {py_version}")
        raise typer.Exit()

@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
):
    """
    PaperBanana CLI – Generate publication-quality academic figures.
    """
    pass

if __name__ == "__main__":
    app()