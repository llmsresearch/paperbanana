"""PaperBanana CLI — Generate publication-quality academic illustrations."""

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
from paperbanana.core.utils import generate_run_id

app = typer.Typer(
    name="paperbanana",
    help="Generate publication-quality academic illustrations from text.",
    no_args_is_help=True,
)
console = Console()

# ── Data subcommand group ─────────────────────────────────────────
data_app = typer.Typer(
    name="data",
    help="Manage reference datasets (download, info, clear).",
    no_args_is_help=True,
)
app.add_typer(data_app, name="data")


def _upsert_env_vars(env_path: Path, updates: dict[str, str]) -> None:
    """Update or append environment variables in a .env file."""
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    key_to_index: dict[str, int] = {}
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        if key not in key_to_index:
            key_to_index[key] = index

    for key, value in updates.items():
        new_line = f"{key}={value}"
        if key in key_to_index:
            lines[key_to_index[key]] = new_line
        else:
            lines.append(new_line)

    env_path.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")


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
    aspect_ratio: Optional[str] = typer.Option(
        None,
        "--aspect-ratio",
        "-ar",
        help="Target aspect ratio: 1:1, 2:3, 3:2, 3:4, 4:3, 9:16, 16:9, 21:9",
    ),
    format: str = typer.Option(
        "png",
        "--format",
        "-f",
        help="Output image format (png, jpeg, or webp)",
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Path to config YAML file"),
    save_prompts: Optional[bool] = typer.Option(
        None,
        "--save-prompts/--no-save-prompts",
        help="Save formatted prompts into the run directory (for debugging)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate inputs and show what would happen without making API calls",
    ),
    auto_download_data: bool = typer.Option(
        False,
        "--auto-download-data",
        help="Auto-download expanded reference set (~257MB) on first run if not cached",
    ),
    exemplar_retrieval: bool = typer.Option(
        False,
        "--exemplar-retrieval",
        help="Enable external exemplar retrieval before planning",
    ),
    exemplar_endpoint: Optional[str] = typer.Option(
        None,
        "--exemplar-endpoint",
        help="External exemplar retrieval endpoint URL",
    ),
    exemplar_mode: Optional[str] = typer.Option(
        None,
        "--exemplar-mode",
        help="Exemplar retrieval mode: external_then_rerank or external_only",
    ),
    exemplar_top_k: Optional[int] = typer.Option(
        None,
        "--exemplar-top-k",
        help="Top-k exemplars requested from external retriever",
    ),
    exemplar_timeout: Optional[float] = typer.Option(
        None,
        "--exemplar-timeout",
        help="External exemplar retrieval timeout (seconds)",
    ),
    exemplar_retries: Optional[int] = typer.Option(
        None,
        "--exemplar-retries",
        help="Retry attempts for external exemplar retrieval on transient errors",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="Random seed for reproducible image generation",
    ),
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
    if exemplar_mode and exemplar_mode not in ("external_then_rerank", "external_only"):
        console.print(
            "[red]Error: --exemplar-mode must be external_then_rerank or external_only[/red]"
        )
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
    if save_prompts is not None:
        overrides["save_prompts"] = save_prompts
    if output:
        overrides["output_dir"] = str(Path(output).parent)
    overrides["output_format"] = format
    if exemplar_retrieval:
        overrides["exemplar_retrieval_enabled"] = True
    if exemplar_endpoint:
        overrides["exemplar_retrieval_endpoint"] = exemplar_endpoint
    if exemplar_mode:
        overrides["exemplar_retrieval_mode"] = exemplar_mode
    if exemplar_top_k is not None:
        overrides["exemplar_retrieval_top_k"] = exemplar_top_k
    if exemplar_timeout is not None:
        overrides["exemplar_retrieval_timeout_seconds"] = exemplar_timeout
    if exemplar_retries is not None:
        overrides["exemplar_retrieval_max_retries"] = exemplar_retries
    if seed is not None:
        overrides["seed"] = seed

    if config:
        settings = Settings.from_yaml(config, **overrides)
    else:
        from dotenv import load_dotenv

        load_dotenv()
        settings = Settings(**overrides)

    from paperbanana.core.pipeline import PaperBananaPipeline

    # ── Auto-download expanded reference set if requested ──────────────
    if auto_download_data:
        from paperbanana.data.manager import DatasetManager

        dm = DatasetManager(cache_dir=settings.cache_dir)
        if not dm.is_downloaded():
            console.print()
            console.print("  [dim]●[/dim] Downloading expanded reference set (~257MB)...", end="")
            try:
                count = dm.download()
                console.print(f" [green]✓[/green] [dim]{count} examples cached[/dim]")
            except Exception as e:
                console.print(f" [red]✗[/red] Download failed: {e}")
                console.print("    [dim]Falling back to built-in reference set (13 examples)[/dim]")

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
        aspect_ratio=aspect_ratio,
    )

    # Determine expected output file extension based on settings.output_format
    output_ext = "jpg" if settings.output_format == "jpeg" else settings.output_format

    if dry_run:
        expected_output = (
            Path(output)
            if output
            else Path(settings.output_dir) / generate_run_id() / f"final_output.{output_ext}"
        )
        console.print(
            Panel.fit(
                "[bold]PaperBanana[/bold] - Dry Run\n\n"
                f"Input: {input_path}\n"
                f"Caption: {caption}\n"
                f"VLM: {settings.vlm_provider} / {settings.vlm_model}\n"
                f"Image: {settings.image_provider} / {settings.image_model}\n"
                f"Iterations: {settings.refinement_iterations}\n"
                f"Output: {expected_output}",
                border_style="yellow",
            )
        )
        return
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

        # Hint: show if using small built-in reference set
        ref_count = pipeline.reference_store.count
        if ref_count <= 20 and not auto_download_data:
            console.print(
                "  [dim]Using built-in reference set"
                f" ({ref_count} examples). For better results:[/dim]"
            )
            console.print("  [dim]  paperbanana data download   # or --auto-download-data[/dim]")
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
            desc, ratio = result
            info = f"{len(desc)} chars"
            if ratio:
                info += f", ratio={ratio}"
            elapsed = time.perf_counter() - t
            console.print(f" [green]\u2713[/green] [dim]{elapsed:.1f}s ({info})[/dim]")
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
def batch(
    manifest: str = typer.Option(
        ..., "--manifest", "-m", help="Path to batch manifest (YAML or JSON)"
    ),
    output_dir: str = typer.Option(
        "outputs",
        "--output-dir",
        "-o",
        help="Parent directory for batch run (batch_<id> will be created here)",
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Path to config YAML file"),
    vlm_provider: Optional[str] = typer.Option(None, "--vlm-provider", help="VLM provider"),
    vlm_model: Optional[str] = typer.Option(None, "--vlm-model", help="VLM model name"),
    image_provider: Optional[str] = typer.Option(
        None, "--image-provider", help="Image gen provider"
    ),
    image_model: Optional[str] = typer.Option(None, "--image-model", help="Image gen model name"),
    iterations: Optional[int] = typer.Option(
        None, "--iterations", "-n", help="Refinement iterations"
    ),
    auto: bool = typer.Option(
        False, "--auto", help="Loop until critic satisfied (with safety cap)"
    ),
    max_iterations: Optional[int] = typer.Option(
        None, "--max-iterations", help="Safety cap for --auto"
    ),
    optimize: bool = typer.Option(
        False, "--optimize", help="Preprocess inputs for better generation"
    ),
    format: str = typer.Option(
        "png", "--format", "-f", help="Output image format (png, jpeg, webp)"
    ),
    save_prompts: Optional[bool] = typer.Option(
        None, "--save-prompts/--no-save-prompts", help="Save prompts per run"
    ),
    auto_download_data: bool = typer.Option(
        False, "--auto-download-data", help="Auto-download reference set if needed"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed progress"),
):
    """Generate multiple methodology diagrams from a manifest file (YAML or JSON)."""
    if format not in ("png", "jpeg", "webp"):
        console.print(f"[red]Error: Format must be png, jpeg, or webp. Got: {format}[/red]")
        raise typer.Exit(1)

    configure_logging(verbose=verbose)
    manifest_path = Path(manifest)
    if not manifest_path.exists():
        console.print(f"[red]Error: Manifest not found: {manifest}[/red]")
        raise typer.Exit(1)

    from paperbanana.core.batch import generate_batch_id, load_batch_manifest
    from paperbanana.core.utils import ensure_dir, save_json

    try:
        items = load_batch_manifest(manifest_path)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        console.print(f"[red]Error loading manifest: {e}[/red]")
        raise typer.Exit(1)

    batch_id = generate_batch_id()
    batch_dir = Path(output_dir) / batch_id
    ensure_dir(batch_dir)

    overrides = {"output_dir": str(batch_dir), "output_format": format}
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
    if save_prompts is not None:
        overrides["save_prompts"] = save_prompts

    if config:
        settings = Settings.from_yaml(config, **overrides)
    else:
        from dotenv import load_dotenv

        load_dotenv()
        settings = Settings(**overrides)

    if auto_download_data:
        from paperbanana.data.manager import DatasetManager

        dm = DatasetManager(cache_dir=settings.cache_dir)
        if not dm.is_downloaded():
            console.print("  [dim]Downloading expanded reference set...[/dim]")
            try:
                dm.download()
            except Exception as e:
                console.print(f"  [yellow]Download failed: {e}, using built-in set[/yellow]")

    console.print(
        Panel.fit(
            f"[bold]PaperBanana[/bold] — Batch Generation\n\n"
            f"Manifest: {manifest_path.name}\n"
            f"Items: {len(items)}\n"
            f"Output: {batch_dir}",
            border_style="blue",
        )
    )
    console.print()

    from paperbanana.core.pipeline import PaperBananaPipeline

    report = {"batch_id": batch_id, "manifest": str(manifest_path), "items": []}
    total_start = time.perf_counter()

    for idx, item in enumerate(items):
        item_id = item["id"]
        input_path = Path(item["input"])
        if not input_path.exists():
            console.print(f"[red]Skipping item '{item_id}': input not found: {input_path}[/red]")
            report["items"].append(
                {
                    "id": item_id,
                    "input": item["input"],
                    "caption": item["caption"],
                    "run_id": None,
                    "output_path": None,
                    "error": "input file not found",
                }
            )
            continue
        source_context = input_path.read_text(encoding="utf-8")
        gen_input = GenerationInput(
            source_context=source_context,
            communicative_intent=item["caption"],
            diagram_type=DiagramType.METHODOLOGY,
        )
        console.print(f"[bold]Item {idx + 1}/{len(items)}[/bold] — {item_id}")
        pipeline = PaperBananaPipeline(settings=settings)
        try:
            result = asyncio.run(pipeline.generate(gen_input))
            report["items"].append(
                {
                    "id": item_id,
                    "input": item["input"],
                    "caption": item["caption"],
                    "run_id": result.metadata.get("run_id"),
                    "output_path": result.image_path,
                    "iterations": len(result.iterations),
                }
            )
            console.print(f"  [green]✓[/green] [dim]{result.image_path}[/dim]\n")
        except Exception as e:
            console.print(f"  [red]✗[/red] {e}\n")
            report["items"].append(
                {
                    "id": item_id,
                    "input": item["input"],
                    "caption": item["caption"],
                    "run_id": None,
                    "output_path": None,
                    "error": str(e),
                }
            )

    total_elapsed = time.perf_counter() - total_start
    report["total_seconds"] = round(total_elapsed, 1)
    report_path = batch_dir / "batch_report.json"
    save_json(report, report_path)

    succeeded = sum(1 for x in report["items"] if x.get("output_path"))
    console.print(
        f"[green]Batch complete.[/green] [dim]{total_elapsed:.1f}s · "
        f"{succeeded}/{len(items)} succeeded[/dim]"
    )
    console.print(f"  Report: [bold]{report_path}[/bold]")


@app.command("batch-report")
def batch_report(
    batch_dir: Optional[str] = typer.Option(
        None,
        "--batch-dir",
        "-b",
        help="Path to batch run directory (e.g. outputs/batch_20250109_123456_abc)",
    ),
    batch_id: Optional[str] = typer.Option(
        None,
        "--batch-id",
        help="Batch ID (e.g. batch_20250109_123456_abc); resolved under --output-dir",
    ),
    output_dir: str = typer.Option(
        "outputs",
        "--output-dir",
        "-o",
        help="Parent directory for batch runs (used with --batch-id)",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        help="Output path for the report file (default: <batch_dir>/batch_report.<md|html>)",
    ),
    format: str = typer.Option(
        "markdown",
        "--format",
        "-f",
        help="Report format: markdown or html",
    ),
):
    """Generate a human-readable report from an existing batch run (batch_report.json)."""
    if format not in ("markdown", "html", "md"):
        console.print(f"[red]Error: Format must be markdown or html. Got: {format}[/red]")
        raise typer.Exit(1)
    if batch_dir is None and batch_id is None:
        console.print("[red]Error: Provide either --batch-dir or --batch-id[/red]")
        raise typer.Exit(1)
    if batch_dir is not None and batch_id is not None:
        console.print("[red]Error: Provide only one of --batch-dir or --batch-id[/red]")
        raise typer.Exit(1)

    from paperbanana.core.batch import write_batch_report

    if batch_dir is not None:
        path = Path(batch_dir)
    else:
        path = Path(output_dir) / batch_id

    output_path = Path(output) if output else None
    fmt = "markdown" if format == "md" else format
    try:
        written = write_batch_report(path, output_path=output_path, format=fmt)
        console.print(f"[green]Report written to:[/green] [bold]{written}[/bold]")
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


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
    aspect_ratio: Optional[str] = typer.Option(
        None,
        "--aspect-ratio",
        "-ar",
        help="Target aspect ratio: 1:1, 2:3, 3:2, 3:4, 4:3, 9:16, 16:9, 21:9",
    ),
    optimize: bool = typer.Option(
        False, "--optimize", help="Enrich context and sharpen caption before generation"
    ),
    auto: bool = typer.Option(
        False, "--auto", help="Let critic loop until satisfied (max 30 iterations)"
    ),
    save_prompts: Optional[bool] = typer.Option(
        None,
        "--save-prompts/--no-save-prompts",
        help="Save formatted prompts into the run directory (for debugging)",
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
        optimize_inputs=optimize,
        auto_refine=auto,
        save_prompts=True if save_prompts is None else save_prompts,
    )

    gen_input = GenerationInput(
        source_context=source_context,
        communicative_intent=intent,
        diagram_type=DiagramType.STATISTICAL_PLOT,
        raw_data={"data": raw_data},
        aspect_ratio=aspect_ratio,
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

    console.print("\n[bold]Step 1: Gemini API Configuration[/bold]")
    use_official_api = Prompt.ask(
        "Use official Google Gemini API?",
        choices=["y", "n"],
        default="y",
    )

    # Save to .env
    env_path = Path(".env")
    if use_official_api == "y":
        console.print("Using official Google AI Studio endpoint (free, no credit card).")
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
        env_updates = {
            "GOOGLE_API_KEY": gemini_key,
            "GOOGLE_BASE_URL": "",
        }
    else:
        console.print("Using custom Gemini-compatible endpoint.\n")
        google_base_url = ""
        while not google_base_url.strip():
            google_base_url = Prompt.ask("Gemini base URL")
            if not google_base_url.strip():
                console.print("[red]URL cannot be empty. Please try again.[/red]")

        gemini_key = Prompt.ask("Paste your Gemini API key")
        env_updates = {
            "GOOGLE_API_KEY": gemini_key,
            "GOOGLE_BASE_URL": google_base_url.strip(),
        }

    _upsert_env_vars(env_path, env_updates)

    console.print(f"\n[green]Setup complete![/green] Configuration saved to {env_path}")
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
    from paperbanana.core.utils import find_prompt_dir
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

    judge = VLMJudge(vlm, prompt_dir=find_prompt_dir())

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


@app.command("ablate-retrieval")
def ablate_retrieval(
    input: str = typer.Option(..., "--input", "-i", help="Path to methodology text file"),
    caption: str = typer.Option(
        ..., "--caption", "-c", help="Figure caption / communicative intent"
    ),
    exemplar_endpoint: str = typer.Option(
        ..., "--exemplar-endpoint", help="External exemplar retrieval endpoint URL"
    ),
    top_k: str = typer.Option(
        "1,3,5", "--top-k", help="Comma-separated top-k values (e.g., 1,3,5)"
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="Random seed used for all variants (default: 42 if omitted)",
    ),
    exemplar_retries: Optional[int] = typer.Option(
        None,
        "--exemplar-retries",
        help="Retry attempts for external exemplar retrieval on transient errors",
    ),
    reference: Optional[str] = typer.Option(
        None,
        "--reference",
        "-r",
        help="Optional human reference image for judge-based preference proxy",
    ),
    output_report: Optional[str] = typer.Option(
        None,
        "--output-report",
        "-o",
        help="Output JSON report path (default: outputs/retrieval_ablation_<runid>.json)",
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Path to config YAML file"),
    vlm_provider: Optional[str] = typer.Option(
        None, "--vlm-provider", help="VLM provider override for generation and judge"
    ),
    image_provider: Optional[str] = typer.Option(
        None, "--image-provider", help="Image generation provider override"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed agent progress and timing"
    ),
):
    """Run baseline vs retrieval ablation (k sweep) and save a JSON report."""
    configure_logging(verbose=verbose)

    input_path = Path(input)
    if not input_path.exists():
        console.print(f"[red]Error: Input file not found: {input}[/red]")
        raise typer.Exit(1)

    reference_path: Optional[Path] = None
    if reference:
        reference_path = Path(reference)
        if not reference_path.exists():
            console.print(f"[red]Error: Reference image not found: {reference}[/red]")
            raise typer.Exit(1)

    from dotenv import load_dotenv

    load_dotenv()

    from paperbanana.core.types import DiagramType, GenerationInput
    from paperbanana.core.utils import generate_run_id
    from paperbanana.evaluation.retrieval_ablation import (
        RetrievalAblationRunner,
        parse_top_k_values,
    )

    try:
        k_values = parse_top_k_values(top_k)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    overrides = {
        "exemplar_retrieval_endpoint": exemplar_endpoint,
        "exemplar_retrieval_enabled": True,
    }
    if vlm_provider:
        overrides["vlm_provider"] = vlm_provider
    if image_provider:
        overrides["image_provider"] = image_provider
    if seed is not None:
        overrides["seed"] = seed
    if exemplar_retries is not None:
        overrides["exemplar_retrieval_max_retries"] = exemplar_retries

    if config:
        settings = Settings.from_yaml(config, **overrides)
    else:
        settings = Settings(**overrides)

    gen_input = GenerationInput(
        source_context=input_path.read_text(encoding="utf-8"),
        communicative_intent=caption,
        diagram_type=DiagramType.METHODOLOGY,
    )

    runner = RetrievalAblationRunner(
        settings,
        reference_image_path=str(reference_path) if reference_path else None,
    )

    async def _run():
        return await runner.run(gen_input, top_k_values=k_values)

    console.print(
        Panel.fit(
            f"[bold]PaperBanana[/bold] - Retrieval Ablation\n\n"
            f"Top-k sweep: {k_values}\n"
            f"Endpoint: {exemplar_endpoint}\n"
            f"Seed: {settings.seed if settings.seed is not None else 42}\n"
            f"Reference: {reference_path if reference_path else 'none'}",
            border_style="magenta",
        )
    )

    report = asyncio.run(_run())

    default_report_path = Path(settings.output_dir) / f"retrieval_ablation_{generate_run_id()}.json"
    report_path = Path(output_report) if output_report else default_report_path
    saved_path = runner.save_report(report, report_path)

    summary = report.summary
    human_pref_line = ""
    if summary.get("best_human_preference_variant") is not None:
        human_pref_line = (
            f"Best human preference: {summary.get('best_human_preference_variant')} "
            f"({summary.get('best_human_preference_score')})\n"
        )
    console.print(
        Panel.fit(
            "[bold]Ablation Summary[/bold]\n\n"
            f"Best alignment: {summary.get('best_alignment_variant')} "
            f"({summary.get('best_alignment_score')})\n"
            f"{human_pref_line}"
            f"Fastest: {summary.get('fastest_variant')} "
            f"({summary.get('fastest_total_seconds')}s)\n"
            f"Fewest iterations: {summary.get('fewest_iterations_variant')} "
            f"({summary.get('fewest_iterations')})\n\n"
            f"Report: [bold]{saved_path}[/bold]",
            border_style="cyan",
        )
    )


@app.command()
def benchmark(
    config: Optional[str] = typer.Option(None, "--config", help="Path to config YAML file"),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", "-o", help="Output directory for benchmark run"
    ),
    vlm_provider: Optional[str] = typer.Option(None, "--vlm-provider", help="VLM provider"),
    vlm_model: Optional[str] = typer.Option(None, "--vlm-model", help="VLM model name"),
    image_provider: Optional[str] = typer.Option(
        None, "--image-provider", help="Image gen provider"
    ),
    image_model: Optional[str] = typer.Option(None, "--image-model", help="Image gen model name"),
    iterations: Optional[int] = typer.Option(
        None, "--iterations", "-n", help="Refinement iterations per entry"
    ),
    auto: bool = typer.Option(False, "--auto", help="Loop until critic satisfied per entry"),
    optimize: bool = typer.Option(False, "--optimize", help="Preprocess inputs per entry"),
    category: Optional[str] = typer.Option(
        None, "--category", help="Only run entries in this category"
    ),
    ids: Optional[str] = typer.Option(
        None, "--ids", help="Comma-separated entry IDs to run (e.g., 2601.03570v1,2601.05110v1)"
    ),
    limit: Optional[int] = typer.Option(None, "--limit", help="Max number of entries to process"),
    eval_only: Optional[str] = typer.Option(
        None,
        "--eval-only",
        help="Skip generation; evaluate existing images from this directory",
    ),
    image_format: str = typer.Option(
        "png", "--format", "-f", help="Output image format (png, jpeg, webp)"
    ),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for reproducibility"),
    concurrency: int = typer.Option(
        1,
        "--concurrency",
        "-c",
        help="Maximum number of benchmark entries to process in parallel",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed progress"),
):
    """Run generation + evaluation across PaperBananaBench entries."""
    if image_format not in ("png", "jpeg", "webp"):
        console.print(f"[red]Error: Format must be png, jpeg, or webp. Got: {image_format}[/red]")
        raise typer.Exit(1)
    if concurrency < 1:
        console.print("[red]Error: --concurrency must be at least 1[/red]")
        raise typer.Exit(1)

    configure_logging(verbose=verbose)

    from dotenv import load_dotenv

    load_dotenv()

    overrides: dict = {"output_format": image_format, "benchmark_concurrency": concurrency}
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
    if optimize:
        overrides["optimize_inputs"] = True
    if output_dir:
        overrides["output_dir"] = output_dir
    if seed is not None:
        overrides["seed"] = seed

    if config:
        settings = Settings.from_yaml(config, **overrides)
    else:
        settings = Settings(**overrides)

    from paperbanana.evaluation.benchmark import BenchmarkRunner

    runner = BenchmarkRunner(settings)

    # Load and filter entries
    id_list = [s.strip() for s in ids.split(",") if s.strip()] if ids else None
    try:
        entries = runner.load_entries(category=category, ids=id_list, limit=limit)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if not entries:
        console.print("[red]Error: No entries match the given filters.[/red]")
        raise typer.Exit(1)

    mode = "eval-only" if eval_only else "generate + evaluate"
    console.print(
        Panel.fit(
            f"[bold]PaperBanana[/bold] — Benchmark\n\n"
            f"Entries: {len(entries)}\n"
            f"Mode: {mode}\n"
            f"VLM: {settings.vlm_provider} / {settings.effective_vlm_model}\n"
            f"Image: {settings.image_provider} / {settings.effective_image_model}",
            border_style="magenta",
        )
    )
    console.print()

    bench_output_dir = Path(output_dir) if output_dir else None

    async def _run():
        return await runner.run(entries, output_dir=bench_output_dir, eval_only_dir=eval_only)

    report = asyncio.run(_run())
    summary = report.summary

    if not summary:
        console.print("[yellow]No entries were successfully evaluated.[/yellow]")
        return

    # Print summary table
    console.print(
        Panel.fit(
            "[bold]Benchmark Summary[/bold]\n\n"
            f"Evaluated: {summary.get('evaluated', 0)}\n"
            f"Model wins: {summary.get('model_wins', 0)}  "
            f"Human wins: {summary.get('human_wins', 0)}  "
            f"Ties: {summary.get('ties', 0)}\n"
            f"Model win rate: {summary.get('model_win_rate', 0)}%\n"
            f"Mean overall score: {summary.get('mean_overall_score', 0)}/100\n"
            f"Mean generation time: {summary.get('mean_generation_seconds', 0)}s\n\n"
            f"Completed: {report.completed}  "
            f"Failed: {report.failed}  "
            f"Total: {report.total_seconds}s",
            border_style="cyan",
        )
    )

    # Per-dimension breakdown
    dim_means = summary.get("dimension_means", {})
    if dim_means:
        console.print("\n[bold]Per-dimension scores:[/bold]")
        for dim, score in dim_means.items():
            console.print(f"  {dim.capitalize():14s} {score}/100")

    # Per-category breakdown
    cat_breakdown = summary.get("category_breakdown", {})
    if cat_breakdown:
        console.print("\n[bold]Per-category breakdown:[/bold]")
        for cat, stats in cat_breakdown.items():
            console.print(
                f"  {cat:30s} n={stats['count']:3d}  "
                f"win_rate={stats['model_win_rate']:5.1f}%  "
                f"mean={stats['mean_score']:.1f}"
            )

    if report.run_dir:
        report_path = Path(report.run_dir)
    else:
        report_path = Path(settings.output_dir) / report.created_at.replace(":", "")
    console.print(f"\nReport: [bold]{report_path / 'benchmark_report.json'}[/bold]")


# ── Data subcommands ──────────────────────────────────────────────


@data_app.command()
def download(
    task: str = typer.Option(
        "diagram",
        "--task",
        help="Which references to import: diagram, plot, or both",
    ),
    force: bool = typer.Option(False, "--force", help="Re-download even if already cached"),
):
    """Download the expanded reference set from official PaperBananaBench (~257MB)."""
    from paperbanana.data.manager import DatasetManager

    dm = DatasetManager()
    if dm.is_downloaded() and not force:
        info = dm.get_info() or {}
        console.print(
            Panel.fit(
                f"[bold]Reference Set — Already Cached[/bold]\n\n"
                f"Location: {dm.reference_dir}\n"
                f"Examples: {dm.get_example_count()}\n"
                f"Version: {info.get('version', 'unknown')}\n"
                f"Revision: {info.get('revision', 'unknown')}",
                border_style="green",
            )
        )
        console.print("\nUse [bold]--force[/bold] to re-download.")
        return

    console.print("[bold]PaperBanana[/bold] — Downloading Reference Set\n")
    try:
        count = dm.download(
            task=task,
            force=force,
            progress_callback=lambda msg: console.print(f"  [dim]●[/dim] {msg}"),
        )
        console.print(f"\n[green]Done![/green] {count} reference examples cached to:")
        console.print(f"  [bold]{dm.reference_dir}[/bold]")
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        raise typer.Exit(1)


@data_app.command()
def info():
    """Show information about the cached reference dataset."""
    from paperbanana.data.manager import DatasetManager

    dm = DatasetManager()
    dataset_info = dm.get_info()

    if not dataset_info:
        console.print("No expanded reference set cached.")
        console.print("\nDownload with: [bold]paperbanana data download[/bold]")
        return

    console.print(
        Panel.fit(
            f"[bold]Cached Reference Set[/bold]\n\n"
            f"Location: {dm.reference_dir}\n"
            f"Examples: {dataset_info.get('example_count', '?')}\n"
            f"Version: {dataset_info.get('version', 'unknown')}\n"
            f"Revision: {dataset_info.get('revision', 'unknown')}\n"
            f"Source: {dataset_info.get('source', 'unknown')}",
            border_style="blue",
        )
    )


@data_app.command()
def clear():
    """Remove cached reference dataset."""
    from paperbanana.data.manager import DatasetManager

    dm = DatasetManager()
    if not dm.is_downloaded():
        console.print("No cached dataset to clear.")
        return

    dm.clear()
    console.print("[green]Cached reference set cleared.[/green]")


if __name__ == "__main__":
    app()
