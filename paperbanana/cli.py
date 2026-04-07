"""PaperBanana CLI — Generate publication-quality academic illustrations."""

from __future__ import annotations

import asyncio
import json as json_mod
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from paperbanana.core.config import Settings
from paperbanana.core.logging import configure_logging
from paperbanana.core.types import (
    DiagramType,
    GenerationInput,
    PipelineProgressEvent,
    PipelineProgressStage,
)
from paperbanana.core.utils import ensure_dir, generate_run_id, save_json

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
        None,
        "--input",
        "-i",
        help="Path to methodology text file or PDF (.pdf requires: pip install 'paperbanana[pdf]')",
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
    cost_only: bool = typer.Option(
        False,
        "--cost-only",
        help="Estimate cost without making API calls (implies --dry-run)",
    ),
    budget: Optional[float] = typer.Option(
        None,
        "--budget",
        help="Budget cap in USD; pipeline aborts gracefully when exceeded",
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
    prompt_dir: Optional[str] = typer.Option(
        None,
        "--prompt-dir",
        help="Path to alternative prompt templates directory (for A/B testing)",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="Random seed for reproducible image generation",
    ),
    venue: Optional[str] = typer.Option(
        None,
        "--venue",
        help="Target venue style (neurips, icml, acl, ieee, custom)",
    ),
    progress_json: bool = typer.Option(
        False,
        "--progress-json",
        help="Emit machine-readable JSON progress events to stdout during generation",
    ),
    pdf_pages: Optional[str] = typer.Option(
        None,
        "--pdf-pages",
        help=("PDF input only: 1-based pages (e.g. '1-5', '3', '1-3,7,10-12'); default: all pages"),
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
    if venue and venue.lower() not in ("neurips", "icml", "acl", "ieee", "custom"):
        console.print(
            f"[red]Error: --venue must be neurips, icml, acl, ieee, or custom. Got: {venue}[/red]"
        )
        raise typer.Exit(1)
    if pdf_pages and (continue_last or continue_run):
        console.print(
            "[red]Error: --pdf-pages cannot be used with --continue or --continue-run[/red]"
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
    if budget is not None:
        overrides["budget_usd"] = budget
    if venue:
        overrides["venue"] = venue
    if prompt_dir:
        overrides["prompt_dir"] = prompt_dir

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

        def on_progress(event: PipelineProgressEvent) -> None:
            if event.stage == PipelineProgressStage.VISUALIZER_START:
                extra = event.extra or {}
                total = extra.get("total_iterations", 0)
                if event.iteration and total:
                    label = f"{event.iteration}/{total}"
                else:
                    label = str(event.iteration or "")
                if settings.auto_refine:
                    label += " (auto)"
                console.print(
                    f"  [dim]●[/dim] Generating image (iter {event.iteration})...",
                    end="",
                )
            elif event.stage == PipelineProgressStage.VISUALIZER_END:
                console.print(
                    f" [green]✓[/green] [dim]{event.seconds:.1f}s[/dim]"
                    if event.seconds is not None
                    else " [green]✓[/green]"
                )
            elif event.stage == PipelineProgressStage.CRITIC_START:
                console.print("  [dim]●[/dim] Critic reviewing...", end="")
            elif event.stage == PipelineProgressStage.CRITIC_END:
                console.print(
                    f" [green]✓[/green] [dim]{event.seconds:.1f}s[/dim]"
                    if event.seconds is not None
                    else " [green]✓[/green]"
                )
                extra = event.extra or {}
                if extra.get("needs_revision"):
                    console.print(
                        "    [yellow]↻[/yellow] Revision needed: "
                        f"[dim]{extra.get('summary', '')}[/dim]"
                    )
                else:
                    console.print("    [green]✓[/green] [bold green]Critic satisfied[/bold green]")

        async def _run_continue():
            pipeline = PaperBananaPipeline(settings=settings)
            return await pipeline.continue_run(
                resume_state=resume_state,
                additional_iterations=iterations,
                user_feedback=feedback,
                progress_callback=on_progress,
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

    # Load source text (plain UTF-8 or PDF)
    input_path = Path(input)
    if not input_path.exists():
        console.print(f"[red]Error: Input file not found: {input}[/red]")
        raise typer.Exit(1)

    from paperbanana.core.source_loader import load_methodology_source

    try:
        source_context = load_methodology_source(input_path, pdf_pages=pdf_pages)
    except ImportError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    # Build generation input
    gen_input = GenerationInput(
        source_context=source_context,
        communicative_intent=caption,
        diagram_type=DiagramType.METHODOLOGY,
        aspect_ratio=aspect_ratio,
    )

    # Determine expected output file extension based on settings.output_format
    output_ext = "jpg" if settings.output_format == "jpeg" else settings.output_format

    if cost_only:
        from paperbanana.core.cost_estimator import estimate_cost

        estimate = estimate_cost(settings, gen_input.diagram_type)
        iter_est = (
            f"auto (max {settings.max_iterations})"
            if settings.auto_refine
            else str(settings.refinement_iterations)
        )
        lines = [
            "[bold]PaperBanana[/bold] - Cost Estimate\n",
            f"VLM: {settings.vlm_provider} / {settings.effective_vlm_model}",
            f"Image: {settings.image_provider} / {settings.effective_image_model}",
            f"Iterations: {iter_est}",
            f"Optimize: {'yes' if settings.optimize_inputs else 'no'}",
            "",
            f"Estimated VLM calls: {estimate['vlm_calls']}",
            f"Estimated image calls: {estimate['image_calls']}",
            f"[bold]Estimated cost: ${estimate['estimated_total_usd']:.4f}[/bold]",
        ]
        if estimate.get("pricing_note"):
            lines.append(f"\n[yellow]Note: {estimate['pricing_note']}[/yellow]")
        console.print(Panel.fit("\n".join(lines), border_style="green"))
        return

    if dry_run:
        expected_output = (
            Path(output)
            if output
            else Path(settings.output_dir) / generate_run_id() / f"final_output.{output_ext}"
        )
        pdf_note = ""
        if input_path.suffix.lower() == ".pdf":
            pdf_note = f"\nPDF pages: {pdf_pages.strip() if pdf_pages else 'all'}"
        console.print(
            Panel.fit(
                "[bold]PaperBanana[/bold] - Dry Run\n\n"
                f"Input: {input_path}{pdf_note}\n"
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

    if not progress_json:
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
        def _on_progress(event: str, payload: dict) -> None:
            if progress_json:
                console.print(
                    json_mod.dumps({"event": event, **payload}),
                    highlight=False,
                )

        pipeline = PaperBananaPipeline(
            settings=settings,
            progress_callback=_on_progress if progress_json else None,
        )

        # Hint: show if using small built-in reference set
        ref_count = pipeline.reference_store.count
        if ref_count <= 20 and not auto_download_data and not progress_json:
            console.print(
                "  [dim]Using built-in reference set"
                f" ({ref_count} examples). For better results:[/dim]"
            )
            console.print("  [dim]  paperbanana data download   # or --auto-download-data[/dim]")

        def on_progress(event: PipelineProgressEvent) -> None:
            if event.stage == PipelineProgressStage.OPTIMIZER_START:
                console.print("[bold]Phase 0[/bold] — Input Optimization")
                console.print("  [dim]●[/dim] Optimizing inputs (parallel)...", end="")
            elif event.stage == PipelineProgressStage.OPTIMIZER_END:
                console.print(
                    f" [green]✓[/green] [dim]{event.seconds:.1f}s[/dim]"
                    if event.seconds is not None
                    else ""
                )
            elif event.stage == PipelineProgressStage.RETRIEVER_START:
                console.print("[bold]Phase 1[/bold] — Planning")
                console.print("  [dim]●[/dim] Retrieving examples...", end="")
            elif event.stage == PipelineProgressStage.RETRIEVER_END:
                extra = event.extra or {}
                n = extra.get("examples_count", 0)
                console.print(
                    f" [green]✓[/green] [dim]{event.seconds:.1f}s ({n} examples)[/dim]"
                    if event.seconds is not None
                    else f" [green]✓[/green] [dim]({n} examples)[/dim]"
                )
            elif event.stage == PipelineProgressStage.PLANNER_START:
                console.print("  [dim]●[/dim] Planning description...", end="")
            elif event.stage == PipelineProgressStage.PLANNER_END:
                extra = event.extra or {}
                ratio = extra.get("recommended_ratio")
                info = f"{event.seconds:.1f}s" if event.seconds is not None else ""
                if ratio:
                    info += f", ratio={ratio}"
                console.print(f" [green]✓[/green] [dim]{info}[/dim]")
            elif event.stage == PipelineProgressStage.STYLIST_START:
                console.print("  [dim]●[/dim] Styling description...", end="")
            elif event.stage == PipelineProgressStage.STYLIST_END:
                console.print(
                    f" [green]✓[/green] [dim]{event.seconds:.1f}s[/dim]"
                    if event.seconds is not None
                    else " [green]✓[/green]"
                )
            elif event.stage == PipelineProgressStage.VISUALIZER_START:
                if event.iteration == 1:
                    console.print("[bold]Phase 2[/bold] — Iterative Refinement")
                extra = event.extra or {}
                total = extra.get("total_iterations", 0)
                if event.iteration and total:
                    label = f"{event.iteration}/{total}"
                else:
                    label = str(event.iteration or "")
                if settings.auto_refine:
                    label += " (auto)"
                console.print(f"  [dim]●[/dim] Generating image [{label}]...", end="")
            elif event.stage == PipelineProgressStage.VISUALIZER_END:
                console.print(
                    f" [green]✓[/green] [dim]{event.seconds:.1f}s[/dim]"
                    if event.seconds is not None
                    else " [green]✓[/green]"
                )
            elif event.stage == PipelineProgressStage.CRITIC_START:
                console.print("  [dim]●[/dim] Critic reviewing...", end="")
            elif event.stage == PipelineProgressStage.CRITIC_END:
                console.print(
                    f" [green]✓[/green] [dim]{event.seconds:.1f}s[/dim]"
                    if event.seconds is not None
                    else " [green]✓[/green]"
                )
                extra = event.extra or {}
                if extra.get("needs_revision"):
                    for s in (extra.get("critic_suggestions") or [])[:3]:
                        console.print(f"    [yellow]↻[/yellow] [dim]{s}[/dim]")
                else:
                    console.print("    [green]✓[/green] [bold green]Critic satisfied[/bold green]")

        return await pipeline.generate(
            gen_input,
            progress_callback=on_progress if not progress_json else None,
        )

    result = asyncio.run(_run_with_progress())
    total_elapsed = time.perf_counter() - total_start

    console.print(
        f"\n[green]✓ Done![/green] [dim]{total_elapsed:.1f}s total"
        f" · {len(result.iterations)} iterations[/dim]\n"
    )
    console.print(f"  Output: [bold]{result.image_path}[/bold]")
    console.print(f"  Run ID: [dim]{result.metadata.get('run_id', 'unknown')}[/dim]")

    cost_data = result.metadata.get("cost")
    if cost_data:
        console.print(
            f"  Cost:   [bold]${cost_data['total_usd']:.4f}[/bold]"
            f" [dim](VLM: ${cost_data['vlm_usd']:.4f},"
            f" Image: ${cost_data['image_usd']:.4f})[/dim]"
        )
        if cost_data.get("budget_exceeded"):
            console.print(
                f"  [yellow]Budget exceeded: ${cost_data['total_usd']:.4f}"
                f" / ${cost_data.get('budget_usd', '?')}[/yellow]"
            )
        elif not cost_data.get("pricing_complete", True):
            console.print(
                "  [yellow]Note: Some model prices unknown; actual cost may differ[/yellow]"
            )


@app.command()
def sweep(
    input: str = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to methodology text file or PDF (.pdf requires: pip install 'paperbanana[pdf]')",
    ),
    caption: str = typer.Option(
        ...,
        "--caption",
        "-c",
        help="Figure caption / communicative intent",
    ),
    pdf_pages: Optional[str] = typer.Option(
        None,
        "--pdf-pages",
        help=("PDF input only: 1-based pages (e.g. '1-5', '3', '1-3,7'); default: all pages"),
    ),
    output_dir: str = typer.Option(
        "outputs",
        "--output-dir",
        "-o",
        help="Parent directory for sweep outputs (sweep_<id> will be created here)",
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Path to config YAML file"),
    vlm_providers: Optional[str] = typer.Option(
        None,
        "--vlm-providers",
        help="Comma-separated VLM providers (e.g. gemini,openai)",
    ),
    vlm_models: Optional[str] = typer.Option(
        None,
        "--vlm-models",
        help="Comma-separated VLM models (paired as full cartesian combinations)",
    ),
    image_providers: Optional[str] = typer.Option(
        None,
        "--image-providers",
        help="Comma-separated image providers (e.g. google_imagen,openai_imagen)",
    ),
    image_models: Optional[str] = typer.Option(
        None,
        "--image-models",
        help="Comma-separated image models (paired as full cartesian combinations)",
    ),
    iterations: Optional[str] = typer.Option(
        None,
        "--iterations",
        help="Comma-separated refinement iterations (e.g. 2,3,4)",
    ),
    optimize_modes: Optional[str] = typer.Option(
        None,
        "--optimize-modes",
        help="Comma-separated booleans for optimize_inputs axis (e.g. on,off)",
    ),
    auto_modes: Optional[str] = typer.Option(
        None,
        "--auto-modes",
        help="Comma-separated booleans for auto_refine axis (e.g. off,on)",
    ),
    max_variants: Optional[int] = typer.Option(
        None,
        "--max-variants",
        help="Optional cap on total generated variants",
    ),
    format: str = typer.Option(
        "png",
        "--format",
        "-f",
        help="Output image format (png, jpeg, webp)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show planned variant matrix without API calls",
    ),
    auto_download_data: bool = typer.Option(
        False,
        "--auto-download-data",
        help="Auto-download expanded reference set (~257MB) on first run if not cached",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed progress"),
):
    """Run a parameter sweep for one input and rank generated variants.

    Successful variants are ranked by *quality_proxy_score*: max(0, 100 − 12.5 × N) where N is
    the number of critic suggestions on the **final** refinement iteration. This is a rough
    proxy for comparing runs, not a substitute for human evaluation.
    """
    if format not in ("png", "jpeg", "webp"):
        console.print(f"[red]Error: Format must be png, jpeg, or webp. Got: {format}[/red]")
        raise typer.Exit(1)
    if max_variants is not None and max_variants < 1:
        console.print("[red]Error: --max-variants must be >= 1[/red]")
        raise typer.Exit(1)

    configure_logging(verbose=verbose)

    input_path = Path(input)
    if not input_path.exists():
        console.print(f"[red]Error: Input file not found: {input}[/red]")
        raise typer.Exit(1)

    from dotenv import load_dotenv

    load_dotenv()

    from paperbanana.core.source_loader import load_methodology_source
    from paperbanana.core.sweep import (
        build_sweep_variants,
        parse_csv_bools,
        parse_csv_ints,
        parse_csv_values,
        quality_proxy_score,
        rank_sweep_results,
        summarize_sweep,
    )

    try:
        variant_list = build_sweep_variants(
            vlm_providers=parse_csv_values(vlm_providers),
            vlm_models=parse_csv_values(vlm_models),
            image_providers=parse_csv_values(image_providers),
            image_models=parse_csv_values(image_models),
            refinement_iterations=parse_csv_ints(iterations, field_name="--iterations"),
            optimize_inputs=parse_csv_bools(optimize_modes, field_name="--optimize-modes"),
            auto_refine=parse_csv_bools(auto_modes, field_name="--auto-modes"),
            max_variants=max_variants,
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if not variant_list:
        console.print("[red]Error: Sweep generated zero variants[/red]")
        raise typer.Exit(1)

    try:
        source_context = load_methodology_source(input_path, pdf_pages=pdf_pages)
    except (ImportError, ValueError) as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    sweep_id = f"sweep_{generate_run_id()}"
    sweep_dir = ensure_dir(Path(output_dir) / sweep_id)
    iter_label = f"{len(variant_list)} variants"
    console.print(
        Panel.fit(
            f"[bold]PaperBanana[/bold] — Parameter Sweep\n\n"
            f"Input: {input_path.name}\n"
            f"Caption: {caption}\n"
            f"Plan: {iter_label}\n"
            f"Output: {sweep_dir}",
            border_style="magenta",
        )
    )

    if dry_run:
        preview = [variant.as_dict() for variant in variant_list[: min(10, len(variant_list))]]
        report = {
            "sweep_id": sweep_id,
            "status": "dry_run",
            "total_variants": len(variant_list),
            "preview": preview,
        }
        report_path = sweep_dir / "sweep_report.json"
        save_json(report, report_path)
        console.print(f"\n[green]Dry run complete.[/green] Planned {len(variant_list)} variants")
        console.print(f"  Report: [bold]{report_path}[/bold]")
        return

    from paperbanana.core.pipeline import PaperBananaPipeline

    if config:
        base_settings = Settings.from_yaml(config)
    else:
        base_settings = Settings()

    if auto_download_data:
        from paperbanana.data.manager import DatasetManager

        dm = DatasetManager(cache_dir=base_settings.cache_dir)
        if not dm.is_downloaded():
            console.print("  [dim]Downloading expanded reference set...[/dim]")
            try:
                dm.download()
            except Exception as e:
                console.print(f"  [yellow]Download failed: {e}, using built-in set[/yellow]")

    all_results: list[dict] = []
    total_start = time.perf_counter()

    for idx, variant in enumerate(variant_list, start=1):
        variant_dir = ensure_dir(sweep_dir / variant.variant_id)
        overrides: dict = {
            "output_dir": str(variant_dir),
            "output_format": format,
            "vlm_provider": variant.vlm_provider,
            "image_provider": variant.image_provider,
            "refinement_iterations": variant.refinement_iterations,
            "optimize_inputs": variant.optimize_inputs,
            "auto_refine": variant.auto_refine,
        }
        if variant.vlm_model:
            overrides["vlm_model"] = variant.vlm_model
        if variant.image_model:
            overrides["image_model"] = variant.image_model

        settings = base_settings.model_copy(update=overrides)

        gen_input = GenerationInput(
            source_context=source_context,
            communicative_intent=caption,
            diagram_type=DiagramType.METHODOLOGY,
        )
        console.print(f"[bold]Variant {idx}/{len(variant_list)}[/bold] — {variant.variant_id}")

        try:
            variant_start = time.perf_counter()
            pipeline = PaperBananaPipeline(settings=settings)
            result = asyncio.run(pipeline.generate(gen_input))
            variant_seconds = time.perf_counter() - variant_start
            final_critique = result.iterations[-1].critique if result.iterations else None
            suggestion_count = len(final_critique.critic_suggestions) if final_critique else 0
            quality_proxy = quality_proxy_score(suggestion_count)
            all_results.append(
                {
                    "status": "success",
                    **variant.as_dict(),
                    "run_id": result.metadata.get("run_id"),
                    "output_path": result.image_path,
                    "iterations_used": len(result.iterations),
                    "critic_suggestions": suggestion_count,
                    "quality_proxy_score": round(quality_proxy, 2),
                    "total_seconds": round(variant_seconds, 2),
                }
            )
            console.print(
                f"  [green]✓[/green] score={quality_proxy:.1f} [dim]{variant_seconds:.1f}s[/dim]"
            )
        except Exception as e:
            all_results.append(
                {
                    "status": "failed",
                    **variant.as_dict(),
                    "error": str(e),
                }
            )
            console.print(f"  [red]✗[/red] {e}")

    successful = [item for item in all_results if item["status"] == "success"]
    ranked_results = rank_sweep_results(successful)
    summary = summarize_sweep(all_results)
    elapsed = time.perf_counter() - total_start
    report = {
        "sweep_id": sweep_id,
        "status": "completed",
        "input": str(input_path),
        "caption": caption,
        "total_seconds": round(elapsed, 2),
        "summary": summary,
        "results": all_results,
        "ranked_results": ranked_results,
        "quality_proxy_note": (
            "quality_proxy_score = max(0, 100 - 12.5 * N) where N is critic suggestion "
            "count on the final iteration"
        ),
    }
    report_path = sweep_dir / "sweep_report.json"
    save_json(report, report_path)

    console.print(
        Panel.fit(
            "[bold]Sweep Complete[/bold]\n\n"
            f"Completed: {summary.get('completed', 0)}\n"
            f"Failed: {summary.get('failed', 0)}\n"
            f"Best variant: {summary.get('best_variant')}\n"
            f"Mean proxy score: {summary.get('mean_quality_proxy_score')}\n"
            f"Report: {report_path}",
            border_style="cyan",
        )
    )


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
    venue: Optional[str] = typer.Option(
        None,
        "--venue",
        help="Target venue style (neurips, icml, acl, ieee, custom)",
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
    if venue and venue.lower() not in ("neurips", "icml", "acl", "ieee", "custom"):
        console.print(
            f"[red]Error: --venue must be neurips, icml, acl, ieee, or custom. Got: {venue}[/red]"
        )
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
    if venue:
        overrides["venue"] = venue

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
        from paperbanana.core.source_loader import load_methodology_source

        try:
            source_context = load_methodology_source(input_path, pdf_pages=item.get("pdf_pages"))
        except ImportError as e:
            console.print(f"[red]Skipping item '{item_id}': {e}[/red]")
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
            continue
        except ValueError as e:
            console.print(f"[red]Skipping item '{item_id}': {e}[/red]")
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
            continue

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
    venue: Optional[str] = typer.Option(
        None,
        "--venue",
        help="Target venue style (neurips, icml, acl, ieee, custom)",
    ),
    cost_only: bool = typer.Option(
        False,
        "--cost-only",
        help="Estimate cost without making API calls",
    ),
    budget: Optional[float] = typer.Option(
        None,
        "--budget",
        help="Budget cap in USD; pipeline aborts gracefully when exceeded",
    ),
):
    """Generate a statistical plot from data."""
    if format not in ("png", "jpeg", "webp"):
        console.print(f"[red]Error: Format must be png, jpeg, or webp. Got: {format}[/red]")
        raise typer.Exit(1)
    if venue and venue.lower() not in ("neurips", "icml", "acl", "ieee", "custom"):
        console.print(
            f"[red]Error: --venue must be neurips, icml, acl, ieee, or custom. Got: {venue}[/red]"
        )
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
        venue=venue,
        budget_usd=budget,
    )

    gen_input = GenerationInput(
        source_context=source_context,
        communicative_intent=intent,
        diagram_type=DiagramType.STATISTICAL_PLOT,
        raw_data={"data": raw_data},
        aspect_ratio=aspect_ratio,
    )

    if cost_only:
        from paperbanana.core.cost_estimator import estimate_cost

        estimate = estimate_cost(settings, gen_input.diagram_type)
        iter_est = (
            f"auto (max {settings.max_iterations})"
            if settings.auto_refine
            else str(settings.refinement_iterations)
        )
        lines = [
            "[bold]PaperBanana[/bold] - Cost Estimate (Plot)\n",
            f"VLM: {settings.vlm_provider} / {settings.effective_vlm_model}",
            f"Iterations: {iter_est}",
            "",
            f"Estimated VLM calls: {estimate['vlm_calls']}",
            f"[bold]Estimated cost: ${estimate['estimated_total_usd']:.4f}[/bold]",
        ]
        if estimate.get("pricing_note"):
            lines.append(f"\n[yellow]Note: {estimate['pricing_note']}[/yellow]")
        console.print(Panel.fit("\n".join(lines), border_style="green"))
        return

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

    cost_data = result.metadata.get("cost")
    if cost_data:
        console.print(
            f"  Cost: [bold]${cost_data['total_usd']:.4f}[/bold]"
            f" [dim](VLM: ${cost_data['vlm_usd']:.4f})[/dim]"
        )


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
    context: str = typer.Option(..., "--context", help="Path to source context text file or PDF"),
    caption: str = typer.Option(..., "--caption", "-c", help="Figure caption"),
    reference: str = typer.Option(..., "--reference", "-r", help="Path to human reference image"),
    vlm_provider: str = typer.Option(
        "gemini", "--vlm-provider", help="VLM provider for evaluation"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed agent progress and timing"
    ),
    pdf_pages: Optional[str] = typer.Option(
        None,
        "--pdf-pages",
        help="PDF context only: 1-based page selection (default: all pages)",
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

    context_path = Path(context)
    if not context_path.exists():
        console.print(f"[red]Error: Context file not found: {context}[/red]")
        raise typer.Exit(1)

    from paperbanana.core.source_loader import load_methodology_source

    try:
        context_text = load_methodology_source(context_path, pdf_pages=pdf_pages)
    except ImportError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

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
    input: str = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to methodology text file or PDF",
    ),
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
    pdf_pages: Optional[str] = typer.Option(
        None,
        "--pdf-pages",
        help="PDF input only: 1-based page selection (default: all pages)",
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

    from paperbanana.core.source_loader import load_methodology_source

    try:
        source_context = load_methodology_source(input_path, pdf_pages=pdf_pages)
    except ImportError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    gen_input = GenerationInput(
        source_context=source_context,
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


@app.command("ablate-prompts")
def ablate_prompts(
    variant_prompt_dir: str = typer.Option(
        ..., "--variant-dir", help="Path to the variant prompt templates directory"
    ),
    baseline_prompt_dir: Optional[str] = typer.Option(
        None, "--baseline-dir", help="Path to baseline prompt templates (default: built-in prompts)"
    ),
    variant_name: str = typer.Option("variant", "--variant-name", help="Label for the variant"),
    baseline_name: str = typer.Option("baseline", "--baseline-name", help="Label for the baseline"),
    category: Optional[str] = typer.Option(
        None, "--category", help="Only run entries in this category"
    ),
    ids: Optional[str] = typer.Option(None, "--ids", help="Comma-separated entry IDs to compare"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Max entries to compare"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for reproducibility"),
    output_report: Optional[str] = typer.Option(
        None, "--output-report", "-o", help="Output JSON report path"
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Path to config YAML file"),
    vlm_provider: Optional[str] = typer.Option(None, "--vlm-provider", help="VLM provider"),
    image_provider: Optional[str] = typer.Option(
        None, "--image-provider", help="Image generation provider"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed progress"),
):
    """Run A/B comparison of two prompt configurations and produce a scored report."""
    configure_logging(verbose=verbose)

    from dotenv import load_dotenv

    load_dotenv()

    overrides: dict = {}
    if vlm_provider:
        overrides["vlm_provider"] = vlm_provider
    if image_provider:
        overrides["image_provider"] = image_provider
    if seed is not None:
        overrides["seed"] = seed

    if config:
        settings = Settings.from_yaml(config, **overrides)
    else:
        settings = Settings(**overrides)

    from paperbanana.evaluation.benchmark import filter_examples
    from paperbanana.evaluation.prompt_ablation import (
        PromptAblationRunner,
        validate_prompt_dir,
    )
    from paperbanana.reference.store import ReferenceStore

    try:
        validate_prompt_dir(variant_prompt_dir)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if baseline_prompt_dir:
        try:
            validate_prompt_dir(baseline_prompt_dir)
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

    runner = PromptAblationRunner(
        settings,
        baseline_prompt_dir=baseline_prompt_dir,
        variant_prompt_dir=variant_prompt_dir,
        baseline_name=baseline_name,
        variant_name=variant_name,
    )

    id_list = [s.strip() for s in ids.split(",") if s.strip()] if ids else None
    try:
        store = ReferenceStore.from_settings(settings)
        examples = store.get_all()
        if not examples:
            raise ValueError("No benchmark entries found. Run 'paperbanana data download' first.")
        entries = filter_examples(examples, category=category, ids=id_list, limit=limit)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if not entries:
        console.print("[red]Error: No entries match the given filters.[/red]")
        raise typer.Exit(1)

    console.print(
        Panel.fit(
            f"[bold]PaperBanana[/bold] — Prompt A/B Comparison\n\n"
            f"Baseline: {runner.baseline_prompt_dir} ({baseline_name})\n"
            f"Variant:  {variant_prompt_dir} ({variant_name})\n"
            f"Entries:  {len(entries)}\n"
            f"Seed:     {settings.seed or 'none'}",
            border_style="magenta",
        )
    )

    report = asyncio.run(runner.run(entries))

    default_path = Path(settings.output_dir) / f"prompt_ablation_{generate_run_id()}.json"
    report_path = Path(output_report) if output_report else default_path
    saved_path = PromptAblationRunner.save_report(report, report_path)

    summary = report.summary
    if not summary:
        console.print("[yellow]No entries were successfully scored.[/yellow]")
        console.print(f"\nReport: [bold]{saved_path}[/bold]")
        return

    # Display results
    deltas = summary.get("mean_dimension_deltas", {})
    delta_lines = []
    for dim, delta in deltas.items():
        sign = "+" if delta > 0 else ""
        delta_lines.append(f"  {dim.capitalize():14s} {sign}{delta}")

    console.print(
        Panel.fit(
            "[bold]Prompt Ablation Summary[/bold]\n\n"
            f"Scored:           {summary.get('scored', 0)}\n"
            f"Variant wins:     {summary.get('variant_wins', 0)}  "
            f"({summary.get('variant_win_rate', 0)}%)\n"
            f"Baseline wins:    {summary.get('baseline_wins', 0)}  "
            f"({summary.get('baseline_win_rate', 0)}%)\n"
            f"Ties:             {summary.get('ties', 0)}\n\n"
            f"Mean baseline:    {summary.get('mean_baseline_score', 0)}/100\n"
            f"Mean variant:     {summary.get('mean_variant_score', 0)}/100\n"
            f"Mean delta:       {summary.get('mean_overall_delta', 0):+.1f}\n\n"
            "[bold]Per-dimension deltas (variant - baseline):[/bold]\n" + "\n".join(delta_lines),
            border_style="cyan",
        )
    )

    console.print(f"\nReport: [bold]{saved_path}[/bold]")


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
    prompt_dir: Optional[str] = typer.Option(
        None,
        "--prompt-dir",
        help="Path to alternative prompt templates directory",
    ),
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
    if prompt_dir:
        overrides["prompt_dir"] = prompt_dir

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


@app.command()
def studio(
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Bind address for the Studio server",
    ),
    port: int = typer.Option(
        7860,
        "--port",
        help="TCP port for the Studio server",
    ),
    share: bool = typer.Option(
        False,
        "--share",
        help="Create a temporary public Gradio share link",
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path to YAML config (same as CLI generate)",
    ),
    output_dir: str = typer.Option(
        "outputs",
        "--output-dir",
        "-o",
        help="Default output directory (overridable in the Studio UI)",
    ),
    root_path: Optional[str] = typer.Option(
        None,
        "--root-path",
        help="Root URL path when behind a reverse proxy",
    ),
):
    """Launch PaperBanana Studio — local web UI for diagrams, plots, and evaluation."""
    try:
        from paperbanana.studio.app import launch_studio as launch_studio_ui
    except ImportError as e:
        console.print(
            "[red]PaperBanana Studio requires Gradio. Install with:[/red]\n"
            "  pip install 'paperbanana[studio]'"
        )
        console.print(f"[dim]{e}[/dim]")
        raise typer.Exit(1)

    configure_logging(verbose=False)
    from dotenv import load_dotenv

    load_dotenv()

    url = f"http://{host}:{port}/"
    console.print(
        Panel.fit(
            f"[bold]PaperBanana Studio[/bold]\n\n"
            f"Open in browser: [link={url}]{url}[/link]\n"
            f"Default output directory: {output_dir}",
            border_style="green",
        )
    )
    launch_studio_ui(
        host=host,
        port=port,
        share=share,
        config_path=config,
        default_output_dir=output_dir,
        root_path=root_path,
    )


if __name__ == "__main__":
    app()
