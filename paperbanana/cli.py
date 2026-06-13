"""PaperBanana CLI — Generate publication-quality academic illustrations."""

from __future__ import annotations

import asyncio
import json as json_mod
import os
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from paperbanana.analytics import (
    load_analytics_records,
    render_markdown_summary,
    summarize_records,
)
from paperbanana.core.config import Settings
from paperbanana.core.logging import configure_logging
from paperbanana.core.pipeline import PaperBananaPipeline
from paperbanana.core.types import (
    DiagramIR,
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


def _version_callback(value: bool) -> None:
    if value:
        import platform

        from paperbanana import __version__

        console.print(f"paperbanana {__version__} (Python {platform.python_version()})")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show paperbanana and Python versions, then exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Generate publication-quality academic illustrations from text."""


# ── Data subcommand group ─────────────────────────────────────────
data_app = typer.Typer(
    name="data",
    help="Manage reference datasets (download, info, clear).",
    no_args_is_help=True,
)
app.add_typer(data_app, name="data")

# ── References subcommand group ──────────────────────────────────
references_app = typer.Typer(
    name="references",
    help="Inspect built-in reference examples (list, show, categories).",
    no_args_is_help=True,
)
app.add_typer(references_app, name="references")

# ── Runs subcommand group ─────────────────────────────────────────
runs_app = typer.Typer(
    name="runs",
    help="Inspect and manage prior outputs (runs and batches).",
    no_args_is_help=True,
)
app.add_typer(runs_app, name="runs")

# ── Guidelines subcommand group ───────────────────────────────────
guidelines_app = typer.Typer(
    name="guidelines",
    help="Manage style guides (synthesize from reference corpus).",
    no_args_is_help=True,
)
app.add_typer(guidelines_app, name="guidelines")

# ── Venues subcommand group ───────────────────────────────────────
venues_app = typer.Typer(
    name="venues",
    help="Manage venue style packs (built-in and user-supplied).",
    no_args_is_help=True,
)
app.add_typer(venues_app, name="venues")

# ── Posters subcommand group (poster memory / corpus) ─────────────
posters_app = typer.Typer(
    name="posters",
    help="Poster layout memory: ingest exemplar posters, inspect, purge.",
    no_args_is_help=True,
)
app.add_typer(posters_app, name="posters")


def _validate_venue_or_exit(venue: Optional[str], venue_dir: Optional[str] = None) -> None:
    """Validate a --venue value against built-in and user style packs."""
    from paperbanana.guidelines.venues import UnknownVenueError, validate_venue

    try:
        validate_venue(venue, extra_dir=venue_dir)
    except UnknownVenueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


def _require_pdf_dep() -> None:
    """Raise a clean error if PyMuPDF is not installed."""
    try:
        import fitz  # noqa: F401
    except ImportError:
        console.print(
            "[red]PDF input requires PyMuPDF.[/red] Install it with:\n"
            r"  pip install 'paperbanana\[pdf]'"
        )
        raise typer.Exit(1)


def _check_pdf_dep(path: Path) -> None:
    """Raise a clean error if PyMuPDF is not installed and the path is a PDF."""
    if path.suffix.lower() == ".pdf":
        _require_pdf_dep()


def _require_studio_dep() -> None:
    """Raise a clean error if Gradio is not installed."""
    try:
        import gradio  # noqa: F401
    except ImportError:
        console.print(
            "[red]PaperBanana Studio requires Gradio. Install with:[/red]\n"
            r"  pip install 'paperbanana\[studio]'"
        )
        raise typer.Exit(1)


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


def _format_mtime(path: Path) -> str:
    try:
        ts = path.stat().st_mtime
    except OSError:
        return "?"
    # Keep it simple: local time, second resolution.
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _safe_read_json(path: Path) -> dict:
    try:
        return json_mod.loads(path.read_text(encoding="utf-8"))
    except (OSError, json_mod.JSONDecodeError):
        return {}


def _summarize_run_row(output_dir: Path, run_id: str) -> dict[str, str]:
    run_dir = output_dir / run_id
    info = _safe_read_json(run_dir / "run_input.json")
    caption = str(info.get("communicative_intent", "")).strip()
    if len(caption) > 60:
        caption = caption[:57] + "..."
    diagram_type = str(info.get("diagram_type", "")).strip()

    final_image = None
    for ext in ("png", "jpg", "jpeg", "webp"):
        p = run_dir / f"final_output.{ext}"
        if p.is_file():
            final_image = p
            break
    status = "ok" if final_image else "no_final"

    # Best-effort: count iter dirs.
    try:
        iters = len([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("iter_")])
    except OSError:
        iters = 0

    return {
        "id": run_id,
        "modified": _format_mtime(run_dir),
        "type": diagram_type or "?",
        "iters": str(iters),
        "status": status,
        "caption": caption or "(no caption found)",
    }


def _summarize_batch_row(output_dir: Path, batch_id: str) -> dict[str, str]:
    batch_dir = output_dir / batch_id
    report = _safe_read_json(batch_dir / "batch_report.json")
    kind = str(report.get("batch_kind") or "").strip() or "?"
    items = report.get("items") if isinstance(report, dict) else None
    n_items = len(items) if isinstance(items, list) else 0

    status_counts: dict[str, int] = {}
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            status = item.get("status")
            if not status:
                status = "success" if item.get("output_path") else "failed"
            status_counts[str(status)] = status_counts.get(str(status), 0) + 1

    status = "?"
    if status_counts:
        parts = [f"{k}:{v}" for k, v in sorted(status_counts.items())]
        status = ", ".join(parts)
        if len(status) > 60:
            status = status[:57] + "..."

    return {
        "id": batch_id,
        "modified": _format_mtime(batch_dir),
        "kind": kind,
        "items": str(n_items),
        "status": status,
    }


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
    image: Optional[list[str]] = typer.Option(
        None,
        "--image",
        help=(
            "Path to a reference/sketch image (hand-drawn sketch, whiteboard photo, "
            "prior figure) that guides generation. Repeatable for multiple images."
        ),
    ),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output image path"),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        help=(
            "Directory for run outputs; also where --continue/--continue-run "
            "look up the existing run"
        ),
    ),
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
    vector: bool = typer.Option(
        False,
        "--vector/--no-vector",
        help="Export SVG and PDF vector formats for statistical plots.",
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
    num_candidates: Optional[int] = typer.Option(
        None,
        "--num-candidates",
        "-k",
        min=1,
        max=8,
        help=(
            "Generate N candidate images in parallel (1-8). Planning runs once; "
            "the Visualizer-Critic refinement fans out per candidate"
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate inputs and show what would happen without making API calls",
    ),
    auto_download_data: bool = typer.Option(
        False,
        "--auto-download-data",
        help=(
            "Auto-download the PaperBananaBench reference set (~254 MB) on first run if not cached"
        ),
    ),
    reference_ids: Optional[str] = typer.Option(
        None,
        "--reference-ids",
        help="Comma-separated reference example IDs to use (bypasses automatic retrieval)",
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
        help=(
            "Venue style pack: built-in (neurips, icml, acl, ieee), a user pack "
            "(see 'paperbanana venues list'), or 'custom' (flat guideline files)"
        ),
    ),
    venue_dir: Optional[str] = typer.Option(
        None,
        "--venue-dir",
        help=(
            "User venue style pack directory "
            "(default: $PAPERBANANA_VENUE_DIR or ~/.config/paperbanana/venues)"
        ),
    ),
    export_tikz: bool = typer.Option(
        False,
        "--export-tikz",
        help=(
            "Export a TikZ/LaTeX source file alongside the generated image. "
            "Generated LaTeX is not compile-validated; test in your LaTeX environment."
        ),
    ),
    vector_export: Optional[str] = typer.Option(
        None,
        "--vector-export",
        help="Export structured vector diagram (Graphviz): none, svg, pdf, both",
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
    generate_caption: bool = typer.Option(
        False,
        "--generate-caption",
        help="Auto-generate a publication-ready figure caption (one extra VLM call)",
    ),
    reference_category: Optional[str] = typer.Option(
        None,
        "--reference-category",
        help=(
            "Filter reference examples by category (comma-separated). "
            "Valid: agent_reasoning, generative_learning, healthcare_medical, "
            "multimodal_fusion, nlp_language, optimization_theory, robotics_control, "
            "science_applications, systems_networking, vision_perception"
        ),
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
    _validate_venue_or_exit(venue, venue_dir)
    if vector_export and vector_export.lower() not in ("none", "svg", "pdf", "both"):
        console.print("[red]Error: --vector-export must be none, svg, pdf, or both[/red]")
        raise typer.Exit(1)
    if pdf_pages and (continue_last or continue_run):
        console.print(
            "[red]Error: --pdf-pages cannot be used with --continue or --continue-run[/red]"
        )
        raise typer.Exit(1)
    if image and (continue_last or continue_run):
        console.print("[red]Error: --image cannot be used with --continue or --continue-run[/red]")
        raise typer.Exit(1)

    # Validate reference/sketch images before any pipeline work starts.
    input_images: list[str] = []
    if image:
        from PIL import Image as PILImage
        from PIL import UnidentifiedImageError

        for image_path in image:
            img_file = Path(image_path)
            if not img_file.is_file():
                console.print(f"[red]Error: Image file not found: {image_path}[/red]")
                raise typer.Exit(1)
            try:
                with PILImage.open(img_file) as im:
                    im.verify()
            except (UnidentifiedImageError, OSError, ValueError):
                console.print(
                    f"[red]Error: Not a valid raster image (e.g. PNG, JPEG, WebP): "
                    f"{image_path}[/red]"
                )
                raise typer.Exit(1)
            input_images.append(str(img_file))

    _valid_categories = {
        "agent_reasoning",
        "generative_learning",
        "healthcare_medical",
        "multimodal_fusion",
        "nlp_language",
        "optimization_theory",
        "robotics_control",
        "science_applications",
        "systems_networking",
        "vision_perception",
    }
    parsed_categories: Optional[list[str]] = None
    if reference_category:
        parsed_categories = [c.strip() for c in reference_category.split(",") if c.strip()]
        unknown = [c for c in parsed_categories if c not in _valid_categories]
        if unknown:
            console.print(
                f"[red]Error: Unknown reference category: {', '.join(unknown)}.\n"
                f"Valid categories: {', '.join(sorted(_valid_categories))}[/red]"
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
    if output_dir:
        # Explicit --output-dir wins over the directory inferred from --output.
        overrides["output_dir"] = output_dir
    overrides["output_format"] = format
    if vector:
        overrides["vector_export"] = True
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
    if num_candidates is not None:
        overrides["num_candidates"] = num_candidates
    if venue:
        overrides["venue"] = venue
    if venue_dir:
        overrides["venue_dir"] = venue_dir
    if vector_export is not None:
        overrides["vector_export"] = vector_export.lower()
    if prompt_dir:
        overrides["prompt_dir"] = prompt_dir
    if generate_caption:
        overrides["generate_caption"] = True
    if export_tikz:
        overrides["export_tikz"] = True
    if parsed_categories:
        overrides["reference_category"] = parsed_categories

    if config:
        settings = Settings.from_yaml(config, **overrides)
    else:
        from dotenv import load_dotenv

        load_dotenv()
        settings = Settings(**overrides)

    from paperbanana.core.pipeline import PaperBananaPipeline

    # ── Auto-download reference set if requested ────────────────────────
    if auto_download_data:
        from paperbanana.data.manager import DatasetManager

        dm = DatasetManager(cache_dir=settings.cache_dir)
        if not dm.is_downloaded():
            console.print()
            console.print("  [dim]●[/dim] Downloading PaperBananaBench reference set...", end="")
            try:
                count = dm.download()
                console.print(f" [green]✓[/green] [dim]{count} examples cached[/dim]")
            except Exception as e:
                console.print(f" [red]✗[/red] Download failed: {e}")
                console.print("    [dim]Falling back to built-in reference set[/dim]")

    # ── Continue-run mode ─────────────────────────────────────────
    if continue_run is not None or continue_last:
        from paperbanana.core.resume import find_latest_run, load_resume_state

        resume_output_dir = settings.output_dir
        if continue_run:
            if "/" in continue_run or os.sep in continue_run:
                # --continue-run was given as a path to the run directory itself.
                run_path = Path(continue_run).expanduser()
                if not run_path.is_dir():
                    console.print(
                        f"[red]Error: Run directory not found: {run_path.resolve()}[/red]"
                    )
                    raise typer.Exit(1)
                resume_output_dir = str(run_path.parent)
                run_id = run_path.name
            else:
                run_id = continue_run
        else:
            try:
                run_id = find_latest_run(resume_output_dir)
                console.print(f"  [dim]Using latest run:[/dim] [bold]{run_id}[/bold]")
            except FileNotFoundError as e:
                console.print(f"[red]Error: {e}[/red]")
                raise typer.Exit(1)

        try:
            resume_state = load_resume_state(resume_output_dir, run_id)
        except (FileNotFoundError, ValueError) as e:
            console.print(f"[red]Error: {e}[/red]")
            console.print(
                "[dim]Hint: if the run lives outside the default output directory, "
                "pass --output-dir <dir> or --continue-run <path/to/run_dir>.[/dim]"
            )
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
            elif event.stage == PipelineProgressStage.STRUCTURER_START:
                console.print("  [dim]●[/dim] Vector export (structurer)...", end="")
            elif event.stage == PipelineProgressStage.STRUCTURER_END:
                extra = event.extra or {}
                err = extra.get("error")
                if err:
                    console.print(f" [yellow]![/yellow] [dim]{str(err)[:120]}[/dim]")
                else:
                    console.print(
                        f" [green]✓[/green] [dim]{event.seconds:.1f}s[/dim]"
                        if event.seconds is not None
                        else " [green]✓[/green]"
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
        if result.vector_svg_path:
            console.print(f"SVG: {result.vector_svg_path}")
        if result.vector_pdf_path:
            console.print(f"PDF: {result.vector_pdf_path}")
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
    _check_pdf_dep(input_path)

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
    ref_id_list = None
    if reference_ids:
        ref_id_list = [rid.strip() for rid in reference_ids.split(",") if rid.strip()]
    gen_input = GenerationInput(
        source_context=source_context,
        communicative_intent=caption,
        diagram_type=DiagramType.METHODOLOGY,
        aspect_ratio=aspect_ratio,
        reference_ids=ref_id_list,
        input_images=input_images,
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
        ]
        if settings.num_candidates > 1:
            lines.append(f"Candidates: {settings.num_candidates} (parallel)")
        lines += [
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
        if input_images:
            pdf_note += f"\nReference images: {', '.join(input_images)}"
        console.print(
            Panel.fit(
                "[bold]PaperBanana[/bold] - Dry Run\n\n"
                f"Input: {input_path}{pdf_note}\n"
                f"Caption: {caption}\n"
                f"VLM: {settings.vlm_provider} / {settings.effective_vlm_model}\n"
                f"Image: {settings.image_provider} / {settings.effective_image_model}\n"
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

    candidates_label = (
        f"\nCandidates: {settings.num_candidates} (parallel)" if settings.num_candidates > 1 else ""
    )
    if not progress_json:
        console.print(
            Panel.fit(
                f"[bold]PaperBanana[/bold] - Generating Methodology Diagram\n\n"
                f"VLM: {settings.vlm_provider} / {settings.effective_vlm_model}\n"
                f"Image: {settings.image_provider} / {settings.effective_image_model}\n"
                f"Iterations: {iter_label}{candidates_label}",
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
            elif event.stage == PipelineProgressStage.STRUCTURER_START:
                console.print("[bold]Vector[/bold] — Structured export")
                console.print("  [dim]●[/dim] Building diagram IR (structurer)...", end="")
            elif event.stage == PipelineProgressStage.STRUCTURER_END:
                extra = event.extra or {}
                err = extra.get("error")
                if err:
                    console.print(f" [yellow]![/yellow] [dim]{err[:120]}[/dim]")
                else:
                    console.print(
                        f" [green]✓[/green] [dim]{event.seconds:.1f}s[/dim]"
                        if event.seconds is not None
                        else " [green]✓[/green]"
                    )
            elif event.stage == PipelineProgressStage.VISUALIZER_START:
                extra = event.extra or {}
                candidate = extra.get("candidate")
                if event.iteration == 1 and candidate in (None, 1):
                    console.print("[bold]Phase 2[/bold] — Iterative Refinement")
                total = extra.get("total_iterations", 0)
                if event.iteration and total:
                    label = f"{event.iteration}/{total}"
                else:
                    label = str(event.iteration or "")
                if settings.auto_refine:
                    label += " (auto)"
                if candidate is not None:
                    label = f"cand {candidate} · {label}"
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
            elif event.stage == PipelineProgressStage.CAPTION_START:
                console.print("[bold]Phase 3[/bold] — Caption Generation")
                console.print("  [dim]●[/dim] Generating figure caption...", end="")
            elif event.stage == PipelineProgressStage.CAPTION_END:
                console.print(
                    f" [green]✓[/green] [dim]{event.seconds:.1f}s[/dim]"
                    if event.seconds is not None
                    else " [green]✓[/green]"
                )
            elif event.stage == PipelineProgressStage.TIKZ_EXPORTER_START:
                console.print("  [dim]●[/dim] Exporting LaTeX/TikZ...", end="")
            elif event.stage == PipelineProgressStage.TIKZ_EXPORTER_END:
                extra = event.extra or {}
                err = extra.get("error")
                if err:
                    console.print(f" [yellow]![/yellow] [dim]{str(err)[:120]}[/dim]")
                    console.print(
                        "    [yellow]TikZ export failed — no .tex file was produced "
                        "(the generated image is unaffected).[/yellow]"
                    )
                else:
                    console.print(
                        f" [green]✓[/green] [dim]{event.seconds:.1f}s[/dim]"
                        if event.seconds is not None
                        else " [green]✓[/green]"
                    )

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
    for cand in result.metadata.get("candidates") or []:
        if cand.get("error"):
            console.print(
                f"  Candidate {cand['index']}: [yellow]failed[/yellow] "
                f"[dim]{str(cand['error'])[:120]}[/dim]"
            )
        elif cand.get("image_path"):
            console.print(f"  Candidate {cand['index']}: {cand['image_path']}")
    if result.tikz_path:
        console.print(f"  TikZ:   [bold]{result.tikz_path}[/bold]")
    elif settings.export_tikz:
        console.print(
            "  [yellow]TikZ export failed — no .tex file was produced "
            "(the generated image is unaffected).[/yellow]"
        )
    console.print(f"  Run ID: [dim]{result.metadata.get('run_id', 'unknown')}[/dim]")
    if result.generated_caption:
        console.print("\n  [bold]Generated Caption:[/bold]")
        console.print(f"  {result.generated_caption}")
    if result.vector_svg_path:
        console.print(f"  SVG:    [bold]{result.vector_svg_path}[/bold]")
    if result.vector_pdf_path:
        console.print(f"  PDF:    [bold]{result.vector_pdf_path}[/bold]")
    ve = result.metadata.get("vector_export") or {}
    no_vector_files = not result.vector_svg_path and not result.vector_pdf_path
    if ve.get("mode") not in (None, "none") and no_vector_files:
        if not ve.get("graphviz_available"):
            console.print(
                "  [yellow]Vector export: install Graphviz and ensure `dot` is on PATH "
                "to render SVG/PDF (diagram_ir.json was still saved).[/yellow]"
            )

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


@app.command("regenerate")
def regenerate_from_ir(
    diagram_ir: str = typer.Option(
        ...,
        "--diagram-ir",
        help="Path to edited diagram_ir.json",
    ),
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
    config: Optional[str] = typer.Option(None, "--config", help="Path to config YAML file"),
    vlm_provider: Optional[str] = typer.Option(None, "--vlm-provider", help="VLM provider"),
    vlm_model: Optional[str] = typer.Option(None, "--vlm-model", help="VLM model name"),
    image_provider: Optional[str] = typer.Option(
        None,
        "--image-provider",
        help="Image gen provider",
    ),
    image_model: Optional[str] = typer.Option(None, "--image-model", help="Image gen model name"),
    iterations: Optional[int] = typer.Option(
        None,
        "--iterations",
        "-n",
        help="Refinement iterations",
    ),
    auto: bool = typer.Option(
        False, "--auto", help="Loop until critic is satisfied (with safety cap)"
    ),
    max_iterations: Optional[int] = typer.Option(
        None, "--max-iterations", help="Safety cap for --auto mode (default: 30)"
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
    save_prompts: Optional[bool] = typer.Option(
        None,
        "--save-prompts/--no-save-prompts",
        help="Save formatted prompts into the run directory (for debugging)",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="Random seed for reproducible image generation",
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
    generate_caption: bool = typer.Option(
        False,
        "--generate-caption",
        help="Auto-generate a publication-ready figure caption (one extra VLM call)",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed agent progress and timing"
    ),
):
    """Regenerate from an edited DiagramIR, preserving locked elements."""
    if format not in ("png", "jpeg", "webp", "svg"):
        console.print(f"[red]Error: Format must be png, jpeg, webp, or svg. Got: {format}[/red]")
        raise typer.Exit(1)

    configure_logging(verbose=verbose)
    overrides = {"output_format": format, "auto_refine": auto, "generate_caption": generate_caption}
    if iterations is not None:
        overrides["refinement_iterations"] = iterations
    if max_iterations is not None:
        overrides["max_iterations"] = max_iterations
    if vlm_provider:
        overrides["vlm_provider"] = vlm_provider
    if vlm_model:
        overrides["vlm_model"] = vlm_model
    if image_provider:
        overrides["image_provider"] = image_provider
    if image_model:
        overrides["image_model"] = image_model
    if seed is not None:
        overrides["seed"] = seed
    if save_prompts is not None:
        overrides["save_prompts"] = save_prompts

    settings = Settings.from_yaml(config, **overrides) if config else Settings(**overrides)

    input_path = Path(input)
    if not input_path.exists():
        console.print(f"[red]Error: Input file not found: {input}[/red]")
        raise typer.Exit(1)
    _check_pdf_dep(input_path)
    from paperbanana.core.source_loader import load_methodology_source

    try:
        source_context = load_methodology_source(input_path, pdf_pages=pdf_pages)
    except (ImportError, ValueError) as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    ir_path = Path(diagram_ir)
    if not ir_path.exists():
        console.print(f"[red]Error: Diagram IR not found: {diagram_ir}[/red]")
        raise typer.Exit(1)
    try:
        diagram_ir_model = DiagramIR.model_validate_json(ir_path.read_text(encoding="utf-8"))
    except Exception as e:
        console.print(f"[red]Error: Invalid diagram IR JSON: {e}[/red]")
        raise typer.Exit(1)

    if not progress_json:
        lock_counts = (
            len(diagram_ir_model.locks.locked_node_ids),
            len(diagram_ir_model.locks.locked_edge_refs),
            len(diagram_ir_model.locks.locked_group_ids),
        )
        console.print(
            Panel.fit(
                f"[bold]PaperBanana[/bold] - Lock-aware IR Regeneration\n\n"
                f"VLM: {settings.vlm_provider} / {settings.effective_vlm_model}\n"
                f"Image: {settings.image_provider} / {settings.effective_image_model}\n"
                f"Locked nodes/edges/groups: {lock_counts[0]}/{lock_counts[1]}/{lock_counts[2]}",
                border_style="magenta",
            )
        )

    total_start = time.perf_counter()

    async def _run_with_progress():
        def _on_progress(event: str, payload: dict) -> None:
            if progress_json:
                console.print(json_mod.dumps({"event": event, **payload}), highlight=False)

        pipeline = PaperBananaPipeline(
            settings=settings,
            progress_callback=_on_progress if progress_json else None,
        )

        def on_progress(event: PipelineProgressEvent) -> None:
            if event.stage == PipelineProgressStage.VISUALIZER_START:
                it = event.iteration or "?"
                total = (event.extra or {}).get("total_iterations", "?")
                console.print(f"  [dim]●[/dim] Generating image [{it}/{total}]...", end="")
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

        return await pipeline.regenerate_from_ir(
            diagram_ir=diagram_ir_model,
            source_context=source_context,
            caption=caption,
            aspect_ratio=aspect_ratio,
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


@app.command()
def sweep(
    input: Optional[str] = typer.Option(
        None,
        "--input",
        "-i",
        help="Path to methodology text file or PDF (.pdf requires: pip install 'paperbanana[pdf]')",
    ),
    caption: Optional[str] = typer.Option(
        None,
        "--caption",
        "-c",
        help="Figure caption / communicative intent",
    ),
    manifest: Optional[str] = typer.Option(
        None,
        "--manifest",
        "-m",
        help="Path to sweep manifest (YAML or JSON). Mutually exclusive with axis flags.",
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
        help=(
            "Auto-download the PaperBananaBench reference set (~254 MB) on first run if not cached"
        ),
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

    axis_flag_values = (
        vlm_providers,
        vlm_models,
        image_providers,
        image_models,
        iterations,
        optimize_modes,
        auto_modes,
    )
    if manifest is not None and any(v is not None for v in axis_flag_values):
        console.print(
            "[red]Error: --manifest is mutually exclusive with axis flags "
            "(--vlm-providers, --vlm-models, --image-providers, --image-models, "
            "--iterations, --optimize-modes, --auto-modes)[/red]"
        )
        raise typer.Exit(1)
    if manifest is None and (not input or not caption):
        console.print(
            "[red]Error: --input and --caption are required unless --manifest is set[/red]"
        )
        raise typer.Exit(1)

    configure_logging(verbose=verbose)

    from paperbanana.core.sweep import (
        build_sweep_variants,
        load_sweep_manifest,
        parse_csv_bools,
        parse_csv_ints,
        parse_csv_values,
        quality_proxy_score,
        rank_sweep_results,
        summarize_sweep,
    )

    axes_from_manifest: dict[str, list] | None = None
    if manifest is not None:
        try:
            parsed = load_sweep_manifest(Path(manifest))
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
        input = parsed["input"]
        caption = parsed["caption"]
        if parsed["pdf_pages"] is not None:
            pdf_pages = parsed["pdf_pages"]
        if parsed["max_variants"] is not None:
            max_variants = parsed["max_variants"]
        axes_from_manifest = parsed["axes"]

    input_path = Path(input)
    if not input_path.exists():
        console.print(f"[red]Error: Input file not found: {input}[/red]")
        raise typer.Exit(1)
    _check_pdf_dep(input_path)

    from dotenv import load_dotenv

    load_dotenv()

    from paperbanana.core.source_loader import load_methodology_source

    try:
        if axes_from_manifest is not None:
            variant_list = build_sweep_variants(
                vlm_providers=[str(x) for x in axes_from_manifest["vlm_providers"]],
                vlm_models=[str(x) for x in axes_from_manifest["vlm_models"]],
                image_providers=[str(x) for x in axes_from_manifest["image_providers"]],
                image_models=[str(x) for x in axes_from_manifest["image_models"]],
                refinement_iterations=[int(x) for x in axes_from_manifest["refinement_iterations"]],
                optimize_inputs=[bool(x) for x in axes_from_manifest["optimize_inputs"]],
                auto_refine=[bool(x) for x in axes_from_manifest["auto_refine"]],
                max_variants=max_variants,
            )
        else:
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
        help=(
            "Venue style pack: built-in (neurips, icml, acl, ieee), a user pack "
            "(see 'paperbanana venues list'), or 'custom' (flat guideline files)"
        ),
    ),
    auto_download_data: bool = typer.Option(
        False,
        "--auto-download-data",
        help=(
            "Auto-download the PaperBananaBench reference set (~254 MB) on first run if not cached"
        ),
    ),
    resume_batch: Optional[str] = typer.Option(
        None, "--resume-batch", help="Batch ID or batch directory to resume"
    ),
    retry_failed: bool = typer.Option(
        False, "--retry-failed", help="Retry previously failed items during resume"
    ),
    max_retries: int = typer.Option(
        0, "--max-retries", help="Extra retries per item after first failure"
    ),
    concurrency: int = typer.Option(1, "--concurrency", help="Parallel item workers"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed progress"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress per-item status table"),
):
    """Generate multiple methodology diagrams from a manifest file (YAML or JSON)."""
    if format not in ("png", "jpeg", "webp"):
        console.print(f"[red]Error: Format must be png, jpeg, or webp. Got: {format}[/red]")
        raise typer.Exit(1)
    _validate_venue_or_exit(venue)
    if max_retries < 0:
        console.print("[red]Error: --max-retries must be >= 0[/red]")
        raise typer.Exit(1)
    if concurrency < 1:
        console.print("[red]Error: --concurrency must be >= 1[/red]")
        raise typer.Exit(1)

    configure_logging(verbose=verbose)
    manifest_path = Path(manifest)
    if not manifest_path.exists():
        console.print(f"[red]Error: Manifest not found: {manifest}[/red]")
        raise typer.Exit(1)

    from paperbanana.core.batch import load_batch_manifest_with_composite
    from paperbanana.core.workflow_runner import run_methodology_batch

    try:
        items, _composite = load_batch_manifest_with_composite(manifest_path)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        console.print(f"[red]Error loading manifest: {e}[/red]")
        raise typer.Exit(1)

    if any(str(item.get("input", "")).lower().endswith(".pdf") for item in items):
        try:
            _require_pdf_dep()
        except typer.Exit:
            raise

    console.print(
        Panel.fit(
            f"[bold]PaperBanana[/bold] — {'Resume ' if resume_batch else ''}Batch Generation\n\n"
            f"Manifest: {manifest_path.name}\n"
            f"Items: {len(items)}\n"
            f"Output parent: {Path(output_dir).resolve()}\n"
            f"Concurrency: {concurrency}",
            border_style="blue",
        )
    )
    console.print()

    try:
        result = run_methodology_batch(
            manifest_path=manifest_path,
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
            progress_callback=console.print,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    batch_dir = Path(result["batch_dir"])
    report_path = Path(result["batch_report_path"])
    if not result.get("had_work", True):
        console.print("[yellow]Nothing to run: all items already completed.[/yellow]")
        console.print(f"  Report: [bold]{report_path}[/bold]")
        return

    succeeded = int(result["succeeded"])
    failed = int(result["failed"])
    skipped = int(result["skipped"])
    total_elapsed = 0.0
    report_file = batch_dir / "batch_report.json"
    if report_file.exists():
        try:
            report = json_mod.loads(report_file.read_text(encoding="utf-8"))
            total_elapsed = float(report.get("total_seconds") or 0.0)
        except Exception:
            pass
    ri = result.get("items_summary", [])
    console.print(
        f"[green]Batch complete.[/green] [dim]{total_elapsed:.1f}s · "
        f"{succeeded} succeeded · {failed} failed · {skipped} skipped[/dim]"
    )
    console.print(f"  Report: [bold]{report_path}[/bold]")
    if not quiet:
        large_batch = len(ri) > 20
        t = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
        t.add_column("#", style="dim", width=4)
        t.add_column("Item", min_width=20)
        t.add_column("Status", width=10)
        t.add_column("Output / Error")
        for idx, item in enumerate(ri):
            if large_batch and item.get("status") == "success":
                continue
            ok = item.get("status") == "success"
            status_str = "[green]✓ done[/green]" if ok else "[red]✗ failed[/red]"
            detail = str(item.get("output_path" if ok else "error") or "—")
            t.add_row(str(idx + 1), str(item.get("id", "—")), status_str, detail)
        if large_batch and succeeded > 0:
            console.print(
                f"[dim]{succeeded} succeeded (hidden), {failed} failed (shown above)[/dim]"
            )
        console.print(t)
    if failed > 0:
        raise typer.Exit(1)

    comp_path = result.get("composite_path")
    if comp_path:
        console.print(f"  Composite: [bold]{comp_path}[/bold]")


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


@app.command("sweep-report")
def sweep_report(
    sweep_dir: Optional[str] = typer.Option(
        None,
        "--sweep-dir",
        "-s",
        help="Path to sweep run directory (e.g. outputs/sweep_20250109_123456_abc)",
    ),
    sweep_id: Optional[str] = typer.Option(
        None,
        "--sweep-id",
        help="Sweep ID (e.g. sweep_20250109_123456_abc); resolved under --output-dir",
    ),
    output_dir: str = typer.Option(
        "outputs",
        "--output-dir",
        "-o",
        help="Parent directory for sweep runs (used with --sweep-id)",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        help="Output path for the report file (default: <sweep_dir>/sweep_report.<md|html>)",
    ),
    format: str = typer.Option(
        "markdown",
        "--format",
        "-f",
        help="Report format: markdown or html",
    ),
    no_thumbnails: bool = typer.Option(
        False,
        "--no-thumbnails",
        help="HTML only: skip the top-ranked thumbnail grid",
    ),
):
    """Generate a human-readable report from an existing sweep run (sweep_report.json)."""
    if format not in ("markdown", "html", "md"):
        console.print(f"[red]Error: Format must be markdown or html. Got: {format}[/red]")
        raise typer.Exit(1)
    if sweep_dir is None and sweep_id is None:
        console.print("[red]Error: Provide either --sweep-dir or --sweep-id[/red]")
        raise typer.Exit(1)
    if sweep_dir is not None and sweep_id is not None:
        console.print("[red]Error: Provide only one of --sweep-dir or --sweep-id[/red]")
        raise typer.Exit(1)

    from paperbanana.core.sweep import write_sweep_report

    if sweep_dir is not None:
        path = Path(sweep_dir)
    else:
        path = Path(output_dir) / sweep_id

    output_path = Path(output) if output else None
    fmt = "markdown" if format == "md" else format
    try:
        written = write_sweep_report(
            path,
            output_path=output_path,
            format=fmt,
            include_thumbnails=not no_thumbnails,
        )
        console.print(f"[green]Report written to:[/green] [bold]{written}[/bold]")
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def orchestrate(
    paper: Optional[str] = typer.Option(
        None,
        "--paper",
        "-p",
        help="Path to paper source (.txt/.md/.pdf) used to plan and generate a full figure package",
    ),
    resume_orchestrate: Optional[str] = typer.Option(
        None,
        "--resume-orchestrate",
        help="Orchestration ID or orchestration directory to resume",
    ),
    retry_failed: bool = typer.Option(
        False,
        "--retry-failed",
        help="Retry previously failed items during resume",
    ),
    max_retries: int = typer.Option(
        0,
        "--max-retries",
        help="Extra retries per task after first failure",
    ),
    data_dir: Optional[str] = typer.Option(
        None,
        "--data-dir",
        help="Optional directory with CSV/JSON files for auto-planned statistical plots",
    ),
    output_dir: str = typer.Option(
        "outputs",
        "--output-dir",
        "-o",
        help="Parent directory for orchestration output package",
    ),
    max_method_figures: int = typer.Option(
        4,
        "--max-method-figures",
        help="Maximum methodology figures to plan and generate from paper sections",
    ),
    max_plot_figures: int = typer.Option(
        4,
        "--max-plot-figures",
        help="Maximum statistical plots to plan from data files",
    ),
    pdf_pages: Optional[str] = typer.Option(
        None,
        "--pdf-pages",
        help="PDF input only: 1-based pages (e.g. '1-5', '2,4,6-8'); default: all pages",
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Path to config YAML file"),
    vlm_provider: Optional[str] = typer.Option(None, "--vlm-provider", help="VLM provider"),
    vlm_model: Optional[str] = typer.Option(None, "--vlm-model", help="VLM model name"),
    image_provider: Optional[str] = typer.Option(
        None, "--image-provider", help="Image generation provider"
    ),
    image_model: Optional[str] = typer.Option(None, "--image-model", help="Image model name"),
    iterations: Optional[int] = typer.Option(
        None, "--iterations", "-n", help="Refinement iterations per figure"
    ),
    auto: bool = typer.Option(
        False, "--auto", help="Loop until critic is satisfied (with safety cap)"
    ),
    max_iterations: Optional[int] = typer.Option(
        None, "--max-iterations", help="Safety cap for --auto mode"
    ),
    optimize: bool = typer.Option(
        False, "--optimize", help="Enable input optimization before generation"
    ),
    format: str = typer.Option(
        "png", "--format", "-f", help="Output image format (png, jpeg, webp)"
    ),
    save_prompts: Optional[bool] = typer.Option(
        None,
        "--save-prompts/--no-save-prompts",
        help="Save prompts for each generated figure run",
    ),
    venue: Optional[str] = typer.Option(
        None,
        "--venue",
        help=(
            "Venue style pack: built-in (neurips, icml, acl, ieee), a user pack "
            "(see 'paperbanana venues list'), or 'custom' (flat guideline files)"
        ),
    ),
    concurrency: int = typer.Option(
        1, "--concurrency", help="Maximum concurrent figure generations"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Plan orchestration package without generating figures",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed progress"),
):
    """Generate a publication-ready multi-figure package from a full paper."""
    is_resume = bool(resume_orchestrate)
    if format not in ("png", "jpeg", "webp"):
        console.print(f"[red]Error: Format must be png, jpeg, or webp. Got: {format}[/red]")
        raise typer.Exit(1)
    _validate_venue_or_exit(venue)
    if max_method_figures < 1:
        console.print("[red]Error: --max-method-figures must be >= 1[/red]")
        raise typer.Exit(1)
    if max_plot_figures < 0:
        console.print("[red]Error: --max-plot-figures must be >= 0[/red]")
        raise typer.Exit(1)
    if concurrency < 1:
        console.print("[red]Error: --concurrency must be >= 1[/red]")
        raise typer.Exit(1)
    if max_retries < 0:
        console.print("[red]Error: --max-retries must be >= 0[/red]")
        raise typer.Exit(1)
    if is_resume and paper:
        console.print("[red]Error: provide only one of --paper or --resume-orchestrate[/red]")
        raise typer.Exit(1)
    if not is_resume and not paper:
        console.print("[red]Error: provide --paper for new orchestrations[/red]")
        raise typer.Exit(1)
    if is_resume and data_dir:
        console.print("[red]Error: --data-dir is only valid for new orchestrations[/red]")
        raise typer.Exit(1)
    if is_resume and pdf_pages:
        console.print("[red]Error: --pdf-pages is only valid for new orchestrations[/red]")
        raise typer.Exit(1)

    configure_logging(verbose=verbose)

    from paperbanana.core.workflow_runner import run_orchestration_package

    def _orchestration_header_panel(summary: dict) -> None:
        orch_dir = Path(summary["orchestrate_dir"])
        paper_display = Path(str(summary.get("paper_path") or "paper")).name
        header = "Resume " if summary.get("resumed") else ""
        console.print(
            Panel.fit(
                f"[bold]PaperBanana[/bold] — {header}Figure Package Orchestration\n\n"
                f"Paper: {paper_display}\n"
                f"Planned methodology figures: {summary['methodology_items_planned']}\n"
                f"Planned plot figures: {summary['plot_items_planned']}\n"
                f"Package dir: {orch_dir}",
                border_style="magenta",
            )
        )

    try:
        result = run_orchestration_package(
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
            progress_callback=console.print,
            after_plan_callback=_orchestration_header_panel,
        )
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    orchestrate_dir = Path(result["orchestrate_dir"])
    plan_path = Path(result["orchestration_plan_path"])
    if dry_run:
        console.print("[green]Dry run complete.[/green] Orchestration plan:")
        console.print(f"  [bold]{plan_path}[/bold]")
        return

    if not result.get("had_work", True):
        console.print("[yellow]Nothing to run: all tasks already completed.[/yellow]")
        console.print(f"  Package: [bold]{orchestrate_dir / 'figure_package.json'}[/bold]")
        return

    total_seconds = float(result.get("total_seconds") or 0.0)
    success_count = int(result.get("generated_count", 0))
    fail_count = int(result.get("failed_count", 0))
    package_manifest_path = Path(result["figure_package_path"])
    console.print(
        f"[green]Orchestration complete.[/green] [dim]{success_count} generated · "
        f"{fail_count} failed · {total_seconds:.1f}s[/dim]"
    )
    console.print(f"  Package: [bold]{package_manifest_path}[/bold]")
    console.print(f"  LaTeX: [bold]{orchestrate_dir / 'figures.tex'}[/bold]")
    console.print(f"  Captions: [bold]{orchestrate_dir / 'captions.md'}[/bold]")

    if fail_count > 0:
        raise typer.Exit(1)


@app.command()
def composite(
    images: list[str] = typer.Argument(..., help="Paths to images to compose into a single figure"),
    layout: str = typer.Option(
        "auto", "--layout", "-l", help="Grid layout: 'RxC' (e.g. '1x3', '2x2') or 'auto'"
    ),
    labels: Optional[str] = typer.Option(
        None,
        "--labels",
        help="Comma-separated labels (e.g. '(a),(b),(c)') or 'none' to disable",
    ),
    spacing: int = typer.Option(20, "--spacing", "-s", help="Pixel spacing between panels"),
    label_position: str = typer.Option(
        "bottom", "--label-position", help="Label placement: 'top' or 'bottom'"
    ),
    label_font_size: int = typer.Option(32, "--label-font-size", help="Font size for panel labels"),
    output: str = typer.Option(
        "composite_output.png", "--output", "-o", help="Output path for the composite image"
    ),
):
    """Compose multiple images into a single labeled multi-panel figure."""
    if label_position not in ("top", "bottom"):
        console.print(
            f"[red]Error: --label-position must be 'top' or 'bottom'. Got: {label_position}[/red]"
        )
        raise typer.Exit(1)

    for img_path in images:
        if not Path(img_path).exists():
            console.print(f"[red]Error: Image not found: {img_path}[/red]")
            raise typer.Exit(1)

    from paperbanana.core.composite import compose_images

    label_list: list[str] | None = None
    auto_label = True
    if labels is not None:
        if labels.lower() == "none":
            auto_label = False
        else:
            label_list = [item.strip() for item in labels.split(",")]
            auto_label = False

    try:
        compose_images(
            image_paths=images,
            layout=layout,
            labels=label_list,
            auto_label=auto_label,
            spacing=spacing,
            label_position=label_position,
            label_font_size=label_font_size,
            output_path=output,
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Composite saved to:[/green] [bold]{output}[/bold]")


@app.command("plot-batch")
def plot_batch(
    manifest: str = typer.Option(
        ..., "--manifest", "-m", help="Path to plot batch manifest (YAML or JSON)"
    ),
    output_dir: str = typer.Option(
        "outputs",
        "--output-dir",
        "-o",
        help="Parent directory for batch run (batch_<id> will be created here)",
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Path to config YAML file"),
    vlm_provider: Optional[str] = typer.Option(
        None, "--vlm-provider", help="VLM provider (default: gemini)"
    ),
    vlm_model: Optional[str] = typer.Option(None, "--vlm-model", help="VLM model name"),
    image_provider: Optional[str] = typer.Option(
        None, "--image-provider", help="Image gen provider"
    ),
    image_model: Optional[str] = typer.Option(None, "--image-model", help="Image gen model name"),
    iterations: Optional[int] = typer.Option(
        None, "--iterations", "-n", help="Refinement iterations per plot"
    ),
    auto: bool = typer.Option(
        False, "--auto", help="Loop until critic satisfied per item (with safety cap)"
    ),
    max_iterations: Optional[int] = typer.Option(
        None, "--max-iterations", help="Safety cap for --auto"
    ),
    optimize: bool = typer.Option(
        False, "--optimize", help="Preprocess inputs per item (enrich context, sharpen intent)"
    ),
    format: str = typer.Option(
        "png", "--format", "-f", help="Output image format (png, jpeg, webp)"
    ),
    save_prompts: Optional[bool] = typer.Option(
        None,
        "--save-prompts/--no-save-prompts",
        help="Save prompts per run",
    ),
    venue: Optional[str] = typer.Option(
        None,
        "--venue",
        help=(
            "Venue style pack: built-in (neurips, icml, acl, ieee), a user pack "
            "(see 'paperbanana venues list'), or 'custom' (flat guideline files)"
        ),
    ),
    aspect_ratio: Optional[str] = typer.Option(
        None,
        "--aspect-ratio",
        "-ar",
        help="Default aspect ratio when not set per manifest item",
    ),
    resume_batch: Optional[str] = typer.Option(
        None, "--resume-batch", help="Batch ID or batch directory to resume"
    ),
    retry_failed: bool = typer.Option(
        False, "--retry-failed", help="Retry previously failed items during resume"
    ),
    max_retries: int = typer.Option(
        0, "--max-retries", help="Extra retries per item after first failure"
    ),
    concurrency: int = typer.Option(1, "--concurrency", help="Parallel item workers"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed progress"),
):
    """Generate multiple statistical plots from a manifest (data + intent per item)."""
    if format not in ("png", "jpeg", "webp"):
        console.print(f"[red]Error: Format must be png, jpeg, or webp. Got: {format}[/red]")
        raise typer.Exit(1)
    _validate_venue_or_exit(venue)
    if max_retries < 0:
        console.print("[red]Error: --max-retries must be >= 0[/red]")
        raise typer.Exit(1)
    if concurrency < 1:
        console.print("[red]Error: --concurrency must be >= 1[/red]")
        raise typer.Exit(1)

    configure_logging(verbose=verbose)
    manifest_path = Path(manifest)
    if not manifest_path.exists():
        console.print(f"[red]Error: Manifest not found: {manifest}[/red]")
        raise typer.Exit(1)

    from paperbanana.core.batch import load_plot_batch_manifest
    from paperbanana.core.workflow_runner import run_plot_batch

    try:
        items = load_plot_batch_manifest(manifest_path)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        console.print(f"[red]Error loading manifest: {e}[/red]")
        raise typer.Exit(1)

    _pb = "Resume " if resume_batch else ""
    console.print(
        Panel.fit(
            f"[bold]PaperBanana[/bold] — {_pb}Batch Plot Generation\n\n"
            f"Manifest: {manifest_path.name}\n"
            f"Items: {len(items)}\n"
            f"Output parent: {Path(output_dir).resolve()}\n"
            f"Concurrency: {concurrency}",
            border_style="green",
        )
    )
    console.print()

    try:
        result = run_plot_batch(
            manifest_path=manifest_path,
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
            progress_callback=console.print,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    batch_dir = Path(result["batch_dir"])
    report_path = Path(result["batch_report_path"])
    if not result.get("had_work", True):
        console.print("[yellow]Nothing to run: all items already completed.[/yellow]")
        console.print(f"  Report: [bold]{report_path}[/bold]")
        return

    succeeded = int(result["succeeded"])
    total_elapsed = 0.0
    report_file = batch_dir / "batch_report.json"
    if report_file.exists():
        try:
            report = json_mod.loads(report_file.read_text(encoding="utf-8"))
            total_elapsed = float(report.get("total_seconds") or 0.0)
        except Exception:
            pass
    console.print(
        f"[green]Plot batch complete.[/green] [dim]{total_elapsed:.1f}s · "
        f"{succeeded}/{len(items)} succeeded[/dim]"
    )
    console.print(f"  Report: [bold]{report_path}[/bold]")


@app.command()
def plot(
    data: str = typer.Option(..., "--data", "-d", help="Path to data file (CSV or JSON)"),
    intent: str = typer.Option(..., "--intent", help="Communicative intent for the plot"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output image path"),
    vlm_provider: Optional[str] = typer.Option(
        None, "--vlm-provider", help="VLM provider (gemini, openai)"
    ),
    vlm_model: Optional[str] = typer.Option(None, "--vlm-model", help="VLM model name"),
    iterations: Optional[int] = typer.Option(
        None, "--iterations", "-n", help="Number of refinement iterations"
    ),
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
    max_iterations: Optional[int] = typer.Option(
        None, "--max-iterations", help="Safety cap for --auto mode (default: 30)"
    ),
    save_prompts: Optional[bool] = typer.Option(
        None,
        "--save-prompts/--no-save-prompts",
        help="Save formatted prompts into the run directory (for debugging)",
    ),
    venue: Optional[str] = typer.Option(
        None,
        "--venue",
        help=(
            "Venue style pack: built-in (neurips, icml, acl, ieee), a user pack "
            "(see 'paperbanana venues list'), or 'custom' (flat guideline files)"
        ),
    ),
    venue_dir: Optional[str] = typer.Option(
        None,
        "--venue-dir",
        help=(
            "User venue style pack directory "
            "(default: $PAPERBANANA_VENUE_DIR or ~/.config/paperbanana/venues)"
        ),
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
    generate_caption: bool = typer.Option(
        False,
        "--generate-caption",
        help="Auto-generate a publication-ready figure caption (one extra VLM call)",
    ),
    export_pgfplots: bool = typer.Option(
        False,
        "--export-pgfplots",
        help=(
            "Export a PGFPlots/LaTeX source file alongside the generated plot. "
            "Generated LaTeX is not compile-validated; test in your LaTeX environment."
        ),
    ),
    vector: bool = typer.Option(
        False,
        "--vector/--no-vector",
        help="Also export SVG and PDF vector formats alongside the raster output.",
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Path to config YAML file"),
):
    """Generate a statistical plot from data."""
    if format not in ("png", "jpeg", "webp"):
        console.print(f"[red]Error: Format must be png, jpeg, or webp. Got: {format}[/red]")
        raise typer.Exit(1)
    _validate_venue_or_exit(venue, venue_dir)

    configure_logging(verbose=verbose)
    data_path = Path(data)
    if not data_path.exists():
        console.print(f"[red]Error: Data file not found: {data}[/red]")
        raise typer.Exit(1)

    from paperbanana.core.plot_data import load_statistical_plot_payload

    try:
        source_context, raw_data = load_statistical_plot_payload(data_path)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    overrides = {}
    # Plots are rendered via VLM-generated matplotlib code; the image-gen
    # provider is never invoked, so "none" avoids demanding credentials
    # the workflow doesn't use (see DummyImageGen).
    overrides["image_provider"] = "none"
    if vlm_provider:
        overrides["vlm_provider"] = vlm_provider
    if vlm_model:
        overrides["vlm_model"] = vlm_model
    if iterations is not None:
        overrides["refinement_iterations"] = iterations
    if auto:
        overrides["auto_refine"] = True
    if max_iterations is not None:
        overrides["max_iterations"] = max_iterations
    if optimize:
        overrides["optimize_inputs"] = True
    # Plot runs save prompts by default (parity with prior behavior on main).
    overrides["save_prompts"] = True if save_prompts is None else save_prompts
    if output:
        overrides["output_dir"] = str(Path(output).parent)
    overrides["output_format"] = format
    if venue:
        overrides["venue"] = venue
    if venue_dir:
        overrides["venue_dir"] = venue_dir
    if budget is not None:
        overrides["budget_usd"] = budget
    if generate_caption:
        overrides["generate_caption"] = True
    if vector:
        overrides["vector_export"] = True
    if export_pgfplots:
        overrides["export_pgfplots"] = True

    if config:
        settings = Settings.from_yaml(config, **overrides)
    else:
        from dotenv import load_dotenv

        load_dotenv()
        settings = Settings(**overrides)

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

    if auto:
        iter_label = f"auto (max {settings.max_iterations})"
    else:
        iter_label = str(settings.refinement_iterations)

    console.print(
        Panel.fit(
            f"[bold]PaperBanana[/bold] - Generating Statistical Plot\n\n"
            f"Data: {data_path.name}\n"
            f"Intent: {intent}\n"
            f"VLM: {settings.vlm_provider} / {settings.effective_vlm_model}\n"
            f"Iterations: {iter_label}",
            border_style="green",
        )
    )

    from paperbanana.core.pipeline import PaperBananaPipeline

    async def _run():
        pipeline = PaperBananaPipeline(settings=settings)
        return await pipeline.generate(gen_input)

    result = asyncio.run(_run())
    console.print(f"\n[green]Done![/green] Plot saved to: [bold]{result.image_path}[/bold]")
    if result.tikz_path:
        console.print(f"  PGFPlots: [bold]{result.tikz_path}[/bold]")
    elif settings.export_pgfplots:
        console.print(
            "  [yellow]PGFPlots export failed — no .tex file was produced "
            "(the generated plot is unaffected).[/yellow]"
        )
    vector_paths = result.metadata.get("vector_output_paths", {})
    for fmt, path in vector_paths.items():
        console.print(f"[green]Vector ({fmt.upper()}):[/green] [bold]{path}[/bold]")
    if result.generated_caption:
        console.print("\n  [bold]Generated Caption:[/bold]")
        console.print(f"  {result.generated_caption}")

    cost_data = result.metadata.get("cost")
    if cost_data:
        console.print(
            f"  Cost: [bold]${cost_data['total_usd']:.4f}[/bold]"
            f" [dim](VLM: ${cost_data['vlm_usd']:.4f})[/dim]"
        )


@app.command()
def tikz(
    input: str = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to an existing generated image to convert to LaTeX/TikZ",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output .tex file path (default: same directory as input, with .tex extension)",
    ),
    source_context: Optional[str] = typer.Option(
        None,
        "--source-context",
        help="Path to a methodology text file for context (improves TikZ fidelity)",
    ),
    caption: str = typer.Option(
        "",
        "--caption",
        "-c",
        help="Figure caption / communicative intent (improves TikZ fidelity)",
    ),
    diagram_type: str = typer.Option(
        "diagram",
        "--diagram-type",
        help="Type of illustration: diagram (→ TikZ) or plot (→ PGFPlots)",
    ),
    vlm_provider: Optional[str] = typer.Option(
        None, "--vlm-provider", help="VLM provider (gemini, openai, anthropic, …)"
    ),
    vlm_model: Optional[str] = typer.Option(None, "--vlm-model", help="VLM model name override"),
    venue: Optional[str] = typer.Option(
        None,
        "--venue",
        help=(
            "Venue style pack: built-in (neurips, icml, acl, ieee), a user pack "
            "(see 'paperbanana venues list'), or 'custom' (flat guideline files)"
        ),
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Path to a YAML config file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed progress"),
):
    """Convert an existing generated image to a compilable LaTeX/TikZ source file."""
    input_path = Path(input)
    if not input_path.exists():
        console.print(f"[red]Error: Input file not found: {input}[/red]")
        raise typer.Exit(1)

    if diagram_type not in ("diagram", "plot"):
        console.print("[red]Error: --diagram-type must be 'diagram' or 'plot'[/red]")
        raise typer.Exit(1)

    _validate_venue_or_exit(venue)

    configure_logging(verbose=verbose)

    # Resolve output path
    tex_path = Path(output) if output else input_path.with_suffix(".tex")

    # Load optional source context
    context_text = ""
    if source_context:
        sc_path = Path(source_context)
        if not sc_path.exists():
            console.print(f"[red]Error: Source context file not found: {source_context}[/red]")
            raise typer.Exit(1)
        context_text = sc_path.read_text(encoding="utf-8")

    # Build settings
    overrides: dict = {}
    if vlm_provider:
        overrides["vlm_provider"] = vlm_provider
    if vlm_model:
        overrides["vlm_model"] = vlm_model
    if venue:
        overrides["venue"] = venue

    if config:
        settings = Settings.from_yaml(config, **overrides)
    else:
        from dotenv import load_dotenv

        load_dotenv()
        settings = Settings(**overrides)

    from paperbanana.agents.tikz_exporter import TikZExporterAgent
    from paperbanana.core.types import DiagramType
    from paperbanana.providers.registry import ProviderRegistry

    dtype = DiagramType.METHODOLOGY if diagram_type == "diagram" else DiagramType.STATISTICAL_PLOT

    console.print(
        Panel.fit(
            f"[bold]PaperBanana[/bold] — Export to LaTeX/TikZ\n\n"
            f"Input:  {input_path}\n"
            f"Output: {tex_path}\n"
            f"Type:   {'TikZ' if dtype == DiagramType.METHODOLOGY else 'PGFPlots'}\n"
            f"VLM:    {settings.vlm_provider} / {settings.effective_vlm_model}",
            border_style="blue",
        )
    )

    async def _run():
        vlm = ProviderRegistry.create_vlm(settings)
        agent = TikZExporterAgent(vlm)
        return await agent.run(
            image_path=str(input_path),
            source_context=context_text,
            caption=caption,
            diagram_type=dtype,
            venue=settings.venue,
        )

    console.print()
    console.print("  [dim]●[/dim] Generating TikZ source...", end="")
    try:
        tikz_source = asyncio.run(_run())
    except Exception as e:
        console.print(f" [red]✗[/red]\n[red]Error: {e}[/red]")
        raise typer.Exit(1)

    console.print(" [green]✓[/green]")

    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(tikz_source, encoding="utf-8")

    console.print(f"\n[green]Done![/green] LaTeX source saved to: [bold]{tex_path}[/bold]")


@app.command()
def polish(
    input: str = typer.Option(
        ..., "--input", "-i", help="Path to the existing figure image to polish"
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output image path (default: outputs/polish_<timestamp>/final_output.png)",
    ),
    venue: Optional[str] = typer.Option(
        None,
        "--venue",
        help="Target venue style (neurips, icml, acl, ieee, custom)",
    ),
    iterations: int = typer.Option(
        1,
        "--iterations",
        "-n",
        min=1,
        help="Polish rounds: each round suggests improvements and applies them to the result",
    ),
    aspect_ratio: Optional[str] = typer.Option(
        None,
        "--aspect-ratio",
        "-ar",
        help="Target aspect ratio: 1:1, 2:3, 3:2, 3:4, 4:3, 9:16, 16:9, 21:9 "
        "(default: preserve the input figure's ratio)",
    ),
    vlm_provider: Optional[str] = typer.Option(
        None, "--vlm-provider", help="VLM provider (gemini)"
    ),
    vlm_model: Optional[str] = typer.Option(None, "--vlm-model", help="VLM model name"),
    image_provider: Optional[str] = typer.Option(
        None, "--image-provider", help="Image gen provider (must support guided image edits)"
    ),
    image_model: Optional[str] = typer.Option(None, "--image-model", help="Image gen model name"),
    budget: Optional[float] = typer.Option(
        None,
        "--budget",
        help="Budget cap in USD; polishing stops gracefully when exceeded",
    ),
    num_candidates: Optional[int] = typer.Option(
        None,
        "--num-candidates",
        "-k",
        min=1,
        max=8,
        help=(
            "Apply each round's suggestions N times in parallel (1-8); the first "
            "successful candidate is kept as the primary result"
        ),
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="Random seed for reproducible image generation"
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Path to config YAML file"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed agent progress and timing"
    ),
):
    """Polish an existing figure: style-guided suggestions applied as a guided image edit."""
    configure_logging(verbose=verbose)

    if venue and venue.lower() not in ("neurips", "icml", "acl", "ieee", "custom"):
        console.print(
            f"[red]Error: --venue must be neurips, icml, acl, ieee, or custom. Got: {venue}[/red]"
        )
        raise typer.Exit(1)

    input_path = Path(input)
    if not input_path.exists():
        console.print(f"[red]Error: Input image not found: {input}[/red]")
        raise typer.Exit(1)

    from PIL import Image as PILImage
    from PIL import UnidentifiedImageError

    try:
        with PILImage.open(input_path) as probe:
            probe.verify()
    except (UnidentifiedImageError, OSError, ValueError) as e:
        console.print(f"[red]Error: Input is not a readable image: {input} ({e})[/red]")
        raise typer.Exit(1)

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
    if budget is not None:
        overrides["budget_usd"] = budget
    if num_candidates is not None:
        overrides["num_candidates"] = num_candidates
    if seed is not None:
        overrides["seed"] = seed
    if venue:
        overrides["venue"] = venue

    if config:
        settings = Settings.from_yaml(config, **overrides)
    else:
        from dotenv import load_dotenv

        load_dotenv()
        settings = Settings(**overrides)

    import datetime

    from paperbanana.agents.polish import PolishAgent
    from paperbanana.core.cost_tracker import CostTracker
    from paperbanana.core.utils import find_prompt_dir, load_image, save_image
    from paperbanana.guidelines.methodology import load_methodology_guidelines
    from paperbanana.providers.registry import ProviderRegistry

    if output:
        final_path = Path(output)
        run_dir = ensure_dir(final_path.parent)
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = ensure_dir(Path(settings.output_dir) / f"polish_{ts}")
        final_path = run_dir / "final_output.png"

    vlm = ProviderRegistry.create_vlm(settings)
    image_gen = ProviderRegistry.create_image_gen(settings)

    if not PolishAgent.supports_guided_edit(image_gen):
        console.print(
            f"[red]Error: image provider '{getattr(image_gen, 'name', 'unknown')}' does not "
            "support guided image editing required by polish mode. "
            "Use --image-provider google.[/red]"
        )
        raise typer.Exit(1)

    cost_tracker = CostTracker(budget=settings.budget_usd)
    if hasattr(vlm, "cost_tracker"):
        vlm.cost_tracker = cost_tracker
    if hasattr(image_gen, "cost_tracker"):
        image_gen.cost_tracker = cost_tracker
    cost_tracker.set_agent("polish")

    style_guide = load_methodology_guidelines(settings.guidelines_path, venue=settings.venue)

    agent = PolishAgent(
        image_gen,
        vlm,
        prompt_dir=settings.prompt_dir or find_prompt_dir(),
        output_dir=str(run_dir),
        image_quality=settings.image_quality,
    )

    console.print(
        Panel.fit(
            f"[bold]PaperBanana[/bold] - Polish Mode\n\n"
            f"Input:      {input_path}\n"
            f"Venue:      {settings.venue}\n"
            f"Rounds:     {iterations}\n"
            f"Candidates: {settings.num_candidates}\n"
            f"Output:     {final_path}",
            border_style="yellow",
        )
    )

    async def _apply_fanned_out(image, suggestions, round_idx: int) -> str:
        """Fan the apply step out over num_candidates parallel guided edits."""
        candidates_root = ensure_dir(run_dir / "candidates")
        tasks = []
        for k in range(1, settings.num_candidates + 1):
            cand_dir = ensure_dir(candidates_root / f"cand_{k}")
            cand_seed = settings.seed + (k - 1) if settings.seed is not None else None
            tasks.append(
                agent.apply(
                    image,
                    suggestions,
                    output_path=str(cand_dir / f"polished_iter_{round_idx}.png"),
                    iteration=round_idx,
                    aspect_ratio=aspect_ratio,
                    seed=cand_seed,
                )
            )
        results = await asyncio.gather(*tasks, return_exceptions=True)
        primary: Optional[str] = None
        for k, res in enumerate(results, start=1):
            if isinstance(res, BaseException):
                console.print(f"  [yellow]![/yellow] Candidate {k} failed: {res}")
            else:
                console.print(f"  [green]✓[/green] Candidate {k}: {res}")
                if primary is None:
                    primary = res
        if primary is None:
            raise RuntimeError(f"All {settings.num_candidates} polish candidates failed")
        return primary

    async def _run() -> tuple[str, int]:
        current_path = str(input_path)
        rounds_applied = 0
        for round_idx in range(1, iterations + 1):
            image = load_image(current_path)
            console.print(f"\n  [dim]●[/dim] Round {round_idx}/{iterations}: analyzing figure...")
            suggestions = await agent.suggest(image, style_guide, iteration=round_idx)
            if not suggestions:
                console.print(
                    f"  [green]✓[/green] Round {round_idx}: no further suggestions — "
                    "figure already conforms to the style guide"
                )
                break

            console.print(f"\n[bold]Round {round_idx} suggestions:[/bold]")
            for j, suggestion in enumerate(suggestions, start=1):
                console.print(f"  {j}. {suggestion}")

            console.print("  [dim]●[/dim] Applying suggestions (guided edit)...")
            if settings.num_candidates > 1:
                current_path = await _apply_fanned_out(image, suggestions, round_idx)
            else:
                current_path = await agent.apply(
                    image,
                    suggestions,
                    output_path=str(run_dir / f"polished_iter_{round_idx}.png"),
                    iteration=round_idx,
                    aspect_ratio=aspect_ratio,
                    seed=settings.seed,
                )
            rounds_applied += 1

            if cost_tracker.is_over_budget:
                console.print(
                    f"  [yellow]Budget exceeded (${cost_tracker.total_cost:.4f} / "
                    f"${settings.budget_usd}); stopping after round {round_idx}[/yellow]"
                )
                break
        return current_path, rounds_applied

    try:
        result_path, rounds_applied = asyncio.run(_run())
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if rounds_applied == 0:
        console.print(
            "\n[green]Done![/green] No changes applied — "
            "the figure already matches the style guide."
        )
    else:
        final_image = load_image(result_path)
        save_image(final_image, str(final_path))
        console.print(f"\n[green]Done![/green] Polished figure saved to: [bold]{final_path}[/bold]")
        console.print(f"Rounds applied: {rounds_applied}")

    cost_summary = cost_tracker.summary()
    if cost_summary["num_vlm_calls"] or cost_summary["num_image_calls"]:
        console.print(
            f"  Cost: [bold]${cost_summary['total_usd']:.4f}[/bold]"
            f" [dim](VLM: ${cost_summary['vlm_usd']:.4f},"
            f" Image: ${cost_summary['image_usd']:.4f})[/dim]"
        )
        if not cost_summary.get("pricing_complete", True):
            console.print(
                "  [yellow]Note: Some model prices unknown; actual cost may differ[/yellow]"
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


@app.command("evaluate-plot")
def evaluate_plot(
    generated: str = typer.Option(..., "--generated", "-g", help="Path to generated plot image"),
    data: str = typer.Option(
        ...,
        "--data",
        "-d",
        help="Path to source data file used for plotting (CSV or JSON)",
    ),
    intent: str = typer.Option(..., "--intent", help="Communicative intent used for the plot"),
    reference: str = typer.Option(
        ...,
        "--reference",
        "-r",
        help="Path to human reference plot image",
    ),
    vlm_provider: str = typer.Option(
        "gemini", "--vlm-provider", help="VLM provider for evaluation"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed agent progress and timing"
    ),
):
    """Evaluate a generated statistical plot vs human reference (comparative)."""
    configure_logging(verbose=verbose)
    from paperbanana.core.plot_data import load_statistical_plot_payload
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

    data_path = Path(data)
    if not data_path.exists():
        console.print(f"[red]Error: Data file not found: {data}[/red]")
        raise typer.Exit(1)

    try:
        source_context, _ = load_statistical_plot_payload(data_path)
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
            source_context=source_context,
            caption=intent,
            reference_path=str(reference_path),
            task=DiagramType.STATISTICAL_PLOT,
        )

    scores = asyncio.run(_run())

    dims = ["faithfulness", "conciseness", "readability", "aesthetics"]
    dim_lines = []
    for dim in dims:
        result = getattr(scores, dim)
        dim_lines.append(f"{dim.capitalize():14s} {result.winner}")

    console.print(
        Panel.fit(
            "[bold]Evaluation Results (Plot Comparative)[/bold]\n\n"
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
    _check_pdf_dep(input_path)

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
    split: str = typer.Option(
        "reference",
        "--split",
        help=(
            "Entry source: 'reference' (curated pool) or 'test' "
            "(official PaperBananaBench test split — comparable to the paper)"
        ),
    ),
    task: str = typer.Option(
        "diagram",
        "--task",
        help="Task for --split test: 'diagram' (292 entries) or 'plot' (240 entries)",
    ),
    mode: str = typer.Option(
        "full",
        "--mode",
        help=(
            "'full' runs the complete agentic pipeline; 'vanilla' is the no-pipeline "
            "baseline (single direct visualizer call, no planning/critique)"
        ),
    ),
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
    if split not in ("reference", "test"):
        console.print(f"[red]Error: --split must be 'reference' or 'test'. Got: {split}[/red]")
        raise typer.Exit(1)
    if task not in ("diagram", "plot"):
        console.print(f"[red]Error: --task must be 'diagram' or 'plot'. Got: {task}[/red]")
        raise typer.Exit(1)
    if mode not in ("full", "vanilla"):
        console.print(f"[red]Error: --mode must be 'full' or 'vanilla'. Got: {mode}[/red]")
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
        if split == "test":
            entries = runner.load_test_entries(task, category=category, ids=id_list, limit=limit)
        else:
            entries = runner.load_entries(category=category, ids=id_list, limit=limit)
    except (ValueError, RuntimeError) as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if not entries:
        console.print("[red]Error: No entries match the given filters.[/red]")
        raise typer.Exit(1)

    run_mode = "eval-only" if eval_only else f"{mode} (generate + evaluate)"
    split_label = f"test ({task})" if split == "test" else "reference"
    console.print(
        Panel.fit(
            f"[bold]PaperBanana[/bold] — Benchmark\n\n"
            f"Entries: {len(entries)}\n"
            f"Split: {split_label}\n"
            f"Mode: {run_mode}\n"
            f"VLM: {settings.vlm_provider} / {settings.effective_vlm_model}\n"
            f"Image: {settings.image_provider} / {settings.effective_image_model}",
            border_style="magenta",
        )
    )
    console.print()

    bench_output_dir = Path(output_dir) if output_dir else None

    async def _run():
        return await runner.run(
            entries,
            output_dir=bench_output_dir,
            eval_only_dir=eval_only,
            mode=mode,
            split=split,
            task=task if split == "test" else None,
        )

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

    # Per-difficulty breakdown (official plot test split)
    diff_breakdown = summary.get("difficulty_breakdown", {})
    if diff_breakdown:
        console.print("\n[bold]Per-difficulty breakdown:[/bold]")
        for diff, stats in diff_breakdown.items():
            console.print(
                f"  {diff:30s} n={stats['count']:3d}  "
                f"win_rate={stats['model_win_rate']:5.1f}%  "
                f"mean={stats['mean_score']:.1f}"
            )

    if report.run_dir:
        report_path = Path(report.run_dir)
    else:
        report_path = Path(settings.output_dir) / report.created_at.replace(":", "")
    console.print(f"\nReport: [bold]{report_path / 'benchmark_report.json'}[/bold]")


# ── Runs subcommands ──────────────────────────────────────────────


@runs_app.command("list")
def runs_list(
    output_dir: str = typer.Option(
        "outputs",
        "--output-dir",
        "-o",
        help="Output directory containing run_* / batch_* folders",
    ),
    kind: str = typer.Option(
        "run",
        "--kind",
        help="What to list: run, batch, or all",
    ),
    limit: int = typer.Option(50, "--limit", help="Max rows to show (per kind)"),
    oldest_first: bool = typer.Option(False, "--oldest-first", help="Sort oldest→newest"),
    plain: bool = typer.Option(
        False,
        "--plain",
        help="Plain text output (newline-delimited, easier to parse in scripts/tests).",
    ),
) -> None:
    """List recent runs and/or batch outputs under an output directory."""
    from paperbanana.studio.runs import list_batch_ids, list_run_ids

    out = Path(output_dir)
    kind_norm = kind.strip().lower()
    if kind_norm not in {"run", "batch", "all"}:
        console.print("[red]Error:[/red] --kind must be one of: run, batch, all")
        raise typer.Exit(1)

    if not out.exists():
        console.print(f"[red]Error:[/red] Output directory not found: {output_dir}")
        raise typer.Exit(1)

    if kind_norm in {"run", "all"}:
        runs = list_run_ids(str(out))
        if not oldest_first:
            runs = list(reversed(runs))
        if limit > 0:
            runs = runs[:limit]

        if plain:
            if not runs:
                typer.echo("No runs found.")
            for run_id in runs:
                row = _summarize_run_row(out, run_id)
                typer.echo(
                    "\t".join(
                        [
                            row["id"],
                            row["modified"],
                            row["type"],
                            row["iters"],
                            row["status"],
                            row["caption"],
                        ]
                    )
                )
        else:
            table = Table(title="Runs", show_lines=False)
            table.add_column("run_id", style="bold")
            table.add_column("modified", style="dim")
            table.add_column("type")
            table.add_column("iters", justify="right")
            table.add_column("status")
            table.add_column("caption")

            for run_id in runs:
                row = _summarize_run_row(out, run_id)
                table.add_row(
                    row["id"],
                    row["modified"],
                    row["type"],
                    row["iters"],
                    row["status"],
                    row["caption"],
                )

            if runs:
                console.print(table)
            else:
                console.print("No runs found.")

    if kind_norm in {"batch", "all"}:
        batches = list_batch_ids(str(out))
        if not oldest_first:
            batches = list(reversed(batches))
        if limit > 0:
            batches = batches[:limit]

        if plain:
            if not batches:
                typer.echo("No batches found.")
            for batch_id in batches:
                row = _summarize_batch_row(out, batch_id)
                typer.echo(
                    "\t".join(
                        [
                            row["id"],
                            row["modified"],
                            row["kind"],
                            row["items"],
                            row["status"],
                        ]
                    )
                )
        else:
            table = Table(title="Batches", show_lines=False)
            table.add_column("batch_id", style="bold")
            table.add_column("modified", style="dim")
            table.add_column("kind")
            table.add_column("items", justify="right")
            table.add_column("status")

            for batch_id in batches:
                row = _summarize_batch_row(out, batch_id)
                table.add_row(row["id"], row["modified"], row["kind"], row["items"], row["status"])

            if batches:
                console.print(table)
            else:
                console.print("No batches found.")


@runs_app.command("show")
def runs_show(
    id: str = typer.Argument(..., help="Run or batch id (e.g. run_..., batch_...)"),
    output_dir: str = typer.Option(
        "outputs",
        "--output-dir",
        "-o",
        help="Output directory containing run_* / batch_* folders",
    ),
    kind: Optional[str] = typer.Option(
        None,
        "--kind",
        help="Optional override: run or batch. If omitted, inferred from the id prefix.",
    ),
    plain: bool = typer.Option(
        False,
        "--plain",
        help="Plain text output (newline-delimited, easier to parse in scripts/tests).",
    ),
) -> None:
    """Show details for a single run or batch output."""
    from paperbanana.studio.runs import load_batch_summary, load_run_summary

    out = Path(output_dir)
    if not out.exists():
        console.print(f"[red]Error:[/red] Output directory not found: {output_dir}")
        raise typer.Exit(1)

    inferred = "batch" if id.startswith("batch_") else "run"
    kind_norm = kind.strip().lower() if kind else inferred
    if kind_norm not in {"run", "batch"}:
        console.print("[red]Error:[/red] --kind must be one of: run, batch")
        raise typer.Exit(1)

    if kind_norm == "run":
        summary = load_run_summary(str(out), id)
        if not summary.get("exists"):
            console.print(f"[red]Error:[/red] {summary.get('error')}")
            raise typer.Exit(1)

        lines = [
            f"[bold]Run[/bold] {summary.get('run_id')}",
            f"Directory: {summary.get('run_dir')}",
            f"Final image: {summary.get('final_image') or '(none)'}",
            f"Metadata: {summary.get('metadata_path') or '(none)'}",
        ]
        if summary.get("iteration_images"):
            lines.append(f"Iterations: {len(summary['iteration_images'])}")

        if plain:
            plain_lines = [
                f"run_id\t{summary.get('run_id')}",
                f"run_dir\t{summary.get('run_dir')}",
                f"final_image\t{summary.get('final_image') or '(none)'}",
                f"metadata_path\t{summary.get('metadata_path') or '(none)'}",
            ]
            if summary.get("iteration_images"):
                plain_lines.append(f"iterations\t{len(summary['iteration_images'])}")
            typer.echo("\n".join(plain_lines))
        else:
            console.print(Panel.fit("\n".join(lines), border_style="green"))
        if summary.get("run_input_preview"):
            console.print("\n[bold]run_input.json (preview)[/bold]")
            console.print(summary["run_input_preview"])
        if summary.get("metadata_preview"):
            console.print("\n[bold]metadata.json (preview)[/bold]")
            console.print(summary["metadata_preview"])
        return

    summary = load_batch_summary(str(out), id)
    if not summary.get("exists"):
        console.print(f"[red]Error:[/red] {summary.get('error')}")
        raise typer.Exit(1)

    status_counts = summary.get("status_counts") or {}
    status_line = (
        ", ".join([f"{k}:{v}" for k, v in sorted(status_counts.items())]) if status_counts else "?"
    )
    lines = [
        f"[bold]Batch[/bold] {summary.get('batch_id')}",
        f"Directory: {summary.get('batch_dir')}",
        f"Status: {status_line}",
    ]
    if plain:
        plain_lines = [
            f"batch_id\t{summary.get('batch_id')}",
            f"batch_dir\t{summary.get('batch_dir')}",
            f"status\t{status_line}",
        ]
        typer.echo("\n".join(plain_lines))
    else:
        console.print(Panel.fit("\n".join(lines), border_style="blue"))
    if summary.get("report_preview"):
        console.print("\n[bold]batch_report.json (preview)[/bold]")
        console.print(summary["report_preview"])


@runs_app.command("delete")
def runs_delete(
    id: str = typer.Argument(..., help="Run or batch id to delete"),
    output_dir: str = typer.Option(
        "outputs",
        "--output-dir",
        "-o",
        help="Output directory containing run_* / batch_* folders",
    ),
    kind: Optional[str] = typer.Option(
        None,
        "--kind",
        help="Optional override: run or batch. If omitted, inferred from the id prefix.",
    ),
    yes: bool = typer.Option(False, "--yes", help="Confirm deletion (required)"),
) -> None:
    """Delete a run_* or batch_* folder under the output directory."""
    import shutil

    if not yes:
        console.print("[red]Error:[/red] Refusing to delete without --yes")
        raise typer.Exit(1)

    out = Path(output_dir)
    if not out.exists():
        console.print(f"[red]Error:[/red] Output directory not found: {output_dir}")
        raise typer.Exit(1)

    inferred = "batch" if id.startswith("batch_") else "run"
    kind_norm = kind.strip().lower() if kind else inferred
    if kind_norm not in {"run", "batch"}:
        console.print("[red]Error:[/red] --kind must be one of: run, batch")
        raise typer.Exit(1)

    if kind_norm == "run" and not id.startswith("run_"):
        console.print("[red]Error:[/red] Run ids must start with run_ (or pass --kind batch)")
        raise typer.Exit(1)
    if kind_norm == "batch" and not id.startswith("batch_"):
        console.print("[red]Error:[/red] Batch ids must start with batch_ (or pass --kind run)")
        raise typer.Exit(1)

    target = (out / id).resolve()
    root = out.resolve()
    if root not in target.parents:
        console.print("[red]Error:[/red] Refusing to delete outside output directory")
        raise typer.Exit(1)
    if not target.is_dir():
        console.print("[red]Error:[/red] Target directory not found")
        raise typer.Exit(1)

    shutil.rmtree(target)
    console.print(f"[green]Deleted[/green] {target}")


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
    """Download the PaperBananaBench reference set (~254 MB).

    Fetches the project-hosted GitHub release mirror and verifies its
    SHA256 checksum before caching.
    """
    from paperbanana.data.manager import DatasetManager

    dm = DatasetManager()
    if dm.is_downloaded() and not force:
        info = dm.get_info() or {}
        meta = info.get("dataset_meta", {})
        ds_version = meta.get("full_bench", {}).get("version", info.get("version", "unknown"))
        console.print(
            Panel.fit(
                f"[bold]Reference Set — Already Cached[/bold]\n\n"
                f"Location: {dm.reference_dir}\n"
                f"Examples: {dm.get_example_count()}\n"
                f"Version: {ds_version}",
                border_style="green",
            )
        )
        console.print("\nUse [bold]--force[/bold] to re-download.")
        return

    console.print("[bold]PaperBanana[/bold] — Downloading PaperBananaBench\n")
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

    datasets = dataset_info.get("datasets", [])
    meta = dataset_info.get("dataset_meta", {})
    lines = [
        "[bold]Cached Reference Set[/bold]\n",
        f"Location: {dm.reference_dir}",
        f"Examples: {dataset_info.get('example_count', '?')}",
        f"Datasets: {', '.join(datasets) if datasets else 'unknown'}",
    ]
    for ds in datasets:
        ds_meta = meta.get(ds, {})
        lines.append(f"  {ds}: v{ds_meta.get('version', '?')} — {ds_meta.get('source', '?')}")

    console.print(Panel.fit("\n".join(lines), border_style="blue"))


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


# ── References subcommands ────────────────────────────────────────


@references_app.command(name="list")
def references_list(
    category: Optional[str] = typer.Option(
        None,
        "--category",
        "-c",
        help="Filter by category (e.g. nlp_language, vision_perception).",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
):
    """List all available reference examples."""
    from paperbanana.reference.store import ReferenceStore

    settings = Settings()
    store = ReferenceStore.from_settings(settings)
    examples = store.get_by_category(category) if category else store.get_all()

    if not examples:
        if category:
            console.print(f"No references found for category [bold]{category}[/bold].")
        else:
            console.print("No reference examples found.")
        raise typer.Exit(0)

    if json_output:
        rows = [
            {
                "id": e.id,
                "category": e.category or "",
                "caption": e.caption[:120],
                "aspect_ratio": e.aspect_ratio,
            }
            for e in examples
        ]
        console.print_json(json_mod.dumps(rows))
        return

    table = Table(title=f"Reference Examples ({len(examples)})")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Category", style="green")
    table.add_column("Caption", max_width=60)
    table.add_column("AR", justify="right")
    for e in examples:
        caption_short = (e.caption[:57] + "...") if len(e.caption) > 60 else e.caption
        table.add_row(
            e.id,
            e.category or "—",
            caption_short,
            f"{e.aspect_ratio:.2f}" if e.aspect_ratio else "—",
        )
    console.print(table)


@references_app.command(name="show")
def references_show(
    example_id: str = typer.Argument(help="Reference example ID to display."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
):
    """Show details of a specific reference example."""
    from paperbanana.reference.store import ReferenceStore

    settings = Settings()
    store = ReferenceStore.from_settings(settings)
    example = store.get_by_id(example_id)

    if example is None:
        console.print(f"[red]Error:[/red] No reference found with ID [bold]{example_id}[/bold].")
        raise typer.Exit(1)

    if json_output:
        console.print_json(json_mod.dumps(example.model_dump(), default=str))
        return

    lines = [
        f"[bold]ID:[/bold]           {example.id}",
        f"[bold]Category:[/bold]     {example.category or '—'}",
        f"[bold]Aspect Ratio:[/bold] {example.aspect_ratio or '—'}",
        f"[bold]Image Path:[/bold]   {example.image_path}",
        f"\n[bold]Caption:[/bold]\n{example.caption}",
    ]
    if example.source_context:
        ctx = example.source_context
        if len(ctx) > 500:
            ctx = ctx[:500] + "…"
        lines.append(f"\n[bold]Source Context (excerpt):[/bold]\n[dim]{ctx}[/dim]")

    console.print(Panel("\n".join(lines), title=f"Reference — {example.id}", border_style="blue"))


@references_app.command(name="categories")
def references_categories(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
):
    """List reference categories and example counts."""
    from paperbanana.reference.store import ReferenceStore

    settings = Settings()
    store = ReferenceStore.from_settings(settings)
    examples = store.get_all()

    if not examples:
        console.print("No reference examples found.")
        raise typer.Exit(0)

    counts: dict[str, int] = {}
    for e in examples:
        cat = e.category or "uncategorized"
        counts[cat] = counts.get(cat, 0) + 1

    if json_output:
        console.print_json(json_mod.dumps(counts))
        return

    table = Table(title="Reference Categories")
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right", style="green")
    for cat in sorted(counts):
        table.add_row(cat, str(counts[cat]))
    table.add_section()
    table.add_row("[bold]Total[/bold]", f"[bold]{sum(counts.values())}[/bold]")
    console.print(table)


# ── Guidelines subcommands ────────────────────────────────────────


@guidelines_app.command(name="synthesize")
def guidelines_synthesize(
    guide_type: str = typer.Option(
        "methodology",
        "--type",
        help="Guide type to synthesize: methodology or plot",
    ),
    venue: Optional[str] = typer.Option(
        None,
        "--venue",
        help=(
            "Write the guide into the guidelines layout: "
            "data/guidelines/<venue>/{methodology|plot}_style_guide.md"
        ),
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Explicit output path for the synthesized guide (e.g. ./style_guide.md)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite the target guide file if it already exists",
    ),
    reference_set: Optional[str] = typer.Option(
        None,
        "--reference-set",
        help="Path to a reference set directory (defaults to the configured/cached set)",
    ),
    sample_size: int = typer.Option(
        50,
        "--sample-size",
        help="Maximum number of reference figures to analyze",
    ),
    batch_size: int = typer.Option(
        20,
        "--batch-size",
        help="Number of figures per VLM analysis call",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="Sampling seed for a reproducible corpus subset",
    ),
    vlm_provider: Optional[str] = typer.Option(None, "--vlm-provider", help="VLM provider"),
    vlm_model: Optional[str] = typer.Option(None, "--vlm-model", help="VLM model name"),
    budget: Optional[float] = typer.Option(
        None,
        "--budget",
        help="Budget cap in USD; synthesis aborts gracefully when exceeded",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed progress"),
    config: Optional[str] = typer.Option(None, "--config", help="Path to config YAML file"),
):
    """Synthesize a corpus-grounded style guide from reference figures.

    Samples up to --sample-size reference figures, VLM-analyzes them in
    batches of --batch-size (palettes, layout, line/shape semantics,
    typography), then merges the analyses into one markdown style guide
    that preserves multiple accepted options and domain-conditional rules.
    """
    from paperbanana.guidelines.synthesis import GUIDE_TYPES, synthesize_style_guide

    if guide_type not in GUIDE_TYPES:
        console.print(
            f"[red]Error: --type must be 'methodology' or 'plot'. Got: {guide_type}[/red]"
        )
        raise typer.Exit(1)
    if venue is None and output is None:
        console.print(
            "[red]Error: Pass --venue NAME (write into the guidelines layout) "
            "or --output PATH (e.g. --output ./style_guide.md).[/red]\n"
            "Built-in guides are never overwritten implicitly."
        )
        raise typer.Exit(1)
    if venue is not None and output is not None:
        console.print("[red]Error: --venue and --output are mutually exclusive.[/red]")
        raise typer.Exit(1)
    if venue is not None and venue.lower() not in ("neurips", "icml", "acl", "ieee"):
        console.print(
            f"[red]Error: --venue must be neurips, icml, acl, or ieee. Got: {venue}[/red]\n"
            "For a custom guide location, use --output PATH and point "
            "GUIDELINES_PATH (or reference.guidelines_path) at its directory."
        )
        raise typer.Exit(1)

    configure_logging(verbose=verbose)

    overrides: dict = {}
    if vlm_provider:
        overrides["vlm_provider"] = vlm_provider
    if vlm_model:
        overrides["vlm_model"] = vlm_model
    if reference_set:
        overrides["reference_set_path"] = reference_set

    if config:
        settings = Settings.from_yaml(config, **overrides)
    else:
        from dotenv import load_dotenv

        load_dotenv()
        settings = Settings(**overrides)

    if venue is not None:
        target = Path(settings.guidelines_path) / venue.lower() / f"{guide_type}_style_guide.md"
    else:
        target = Path(output)  # type: ignore[arg-type]
    if target.exists() and not force:
        console.print(
            f"[red]Error: {target} already exists.[/red] "
            "Pass --force to overwrite it, or choose another --output path."
        )
        raise typer.Exit(1)

    from paperbanana.core.cost_tracker import CostTracker
    from paperbanana.providers.registry import ProviderRegistry
    from paperbanana.reference.store import ReferenceStore

    store = ReferenceStore.from_settings(settings)
    examples = store.get_all()
    if not examples:
        console.print(
            "[red]Error: No reference examples found.[/red]\n"
            "Download the reference set with: [bold]paperbanana data download[/bold]"
        )
        raise typer.Exit(1)

    try:
        vlm = ProviderRegistry.create_vlm(settings)
    except (ValueError, ImportError) as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    cost_tracker: Optional[CostTracker] = None
    if budget is not None:
        cost_tracker = CostTracker(budget=budget)
        if hasattr(vlm, "cost_tracker"):
            vlm.cost_tracker = cost_tracker
            cost_tracker.set_agent("guidelines_synthesis")

    console.print(
        Panel.fit(
            f"[bold]PaperBanana[/bold] - Synthesizing Style Guide\n\n"
            f"Type: {guide_type}\n"
            f"Corpus: {store.count} examples\n"
            f"Sample: up to {sample_size} figures (batches of {batch_size})\n"
            f"VLM: {settings.vlm_provider} / {settings.effective_vlm_model}\n"
            f"Output: {target}",
            border_style="green",
        )
    )

    async def _run() -> str:
        return await synthesize_style_guide(
            vlm,
            examples,
            guide_type=guide_type,
            batch_size=batch_size,
            sample_size=sample_size,
            seed=seed,
            prompt_dir=settings.prompt_dir,
            progress_callback=lambda msg: console.print(f"  [dim]●[/dim] {msg}"),
        )

    try:
        guide = asyncio.run(_run())
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(1)

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(guide, encoding="utf-8")
    except OSError as e:
        console.print(
            f"\n[red]Error: Cannot write to {target} ({e}).[/red]\n"
            "Re-run with --output pointing to a writable path."
        )
        raise typer.Exit(1)

    console.print(f"\n[green]Done![/green] Style guide saved to: [bold]{target}[/bold]")
    if venue is not None:
        console.print(f"  Use it with: [bold]--venue {venue.lower()}[/bold]")
    if cost_tracker is not None:
        console.print(f"  Cost: [bold]${cost_tracker.total_cost:.4f}[/bold]")


# ── venues subcommands ─────────────────────────────────────────────


_VENUE_YAML_TEMPLATE = """\
# PaperBanana venue style pack configuration.
# All fields are optional — delete or comment out anything you don't need.

# Human-readable name shown by `paperbanana venues list`.
display_name: "{display_name}"

# Default --aspect-ratio for runs with this venue (used when the CLI flag
# is not passed). One of: 1:1, 2:3, 3:2, 3:4, 4:3, 9:16, 16:9, 21:9.
# aspect_ratio: "16:9"

# Preferred font families, appended as a note to the style guides.
# fonts:
#   - "Helvetica"
#   - "Arial"
"""


@venues_app.command(name="list")
def venues_list(
    venue_dir: Optional[str] = typer.Option(
        None,
        "--venue-dir",
        help=(
            "User venue style pack directory "
            "(default: $PAPERBANANA_VENUE_DIR or ~/.config/paperbanana/venues)"
        ),
    ),
):
    """List available venue style packs (built-in and user)."""
    from paperbanana.guidelines.venues import (
        list_venues,
        load_venue_config,
        resolve_user_venue_dir,
    )

    venues = list_venues(extra_dir=venue_dir)

    table = Table(title="Venue Style Packs")
    table.add_column("Name", style="bold")
    table.add_column("Source")
    table.add_column("Display Name")
    table.add_column("Aspect Ratio")
    table.add_column("Path", overflow="fold")

    for info in venues.values():
        try:
            config = load_venue_config(info.dir)
        except (ValueError, OSError):
            config = None
        table.add_row(
            info.name,
            info.source,
            (config.display_name if config and config.display_name else "—"),
            (config.aspect_ratio if config and config.aspect_ratio else "—"),
            str(info.dir),
        )

    console.print(table)
    console.print(
        f"\nUser venue directory: [bold]{resolve_user_venue_dir(venue_dir)}[/bold]\n"
        "Scaffold a new pack with: [bold]paperbanana venues init <name>[/bold]"
    )


@venues_app.command(name="init")
def venues_init(
    name: str = typer.Argument(..., help="Name for the new venue pack (used with --venue)"),
    venue_dir: Optional[str] = typer.Option(
        None,
        "--venue-dir",
        help=(
            "User venue style pack directory "
            "(default: $PAPERBANANA_VENUE_DIR or ~/.config/paperbanana/venues)"
        ),
    ),
):
    """Scaffold a user venue style pack from the NeurIPS templates."""
    from paperbanana.guidelines.methodology import load_methodology_guidelines
    from paperbanana.guidelines.plots import load_plot_guidelines
    from paperbanana.guidelines.venues import (
        DEFAULT_BUILTIN_GUIDELINES_DIR,
        METHODOLOGY_GUIDE_FILENAME,
        PLOT_GUIDE_FILENAME,
        RESERVED_VENUE_NAMES,
        VENUE_CONFIG_FILENAME,
        list_venues,
        resolve_user_venue_dir,
    )

    venue_name = name.strip().lower()
    if not venue_name or not all(c.isalnum() or c in ("-", "_") for c in venue_name):
        console.print(
            f"[red]Error: Invalid venue name '{name}'. "
            "Use letters, digits, hyphens, and underscores only.[/red]"
        )
        raise typer.Exit(1)
    if venue_name in RESERVED_VENUE_NAMES:
        console.print(f"[red]Error: Venue name '{venue_name}' is reserved.[/red]")
        raise typer.Exit(1)

    existing = list_venues(extra_dir=venue_dir)
    if venue_name in existing:
        info = existing[venue_name]
        console.print(
            f"[red]Error: Venue '{venue_name}' already exists "
            f"({info.source}: {info.dir}).[/red]\n"
            "Built-in venues cannot be shadowed; pick a different name."
        )
        raise typer.Exit(1)

    target = resolve_user_venue_dir(venue_dir) / venue_name
    try:
        target.mkdir(parents=True, exist_ok=False)
        # Seed both guides from the NeurIPS templates as a starting point.
        methodology = load_methodology_guidelines(DEFAULT_BUILTIN_GUIDELINES_DIR, venue="neurips")
        plot = load_plot_guidelines(DEFAULT_BUILTIN_GUIDELINES_DIR, venue="neurips")
        (target / METHODOLOGY_GUIDE_FILENAME).write_text(methodology, encoding="utf-8")
        (target / PLOT_GUIDE_FILENAME).write_text(plot, encoding="utf-8")
        (target / VENUE_CONFIG_FILENAME).write_text(
            _VENUE_YAML_TEMPLATE.format(display_name=venue_name.upper()),
            encoding="utf-8",
        )
    except OSError as e:
        console.print(f"[red]Error: Cannot create venue pack at {target} ({e}).[/red]")
        raise typer.Exit(1)

    console.print(
        f"[green]Created venue pack:[/green] [bold]{target}[/bold]\n\n"
        f"  {METHODOLOGY_GUIDE_FILENAME}  — methodology diagram style guide\n"
        f"  {PLOT_GUIDE_FILENAME}         — statistical plot style guide\n"
        f"  {VENUE_CONFIG_FILENAME}                  — optional venue metadata\n\n"
        "Next steps:\n"
        f"  1. Edit the two style guides (or generate them from example figures "
        f"with 'paperbanana guidelines synthesize --output {target}/...').\n"
        f"  2. Adjust {VENUE_CONFIG_FILENAME} (display name, aspect ratio, fonts).\n"
        f"  3. Use it: [bold]paperbanana generate --venue {venue_name} ...[/bold]"
    )


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
    _require_studio_dep()

    from paperbanana.studio.app import launch_studio as launch_studio_ui

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


@app.command("validate-manifest")
def validate_manifest(
    manifest: str = typer.Option(
        ...,
        "--manifest",
        "-m",
        help="Path to the YAML or JSON manifest file to validate.",
    ),
    manifest_type: str = typer.Option(
        "auto",
        "--type",
        "-t",
        help="Manifest type: 'batch', 'plot', or 'auto' (detect from content).",
    ),
) -> None:
    """Validate a batch or plot-batch manifest without running any generation."""
    from paperbanana.core.batch import validate_manifest as _validate

    if manifest_type not in ("batch", "plot", "auto"):
        console.print(
            f"[red]Error: --type must be 'batch', 'plot', or 'auto'. Got: {manifest_type}[/red]"
        )
        raise typer.Exit(1)

    manifest_path = Path(manifest)
    if not manifest_path.exists():
        console.print(f"[red]Error: Manifest not found: {manifest}[/red]")
        raise typer.Exit(1)

    errors = _validate(manifest_path, manifest_type=manifest_type)  # type: ignore[arg-type]

    if not errors:
        console.print(f"[green]✓ Manifest is valid: {manifest}[/green]")
        raise typer.Exit(0)

    console.print(f"[red]✗ Manifest validation failed: {manifest}[/red]")
    console.print(f"[red]  {len(errors)} violation(s) found:[/red]")
    for i, err in enumerate(errors, 1):
        console.print(f"  [red]{i}. {err}[/red]")
    raise typer.Exit(1)


@app.command("analytics")
def analytics(
    path: str = typer.Option(
        "outputs",
        "--path",
        "-p",
        help=(
            "Root directory to scan for metadata.json, batch_report.json, and figure_package.json."
        ),
    ),
    format: str = typer.Option(
        "markdown",
        "--format",
        "-f",
        help="Output format: markdown or json.",
    ),
) -> None:
    """Analyze historical run artifacts and report aggregate cost/latency/success KPIs."""
    output_format = format.lower().strip()
    if output_format not in {"markdown", "json"}:
        console.print(f"[red]Error: --format must be 'markdown' or 'json'. Got: {format}[/red]")
        raise typer.Exit(1)
    try:
        records = load_analytics_records(Path(path))
    except FileNotFoundError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1)

    summary = summarize_records(records)
    if output_format == "json":
        from paperbanana.analytics.reporting import summary_to_dict

        console.print_json(json_mod.dumps(summary_to_dict(summary), indent=2))
    else:
        console.print(render_markdown_summary(summary))


@app.command("show-config")
def show_config(
    json_output: bool = typer.Option(False, "--json", help="Emit resolved config as JSON"),
    config: Optional[str] = typer.Option(None, "--config", help="Path to config YAML file"),
) -> None:
    """Print the fully resolved settings (env + config file) without running generation."""
    from dotenv import load_dotenv

    load_dotenv()

    if config:
        settings = Settings.from_yaml(config)
    else:
        settings = Settings()

    # Fields containing sensitive secrets that should be masked
    _secret_fields = {
        "google_api_key",
        "openrouter_api_key",
        "openai_api_key",
        "anthropic_api_key",
    }

    data = settings.model_dump()
    # Mask sensitive values
    for key in _secret_fields:
        val = data.get(key)
        if val:
            data[key] = val[:4] + "****" + val[-4:] if len(val) > 8 else "****"

    # Add effective (resolved) models for clarity
    data["_effective_vlm_model"] = settings.effective_vlm_model
    data["_effective_image_model"] = settings.effective_image_model

    if json_output:
        console.print_json(json_mod.dumps(data, indent=2, default=str))
    else:
        table = Table(title="Resolved PaperBanana Settings", show_lines=True)
        table.add_column("Setting", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")

        for key, value in data.items():
            display = str(value) if value is not None else "[dim]None[/dim]"
            table.add_row(key, display)

        console.print(table)


@app.command()
def doctor(
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON (for CI pipelines)."
    ),
) -> None:
    """Check system health: optional dependencies, API keys, and reference data."""
    from paperbanana.doctor import run_doctor

    raise typer.Exit(run_doctor(output_json=json_output))


@venues_app.command(name="specs")
def venues_specs(
    venue_spec_dir: Optional[str] = typer.Option(
        None,
        "--venue-spec-dir",
        help=(
            "User venue poster-spec directory "
            "(default: $PAPERBANANA_VENUE_SPEC_DIR or ~/.config/paperbanana/venue_specs)"
        ),
    ),
):
    """List the venue poster regulation bank (built-in and user specs)."""
    from paperbanana.poster.venue_spec import (
        list_venue_specs,
        load_venue_spec,
        resolve_user_spec_dir,
    )

    specs = list_venue_specs(extra_dir=venue_spec_dir)
    table = Table(title="Venue Poster Specs")
    table.add_column("Venue", style="bold")
    table.add_column("Years")
    table.add_column("Latest Spec")
    table.add_column("Dimensions")
    table.add_column("Orientation")
    for name, years in specs.items():
        try:
            spec = load_venue_spec(name, extra_dir=venue_spec_dir)
            dims = spec.dimensions
            gen_w, gen_h = dims.generation_size_mm()
            dim_text = (
                f"{dims.mode}: {dims.width_mm:.0f}x{dims.height_mm:.0f}mm "
                f"(generate {gen_w:.0f}x{gen_h:.0f}mm)"
            )
            table.add_row(
                name,
                ", ".join(str(y) for y in years),
                spec.display_name,
                dim_text,
                dims.orientation,
            )
        except ValueError as e:
            table.add_row(name, ", ".join(str(y) for y in years), f"[red]{e}[/red]", "", "")
    console.print(table)
    console.print(
        f"\nUser spec directory: [bold]{resolve_user_spec_dir(venue_spec_dir)}[/bold]\n"
        "Add a venue by dropping [bold]<venue>/<year>.yaml[/bold] there "
        "(see data/venue_specs/ for the schema)."
    )


@app.command()
def poster(
    paper: str = typer.Option(..., "--paper", "-p", help="Path to the paper PDF"),
    venue: Optional[str] = typer.Option(
        None,
        "--venue",
        help="Venue with a poster spec (see 'paperbanana venues specs'); default from config",
    ),
    year: Optional[int] = typer.Option(
        None, "--year", help="Venue spec year (default: latest available)"
    ),
    qr_url: Optional[str] = typer.Option(
        None, "--qr-url", help="URL rendered as a scannable QR code on the poster"
    ),
    figures: str = typer.Option(
        "generated",
        "--figures",
        help="Figure policy: generated (model draws), real (embed paper figures), "
        "auto (decide per figure). real/auto land in the next milestone.",
    ),
    repair_rounds: Optional[int] = typer.Option(
        None,
        "--repair-rounds",
        "-n",
        help="Max faithfulness repair regenerations after the audit (default: 1)",
    ),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", help="Run output directory"),
    budget: Optional[float] = typer.Option(
        None, "--budget", help="Budget cap in USD; pipeline aborts when exceeded"
    ),
    vlm_provider: Optional[str] = typer.Option(None, "--vlm-provider", help="VLM provider"),
    vlm_model: Optional[str] = typer.Option(None, "--vlm-model", help="VLM model name"),
    image_provider: Optional[str] = typer.Option(
        None, "--image-provider", help="Image gen provider (needs guided-edit support)"
    ),
    image_model: Optional[str] = typer.Option(None, "--image-model", help="Image gen model"),
    save_prompts: Optional[bool] = typer.Option(
        None, "--save-prompts/--no-save-prompts", help="Save formatted prompts in the run dir"
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Path to config YAML file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose progress output"),
) -> None:
    """Generate a venue-compliant conference poster (pptx + press-ready PDF) from a paper PDF.

    Figures are never just copied: each one is judged against print physics
    and either reused, re-authored for poster legibility (with a strict
    data-faithfulness gate), or replaced by a newly generated diagram.
    """
    paper_path = Path(paper).expanduser()
    if not paper_path.is_file():
        console.print(f"[red]Error: paper PDF not found: {paper_path}[/red]")
        raise typer.Exit(1)
    if figures not in ("generated", "real", "auto"):
        console.print(f"[red]Error: --figures must be generated|real|auto, got '{figures}'[/red]")
        raise typer.Exit(1)

    overrides: dict = {}
    if output_dir:
        overrides["output_dir"] = output_dir
    if budget is not None:
        overrides["budget_usd"] = budget
    if vlm_provider:
        overrides["vlm_provider"] = vlm_provider
    if vlm_model:
        overrides["vlm_model"] = vlm_model
    if image_provider:
        overrides["image_provider"] = image_provider
    if image_model:
        overrides["image_model"] = image_model
    if save_prompts is not None:
        overrides["save_prompts"] = save_prompts

    if config:
        settings = Settings.from_yaml(config, **overrides)
    else:
        from dotenv import load_dotenv

        load_dotenv()
        settings = Settings(**overrides)

    from paperbanana.poster.generative import GenerativePosterPipeline
    from paperbanana.poster.venue_spec import UnknownVenueSpecError

    def progress(event: str, payload: dict) -> None:
        labels = {
            "ingest_complete": "Paper text extracted",
            "grounding_complete": "Verified facts grounded",
            "generating": "Generating poster",
            "audit_complete": "Faithfulness audit done",
            "poster_complete": "Poster complete",
        }
        label = labels.get(event, event)
        detail = ""
        if verbose and payload:
            detail = "  [dim]" + ", ".join(f"{k}={v}" for k, v in payload.items()) + "[/dim]"
        elif event == "audit_complete":
            n = payload.get("findings", 0)
            detail = f"  [yellow]{n} discrepancies[/yellow]" if n else "  [green]clean[/green]"
        console.print(f"  [dim]●[/dim] {label}{detail}")

    try:
        pipeline = GenerativePosterPipeline(settings=settings, progress_callback=progress)
        output = asyncio.run(
            pipeline.generate(
                paper_path,
                venue=venue,
                year=year,
                qr_url=qr_url,
                figures=figures,
                repair_rounds=repair_rounds if repair_rounds is not None else 1,
            )
        )
    except UnknownVenueSpecError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e
    except NotImplementedError as e:
        console.print(f"[yellow]{e}[/yellow]")
        raise typer.Exit(1) from e

    console.print()
    console.print("[bold green]Poster generated[/bold green]")
    console.print(f"  PNG:                {output.png_path}")
    console.print(f"  PDF (print-ready):  {output.pdf_path}")
    console.print(
        f"  size:               {output.size_mm[0]:.0f}x{output.size_mm[1]:.0f}mm "
        f"({output.venue} {output.venue_spec_year})"
    )
    if output.audit_findings:
        console.print(
            f"  [yellow]audit: {len(output.audit_findings)} unresolved finding(s)[/yellow] "
            f"after {output.repair_rounds} repair round(s) — see {output.run_dir}/audit.json"
        )
    else:
        console.print("  [green]audit: clean (no contradictions vs the paper)[/green]")
    if output.cost_usd:
        console.print(f"  cost:               ${output.cost_usd:.2f}")
    if output.compliance.passed:
        console.print("  [green]compliance: PASSED[/green]")
        for check in output.compliance.warnings:
            console.print(f"  [yellow]warn: {check.id} — {check.detail}[/yellow]")
    else:
        console.print("  [red]compliance: FAILED[/red]")
        for check in output.compliance.failures:
            console.print(
                f"  [red]fail: {check.id} — {check.value} (required {check.threshold})[/red]"
            )
        raise typer.Exit(2)


@app.command("evaluate-poster")
def evaluate_poster_cmd(
    run_dir: str = typer.Option(
        ..., "--run-dir", help="Poster run directory (outputs/poster_*) to evaluate"
    ),
    context: Optional[str] = typer.Option(
        None,
        "--context",
        help="Paper text/PDF grounding the content score (default: the run's ingested paper)",
    ),
    reference: Optional[str] = typer.Option(
        None, "--reference", help="Author-made reference poster image for calibrated judging"
    ),
    vlm_provider: Optional[str] = typer.Option(None, "--vlm-provider", help="Judge VLM provider"),
    vlm_model: Optional[str] = typer.Option(None, "--vlm-model", help="Judge VLM model"),
    dual_judge: bool = typer.Option(
        False,
        "--dual-judge",
        help="Average scores from a second judge model (JUDGE_VLM_MODEL / "
        "poster_judge_vlm_model; same provider)",
    ),
    config: Optional[str] = typer.Option(None, "--config", help="Path to config YAML file"),
) -> None:
    """Evaluate a generated poster: PPTEval-aligned VLM judge (Content/Design/
    Coherence, comparable with Paper2Poster baselines) plus deterministic
    venue-compliance preflight recomputed from the run's IR."""
    run_path = Path(run_dir).expanduser()
    preview = run_path / "preview.png"
    if not preview.is_file():
        console.print(f"[red]Error: no preview.png in {run_path}[/red]")
        raise typer.Exit(1)

    if context:
        from paperbanana.core.orchestrate import load_paper_text

        paper_context = load_paper_text(Path(context))
    else:
        assets_json = run_path / "paper_assets" / "paper_assets.json"
        if not assets_json.is_file():
            console.print(
                f"[red]Error: no ingested paper in {run_path}; pass --context explicitly[/red]"
            )
            raise typer.Exit(1)
        assets = json_mod.loads(assets_json.read_text(encoding="utf-8"))
        sections = "\n\n".join(f"## {s['heading']}\n{s['text']}" for s in assets["sections"])
        paper_context = f"{assets['title']}\n\n{assets['abstract']}\n\n{sections}"

    overrides: dict = {}
    if vlm_provider:
        overrides["vlm_provider"] = vlm_provider
    if vlm_model:
        overrides["vlm_model"] = vlm_model
    if config:
        settings = Settings.from_yaml(config, **overrides)
    else:
        from dotenv import load_dotenv

        load_dotenv()
        settings = Settings(**overrides)

    from paperbanana.core.utils import find_prompt_dir
    from paperbanana.poster.evaluation import evaluate_poster
    from paperbanana.providers.registry import ProviderRegistry

    vlm = ProviderRegistry.create_vlm(settings)
    secondary_vlm = None
    if dual_judge or settings.poster_dual_judge:
        if not settings.poster_judge_vlm_model:
            console.print(
                "[red]Error: --dual-judge needs a second judge model — set "
                "JUDGE_VLM_MODEL (or poster_judge_vlm_model in config) to a model "
                "different from the primary judge.[/red]"
            )
            raise typer.Exit(1)
        secondary_vlm = ProviderRegistry.create_vlm(
            settings.model_copy(update={"vlm_model": settings.poster_judge_vlm_model})
        )
    evaluation = asyncio.run(
        evaluate_poster(
            vlm,
            preview_path=preview,
            paper_context=paper_context,
            prompt_dir=Path(settings.prompt_dir or find_prompt_dir()),
            reference_path=Path(reference).expanduser() if reference else None,
            run_dir=run_path,
            venue_spec_dir=settings.venue_spec_dir,
            secondary_vlm=secondary_vlm,
        )
    )

    table = Table(title=f"Poster Evaluation — {run_path.name}")
    table.add_column("Dimension", style="bold")
    table.add_column("Score (1-5)")
    table.add_column("Rationale", overflow="fold")
    for s in evaluation.scores:
        table.add_row(s.dimension.title(), f"{s.score:g}", s.rationale)
    table.add_row("Overall", f"[bold]{evaluation.overall:.2f}[/bold]", "")
    console.print(table)
    if evaluation.judges > 1:
        console.print(f"Scores averaged across {evaluation.judges} judges.")
    if evaluation.compliance is not None:
        status = "[green]PASSED[/green]" if evaluation.compliance.passed else "[red]FAILED[/red]"
        console.print(
            f"Venue compliance (deterministic): {status} "
            f"({len(evaluation.compliance.failures)} failures, "
            f"{len(evaluation.compliance.warnings)} warnings)"
        )
    (run_path / "evaluation.json").write_text(
        evaluation.model_dump_json(indent=2), encoding="utf-8"
    )
    console.print(f"Saved: {run_path / 'evaluation.json'}")


@posters_app.command(name="ingest")
def posters_ingest(
    paths: list[str] = typer.Argument(..., help="Poster images/PDFs (or directories) to ingest"),
    venue: Optional[str] = typer.Option(None, "--venue", help="Venue tag for these exemplars"),
    vlm_provider: Optional[str] = typer.Option(None, "--vlm-provider", help="Parser VLM provider"),
    vlm_model: Optional[str] = typer.Option(None, "--vlm-model", help="Parser VLM model"),
    quality: Optional[float] = typer.Option(
        4.5,
        "--quality",
        min=1.0,
        max=5.0,
        help="Quality rating for these posters (curated/award posters deserve >=4; "
        "posters rated >=4 also become critic calibration anchors)",
    ),
    continue_on_error: bool = typer.Option(
        False, "--continue-on-error", help="Report per-file failures instead of stopping"
    ),
) -> None:
    """Teach the system from real posters: parse layout structures into memory.

    Every ingested poster's band/column/span skeleton becomes a retrieval
    exemplar that grounds future layout proposals."""
    import datetime as _dt

    from PIL import Image as PILImage

    from paperbanana.core.utils import find_prompt_dir
    from paperbanana.poster.agents.layout_parser import LayoutParserAgent
    from paperbanana.poster.convert import pdf_to_png
    from paperbanana.poster.memory import PosterExemplar, append_exemplar, resolve_memory_dir
    from paperbanana.providers.registry import ProviderRegistry

    files: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            files.extend(
                sorted(
                    f
                    for f in path.iterdir()
                    if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".pdf")
                )
            )
        elif path.is_file():
            files.append(path)
        else:
            console.print(f"[red]Error: not found: {path}[/red]")
            raise typer.Exit(1)
    if not files:
        console.print("[red]Error: no poster files found[/red]")
        raise typer.Exit(1)

    from dotenv import load_dotenv

    load_dotenv()
    overrides: dict = {}
    if vlm_provider:
        overrides["vlm_provider"] = vlm_provider
    if vlm_model:
        overrides["vlm_model"] = vlm_model
    settings = Settings(**overrides)
    vlm = ProviderRegistry.create_vlm(settings)
    parser = LayoutParserAgent(vlm, prompt_dir=settings.prompt_dir or find_prompt_dir())

    async def _ingest_all() -> tuple[int, list[tuple[Path, str]]]:
        ok, failures = 0, []
        for file in files:
            try:
                if file.suffix.lower() == ".pdf":
                    import tempfile

                    with tempfile.TemporaryDirectory() as tmp:
                        png = pdf_to_png(file, Path(tmp) / "poster.png", dpi=110)
                        image = PILImage.open(png).convert("RGB")
                else:
                    image = PILImage.open(file).convert("RGB")
                parsed = await parser.run(poster_image=image, source_name=file.stem)
                # Thumbnail doubles as a critic calibration anchor: the
                # refinement loop shows top-quality exemplar previews as
                # "this is what good looks like".
                thumb_dir = resolve_memory_dir(settings.poster_memory_dir) / "thumbnails"
                thumb_dir.mkdir(parents=True, exist_ok=True)
                thumb = image.copy()
                thumb.thumbnail((1400, 1400))
                thumb_path = thumb_dir / f"ingested_{file.stem}.png"
                thumb.save(thumb_path)
                exemplar = PosterExemplar(
                    id=f"ingested_{file.stem}",
                    source="ingested",
                    venue=venue or parsed.venue,
                    orientation="landscape" if image.width >= image.height else "portrait",
                    aspect_ratio=round(image.width / image.height, 3),
                    n_panels=len(parsed.skeleton.panels),
                    n_figures=parsed.n_figures,
                    visual_share=parsed.visual_share,
                    skeleton=parsed.skeleton,
                    quality=quality,
                    thumbnail_path=str(thumb_path),
                    created=_dt.datetime.now().isoformat(timespec="seconds"),
                    tags=["ingested"],
                )
                append_exemplar(
                    exemplar,
                    memory_dir=settings.poster_memory_dir,
                    max_self_generated=settings.poster_memory_max_self,
                )
                console.print(f"  [green]+[/green] {file.name} -> {exemplar.id}")
                ok += 1
            except Exception as exc:
                if not continue_on_error:
                    console.print(f"[red]Error ingesting {file}: {exc}[/red]")
                    raise typer.Exit(1) from exc
                failures.append((file, str(exc)))
        return ok, failures

    ok, failures = asyncio.run(_ingest_all())
    console.print(f"Ingested {ok}/{len(files)} posters into memory.")
    for file, error in failures:
        console.print(f"  [red]failed[/red] {file.name}: {error[:120]}")
    if failures:
        raise typer.Exit(1)


@posters_app.command(name="memory")
def posters_memory(
    action: str = typer.Argument("list", help="list | purge"),
    self_generated: bool = typer.Option(
        False, "--self-generated", help="Purge only self-generated exemplars"
    ),
) -> None:
    """Inspect or purge the poster layout memory."""
    from paperbanana.poster.memory import load_exemplars, purge_exemplars, resolve_memory_dir

    if action == "purge":
        removed = purge_exemplars(self_generated_only=self_generated)
        console.print(f"Removed {removed} exemplar(s) from {resolve_memory_dir()}.")
        return
    exemplars = load_exemplars()
    table = Table(title=f"Poster Memory ({len(exemplars)} exemplars)")
    table.add_column("Id", overflow="fold")
    table.add_column("Source")
    table.add_column("Venue")
    table.add_column("Orient")
    table.add_column("Panels")
    table.add_column("Figures")
    table.add_column("Quality")
    by_source: dict = {}
    for e in exemplars:
        by_source.setdefault(e.source, []).append(e)
    for source, members in sorted(by_source.items()):
        for e in members[:5]:
            table.add_row(
                e.id,
                e.source,
                e.venue or "—",
                e.orientation,
                str(e.n_panels),
                str(e.n_figures),
                f"{e.quality:.1f}" if e.quality else "—",
            )
        if len(members) > 5:
            table.add_row(f"… +{len(members) - 5} more", source, "", "", "", "", "")
    console.print(table)
    console.print(f"User store: [bold]{resolve_memory_dir()}[/bold]")


if __name__ == "__main__":
    app()
