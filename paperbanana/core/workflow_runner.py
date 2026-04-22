"""Shared batch and orchestration execution for CLI and MCP (no Typer / Rich)."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Callable

import structlog

from paperbanana.core.batch import (
    checkpoint_progress,
    generate_batch_id,
    init_or_load_checkpoint,
    load_batch_manifest_with_composite,
    load_plot_batch_manifest,
    mark_item_failure,
    mark_item_running,
    mark_item_success,
    select_items_for_run,
)
from paperbanana.core.config import Settings
from paperbanana.core.orchestrate import (
    init_or_load_orchestration_checkpoint,
    prepare_orchestration_plan,
    run_orchestration,
)
from paperbanana.core.plot_data import load_statistical_plot_payload
from paperbanana.core.source_loader import load_methodology_source
from paperbanana.core.types import DiagramType, GenerationInput
from paperbanana.core.utils import ensure_dir

logger = structlog.get_logger()


def _require_pdf_dep() -> None:
    try:
        import fitz  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "PDF input requires PyMuPDF. Install with: pip install 'paperbanana[pdf]'"
        ) from e


def _check_pdf_dep(path: Path) -> None:
    if path.suffix.lower() == ".pdf":
        _require_pdf_dep()


def _load_settings(
    *,
    config: str | None,
    overrides: dict[str, Any],
) -> Settings:
    if config:
        return Settings.from_yaml(config, **overrides)
    from dotenv import load_dotenv

    load_dotenv()
    return Settings(**overrides)


def run_methodology_batch(
    *,
    manifest_path: Path,
    output_dir: Path,
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
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run methodology batch; mirrors ``paperbanana batch``."""
    from paperbanana.core.pipeline import PaperBananaPipeline
    from paperbanana.data.manager import DatasetManager

    manifest_path = Path(manifest_path).resolve()
    if format not in ("png", "jpeg", "webp"):
        raise ValueError(f"Format must be png, jpeg, or webp. Got: {format}")
    if venue and venue.lower() not in ("neurips", "icml", "acl", "ieee", "custom"):
        raise ValueError(f"venue must be neurips, icml, acl, ieee, or custom. Got: {venue}")
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    items, composite_config = load_batch_manifest_with_composite(manifest_path)
    if any(str(item.get("input", "")).lower().endswith(".pdf") for item in items):
        _require_pdf_dep()

    is_resume = bool(resume_batch)
    if is_resume:
        resume_ref = Path(resume_batch)
        if resume_ref.is_dir():
            batch_dir = resume_ref.resolve()
            batch_id = batch_dir.name
        else:
            batch_id = resume_batch.strip()
            batch_dir = (Path(output_dir) / batch_id).resolve()
    else:
        batch_id = generate_batch_id()
        batch_dir = (Path(output_dir) / batch_id).resolve()
    ensure_dir(batch_dir)

    overrides: dict[str, Any] = {"output_dir": str(batch_dir), "output_format": format}
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

    settings = _load_settings(config=config, overrides=overrides)

    if auto_download_data:
        dm = DatasetManager(cache_dir=settings.cache_dir)
        if not dm.is_downloaded():
            try:
                dm.download(dataset="curated")
            except Exception as e:
                logger.warning("curated_download_failed", error=str(e))

    state = init_or_load_checkpoint(
        batch_dir=batch_dir,
        batch_id=batch_id,
        manifest_path=manifest_path,
        batch_kind="methodology",
        items=items,
        resume=is_resume,
    )

    def emit(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        else:
            logger.info("workflow_batch", message=msg)

    total_start = time.perf_counter()
    planned = select_items_for_run(state, retry_failed=retry_failed)
    if not planned:
        checkpoint_progress(batch_dir=batch_dir, state=state, mark_complete=True)
        report_path = batch_dir / "batch_report.json"
        emit(f"Nothing to run; report at {report_path}")
        return {
            "batch_dir": str(batch_dir),
            "batch_id": batch_id,
            "batch_report_path": str(report_path),
            "had_work": False,
            "succeeded": 0,
            "failed": 0,
            "skipped": len(state.get("items", {})),
            "composite_path": None,
            "strict_success": True,
        }

    async def _run_all() -> None:
        sem = asyncio.Semaphore(concurrency)

        async def _run_one(idx: int, item: dict[str, object]) -> None:
            item_key = str(item["_item_key"])
            item_id = str(item["id"])
            async with sem:
                for attempt in range(max_retries + 1):
                    mark_item_running(state, item_key)
                    checkpoint_progress(
                        batch_dir=batch_dir,
                        state=state,
                        total_seconds=time.perf_counter() - total_start,
                    )
                    input_path = Path(str(item["input"]))
                    if not input_path.exists():
                        mark_item_failure(state, item_key, "input file not found")
                        checkpoint_progress(
                            batch_dir=batch_dir,
                            state=state,
                            total_seconds=time.perf_counter() - total_start,
                        )
                        emit(f"Item {idx + 1}/{len(items)} {item_id}: input missing")
                        return
                    try:
                        source_context = load_methodology_source(
                            input_path, pdf_pages=item.get("pdf_pages")
                        )
                        gen_input = GenerationInput(
                            source_context=source_context,
                            communicative_intent=str(item["caption"]),
                            diagram_type=DiagramType.METHODOLOGY,
                        )
                        result = await PaperBananaPipeline(settings=settings).generate(gen_input)
                        mark_item_success(
                            state,
                            item_key,
                            result.metadata.get("run_id"),
                            result.image_path,
                            len(result.iterations),
                        )
                        checkpoint_progress(
                            batch_dir=batch_dir,
                            state=state,
                            total_seconds=time.perf_counter() - total_start,
                        )
                        emit(f"Item {idx + 1}/{len(items)} {item_id}: ok -> {result.image_path}")
                        return
                    except Exception as e:
                        mark_item_failure(state, item_key, str(e))
                        checkpoint_progress(
                            batch_dir=batch_dir,
                            state=state,
                            total_seconds=time.perf_counter() - total_start,
                        )
                        if attempt < max_retries:
                            emit(f"Item {item_id}: retry {attempt + 1}/{max_retries} after {e}")
                            continue
                        emit(f"Item {idx + 1}/{len(items)} {item_id}: failed - {e}")
                        return

        await asyncio.gather(*[_run_one(idx, item) for idx, item, _ in planned])

    asyncio.run(_run_all())

    total_elapsed = time.perf_counter() - total_start
    report = checkpoint_progress(
        batch_dir=batch_dir,
        state=state,
        total_seconds=total_elapsed,
        mark_complete=True,
    )
    report_path = batch_dir / "batch_report.json"
    ri = report["items"]
    succeeded = sum(1 for x in ri if x.get("status") == "success")
    failed = sum(1 for x in ri if x.get("status") == "failed")
    skipped = len(ri) - succeeded - failed

    composite_path: str | None = None
    if composite_config is not None:
        output_paths = [x["output_path"] for x in report["items"] if x.get("output_path")]
        if output_paths:
            from paperbanana.core.composite import compose_images

            comp_output = composite_config.get("output") or "composite.png"
            comp_path = batch_dir / str(comp_output)
            try:
                compose_images(
                    image_paths=output_paths,
                    layout=composite_config.get("layout", "auto"),
                    labels=composite_config.get("labels"),
                    auto_label=composite_config.get("auto_label", True),
                    spacing=composite_config.get("spacing", 20),
                    label_position=composite_config.get("label_position", "bottom"),
                    output_path=comp_path,
                )
                composite_path = str(comp_path)
                emit(f"Composite: {composite_path}")
            except Exception as e:
                logger.warning("composite_failed", error=str(e))

    return {
        "batch_dir": str(batch_dir),
        "batch_id": batch_id,
        "batch_report_path": str(report_path),
        "had_work": True,
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "composite_path": composite_path,
        "strict_success": failed == 0,
        "items_summary": [
            {
                "id": x.get("id"),
                "status": x.get("status"),
                "output_path": x.get("output_path"),
                "error": x.get("error"),
            }
            for x in ri
        ],
    }


def run_plot_batch(
    *,
    manifest_path: Path,
    output_dir: Path,
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
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run statistical plot batch; mirrors ``paperbanana plot-batch``."""
    from paperbanana.core.pipeline import PaperBananaPipeline

    manifest_path = Path(manifest_path).resolve()
    if format not in ("png", "jpeg", "webp"):
        raise ValueError(f"Format must be png, jpeg, or webp. Got: {format}")
    if venue and venue.lower() not in ("neurips", "icml", "acl", "ieee", "custom"):
        raise ValueError(f"venue must be neurips, icml, acl, ieee, or custom. Got: {venue}")
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    items = load_plot_batch_manifest(manifest_path)

    is_resume = bool(resume_batch)
    if is_resume:
        resume_ref = Path(resume_batch)
        if resume_ref.is_dir():
            batch_dir = resume_ref.resolve()
            batch_id = batch_dir.name
        else:
            batch_id = resume_batch.strip()
            batch_dir = (Path(output_dir) / batch_id).resolve()
    else:
        batch_id = generate_batch_id()
        batch_dir = (Path(output_dir) / batch_id).resolve()
    ensure_dir(batch_dir)

    overrides: dict[str, Any] = {
        "output_dir": str(batch_dir),
        "output_format": format,
        "optimize_inputs": optimize,
        "auto_refine": auto,
    }
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
    if max_iterations is not None:
        overrides["max_iterations"] = max_iterations
    overrides["save_prompts"] = True if save_prompts is None else save_prompts
    if venue:
        overrides["venue"] = venue
    if not vlm_provider:
        overrides.setdefault("vlm_provider", "gemini")

    settings = _load_settings(config=config, overrides=overrides)

    state = init_or_load_checkpoint(
        batch_dir=batch_dir,
        batch_id=batch_id,
        manifest_path=manifest_path,
        batch_kind="statistical_plot",
        items=items,
        resume=is_resume,
    )

    def emit(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        else:
            logger.info("workflow_plot_batch", message=msg)

    total_start = time.perf_counter()
    planned = select_items_for_run(state, retry_failed=retry_failed)
    if not planned:
        checkpoint_progress(batch_dir=batch_dir, state=state, mark_complete=True)
        report_path = batch_dir / "batch_report.json"
        emit(f"Nothing to run; report at {report_path}")
        return {
            "batch_dir": str(batch_dir),
            "batch_id": batch_id,
            "batch_report_path": str(report_path),
            "had_work": False,
            "succeeded": 0,
            "failed": 0,
            "skipped": len(state.get("items", {})),
            "strict_success": True,
        }

    async def _run_all() -> None:
        sem = asyncio.Semaphore(concurrency)

        async def _run_one(idx: int, item: dict[str, object]) -> None:
            item_key = str(item["_item_key"])
            item_id = str(item["id"])
            async with sem:
                for attempt in range(max_retries + 1):
                    mark_item_running(state, item_key)
                    checkpoint_progress(
                        batch_dir=batch_dir,
                        state=state,
                        total_seconds=time.perf_counter() - total_start,
                    )
                    data_path = Path(str(item["data"]))
                    if not data_path.exists():
                        mark_item_failure(state, item_key, "data file not found")
                        checkpoint_progress(
                            batch_dir=batch_dir,
                            state=state,
                            total_seconds=time.perf_counter() - total_start,
                        )
                        emit(f"Item {idx + 1}/{len(items)} {item_id}: data missing")
                        return
                    try:
                        source_context, raw_data = load_statistical_plot_payload(data_path)
                        ar = item.get("aspect_ratio") or aspect_ratio
                        gen_input = GenerationInput(
                            source_context=source_context,
                            communicative_intent=str(item["intent"]),
                            diagram_type=DiagramType.STATISTICAL_PLOT,
                            raw_data={"data": raw_data},
                            aspect_ratio=ar,
                        )
                        result = await PaperBananaPipeline(settings=settings).generate(gen_input)
                        mark_item_success(
                            state,
                            item_key,
                            result.metadata.get("run_id"),
                            result.image_path,
                            len(result.iterations),
                        )
                        checkpoint_progress(
                            batch_dir=batch_dir,
                            state=state,
                            total_seconds=time.perf_counter() - total_start,
                        )
                        emit(f"Item {idx + 1}/{len(items)} {item_id}: ok -> {result.image_path}")
                        return
                    except Exception as e:
                        mark_item_failure(state, item_key, str(e))
                        checkpoint_progress(
                            batch_dir=batch_dir,
                            state=state,
                            total_seconds=time.perf_counter() - total_start,
                        )
                        if attempt < max_retries:
                            emit(f"Item {item_id}: retry {attempt + 1}/{max_retries} after {e}")
                            continue
                        emit(f"Item {idx + 1}/{len(items)} {item_id}: failed - {e}")
                        return

        await asyncio.gather(*[_run_one(idx, item) for idx, item, _ in planned])

    asyncio.run(_run_all())

    total_elapsed = time.perf_counter() - total_start
    report = checkpoint_progress(
        batch_dir=batch_dir,
        state=state,
        total_seconds=total_elapsed,
        mark_complete=True,
    )
    report_path = batch_dir / "batch_report.json"
    ri = report["items"]
    succeeded = sum(1 for x in ri if x.get("status") == "success")
    failed = sum(1 for x in ri if x.get("status") == "failed")
    skipped = len(ri) - succeeded - failed

    return {
        "batch_dir": str(batch_dir),
        "batch_id": batch_id,
        "batch_report_path": str(report_path),
        "had_work": True,
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "strict_success": failed == 0,
        "items_summary": [
            {
                "id": x.get("id"),
                "status": x.get("status"),
                "output_path": x.get("output_path"),
                "error": x.get("error"),
            }
            for x in ri
        ],
    }


def run_orchestration_package(
    *,
    paper: str | None,
    resume_orchestrate: str | None,
    output_dir: Path,
    data_dir: str | None,
    max_method_figures: int,
    max_plot_figures: int,
    pdf_pages: str | None,
    dry_run: bool,
    config: str | None,
    vlm_provider: str | None,
    vlm_model: str | None,
    image_provider: str | None,
    image_model: str | None,
    iterations: int | None,
    auto: bool,
    max_iterations: int | None,
    optimize: bool,
    format: str,
    save_prompts: bool | None,
    venue: str | None,
    retry_failed: bool,
    max_retries: int,
    concurrency: int,
    progress_callback: Callable[[str], None] | None = None,
    after_plan_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Plan and/or run figure-package orchestration; mirrors ``paperbanana orchestrate``."""
    is_resume = bool(resume_orchestrate)
    if format not in ("png", "jpeg", "webp"):
        raise ValueError(f"Format must be png, jpeg, or webp. Got: {format}")
    if venue and venue.lower() not in ("neurips", "icml", "acl", "ieee", "custom"):
        raise ValueError(f"venue must be neurips, icml, acl, ieee, or custom. Got: {venue}")
    if max_method_figures < 1:
        raise ValueError("max_method_figures must be >= 1")
    if max_plot_figures < 0:
        raise ValueError("max_plot_figures must be >= 0")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    if is_resume and paper:
        raise ValueError("Provide only one of paper or resume_orchestrate")
    if not is_resume and not paper:
        raise ValueError("paper is required for new orchestrations")
    if is_resume and data_dir:
        raise ValueError("data_dir is only valid for new orchestrations")
    if is_resume and pdf_pages:
        raise ValueError("pdf_pages is only valid for new orchestrations")

    orchestration_id, orchestrate_dir, plan, plan_path, resumed = prepare_orchestration_plan(
        paper=paper,
        resume_orchestrate=resume_orchestrate,
        output_dir=str(output_dir),
        data_dir=data_dir,
        max_method_figures=max_method_figures,
        max_plot_figures=max_plot_figures,
        pdf_pages=pdf_pages,
    )

    if not resumed:
        _check_pdf_dep(Path(str(plan.get("paper_path", ""))))

    ensure_dir(orchestrate_dir)
    runs_dir = ensure_dir(orchestrate_dir / "runs")

    plan_header: dict[str, Any] = {
        "orchestration_id": orchestration_id,
        "orchestrate_dir": str(orchestrate_dir),
        "orchestration_plan_path": str(plan_path),
        "paper_path": str(plan.get("paper_path", "")),
        "paper_title": str(plan.get("paper_title", "")),
        "methodology_items_planned": len(plan.get("methodology_items", [])),
        "plot_items_planned": len(plan.get("plot_items", [])),
        "resumed": resumed,
    }
    if after_plan_callback is not None:
        after_plan_callback(plan_header)

    if dry_run:
        return {
            **plan_header,
            "dry_run": True,
            "strict_success": True,
        }

    overrides: dict[str, Any] = {
        "output_dir": str(runs_dir),
        "output_format": format,
        "optimize_inputs": optimize,
        "auto_refine": auto,
    }
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
    if max_iterations is not None:
        overrides["max_iterations"] = max_iterations
    if save_prompts is not None:
        overrides["save_prompts"] = save_prompts
    if venue:
        overrides["venue"] = venue

    settings = _load_settings(config=config, overrides=overrides)

    state = init_or_load_orchestration_checkpoint(
        orchestrate_dir=orchestrate_dir,
        orchestration_id=orchestration_id,
        plan_path=plan_path,
        plan=plan,
        resume=resumed,
    )

    def emit(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        else:
            logger.info("workflow_orchestrate", message=msg)

    report, had_work = run_orchestration(
        state=state,
        plan=plan,
        settings=settings,
        orchestrate_dir=orchestrate_dir,
        retry_failed=retry_failed,
        max_retries=max_retries,
        concurrency=concurrency,
        progress_callback=emit,
    )

    package_path = orchestrate_dir / "figure_package.json"
    figures_tex = orchestrate_dir / "figures.tex"
    captions_md = orchestrate_dir / "captions.md"
    fail_count = len(report.get("failures", []))
    success_count = len(report.get("generated_items", []))

    base: dict[str, Any] = {
        "orchestration_id": orchestration_id,
        "orchestrate_dir": str(orchestrate_dir),
        "orchestration_plan_path": str(plan_path),
        "figure_package_path": str(package_path),
        "figures_tex_path": str(figures_tex),
        "captions_md_path": str(captions_md),
        "runs_dir": str(runs_dir),
        "dry_run": False,
        "had_work": had_work,
        "paper_path": str(report.get("paper_path") or plan.get("paper_path", "")),
        "paper_title": str(report.get("paper_title") or plan.get("paper_title", "")),
        "methodology_items_planned": len(plan.get("methodology_items", [])),
        "plot_items_planned": len(plan.get("plot_items", [])),
        "generated_count": success_count,
        "failed_count": fail_count,
        "total_seconds": report.get("total_seconds"),
        "strict_success": fail_count == 0,
        "failures": report.get("failures", []),
    }
    if not had_work:
        base["note"] = "All tasks already completed; package manifest refreshed."
    return base
