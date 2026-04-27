"""Async pipeline runners with progress text for the Studio UI."""

from __future__ import annotations

import asyncio
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

from paperbanana.core.batch import (
    checkpoint_progress,
    generate_batch_id,
    init_or_load_checkpoint,
    load_batch_manifest,
    load_plot_batch_manifest,
    mark_item_failure,
    mark_item_running,
    mark_item_success,
    select_items_for_run,
)
from paperbanana.core.config import Settings
from paperbanana.core.logging import configure_logging
from paperbanana.core.pipeline import PaperBananaPipeline
from paperbanana.core.plot_data import load_statistical_plot_payload
from paperbanana.core.resume import load_resume_state
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
from paperbanana.core.types import (
    DiagramType,
    GenerationInput,
    PipelineProgressEvent,
    PipelineProgressStage,
)
from paperbanana.core.utils import ensure_dir, find_prompt_dir, generate_run_id, save_json
from paperbanana.core.workflow_runner import run_orchestration_package
from paperbanana.evaluation.judge import VLMJudge
from paperbanana.providers.registry import ProviderRegistry

VLM_PROVIDER_CHOICES = ["gemini", "openai", "openrouter", "bedrock", "anthropic"]
IMAGE_PROVIDER_CHOICES = [
    "google_imagen",
    "openai_imagen",
    "openrouter_imagen",
    "bedrock_imagen",
]
ASPECT_RATIO_CHOICES = [
    "default",
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "9:16",
    "16:9",
    "21:9",
]


def read_text_file(path: str | None, max_chars: int = 500_000) -> str:
    """Read UTF-8 text from a path; empty string if missing."""
    if not path:
        return ""
    p = Path(path)
    if not p.is_file():
        return ""
    text = p.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n[truncated]"
    return text


def merge_context(text: str, file_path: str | None) -> str:
    """Prefer uploaded file content when present; otherwise use text box."""
    from_file = read_text_file(file_path)
    if from_file.strip():
        return from_file
    return (text or "").strip()


def build_settings(
    *,
    config_path: Optional[str],
    output_dir: str,
    vlm_provider: str,
    vlm_model: str,
    image_provider: str,
    image_model: str,
    output_format: str,
    refinement_iterations: int,
    auto_refine: bool,
    max_iterations: int,
    optimize_inputs: bool,
    save_prompts: bool,
    seed: Optional[int] = None,
) -> Settings:
    """Merge YAML config (optional), environment, and Studio overrides."""
    base_defaults = Settings()
    overrides: dict[str, Any] = {
        "output_dir": output_dir,
        "vlm_provider": vlm_provider.strip() or "gemini",
        "vlm_model": vlm_model.strip() or base_defaults.vlm_model,
        "image_provider": image_provider.strip() or "google_imagen",
        "image_model": image_model.strip() or base_defaults.image_model,
        "output_format": output_format.lower(),
        "refinement_iterations": int(refinement_iterations),
        "auto_refine": bool(auto_refine),
        "max_iterations": int(max_iterations),
        "optimize_inputs": bool(optimize_inputs),
        "save_prompts": bool(save_prompts),
    }
    if seed is not None and str(seed).strip() != "":
        try:
            overrides["seed"] = int(seed)
        except ValueError:
            pass

    if config_path and str(config_path).strip():
        return Settings.from_yaml(Path(config_path).expanduser(), **overrides)
    return Settings(**overrides)


class ProgressLog:
    """Collect human-readable lines from ``PipelineProgressEvent`` callbacks."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def append(self, line: str) -> None:
        self.lines.append(line)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def handler(self) -> Callable[[PipelineProgressEvent], None]:
        def _on(event: PipelineProgressEvent) -> None:
            self._dispatch(event)

        return _on

    def _dispatch(self, event: PipelineProgressEvent) -> None:
        st = event.stage
        sec = f" ({event.seconds:.1f}s)" if event.seconds is not None else ""
        if st == PipelineProgressStage.OPTIMIZER_START:
            self.append("Phase 0 — Input optimization: starting…")
        elif st == PipelineProgressStage.OPTIMIZER_END:
            self.append(f"Phase 0 — Input optimization: done{sec}")
        elif st == PipelineProgressStage.RETRIEVER_START:
            self.append("Phase 1 — Retriever: selecting examples…")
        elif st == PipelineProgressStage.RETRIEVER_END:
            n = (event.extra or {}).get("examples_count", "?")
            self.append(f"Phase 1 — Retriever: {n} examples{sec}")
        elif st == PipelineProgressStage.PLANNER_START:
            self.append("Phase 1 — Planner: drafting description…")
        elif st == PipelineProgressStage.PLANNER_END:
            ratio = (event.extra or {}).get("recommended_ratio")
            extra = f", suggested ratio {ratio}" if ratio else ""
            self.append(f"Phase 1 — Planner: done{sec}{extra}")
        elif st == PipelineProgressStage.STYLIST_START:
            self.append("Phase 1 — Stylist: refining aesthetics…")
        elif st == PipelineProgressStage.STYLIST_END:
            self.append(f"Phase 1 — Stylist: done{sec}")
        elif st == PipelineProgressStage.STRUCTURER_START:
            self.append("Vector — Structurer: building diagram IR…")
        elif st == PipelineProgressStage.STRUCTURER_END:
            ex = event.extra or {}
            if ex.get("error"):
                self.append(f"Vector — Structurer: failed{sec}")
            else:
                self.append(f"Vector — export done{sec}")
        elif st == PipelineProgressStage.VISUALIZER_START:
            it = event.iteration or "?"
            tot = (event.extra or {}).get("total_iterations")
            tot_s = f"/{tot}" if tot else ""
            self.append(f"Phase 2 — Visualizer: iteration {it}{tot_s}…")
        elif st == PipelineProgressStage.VISUALIZER_END:
            self.append(f"Phase 2 — Visualizer: image saved{sec}")
        elif st == PipelineProgressStage.CRITIC_START:
            self.append("Phase 2 — Critic: reviewing…")
        elif st == PipelineProgressStage.CRITIC_END:
            ex = event.extra or {}
            if ex.get("needs_revision"):
                self.append(f"Phase 2 — Critic: revision suggested{sec}")
                for s in (ex.get("critic_suggestions") or [])[:5]:
                    self.append(f"  • {s}")
            else:
                self.append(f"Phase 2 — Critic: satisfied{sec}")


def _aspect_ratio_value(label: str) -> Optional[str]:
    if not label or label == "default":
        return None
    return label


def run_methodology(
    settings: Settings,
    source_context: str,
    caption: str,
    aspect_ratio_label: str,
    reference_ids: Optional[str] = None,
    verbose_logging: bool = False,
) -> tuple[str, Optional[str], list[tuple[str, str]], str]:
    """Run methodology diagram generation. Returns (log, final_path, gallery, error)."""
    configure_logging(verbose=verbose_logging)
    log = ProgressLog()
    log.append("Starting methodology diagram pipeline…")
    err = ""
    try:
        ref_id_list = None
        if reference_ids:
            ref_id_list = [rid.strip() for rid in reference_ids.split(",") if rid.strip()]
        gen_in = GenerationInput(
            source_context=source_context,
            communicative_intent=caption.strip(),
            diagram_type=DiagramType.METHODOLOGY,
            aspect_ratio=_aspect_ratio_value(aspect_ratio_label),
            reference_ids=ref_id_list,
        )

        async def _go():
            pipeline = PaperBananaPipeline(settings=settings)
            return await pipeline.generate(gen_in, progress_callback=log.handler())

        result = asyncio.run(_go())
        log.append("")
        log.append(f"Complete. Run ID: {result.metadata.get('run_id', '?')}")
        log.append(f"Final image: {result.image_path}")
        gallery: list[tuple[str, str]] = []
        for rec in result.iterations:
            p = Path(rec.image_path)
            if p.is_file():
                gallery.append((str(p), f"iter {rec.iteration}"))
        final = result.image_path
        fp = final if Path(final).is_file() else None
        return log.text, fp, gallery, ""
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        log.append("")
        log.append("FAILED")
        log.append(err)
        log.append(traceback.format_exc())
        return log.text, None, [], err


def run_plot(
    settings: Settings,
    data_path: str,
    intent: str,
    aspect_ratio_label: str,
    verbose_logging: bool = False,
) -> tuple[str, Optional[str], list[tuple[str, str]], str]:
    """Run statistical plot pipeline from CSV or JSON path."""
    configure_logging(verbose=verbose_logging)
    log = ProgressLog()
    log.append("Starting statistical plot pipeline…")
    path = Path(data_path)
    if not path.is_file():
        msg = f"Data file not found: {data_path}"
        log.append(msg)
        return log.text, None, [], msg

    try:
        source_context, raw_data = load_statistical_plot_payload(path)

        gen_in = GenerationInput(
            source_context=source_context,
            communicative_intent=intent.strip(),
            diagram_type=DiagramType.STATISTICAL_PLOT,
            raw_data={"data": raw_data},
            aspect_ratio=_aspect_ratio_value(aspect_ratio_label),
        )

        async def _go():
            pipeline = PaperBananaPipeline(settings=settings)
            return await pipeline.generate(gen_in, progress_callback=log.handler())

        result = asyncio.run(_go())
        log.append("")
        log.append(f"Complete. Run ID: {result.metadata.get('run_id', '?')}")
        gallery: list[tuple[str, str]] = []
        for rec in result.iterations:
            p = Path(rec.image_path)
            if p.is_file():
                gallery.append((str(p), f"iter {rec.iteration}"))
        fp = result.image_path if Path(result.image_path).is_file() else None
        return log.text, fp, gallery, ""
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        log.append("")
        log.append("FAILED")
        log.append(err)
        log.append(traceback.format_exc())
        return log.text, None, [], err


def run_evaluate(
    settings: Settings,
    generated_path: str,
    reference_path: str,
    source_context: str,
    caption: str,
    evaluation_task: DiagramType = DiagramType.METHODOLOGY,
    plot_data_path: str = "",
    verbose_logging: bool = False,
) -> tuple[str, str]:
    """VLM judge comparative evaluation. Returns (log, formatted results)."""
    configure_logging(verbose=verbose_logging)
    task_label = "plot" if evaluation_task == DiagramType.STATISTICAL_PLOT else "diagram"
    lines: list[str] = [f"Starting comparative evaluation ({task_label}, VLM judge)…"]
    gp = Path(generated_path)
    rp = Path(reference_path)
    if not gp.is_file():
        msg = f"Generated image not found: {generated_path}"
        lines.append(msg)
        return "\n".join(lines), msg
    if not rp.is_file():
        msg = f"Reference image not found: {reference_path}"
        lines.append(msg)
        return "\n".join(lines), msg
    effective_context = source_context
    if evaluation_task == DiagramType.STATISTICAL_PLOT:
        plot_path = Path(plot_data_path)
        if not plot_path.is_file():
            msg = f"Plot data file not found: {plot_data_path}"
            lines.append(msg)
            return "\n".join(lines), msg
        try:
            effective_context, _ = load_statistical_plot_payload(plot_path)
        except ValueError as e:
            msg = f"Invalid plot data: {e}"
            lines.append(msg)
            return "\n".join(lines), msg

    if not effective_context.strip():
        msg = "Source context is empty."
        lines.append(msg)
        return "\n".join(lines), msg

    try:
        vlm = ProviderRegistry.create_vlm(settings)
        judge = VLMJudge(vlm, prompt_dir=find_prompt_dir())

        async def _go():
            return await judge.evaluate(
                image_path=str(gp),
                source_context=effective_context,
                caption=caption.strip(),
                reference_path=str(rp),
                task=evaluation_task,
            )

        scores = asyncio.run(_go())
        lines.append("Done.")
        dims = ["faithfulness", "conciseness", "readability", "aesthetics"]
        out_parts = [f"## Results ({task_label})\n"]
        for dim in dims:
            r = getattr(scores, dim)
            out_parts.append(f"**{dim}** — {r.winner} (score {r.score:.0f})\n")
            if r.reasoning:
                out_parts.append(f"{r.reasoning}\n\n")
        out_parts.append(
            f"### Overall\n**{scores.overall_winner}** — score {scores.overall_score:.0f}\n"
        )
        return "\n".join(lines), "".join(out_parts)
    except Exception as e:
        err = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
        lines.append("FAILED")
        lines.append(err)
        return "\n".join(lines), err


def run_continue(
    settings: Settings,
    output_dir: str,
    run_id: str,
    user_feedback: str,
    additional_iterations: Optional[int],
    verbose_logging: bool = False,
) -> tuple[str, Optional[str], list[tuple[str, str]], str]:
    """Continue an existing run directory."""
    configure_logging(verbose=verbose_logging)
    log = ProgressLog()
    log.append(f"Continuing run {run_id}…")
    try:
        state = load_resume_state(output_dir, run_id.strip())
    except (FileNotFoundError, ValueError) as e:
        msg = str(e)
        log.append(msg)
        return log.text, None, [], msg

    try:
        extra_it = None
        if additional_iterations and additional_iterations > 0:
            extra_it = additional_iterations

        async def _go():
            pipeline = PaperBananaPipeline(settings=settings)
            return await pipeline.continue_run(
                resume_state=state,
                additional_iterations=extra_it,
                user_feedback=user_feedback.strip() or None,
                progress_callback=log.handler(),
            )

        result = asyncio.run(_go())
        log.append("")
        log.append(f"Complete. Final: {result.image_path}")
        gallery: list[tuple[str, str]] = []
        for rec in result.iterations:
            p = Path(rec.image_path)
            if p.is_file():
                gallery.append((str(p), f"iter {rec.iteration}"))
        fp = result.image_path if Path(result.image_path).is_file() else None
        return log.text, fp, gallery, ""
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        log.append("")
        log.append("FAILED")
        log.append(err)
        log.append(traceback.format_exc())
        return log.text, None, [], err


def run_batch(
    settings: Settings,
    manifest_path: str,
    *,
    resume_batch: Optional[str] = None,
    retry_failed: bool = False,
    max_retries: int = 0,
    concurrency: int = 1,
    verbose_logging: bool = False,
) -> tuple[str, str]:
    """Run batch manifest; returns (log, batch_dir path or error note)."""
    configure_logging(verbose=verbose_logging)
    lines: list[str] = []
    mpath = Path(manifest_path)
    if not mpath.is_file():
        msg = f"Manifest not found: {manifest_path}"
        lines.append(msg)
        return "\n".join(lines), msg

    try:
        items = load_batch_manifest(mpath)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        msg = f"Invalid manifest: {e}"
        lines.append(msg)
        return "\n".join(lines), msg

    is_resume = bool(resume_batch)
    if is_resume:
        resume_ref = Path(resume_batch)
        if resume_ref.is_dir():
            batch_dir = resume_ref.resolve()
            batch_id = batch_dir.name
        else:
            batch_id = resume_batch.strip()
            batch_dir = (Path(settings.output_dir) / batch_id).resolve()
    else:
        batch_id = generate_batch_id()
        batch_dir = Path(settings.output_dir) / batch_id
    ensure_dir(batch_dir)

    settings = settings.model_copy(update={"output_dir": str(batch_dir)})
    lines.append(f"Batch ID: {batch_id}")
    lines.append(f"Items: {len(items)}")
    lines.append(f"Output: {batch_dir}")
    lines.append("")

    state = init_or_load_checkpoint(
        batch_dir=batch_dir,
        batch_id=batch_id,
        manifest_path=mpath,
        batch_kind="methodology",
        items=items,
        resume=is_resume,
    )
    planned = select_items_for_run(state, retry_failed=retry_failed)
    if not planned:
        checkpoint_progress(batch_dir=batch_dir, state=state, mark_complete=True)
        lines.append("Nothing to run: all items already completed.")
        lines.append(f"Report written: {batch_dir / 'batch_report.json'}")
        return "\n".join(lines), str(batch_dir.resolve())

    if max_retries < 0:
        max_retries = 0
    if concurrency < 1:
        concurrency = 1

    async def _run_all_items() -> None:
        sem = asyncio.Semaphore(concurrency)
        from paperbanana.core.source_loader import load_methodology_source

        async def _run_one(idx: int, item: dict[str, Any]) -> None:
            item_id = item["id"]
            item_key = item["_item_key"]
            lines.append(f"— Item {idx + 1}/{len(items)} — {item_id}")
            async with sem:
                for attempt in range(max_retries + 1):
                    mark_item_running(state, item_key)
                    checkpoint_progress(batch_dir=batch_dir, state=state)
                    input_path = Path(item["input"])
                    if not input_path.is_file():
                        mark_item_failure(state, item_key, "input file not found")
                        checkpoint_progress(batch_dir=batch_dir, state=state)
                        lines.append(f"  error: input not found ({input_path})")
                        return
                    try:
                        source_context = load_methodology_source(
                            input_path, pdf_pages=item.get("pdf_pages")
                        )
                        gen_in = GenerationInput(
                            source_context=source_context,
                            communicative_intent=item["caption"],
                            diagram_type=DiagramType.METHODOLOGY,
                        )
                        result = await PaperBananaPipeline(settings=settings).generate(gen_in)
                        mark_item_success(
                            state,
                            item_key,
                            result.metadata.get("run_id"),
                            result.image_path,
                            len(result.iterations),
                        )
                        checkpoint_progress(batch_dir=batch_dir, state=state)
                        lines.append(f"  ok: {result.image_path}")
                        return
                    except Exception as e:
                        mark_item_failure(state, item_key, str(e))
                        checkpoint_progress(batch_dir=batch_dir, state=state)
                        if attempt < max_retries:
                            lines.append(f"  retry {attempt + 1}/{max_retries}: {e}")
                            continue
                        lines.append(f"  error: {e}")
                        return

        await asyncio.gather(*[_run_one(idx, item) for idx, item, _ in planned])

    asyncio.run(_run_all_items())

    report = checkpoint_progress(batch_dir=batch_dir, state=state, mark_complete=True)
    report_path = batch_dir / "batch_report.json"
    lines.append("")
    lines.append(f"Report written: {report_path}")
    ok = sum(1 for x in report["items"] if x.get("output_path"))
    lines.append(f"Succeeded: {ok}/{len(items)}")
    return "\n".join(lines), str(batch_dir.resolve())


def run_plot_batch(
    settings: Settings,
    manifest_path: str,
    default_aspect_ratio_label: str = "default",
    *,
    resume_batch: Optional[str] = None,
    retry_failed: bool = False,
    max_retries: int = 0,
    concurrency: int = 1,
    verbose_logging: bool = False,
) -> tuple[str, str]:
    """Run plot batch manifest; returns (log, batch_dir path or error note)."""
    configure_logging(verbose=verbose_logging)
    lines: list[str] = []
    mpath = Path(manifest_path)
    if not mpath.is_file():
        msg = f"Manifest not found: {manifest_path}"
        lines.append(msg)
        return "\n".join(lines), msg

    try:
        items = load_plot_batch_manifest(mpath)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        msg = f"Invalid manifest: {e}"
        lines.append(msg)
        return "\n".join(lines), msg

    is_resume = bool(resume_batch)
    if is_resume:
        resume_ref = Path(resume_batch)
        if resume_ref.is_dir():
            batch_dir = resume_ref.resolve()
            batch_id = batch_dir.name
        else:
            batch_id = resume_batch.strip()
            batch_dir = (Path(settings.output_dir) / batch_id).resolve()
    else:
        batch_id = generate_batch_id()
        batch_dir = Path(settings.output_dir) / batch_id
    ensure_dir(batch_dir)

    settings = settings.model_copy(update={"output_dir": str(batch_dir)})
    lines.append(f"Batch ID: {batch_id}")
    lines.append("Kind: statistical plots")
    lines.append(f"Items: {len(items)}")
    lines.append(f"Output: {batch_dir}")
    lines.append("")

    state = init_or_load_checkpoint(
        batch_dir=batch_dir,
        batch_id=batch_id,
        manifest_path=mpath,
        batch_kind="statistical_plot",
        items=items,
        resume=is_resume,
    )
    planned = select_items_for_run(state, retry_failed=retry_failed)
    if not planned:
        checkpoint_progress(batch_dir=batch_dir, state=state, mark_complete=True)
        lines.append("Nothing to run: all items already completed.")
        lines.append(f"Report written: {batch_dir / 'batch_report.json'}")
        return "\n".join(lines), str(batch_dir.resolve())

    if max_retries < 0:
        max_retries = 0
    if concurrency < 1:
        concurrency = 1

    total_start = time.perf_counter()

    async def _run_all_items() -> None:
        sem = asyncio.Semaphore(concurrency)

        async def _run_one(idx: int, item: dict[str, Any]) -> None:
            item_id = item["id"]
            item_key = item["_item_key"]
            lines.append(f"— Item {idx + 1}/{len(items)} — {item_id}")
            async with sem:
                for attempt in range(max_retries + 1):
                    mark_item_running(state, item_key)
                    checkpoint_progress(batch_dir=batch_dir, state=state)
                    data_path = Path(item["data"])
                    if not data_path.is_file():
                        mark_item_failure(state, item_key, "data file not found")
                        checkpoint_progress(batch_dir=batch_dir, state=state)
                        lines.append(f"  error: data file not found ({data_path})")
                        return
                    try:
                        source_context, raw_data = load_statistical_plot_payload(data_path)
                        ar = item.get("aspect_ratio") or _aspect_ratio_value(
                            default_aspect_ratio_label
                        )
                        gen_in = GenerationInput(
                            source_context=source_context,
                            communicative_intent=item["intent"],
                            diagram_type=DiagramType.STATISTICAL_PLOT,
                            raw_data={"data": raw_data},
                            aspect_ratio=ar,
                        )
                        result = await PaperBananaPipeline(settings=settings).generate(gen_in)
                        mark_item_success(
                            state,
                            item_key,
                            result.metadata.get("run_id"),
                            result.image_path,
                            len(result.iterations),
                        )
                        checkpoint_progress(batch_dir=batch_dir, state=state)
                        lines.append(f"  ok: {result.image_path}")
                        return
                    except Exception as e:
                        mark_item_failure(state, item_key, str(e))
                        checkpoint_progress(batch_dir=batch_dir, state=state)
                        if attempt < max_retries:
                            lines.append(f"  retry {attempt + 1}/{max_retries}: {e}")
                            continue
                        lines.append(f"  error: {e}")
                        return

        await asyncio.gather(*[_run_one(idx, item) for idx, item, _ in planned])

    asyncio.run(_run_all_items())

    total_elapsed = time.perf_counter() - total_start
    report = checkpoint_progress(
        batch_dir=batch_dir,
        state=state,
        total_seconds=total_elapsed,
        mark_complete=True,
    )
    report_path = batch_dir / "batch_report.json"
    lines.append("")
    lines.append(f"Report written: {report_path}")
    ok = sum(1 for x in report["items"] if x.get("output_path"))
    lines.append(f"Succeeded: {ok}/{len(items)}")
    lines.append(f"Total time: {report['total_seconds']}s")
    return "\n".join(lines), str(batch_dir.resolve())


def run_sweep(
    settings: Settings,
    *,
    input_path: str,
    caption: str,
    pdf_pages: Optional[str] = None,
    vlm_providers: str = "",
    vlm_models: str = "",
    image_providers: str = "",
    image_models: str = "",
    iterations: str = "",
    optimize_modes: str = "",
    auto_modes: str = "",
    max_variants: Optional[int] = None,
    dry_run: bool = False,
    verbose_logging: bool = False,
) -> tuple[str, str, str]:
    """Run sweep using core sweep utilities. Returns (log, sweep_dir, report_path)."""
    configure_logging(verbose=verbose_logging)
    lines: list[str] = ["Starting parameter sweep..."]
    input_file = Path(input_path)
    if not input_file.is_file():
        msg = f"Input file not found: {input_path}"
        lines.append(msg)
        return "\n".join(lines), "", ""
    if not caption.strip():
        msg = "Caption is required."
        lines.append(msg)
        return "\n".join(lines), "", ""
    if max_variants is not None and max_variants < 1:
        msg = "max_variants must be >= 1"
        lines.append(msg)
        return "\n".join(lines), "", ""

    try:
        variants = build_sweep_variants(
            vlm_providers=parse_csv_values(vlm_providers),
            vlm_models=parse_csv_values(vlm_models),
            image_providers=parse_csv_values(image_providers),
            image_models=parse_csv_values(image_models),
            refinement_iterations=parse_csv_ints(iterations, field_name="iterations"),
            optimize_inputs=parse_csv_bools(optimize_modes, field_name="optimize_modes"),
            auto_refine=parse_csv_bools(auto_modes, field_name="auto_modes"),
            max_variants=max_variants,
        )
    except ValueError as e:
        lines.append(str(e))
        return "\n".join(lines), "", ""
    if not variants:
        lines.append("Sweep generated zero variants.")
        return "\n".join(lines), "", ""

    try:
        source_context = load_methodology_source(input_file, pdf_pages=pdf_pages)
    except Exception as e:
        lines.append(f"{type(e).__name__}: {e}")
        return "\n".join(lines), "", ""

    sweep_id = f"sweep_{generate_run_id()}"
    sweep_dir = ensure_dir(Path(settings.output_dir) / sweep_id)
    report_path = sweep_dir / "sweep_report.json"
    lines.append(f"Sweep ID: {sweep_id}")
    lines.append(f"Variants: {len(variants)}")
    lines.append(f"Output: {sweep_dir}")

    if dry_run:
        preview = [variant.as_dict() for variant in variants[: min(10, len(variants))]]
        report = {
            "sweep_id": sweep_id,
            "status": "dry_run",
            "input": str(input_file.resolve()),
            "caption": caption,
            "total_variants": len(variants),
            "preview": preview,
        }
        save_json(report, report_path)
        lines.append("Dry run complete.")
        lines.append(f"Report written: {report_path}")
        return "\n".join(lines), str(sweep_dir), str(report_path)

    all_results: list[dict[str, Any]] = []
    total_start = time.perf_counter()
    gen_input = GenerationInput(
        source_context=source_context,
        communicative_intent=caption.strip(),
        diagram_type=DiagramType.METHODOLOGY,
    )

    for idx, variant in enumerate(variants, start=1):
        lines.append(f"Variant {idx}/{len(variants)} — {variant.variant_id}")
        variant_dir = ensure_dir(sweep_dir / variant.variant_id)
        overrides: dict[str, Any] = {
            "output_dir": str(variant_dir),
            "output_format": settings.output_format,
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
        variant_settings = settings.model_copy(update=overrides)
        try:
            variant_start = time.perf_counter()
            result = asyncio.run(PaperBananaPipeline(settings=variant_settings).generate(gen_input))
            variant_seconds = time.perf_counter() - variant_start
            final_critique = result.iterations[-1].critique if result.iterations else None
            suggestion_count = len(final_critique.critic_suggestions) if final_critique else 0
            score = quality_proxy_score(suggestion_count)
            all_results.append(
                {
                    "status": "success",
                    **variant.as_dict(),
                    "run_id": result.metadata.get("run_id"),
                    "output_path": result.image_path,
                    "iterations_used": len(result.iterations),
                    "critic_suggestions": suggestion_count,
                    "quality_proxy_score": round(score, 2),
                    "total_seconds": round(variant_seconds, 2),
                }
            )
            lines.append(f"  ok: score={score:.1f}, {variant_seconds:.1f}s")
        except Exception as e:
            all_results.append(
                {
                    "status": "failed",
                    **variant.as_dict(),
                    "error": str(e),
                }
            )
            lines.append(f"  failed: {e}")

    ranked_results = rank_sweep_results([item for item in all_results if item["status"] == "success"])
    summary = summarize_sweep(all_results)
    report = {
        "sweep_id": sweep_id,
        "status": "completed",
        "input": str(input_file.resolve()),
        "caption": caption,
        "total_seconds": round(time.perf_counter() - total_start, 2),
        "summary": summary,
        "results": all_results,
        "ranked_results": ranked_results,
        "quality_proxy_note": (
            "quality_proxy_score = max(0, 100 - 12.5 * N) where N is critic suggestion "
            "count on the final iteration"
        ),
    }
    save_json(report, report_path)
    lines.append("")
    lines.append(f"Completed: {summary.get('completed', 0)}")
    lines.append(f"Failed: {summary.get('failed', 0)}")
    lines.append(f"Best variant: {summary.get('best_variant')}")
    lines.append(f"Report written: {report_path}")
    return "\n".join(lines), str(sweep_dir), str(report_path)


def run_orchestrate(
    settings: Settings,
    *,
    paper: Optional[str],
    resume_orchestrate: Optional[str],
    data_dir: Optional[str],
    max_method_figures: int,
    max_plot_figures: int,
    pdf_pages: Optional[str],
    dry_run: bool,
    retry_failed: bool,
    max_retries: int,
    concurrency: int,
) -> tuple[str, str]:
    """Run orchestration package workflow. Returns (log, output summary markdown)."""
    lines: list[str] = []

    def _progress(msg: str) -> None:
        lines.append(msg)

    try:
        result = run_orchestration_package(
            paper=paper,
            resume_orchestrate=resume_orchestrate,
            output_dir=Path(settings.output_dir),
            data_dir=data_dir,
            max_method_figures=max_method_figures,
            max_plot_figures=max_plot_figures,
            pdf_pages=pdf_pages,
            dry_run=dry_run,
            config=None,
            vlm_provider=settings.vlm_provider,
            vlm_model=settings.vlm_model,
            image_provider=settings.image_provider,
            image_model=settings.image_model,
            iterations=settings.refinement_iterations,
            auto=settings.auto_refine,
            max_iterations=settings.max_iterations,
            optimize=settings.optimize_inputs,
            format=settings.output_format,
            save_prompts=settings.save_prompts,
            venue=None,
            retry_failed=retry_failed,
            max_retries=max(0, int(max_retries)),
            concurrency=max(1, int(concurrency)),
            progress_callback=_progress,
        )
    except Exception as e:
        lines.append(f"{type(e).__name__}: {e}")
        lines.append(traceback.format_exc())
        return "\n".join(lines), f"FAILED: {e}"

    lines.insert(0, f"Orchestration ID: {result.get('orchestration_id', '?')}")
    lines.insert(1, f"Output: {result.get('orchestrate_dir', '')}")
    if result.get("dry_run"):
        summary = (
            "## Orchestration Dry Run\n\n"
            f"- Orchestration dir: `{result.get('orchestrate_dir', '')}`\n"
            f"- Plan: `{result.get('orchestration_plan_path', '')}`\n"
            f"- Planned methodology figures: {result.get('methodology_items_planned', 0)}\n"
            f"- Planned plot figures: {result.get('plot_items_planned', 0)}\n"
        )
        return "\n".join(lines), summary

    summary_lines = [
        "## Orchestration Complete",
        "",
        f"- Orchestration dir: `{result.get('orchestrate_dir', '')}`",
        f"- Plan: `{result.get('orchestration_plan_path', '')}`",
        f"- Figure package: `{result.get('figure_package_path', '')}`",
        f"- TeX: `{result.get('figures_tex_path', '')}`",
        f"- Captions: `{result.get('captions_md_path', '')}`",
        f"- Generated: {result.get('generated_count', 0)}",
        f"- Failed: {result.get('failed_count', 0)}",
    ]
    failures = result.get("failures") or []
    if failures:
        summary_lines.append("")
        summary_lines.append("### Failures")
        for item in failures[:10]:
            fid = item.get("id") or item.get("item_id") or "unknown"
            ferr = item.get("error") or "unknown error"
            summary_lines.append(f"- `{fid}`: {ferr}")
    return "\n".join(lines), "\n".join(summary_lines)


def _sanitize_output_filename(name: str) -> str:
    """Strip directory components and reject traversal attempts."""
    cleaned = (name or "").strip() or "composite.png"
    base = Path(cleaned).name
    if not base or base in (".", ".."):
        return "composite.png"
    return base


def run_composite(
    image_paths: list[str],
    *,
    output_dir: str,
    layout: str = "auto",
    labels: str = "",
    spacing: int = 20,
    label_position: str = "bottom",
    label_font_size: int = 32,
    output_filename: str = "composite.png",
) -> tuple[str, Optional[str]]:
    """Compose multiple uploaded images into a single labeled multi-panel figure.

    Returns (log, output_path). output_path is None on failure.
    """
    from typing import Literal, cast

    from paperbanana.core.composite import compose_images

    lines: list[str] = ["Starting composite figure generation…"]

    valid_paths = [p for p in image_paths if p and Path(p).is_file()]
    if not valid_paths:
        msg = "No valid image files provided. Upload at least one image."
        lines.append(msg)
        return "\n".join(lines), None

    if label_position not in ("top", "bottom"):
        msg = f"label_position must be 'top' or 'bottom'. Got: {label_position!r}"
        lines.append(msg)
        return "\n".join(lines), None

    if spacing < 0:
        msg = f"spacing must be >= 0. Got: {spacing}"
        lines.append(msg)
        return "\n".join(lines), None

    if label_font_size <= 0:
        msg = f"label_font_size must be > 0. Got: {label_font_size}"
        lines.append(msg)
        return "\n".join(lines), None

    label_list: Optional[list[str]] = None
    auto_label = True
    stripped_labels = labels.strip()
    if stripped_labels:
        if stripped_labels.lower() == "none":
            auto_label = False
        else:
            label_list = [item.strip() for item in labels.split(",") if item.strip()]
            auto_label = False

    out_dir_str = (output_dir or "").strip() or "outputs"
    out_dir = Path(out_dir_str).resolve()
    ensure_dir(out_dir)
    safe_name = _sanitize_output_filename(output_filename)
    output_path = out_dir / safe_name

    lines.append(f"Panels: {len(valid_paths)}")
    lines.append(f"Layout: {layout}")
    lines.append(f"Output: {output_path}")

    try:
        compose_images(
            image_paths=valid_paths,
            layout=layout,
            labels=label_list,
            auto_label=auto_label,
            spacing=spacing,
            label_position=cast(Literal["top", "bottom"], label_position),
            label_font_size=label_font_size,
            output_path=output_path,
        )
    except (ValueError, OSError) as e:
        lines.append("FAILED")
        lines.append(f"{type(e).__name__}: {e}")
        return "\n".join(lines), None
    except Exception as e:
        lines.append("FAILED")
        lines.append(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")
        return "\n".join(lines), None

    lines.append("Done.")
    return "\n".join(lines), str(output_path)
