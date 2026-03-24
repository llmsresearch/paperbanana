"""Async pipeline runners with progress text for the Studio UI."""

from __future__ import annotations

import asyncio
import json
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

from paperbanana.core.batch import generate_batch_id, load_batch_manifest
from paperbanana.core.config import Settings
from paperbanana.core.logging import configure_logging
from paperbanana.core.pipeline import PaperBananaPipeline
from paperbanana.core.resume import load_resume_state
from paperbanana.core.types import (
    DiagramType,
    GenerationInput,
    PipelineProgressEvent,
    PipelineProgressStage,
)
from paperbanana.core.utils import ensure_dir, find_prompt_dir, save_json
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
    verbose_logging: bool = False,
) -> tuple[str, Optional[str], list[tuple[str, str]], str]:
    """Run methodology diagram generation. Returns (log, final_path, gallery, error)."""
    configure_logging(verbose=verbose_logging)
    log = ProgressLog()
    log.append("Starting methodology diagram pipeline…")
    err = ""
    try:
        gen_in = GenerationInput(
            source_context=source_context,
            communicative_intent=caption.strip(),
            diagram_type=DiagramType.METHODOLOGY,
            aspect_ratio=_aspect_ratio_value(aspect_ratio_label),
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
        if path.suffix.lower() == ".csv":
            import pandas as pd

            df = pd.read_csv(path)
            raw_data = df.to_dict(orient="records")
            source_context = (
                f"CSV columns: {list(df.columns)}\nRows: {len(df)}\n\n"
                f"Sample:\n{df.head().to_string()}"
            )
        else:
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw_data = raw if isinstance(raw, list) else raw.get("data", raw)
            source_context = f"JSON data:\n{json.dumps(raw, indent=2)[:8000]}"

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
    verbose_logging: bool = False,
) -> tuple[str, str]:
    """VLM judge comparative evaluation. Returns (log, formatted results)."""
    configure_logging(verbose=verbose_logging)
    lines: list[str] = ["Starting comparative evaluation (VLM judge)…"]
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
    if not source_context.strip():
        msg = "Source context is empty."
        lines.append(msg)
        return "\n".join(lines), msg

    try:
        vlm = ProviderRegistry.create_vlm(settings)
        judge = VLMJudge(vlm, prompt_dir=find_prompt_dir())

        async def _go():
            return await judge.evaluate(
                image_path=str(gp),
                source_context=source_context,
                caption=caption.strip(),
                reference_path=str(rp),
            )

        scores = asyncio.run(_go())
        lines.append("Done.")
        dims = ["faithfulness", "conciseness", "readability", "aesthetics"]
        out_parts = ["## Results\n"]
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

    batch_id = generate_batch_id()
    batch_dir = Path(settings.output_dir) / batch_id
    ensure_dir(batch_dir)

    settings = settings.model_copy(update={"output_dir": str(batch_dir)})
    lines.append(f"Batch ID: {batch_id}")
    lines.append(f"Items: {len(items)}")
    lines.append(f"Output: {batch_dir}")
    lines.append("")

    report: dict[str, Any] = {"batch_id": batch_id, "manifest": str(mpath.resolve()), "items": []}

    async def _run_all_items() -> None:
        for idx, item in enumerate(items):
            item_id = item["id"]
            input_path = Path(item["input"])
            lines.append(f"— Item {idx + 1}/{len(items)} — {item_id}")
            if not input_path.is_file():
                lines.append(f"  skip: input not found ({input_path})")
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
            source_context = input_path.read_text(encoding="utf-8", errors="replace")
            gen_in = GenerationInput(
                source_context=source_context,
                communicative_intent=item["caption"],
                diagram_type=DiagramType.METHODOLOGY,
            )
            pipeline = PaperBananaPipeline(settings=settings)
            try:
                result = await pipeline.generate(gen_in)
                lines.append(f"  ok: {result.image_path}")
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
            except Exception as e:
                lines.append(f"  error: {e}")
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

    asyncio.run(_run_all_items())

    report_path = batch_dir / "batch_report.json"
    save_json(report, report_path)
    lines.append("")
    lines.append(f"Report written: {report_path}")
    ok = sum(1 for x in report["items"] if x.get("output_path"))
    lines.append(f"Succeeded: {ok}/{len(items)}")
    return "\n".join(lines), str(batch_dir.resolve())
