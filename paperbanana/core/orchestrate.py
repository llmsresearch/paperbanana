"""Paper-level figure orchestration utilities."""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from paperbanana.core.config import Settings
from paperbanana.core.plot_data import load_statistical_plot_payload
from paperbanana.core.source_loader import load_methodology_source
from paperbanana.core.types import DiagramType, GenerationInput

_HEADING_NUMBERED_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.+?)\s*$")
_HEADING_SIMPLE_RE = re.compile(r"^\s*([A-Z][A-Za-z0-9 ,:/()\-]{3,100})\s*$")
_PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s+)?\d+(?:\s*/\s*\d+)?\s*$", re.IGNORECASE)

_METHOD_FIGURE_HINTS: list[tuple[str, str]] = [
    ("overview", "System overview and major processing blocks"),
    ("architecture", "Detailed architecture with key module boundaries"),
    ("method", "Method flow from inputs to outputs"),
    ("pipeline", "Training and inference pipeline with stage dependencies"),
    ("training", "Training procedure and optimization workflow"),
    ("inference", "Inference workflow and serving path"),
    ("experiment", "Experimental setup and evaluation pipeline"),
    ("ablation", "Ablation design and comparison setup"),
]

_PLOT_INTENT_HINTS: list[tuple[str, str]] = [
    ("ablation", "Bar chart comparing ablation variants and performance"),
    ("benchmark", "Grouped bar chart comparing benchmark performance across models"),
    ("leaderboard", "Ranked bar chart showing model leaderboard results"),
    ("result", "Comparative chart summarizing key experiment results"),
    ("latency", "Scatter plot of latency versus quality across variants"),
    ("speed", "Line chart showing runtime trend across settings"),
    ("cost", "Bar chart comparing cost and quality trade-offs"),
]

ORCHESTRATION_CHECKPOINT_FILENAME = "orchestration_checkpoint.json"
ORCHESTRATION_REPORT_FILENAME = "figure_package.json"


def generate_orchestration_id() -> str:
    """Generate a unique orchestration run identifier."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"orchestrate_{ts}_{suffix}"


def load_paper_text(paper_path: Path, *, pdf_pages: str | None = None) -> str:
    """Load paper text from a file path (txt/md/pdf)."""
    return load_methodology_source(Path(paper_path), pdf_pages=pdf_pages)


def extract_paper_title(paper_text: str, fallback_path: Path) -> str:
    """Infer a display title from the paper text."""
    for raw in paper_text.splitlines()[:40]:
        line = raw.strip()
        if not line:
            continue
        if len(line) < 8:
            continue
        if len(line) > 140:
            continue
        if line.lower().startswith(("arxiv", "http://", "https://", "doi:")):
            continue
        return line
    return fallback_path.stem.replace("_", " ").strip() or "Untitled Paper"


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.endswith("."):
        return False
    if len(stripped) < 4 or len(stripped) > 110:
        return False
    if _HEADING_NUMBERED_RE.match(stripped):
        return True
    if _HEADING_SIMPLE_RE.match(stripped):
        words = stripped.split()
        if len(words) > 16:
            return False
        if stripped.lower() in {"abstract", "introduction", "conclusion", "references"}:
            return True
        # Allow title-case / uppercase section-like headings.
        uppercase_ratio = sum(1 for c in stripped if c.isupper()) / max(len(stripped), 1)
        if uppercase_ratio > 0.25:
            return True
        if all(w[:1].isupper() for w in words if w and w[0].isalpha()):
            return True
    return False


def _is_pdf_noise_line(line: str, repeated_count: int) -> bool:
    """Filter common PDF extraction noise like page numbers and running headers."""
    stripped = line.strip()
    if not stripped:
        return True
    if _PAGE_NUMBER_RE.match(stripped):
        return True
    if stripped.lower().startswith("page ") and any(ch.isdigit() for ch in stripped):
        return True
    if repeated_count > 1 and not _HEADING_NUMBERED_RE.match(stripped):
        lowered = stripped.lower()
        if lowered not in {"abstract", "introduction", "conclusion", "references"}:
            return True
    return False


def split_paper_sections(paper_text: str) -> list[dict[str, str]]:
    """Split paper text into section chunks by heading heuristics."""
    lines = paper_text.splitlines()
    counts: dict[str, int] = {}
    for line in lines:
        stripped = line.strip()
        if stripped:
            counts[stripped] = counts.get(stripped, 0) + 1

    headings: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if _is_pdf_noise_line(stripped, counts.get(stripped, 0)):
            continue
        if _looks_like_heading(line):
            heading = stripped
            if headings and headings[-1][1] == heading:
                continue
            headings.append((idx, heading))

    if not headings:
        text = paper_text.strip()
        if not text:
            return []
        return [{"heading": "Paper Content", "content": text}]

    sections: list[dict[str, str]] = []
    for i, (start, heading) in enumerate(headings):
        end = headings[i + 1][0] if i + 1 < len(headings) else len(lines)
        content_lines = []
        for raw in lines[start + 1 : end]:
            stripped = raw.strip()
            if _is_pdf_noise_line(stripped, counts.get(stripped, 0)):
                continue
            content_lines.append(raw)
        content = "\n".join(content_lines).strip()
        if not content:
            continue
        sections.append({"heading": heading, "content": content})

    if not sections:
        return [{"heading": "Paper Content", "content": paper_text.strip()}]
    return sections


def _trim_text(text: str, max_chars: int = 3500) -> str:
    s = (text or "").strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip() + "\n\n[truncated]"


def _best_method_hint(heading: str, content: str) -> str:
    source = f"{heading}\n{content}".lower()
    for key, hint in _METHOD_FIGURE_HINTS:
        if key in source:
            return hint
    return "Method component interaction and information flow"


def _build_method_caption(index: int, heading: str, content: str) -> str:
    hint = _best_method_hint(heading, content)
    title = heading.strip() or f"Method Figure {index}"
    return f"{title}: {hint}."


def plan_methodology_figures(
    *,
    paper_text: str,
    max_figures: int,
) -> list[dict[str, str]]:
    """Plan methodology figure items from paper sections."""
    sections = split_paper_sections(paper_text)
    if not sections:
        return []

    selected: list[dict[str, str]] = []
    for section in sections:
        if len(selected) >= max_figures:
            break
        heading = section["heading"]
        content = section["content"]
        caption = _build_method_caption(len(selected) + 1, heading, content)
        context = f"Section: {heading}\n\n{_trim_text(content)}"
        selected.append(
            {
                "id": f"method_{len(selected) + 1:02d}",
                "heading": heading,
                "caption": caption,
                "context": context,
                "label": f"fig:method_{len(selected) + 1:02d}",
            }
        )

    return selected


def _guess_plot_intent(path: Path) -> str:
    name = path.stem.replace("_", " ").replace("-", " ").strip().lower()
    for key, intent in _PLOT_INTENT_HINTS:
        if key in name:
            return f"{intent} from {path.stem}."
    return f"Comparative chart highlighting key metrics from {path.stem}."


def discover_plot_data_files(data_dir: Path) -> list[Path]:
    """Find candidate CSV/JSON files for plot generation."""
    root = Path(data_dir)
    if not root.exists() or not root.is_dir():
        return []
    discovered: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in (".csv", ".json"):
            continue
        # Avoid loading generated report/checkpoint files.
        if path.name in {"batch_report.json", "batch_checkpoint.json", "metadata.json"}:
            continue
        discovered.append(path.resolve())
    discovered.sort(key=lambda p: str(p))
    return discovered


def plan_plot_figures(*, data_dir: Path | None, max_figures: int) -> list[dict[str, str]]:
    """Plan plot figure items from discovered data files."""
    if data_dir is None:
        return []
    files = discover_plot_data_files(data_dir)
    if not files:
        return []
    selected = files[:max_figures]
    items: list[dict[str, str]] = []
    for idx, path in enumerate(selected, start=1):
        items.append(
            {
                "id": f"plot_{idx:02d}",
                "data": str(path),
                "intent": _guess_plot_intent(path),
                "label": f"fig:plot_{idx:02d}",
            }
        )
    return items


def build_orchestration_plan(
    *,
    paper_path: Path,
    paper_text: str,
    data_dir: Path | None,
    max_method_figures: int,
    max_plot_figures: int,
) -> dict[str, Any]:
    """Build a complete figure-package plan for orchestration."""
    title = extract_paper_title(paper_text, paper_path)
    method_items = plan_methodology_figures(paper_text=paper_text, max_figures=max_method_figures)
    plot_items = plan_plot_figures(data_dir=data_dir, max_figures=max_plot_figures)
    return {
        "paper_title": title,
        "paper_path": str(Path(paper_path).resolve()),
        "methodology_items": method_items,
        "plot_items": plot_items,
    }


def prepare_orchestration_plan(
    *,
    paper: str | None,
    resume_orchestrate: str | None,
    output_dir: str,
    data_dir: str | None,
    max_method_figures: int,
    max_plot_figures: int,
    pdf_pages: str | None,
) -> tuple[str, Path, dict[str, Any], Path, bool]:
    """Resolve directories and load/create orchestration plan."""
    is_resume = bool(resume_orchestrate)
    if is_resume:
        resume_ref = Path(str(resume_orchestrate))
        if resume_ref.is_dir():
            orchestrate_dir = resume_ref.resolve()
            orchestration_id = orchestrate_dir.name
        else:
            orchestration_id = str(resume_orchestrate).strip()
            orchestrate_dir = (Path(output_dir) / orchestration_id).resolve()
        plan_path = orchestrate_dir / "orchestration_plan.json"
        if not plan_path.exists():
            raise FileNotFoundError(f"orchestration plan not found: {plan_path}")
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise ValueError(f"Error loading orchestration plan: {e}") from e
        return orchestration_id, orchestrate_dir, plan, plan_path, True

    paper_path = Path(str(paper))
    if not paper_path.exists():
        raise FileNotFoundError(f"Paper file not found: {paper}")
    if pdf_pages and paper_path.suffix.lower() != ".pdf":
        raise ValueError("--pdf-pages can only be used with PDF papers")

    data_root = Path(data_dir).resolve() if data_dir else None
    if data_root is not None and not data_root.exists():
        raise FileNotFoundError(f"Data directory not found: {data_root}")
    if data_root is not None and not data_root.is_dir():
        raise ValueError(f"--data-dir must be a directory: {data_root}")

    orchestration_id = generate_orchestration_id()
    orchestrate_dir = Path(output_dir).resolve() / orchestration_id
    contexts_dir = orchestrate_dir / "contexts"
    contexts_dir.mkdir(parents=True, exist_ok=True)
    paper_text = load_paper_text(paper_path, pdf_pages=pdf_pages)
    plan = build_orchestration_plan(
        paper_path=paper_path,
        paper_text=paper_text,
        data_dir=data_root,
        max_method_figures=max_method_figures,
        max_plot_figures=max_plot_figures,
    )
    for item in plan["methodology_items"]:
        context_path = contexts_dir / f"{item['id']}.txt"
        context_path.write_text(item["context"], encoding="utf-8")
        item["context_path"] = str(context_path)

    plan_path = orchestrate_dir / "orchestration_plan.json"
    _atomic_json_write(plan_path, plan)
    return orchestration_id, orchestrate_dir, plan, plan_path, False


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _task_key(task: dict[str, Any]) -> str:
    return f"{task.get('kind', 'unknown')}::{task.get('id', 'unknown')}"


def flatten_plan_tasks(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert an orchestration plan into normalized task entries."""
    tasks: list[dict[str, Any]] = []
    for item in plan.get("methodology_items", []):
        entry = {
            "kind": "methodology",
            "id": item.get("id"),
            "caption": item.get("caption"),
            "label": item.get("label"),
            "context": item.get("context"),
            "context_path": item.get("context_path"),
        }
        entry["_task_key"] = _task_key(entry)
        tasks.append(entry)
    for item in plan.get("plot_items", []):
        entry = {
            "kind": "plot",
            "id": item.get("id"),
            "intent": item.get("intent"),
            "label": item.get("label"),
            "data": item.get("data"),
        }
        entry["_task_key"] = _task_key(entry)
        tasks.append(entry)
    return tasks


def init_or_load_orchestration_checkpoint(
    *,
    orchestrate_dir: Path,
    orchestration_id: str,
    plan_path: Path,
    plan: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    """Create or load orchestration checkpoint state."""
    cp_path = Path(orchestrate_dir) / ORCHESTRATION_CHECKPOINT_FILENAME
    tasks = flatten_plan_tasks(plan)
    if resume:
        if not cp_path.exists():
            raise FileNotFoundError(f"No {ORCHESTRATION_CHECKPOINT_FILENAME} in {orchestrate_dir}")
        state = json.loads(cp_path.read_text(encoding="utf-8"))
        prev_keys = [x.get("_task_key") for x in state.get("plan_tasks", [])]
        now_keys = [x.get("_task_key") for x in tasks]
        if prev_keys != now_keys:
            raise ValueError("Plan tasks do not match checkpoint. Refusing resume.")
        return state

    state: dict[str, Any] = {
        "orchestration_id": orchestration_id,
        "status": "running",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "plan_path": str(Path(plan_path).resolve()),
        "paper_title": plan.get("paper_title", ""),
        "paper_path": plan.get("paper_path", ""),
        "planned_methodology_items": len(plan.get("methodology_items", [])),
        "planned_plot_items": len(plan.get("plot_items", [])),
        "plan_tasks": tasks,
        "items": {},
    }
    for task in tasks:
        task_key = task["_task_key"]
        state["items"][task_key] = {
            "id": task.get("id"),
            "kind": task.get("kind"),
            "caption": task.get("caption") or task.get("intent") or "",
            "label": task.get("label") or f"fig:{task.get('id')}",
            "status": "pending",
            "attempts": 0,
            "run_id": None,
            "source_output": None,
            "relative_path": None,
            "absolute_path": None,
            "error": None,
            "errors": [],
            "started_at": None,
            "finished_at": None,
        }
    _atomic_json_write(cp_path, state)
    checkpoint_orchestration_progress(orchestrate_dir=orchestrate_dir, state=state)
    return state


def select_orchestration_tasks(
    state: dict[str, Any], *, retry_failed: bool = False
) -> list[tuple[int, dict[str, Any], dict[str, Any]]]:
    """Return tasks selected for execution."""
    selected: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    tasks = state.get("plan_tasks", [])
    task_states = state.get("items", {})
    for idx, task in enumerate(tasks):
        task_state = task_states.get(task["_task_key"], {})
        status = task_state.get("status")
        if status in ("pending", "running"):
            selected.append((idx, task, task_state))
        elif retry_failed and status == "failed":
            selected.append((idx, task, task_state))
    return selected


def mark_orchestration_item_running(state: dict[str, Any], task_key: str) -> None:
    item = state["items"][task_key]
    item["status"] = "running"
    item["attempts"] = int(item.get("attempts") or 0) + 1
    item["started_at"] = _utc_now()
    item["finished_at"] = None
    state["updated_at"] = _utc_now()


def mark_orchestration_item_success(
    state: dict[str, Any],
    task_key: str,
    *,
    run_id: str | None,
    source_output: str,
    relative_path: str,
    absolute_path: str,
) -> None:
    item = state["items"][task_key]
    item["status"] = "success"
    item["run_id"] = run_id
    item["source_output"] = source_output
    item["relative_path"] = relative_path
    item["absolute_path"] = absolute_path
    item["error"] = None
    item["finished_at"] = _utc_now()
    state["updated_at"] = _utc_now()


def mark_orchestration_item_failure(state: dict[str, Any], task_key: str, error: str) -> None:
    item = state["items"][task_key]
    item["status"] = "failed"
    item["error"] = error
    item.setdefault("errors", []).append({"at": _utc_now(), "error": error})
    item["finished_at"] = _utc_now()
    state["updated_at"] = _utc_now()


def checkpoint_orchestration_progress(
    *,
    orchestrate_dir: Path,
    state: dict[str, Any],
    total_seconds: float | None = None,
    mark_complete: bool = False,
) -> dict[str, Any]:
    """Persist orchestration checkpoint and synchronized package report."""
    cp_path = Path(orchestrate_dir) / ORCHESTRATION_CHECKPOINT_FILENAME
    report_path = Path(orchestrate_dir) / ORCHESTRATION_REPORT_FILENAME
    if mark_complete:
        state["status"] = "completed"
    if total_seconds is not None:
        state["total_seconds"] = round(float(total_seconds), 1)
    state["updated_at"] = _utc_now()
    _atomic_json_write(cp_path, state)

    generated_items: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    for task in state.get("plan_tasks", []):
        task_key = task.get("_task_key")
        item = state.get("items", {}).get(task_key, {})
        status = item.get("status")
        if status == "success":
            generated_items.append(
                {
                    "id": str(item.get("id") or task.get("id")),
                    "kind": str(item.get("kind") or task.get("kind")),
                    "caption": str(item.get("caption") or ""),
                    "label": str(item.get("label") or ""),
                    "run_id": str(item.get("run_id") or ""),
                    "source_output": str(item.get("source_output") or ""),
                    "relative_path": str(item.get("relative_path") or ""),
                    "absolute_path": str(item.get("absolute_path") or ""),
                }
            )
        elif status == "failed":
            failures.append(
                {
                    "id": str(item.get("id") or task.get("id")),
                    "kind": str(item.get("kind") or task.get("kind")),
                    "error": str(item.get("error") or "unknown"),
                }
            )

    generated_items.sort(key=lambda x: x["id"])
    report = {
        "orchestration_id": state.get("orchestration_id"),
        "status": state.get("status", "running"),
        "paper_title": state.get("paper_title"),
        "paper_path": state.get("paper_path"),
        "planned_methodology_items": state.get("planned_methodology_items", 0),
        "planned_plot_items": state.get("planned_plot_items", 0),
        "generated_items": generated_items,
        "failures": failures,
        "total_seconds": round(float(state.get("total_seconds") or 0.0), 1),
    }
    _atomic_json_write(report_path, report)
    return report


def render_orchestration_sidecars(orchestrate_dir: Path, report: dict[str, Any]) -> None:
    """Render human-facing package sidecar files from a report."""
    write_latex_figure_snippets(
        output_path=Path(orchestrate_dir) / "figures.tex",
        title=str(report.get("paper_title") or ""),
        generated_items=report.get("generated_items", []),
    )
    write_caption_sheet(
        output_path=Path(orchestrate_dir) / "captions.md",
        title=str(report.get("paper_title") or ""),
        generated_items=report.get("generated_items", []),
    )


async def _run_orchestration_async(
    *,
    state: dict[str, Any],
    settings: Settings,
    orchestrate_dir: Path,
    package_assets_dir: Path,
    planned: list[tuple[int, dict[str, Any], dict[str, Any]]],
    total_items: int,
    max_retries: int,
    concurrency: int,
    previous_seconds: float,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    from paperbanana.core.pipeline import PaperBananaPipeline

    ext = "jpg" if settings.output_format == "jpeg" else settings.output_format
    run_start = time.perf_counter()
    checkpoint_lock = asyncio.Lock()
    sem = asyncio.Semaphore(concurrency)

    def emit(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    async def _checkpoint() -> None:
        checkpoint_orchestration_progress(
            orchestrate_dir=orchestrate_dir,
            state=state,
            total_seconds=previous_seconds + (time.perf_counter() - run_start),
        )

    async def _run_one(task_index: int, task: dict[str, Any]) -> None:
        item = dict(task)
        kind = str(item["kind"])
        item_id = str(item["id"])
        task_key = str(item["_task_key"])
        async with sem:
            for attempt in range(max_retries + 1):
                async with checkpoint_lock:
                    mark_orchestration_item_running(state, task_key)
                    await _checkpoint()

                try:
                    pipeline = PaperBananaPipeline(settings=settings)
                    if kind == "methodology":
                        source_context = str(item.get("context") or "")
                        if not source_context:
                            context_path = Path(str(item.get("context_path") or ""))
                            if not context_path.is_file():
                                raise FileNotFoundError(
                                    f"Context file not found for {item_id}: {context_path}"
                                )
                            source_context = context_path.read_text(encoding="utf-8")
                        gen_input = GenerationInput(
                            source_context=source_context,
                            communicative_intent=str(item["caption"]),
                            diagram_type=DiagramType.METHODOLOGY,
                        )
                    else:
                        data_path = Path(str(item["data"]))
                        if not data_path.is_file():
                            raise FileNotFoundError(f"Data file not found: {data_path}")
                        source_context, raw_data = load_statistical_plot_payload(data_path)
                        gen_input = GenerationInput(
                            source_context=source_context,
                            communicative_intent=str(item["intent"]),
                            diagram_type=DiagramType.STATISTICAL_PLOT,
                            raw_data={"data": raw_data},
                        )
                    result = await pipeline.generate(gen_input)
                    final_path = Path(result.image_path)
                    if not final_path.exists():
                        raise RuntimeError("Pipeline returned no final output image")
                    output_name = f"{item_id}.{ext}"
                    packaged_path = package_assets_dir / output_name
                    shutil.copy2(final_path, packaged_path)
                    relative_path = f"figures/{output_name}"

                    async with checkpoint_lock:
                        mark_orchestration_item_success(
                            state,
                            task_key,
                            run_id=str(result.metadata.get("run_id") or pipeline.run_id),
                            source_output=str(final_path),
                            relative_path=relative_path,
                            absolute_path=str(packaged_path),
                        )
                        await _checkpoint()
                    emit(
                        f"[green]{task_index + 1}/{total_items} {item_id}: ok[/green] "
                        f"[dim]{packaged_path}[/dim]"
                    )
                    return
                except Exception as e:
                    async with checkpoint_lock:
                        # We checkpoint each failed attempt before retrying so interrupted runs
                        # preserve the latest error, even if a subsequent retry has not started yet.
                        mark_orchestration_item_failure(state, task_key, str(e))
                        await _checkpoint()
                    if attempt < max_retries:
                        emit(
                            f"[yellow]{task_index + 1}/{total_items} {item_id}: retry "
                            f"{attempt + 1}/{max_retries} after {e}[/yellow]"
                        )
                        continue
                    emit(f"[red]{task_index + 1}/{total_items} {item_id}: failed - {e}[/red]")
                    return

    await asyncio.gather(*[_run_one(idx, task) for idx, task, _ in planned])
    total_seconds = previous_seconds + (time.perf_counter() - run_start)
    return checkpoint_orchestration_progress(
        orchestrate_dir=orchestrate_dir,
        state=state,
        total_seconds=total_seconds,
        mark_complete=True,
    )


def run_orchestration(
    *,
    state: dict[str, Any],
    plan: dict[str, Any],
    settings: Settings,
    orchestrate_dir: Path,
    retry_failed: bool,
    max_retries: int,
    concurrency: int,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], bool]:
    """Execute orchestration tasks and return (report, had_work)."""
    package_assets_dir = Path(orchestrate_dir) / "figures"
    package_assets_dir.mkdir(parents=True, exist_ok=True)
    planned = select_orchestration_tasks(state, retry_failed=retry_failed)
    if not planned:
        report = checkpoint_orchestration_progress(
            orchestrate_dir=orchestrate_dir,
            state=state,
            total_seconds=float(state.get("total_seconds") or 0.0),
            mark_complete=True,
        )
        render_orchestration_sidecars(orchestrate_dir, report)
        return report, False

    report = asyncio.run(
        _run_orchestration_async(
            state=state,
            settings=settings,
            orchestrate_dir=orchestrate_dir,
            package_assets_dir=package_assets_dir,
            planned=planned,
            total_items=len(flatten_plan_tasks(plan)),
            max_retries=max_retries,
            concurrency=concurrency,
            previous_seconds=float(state.get("total_seconds") or 0.0),
            progress_callback=progress_callback,
        )
    )
    render_orchestration_sidecars(orchestrate_dir, report)
    return report, True


def write_latex_figure_snippets(
    *,
    output_path: Path,
    title: str,
    generated_items: list[dict[str, str]],
) -> Path:
    """Write LaTeX figure snippets for generated package items."""
    lines: list[str] = [
        "% Auto-generated by PaperBanana orchestrate",
        f"% Paper: {title}",
        "",
    ]
    for item in generated_items:
        rel_path = item.get("relative_path", "")
        caption = item.get("caption", "").strip()
        label = item.get("label", "").strip()
        lines.extend(
            [
                r"\begin{figure}[t]",
                r"  \centering",
                f"  \\includegraphics[width=\\linewidth]{{{rel_path}}}",
                f"  \\caption{{{caption}}}",
                f"  \\label{{{label}}}",
                r"\end{figure}",
                "",
            ]
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output


def write_caption_sheet(
    *,
    output_path: Path,
    title: str,
    generated_items: list[dict[str, str]],
) -> Path:
    """Write a markdown caption/reference sheet for generated figures."""
    lines = [f"# Figure Package for {title}", ""]
    for item in generated_items:
        lines.extend(
            [
                f"## {item.get('id', 'figure')}",
                f"- Caption: {item.get('caption', '')}",
                f"- Label: `{item.get('label', '')}`",
                f"- Asset: `{item.get('relative_path', '')}`",
                "",
            ]
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output
