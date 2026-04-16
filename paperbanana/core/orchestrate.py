"""Paper-level figure orchestration utilities."""

from __future__ import annotations

import datetime
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from paperbanana.core.source_loader import load_methodology_source

_HEADING_NUMBERED_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.+?)\s*$")
_HEADING_SIMPLE_RE = re.compile(r"^\s*([A-Z][A-Za-z0-9 ,:/()\-]{3,100})\s*$")

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


def split_paper_sections(paper_text: str) -> list[dict[str, str]]:
    """Split paper text into section chunks by heading heuristics."""
    lines = paper_text.splitlines()
    headings: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        if _looks_like_heading(line):
            heading = line.strip()
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
        content = "\n".join(lines[start + 1 : end]).strip()
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


def write_latex_figure_snippets(
    *,
    output_path: Path,
    title: str,
    generated_items: list[dict[str, str]],
) -> Path:
    """Write LaTeX figure snippets for generated package items."""
    lines: list[str] = [
        f"% Auto-generated by PaperBanana orchestrate",
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
