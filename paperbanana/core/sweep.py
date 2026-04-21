"""Variant sweep planning and result summarization utilities."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Literal

import structlog

logger = structlog.get_logger()

SWEEP_REPORT_FILENAME = "sweep_report.json"

# Heuristic used to rank successful variants in CLI sweep reports (not a human-judgment score).
QUALITY_PROXY_MAX = 100.0
QUALITY_PROXY_PENALTY_PER_SUGGESTION = 12.5


def quality_proxy_score(suggestion_count: int) -> float:
    """Map final-iteration critic suggestion count to a rough ranking score."""
    return max(
        0.0,
        QUALITY_PROXY_MAX - QUALITY_PROXY_PENALTY_PER_SUGGESTION * float(suggestion_count),
    )


@dataclass(frozen=True)
class SweepVariant:
    """Single sweep variant definition."""

    variant_id: str
    vlm_provider: str
    vlm_model: str | None
    image_provider: str
    image_model: str | None
    refinement_iterations: int
    optimize_inputs: bool
    auto_refine: bool

    def as_dict(self) -> dict[str, Any]:
        """Serialize variant for report output."""
        return {
            "variant_id": self.variant_id,
            "vlm_provider": self.vlm_provider,
            "vlm_model": self.vlm_model,
            "image_provider": self.image_provider,
            "image_model": self.image_model,
            "refinement_iterations": self.refinement_iterations,
            "optimize_inputs": self.optimize_inputs,
            "auto_refine": self.auto_refine,
        }


def parse_csv_values(raw: str | None) -> list[str]:
    """Parse comma-separated values into a normalized list."""
    if raw is None:
        return []
    values = []
    for token in raw.split(","):
        item = token.strip()
        if item:
            values.append(item)
    return values


def parse_csv_ints(raw: str | None, *, field_name: str) -> list[int]:
    """Parse comma-separated integer list with validation."""
    values = parse_csv_values(raw)
    if not values:
        return []

    parsed: list[int] = []
    for token in values:
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"{field_name} must contain integers. Got: '{token}'") from exc
        if value < 1:
            raise ValueError(f"{field_name} values must be >= 1. Got: {value}")
        parsed.append(value)
    return parsed


def parse_csv_bools(raw: str | None, *, field_name: str) -> list[bool]:
    """Parse comma-separated booleans (on/off, true/false, 1/0)."""
    values = parse_csv_values(raw)
    if not values:
        return []

    allowed_true = {"1", "true", "t", "yes", "y", "on"}
    allowed_false = {"0", "false", "f", "no", "n", "off"}
    parsed: list[bool] = []
    for token in values:
        lowered = token.lower()
        if lowered in allowed_true:
            parsed.append(True)
            continue
        if lowered in allowed_false:
            parsed.append(False)
            continue
        raise ValueError(f"{field_name} must use booleans (on/off,true/false,1/0). Got: '{token}'")
    return parsed


def build_sweep_variants(
    *,
    vlm_providers: list[str],
    vlm_models: list[str],
    image_providers: list[str],
    image_models: list[str],
    refinement_iterations: list[int],
    optimize_inputs: list[bool],
    auto_refine: list[bool],
    max_variants: int | None = None,
) -> list[SweepVariant]:
    """Build cartesian sweep variants with optional truncation."""
    axes = {
        "vlm_provider": vlm_providers or ["gemini"],
        "vlm_model": vlm_models or [None],
        "image_provider": image_providers or ["google_imagen"],
        "image_model": image_models or [None],
        "refinement_iterations": refinement_iterations or [3],
        "optimize_inputs": optimize_inputs or [False],
        "auto_refine": auto_refine or [False],
    }

    axis_names = list(axes.keys())
    axis_values = [axes[name] for name in axis_names]
    variants: list[SweepVariant] = []

    for index, combo in enumerate(itertools.product(*axis_values), start=1):
        data = dict(zip(axis_names, combo))
        variants.append(
            SweepVariant(
                variant_id=f"variant_{index:03d}",
                vlm_provider=str(data["vlm_provider"]),
                vlm_model=(str(data["vlm_model"]) if data["vlm_model"] is not None else None),
                image_provider=str(data["image_provider"]),
                image_model=(str(data["image_model"]) if data["image_model"] is not None else None),
                refinement_iterations=int(data["refinement_iterations"]),
                optimize_inputs=bool(data["optimize_inputs"]),
                auto_refine=bool(data["auto_refine"]),
            )
        )
        if max_variants is not None and len(variants) >= max_variants:
            break

    return variants


def rank_sweep_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return results ordered by proxy score then runtime."""

    def _sort_key(item: dict[str, Any]) -> tuple[float, float]:
        score = float(item.get("quality_proxy_score", 0.0))
        runtime = float(item.get("total_seconds", 10**9))
        return (-score, runtime)

    return sorted(results, key=_sort_key)


def summarize_sweep(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute sweep-level summary statistics."""
    if not results:
        return {
            "completed": 0,
            "failed": 0,
            "best_variant": None,
            "best_quality_proxy_score": None,
            "mean_quality_proxy_score": None,
            "mean_total_seconds": None,
        }

    completed = [item for item in results if item.get("status") == "success"]
    failed = [item for item in results if item.get("status") != "success"]
    ranked = rank_sweep_results(completed)
    best = ranked[0] if ranked else None

    return {
        "completed": len(completed),
        "failed": len(failed),
        "best_variant": best["variant_id"] if best else None,
        "best_quality_proxy_score": best.get("quality_proxy_score") if best else None,
        "mean_quality_proxy_score": (
            round(mean(float(item.get("quality_proxy_score", 0.0)) for item in completed), 2)
            if completed
            else None
        ),
        "mean_total_seconds": (
            round(mean(float(item.get("total_seconds", 0.0)) for item in completed), 2)
            if completed
            else None
        ),
    }


# ── Report rendering ────────────────────────────────────────────────


def load_sweep_report(sweep_dir: Path) -> dict[str, Any]:
    """Load sweep_report.json from a sweep output directory.

    Args:
        sweep_dir: Path to the sweep run directory (e.g. outputs/sweep_20250109_123456_abc).

    Returns:
        The report dict (sweep_id, status, results or preview, etc.).

    Raises:
        FileNotFoundError: If sweep_dir or sweep_report.json does not exist.
        ValueError: If the JSON is invalid or missing required keys.
    """
    sweep_dir = Path(sweep_dir).resolve()
    report_path = sweep_dir / SWEEP_REPORT_FILENAME
    if not sweep_dir.exists() or not sweep_dir.is_dir():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_dir}")
    if not report_path.exists():
        raise FileNotFoundError(f"No {SWEEP_REPORT_FILENAME} in {sweep_dir}. Run a sweep first.")
    raw = report_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict) or "sweep_id" not in data:
        raise ValueError(f"Invalid report: expected dict with 'sweep_id'. Got: {type(data)}")
    status = data.get("status")
    if status == "dry_run":
        if "preview" not in data:
            raise ValueError("Dry-run report missing 'preview' key")
    elif "results" not in data:
        raise ValueError("Completed sweep report missing 'results' key")
    return data


def _provider_cell(item: dict[str, Any], which: Literal["vlm", "image"]) -> str:
    """Format a provider/model pair into a short display string."""
    provider = item.get(f"{which}_provider") or "—"
    model = item.get(f"{which}_model")
    return f"{provider} / {model}" if model else str(provider)


_MD_PIPE_ESCAPE = "\\|"


def _md_escape(value: Any) -> str:
    """Escape pipe characters for Markdown table cells."""
    return str(value).replace("|", _MD_PIPE_ESCAPE)


def _relative_output(out: str, sweep_dir: Path) -> str:
    """Convert an absolute output_path to a sweep-dir-relative path when possible."""
    if not out:
        return ""
    p = Path(out)
    if not p.is_absolute():
        return out
    try:
        return p.relative_to(sweep_dir).as_posix()
    except ValueError:
        return out


def generate_sweep_report_md(report: dict[str, Any], sweep_dir: Path) -> str:
    """Generate a Markdown report from a sweep report dict."""
    sweep_dir = Path(sweep_dir).resolve()
    sweep_id = report.get("sweep_id", "sweep")
    status = report.get("status", "completed")
    caption = report.get("caption", "")
    input_path = report.get("input", "")

    lines = [f"# Sweep Report: {sweep_id}", ""]
    if input_path:
        lines.append(f"- **Input:** `{input_path}`")
    if caption:
        lines.append(f"- **Caption:** {caption}")
    lines.append(f"- **Status:** {status}")

    if status == "dry_run":
        total = report.get("total_variants", len(report.get("preview", [])))
        lines.extend(
            [
                f"- **Planned variants:** {total}",
                "",
                "## Planned Variants (preview)",
                "",
                "| Variant | VLM | Image | Iters | Optimize | Auto-refine |",
                "|---------|-----|-------|-------|----------|-------------|",
            ]
        )
        for item in report.get("preview", []):
            vlm = _md_escape(_provider_cell(item, "vlm"))
            img = _md_escape(_provider_cell(item, "image"))
            lines.append(
                f"| {item.get('variant_id', '—')} "
                f"| {vlm} "
                f"| {img} "
                f"| {item.get('refinement_iterations', '—')} "
                f"| {item.get('optimize_inputs', '—')} "
                f"| {item.get('auto_refine', '—')} |"
            )
        return "\n".join(lines)

    summary = report.get("summary") or {}
    total_seconds = float(report.get("total_seconds") or 0.0)
    best_score = summary.get("best_quality_proxy_score")
    mean_score = summary.get("mean_quality_proxy_score")
    lines.extend(
        [
            f"- **Completed:** {summary.get('completed', 0)}",
            f"- **Failed:** {summary.get('failed', 0)}",
            f"- **Best variant:** {summary.get('best_variant') or '—'}",
            f"- **Best score:** {best_score if best_score is not None else '—'}",
            f"- **Mean score:** {mean_score if mean_score is not None else '—'}",
            f"- **Total seconds:** {total_seconds:.1f}",
        ]
    )

    ranked = report.get("ranked_results") or []
    top_n = ranked[: min(5, len(ranked))]
    if top_n:
        lines.extend(
            [
                "",
                "## Top Variants (ranked)",
                "",
                "| Rank | Variant | VLM | Image | Iters | Suggestions | Score | Seconds |",
                "|------|---------|-----|-------|-------|-------------|-------|---------|",
            ]
        )
        for rank, item in enumerate(top_n, start=1):
            vlm = _md_escape(_provider_cell(item, "vlm"))
            img = _md_escape(_provider_cell(item, "image"))
            lines.append(
                f"| {rank} "
                f"| {item.get('variant_id', '—')} "
                f"| {vlm} "
                f"| {img} "
                f"| {item.get('iterations_used', '—')} "
                f"| {item.get('critic_suggestions', '—')} "
                f"| {item.get('quality_proxy_score', '—')} "
                f"| {item.get('total_seconds', '—')} |"
            )

    header = (
        "| Variant | VLM | Image | Status | Iters | Suggestions | "
        "Score | Seconds | Output / Error |"
    )
    divider = (
        "|---------|-----|-------|--------|-------|-------------|"
        "-------|---------|----------------|"
    )
    lines.extend(["", "## All Variants", "", header, divider])
    for item in report.get("results", []):
        vid = item.get("variant_id", "—")
        vlm = _md_escape(_provider_cell(item, "vlm"))
        img = _md_escape(_provider_cell(item, "image"))
        if item.get("status") == "success":
            status_cell = "✓ Success"
            iters = item.get("iterations_used", "—")
            suggestions = item.get("critic_suggestions", "—")
            score = item.get("quality_proxy_score", "—")
            seconds = item.get("total_seconds", "—")
            out = _relative_output(item.get("output_path") or "", sweep_dir)
            out_cell = f"`{_md_escape(out)}`" if out else "—"
            lines.append(
                f"| {vid} | {vlm} | {img} | {status_cell} | {iters} "
                f"| {suggestions} | {score} | {seconds} | {out_cell} |"
            )
        else:
            status_cell = "✗ Failed"
            err = _md_escape(item.get("error") or "unknown")[:80]
            lines.append(f"| {vid} | {vlm} | {img} | {status_cell} | — | — | — | — | {err} |")

    note = report.get("quality_proxy_note")
    if note:
        lines.extend(["", f"> **Note:** {note}"])

    return "\n".join(lines)


def generate_sweep_report_html(report: dict[str, Any], sweep_dir: Path) -> str:
    """Generate an HTML report from a sweep report dict."""
    sweep_dir = Path(sweep_dir).resolve()
    sweep_id = report.get("sweep_id", "sweep")
    status = report.get("status", "completed")
    caption = report.get("caption", "")
    input_path = report.get("input", "")

    def escape(s: str) -> str:
        return (
            str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    style = """
    body { font-family: system-ui, sans-serif; margin: 1rem 2rem; max-width: 1100px; }
    h1 { font-size: 1.25rem; color: #333; }
    h2 { font-size: 1.05rem; color: #444; margin-top: 1.5rem; }
    .meta { color: #666; margin-bottom: 1rem; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 1rem; }
    th, td {
      border: 1px solid #ddd; padding: 0.4rem 0.6rem;
      text-align: left; font-size: 0.9rem;
    }
    th { background: #f5f5f5; font-weight: 600; }
    .status.success { color: #0a0; font-weight: 600; }
    .status.fail { color: #c00; font-weight: 600; }
    .note {
      color: #555; background: #fafafa; padding: 0.5rem 0.75rem;
      border-left: 3px solid #ccc;
    }
    a { color: #06c; }
    """

    meta_lines = []
    if input_path:
        meta_lines.append(f"Input: <code>{escape(input_path)}</code>")
    if caption:
        meta_lines.append(f"Caption: {escape(caption)}")
    meta_lines.append(f"Status: <strong>{escape(status)}</strong>")

    if status == "dry_run":
        total = report.get("total_variants", len(report.get("preview", [])))
        meta_lines.append(f"Planned variants: <strong>{escape(str(total))}</strong>")
        rows = []
        for item in report.get("preview", []):
            rows.append(
                f"<tr><td>{escape(item.get('variant_id', '—'))}</td>"
                f"<td>{escape(_provider_cell(item, 'vlm'))}</td>"
                f"<td>{escape(_provider_cell(item, 'image'))}</td>"
                f"<td>{escape(str(item.get('refinement_iterations', '—')))}</td>"
                f"<td>{escape(str(item.get('optimize_inputs', '—')))}</td>"
                f"<td>{escape(str(item.get('auto_refine', '—')))}</td></tr>"
            )
        preview_body = "\n".join(rows)
        body = f"""
  <h2>Planned Variants (preview)</h2>
  <table>
    <thead><tr><th>Variant</th><th>VLM</th><th>Image</th><th>Iters</th><th>Optimize</th>
    <th>Auto-refine</th></tr></thead>
    <tbody>
{preview_body}
    </tbody>
  </table>
"""
    else:
        summary = report.get("summary") or {}
        total_seconds = float(report.get("total_seconds") or 0.0)
        best_variant = summary.get("best_variant") or "—"
        mean_score = summary.get("mean_quality_proxy_score") or "—"
        meta_lines.extend(
            [
                f"Completed: <strong>{escape(str(summary.get('completed', 0)))}</strong>",
                f"Failed: <strong>{escape(str(summary.get('failed', 0)))}</strong>",
                f"Best variant: <strong>{escape(str(best_variant))}</strong>",
                f"Mean score: <strong>{escape(str(mean_score))}</strong>",
                f"Total seconds: <strong>{total_seconds:.1f}</strong>",
            ]
        )

        ranked = report.get("ranked_results") or []
        top_n = ranked[: min(5, len(ranked))]
        top_rows = []
        for rank, item in enumerate(top_n, start=1):
            top_rows.append(
                f"<tr><td>{rank}</td><td>{escape(item.get('variant_id', '—'))}</td>"
                f"<td>{escape(_provider_cell(item, 'vlm'))}</td>"
                f"<td>{escape(_provider_cell(item, 'image'))}</td>"
                f"<td>{escape(str(item.get('iterations_used', '—')))}</td>"
                f"<td>{escape(str(item.get('critic_suggestions', '—')))}</td>"
                f"<td>{escape(str(item.get('quality_proxy_score', '—')))}</td>"
                f"<td>{escape(str(item.get('total_seconds', '—')))}</td></tr>"
            )

        result_rows = []
        for item in report.get("results", []):
            vid = escape(item.get("variant_id", "—"))
            vlm = escape(_provider_cell(item, "vlm"))
            img = escape(_provider_cell(item, "image"))
            if item.get("status") == "success":
                status_cell = '<span class="status success">Success</span>'
                iters = escape(str(item.get("iterations_used", "—")))
                suggestions = escape(str(item.get("critic_suggestions", "—")))
                score = escape(str(item.get("quality_proxy_score", "—")))
                seconds = escape(str(item.get("total_seconds", "—")))
                out = _relative_output(item.get("output_path") or "", sweep_dir)
                out_cell = f'<a href="{escape(out)}">{escape(out)}</a>' if out else "—"
                result_rows.append(
                    f"<tr><td>{vid}</td><td>{vlm}</td><td>{img}</td><td>{status_cell}</td>"
                    f"<td>{iters}</td><td>{suggestions}</td><td>{score}</td><td>{seconds}</td>"
                    f"<td>{out_cell}</td></tr>"
                )
            else:
                status_cell = '<span class="status fail">Failed</span>'
                err = escape((item.get("error") or "unknown")[:200])
                result_rows.append(
                    f"<tr><td>{vid}</td><td>{vlm}</td><td>{img}</td><td>{status_cell}</td>"
                    f'<td colspan="5">{err}</td></tr>'
                )

        top_html = ""
        if top_rows:
            top_body = "\n".join(top_rows)
            top_html = f"""
  <h2>Top Variants (ranked)</h2>
  <table>
    <thead><tr><th>Rank</th><th>Variant</th><th>VLM</th><th>Image</th><th>Iters</th>
    <th>Suggestions</th><th>Score</th><th>Seconds</th></tr></thead>
    <tbody>
{top_body}
    </tbody>
  </table>
"""

        note = report.get("quality_proxy_note")
        note_html = f'<p class="note">{escape(note)}</p>' if note else ""
        result_body = "\n".join(result_rows)

        body = f"""{top_html}
  <h2>All Variants</h2>
  <table>
    <thead><tr><th>Variant</th><th>VLM</th><th>Image</th><th>Status</th><th>Iters</th>
    <th>Suggestions</th><th>Score</th><th>Seconds</th><th>Output / Error</th></tr></thead>
    <tbody>
{result_body}
    </tbody>
  </table>
{note_html}
"""

    meta_html = "<br>\n  ".join(meta_lines)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Sweep Report — {escape(sweep_id)}</title>
  <style>{style}</style>
</head>
<body>
  <h1>Sweep Report: {escape(sweep_id)}</h1>
  <p class="meta">{meta_html}</p>
{body}</body>
</html>
"""


def write_sweep_report(
    sweep_dir: Path,
    output_path: Path | None = None,
    format: Literal["markdown", "html", "md"] = "markdown",
) -> Path:
    """Load the sweep report from sweep_dir, generate a report, and write it to disk.

    Args:
        sweep_dir: Path to the sweep run directory.
        output_path: Where to write the report. If None, writes to sweep_dir/sweep_report.{md|html}.
        format: Report format: markdown, html, or md (alias for markdown).

    Returns:
        The path where the report was written.
    """
    sweep_dir = Path(sweep_dir).resolve()
    report = load_sweep_report(sweep_dir)
    ext = "html" if format == "html" else "md"
    if output_path is None:
        output_path = sweep_dir / f"sweep_report.{ext}"
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if format == "html":
        content = generate_sweep_report_html(report, sweep_dir)
    else:
        content = generate_sweep_report_md(report, sweep_dir)
    output_path.write_text(content, encoding="utf-8")
    logger.info("Wrote sweep report", path=str(output_path), format=format)
    return output_path
