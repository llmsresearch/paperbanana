"""Batch generation: manifest loading, batch run id, and report generation."""

from __future__ import annotations

import datetime
import json
import uuid
from pathlib import Path
from typing import Any, Literal

import structlog

logger = structlog.get_logger()

REPORT_FILENAME = "batch_report.json"


def generate_batch_id() -> str:
    """Generate a unique batch run ID."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    return f"batch_{ts}_{short_uuid}"


def load_batch_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    """Load a batch manifest (YAML or JSON) and return a list of items.

    Each item is a dict with:
      - input: path to methodology text or PDF file (resolved relative to manifest parent)
      - caption: figure caption / communicative intent
      - id: optional string identifier for the item (default: index-based)
      - pdf_pages: optional 1-based page selection for PDF inputs (e.g. "1-5" or "2,4,6-8")

    Paths in the manifest are resolved relative to the manifest file's directory.
    """
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    parent = manifest_path.parent
    raw = manifest_path.read_text(encoding="utf-8")
    suffix = manifest_path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml

            data = yaml.safe_load(raw)
        except ImportError:
            raise RuntimeError(
                "PyYAML is required for YAML manifests. Install with: pip install pyyaml"
            )
    elif suffix == ".json":
        import json

        data = json.loads(raw)
    else:
        raise ValueError(f"Manifest must be .yaml, .yml, or .json. Got: {manifest_path.suffix}")

    if data is None:
        raise ValueError("Manifest is empty")
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "items" in data:
        items = data["items"]
    else:
        raise ValueError("Manifest must be a list of items or an object with an 'items' list")

    result = []
    for i, entry in enumerate(items):
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest item {i} must be an object, got {type(entry).__name__}")
        inp = entry.get("input")
        caption = entry.get("caption")
        if not inp or not caption:
            raise ValueError(f"Manifest item {i}: 'input' and 'caption' are required")
        input_path = Path(inp)
        if not input_path.is_absolute():
            input_path = (parent / input_path).resolve()
        pdf_pages = entry.get("pdf_pages")
        if pdf_pages is not None and not isinstance(pdf_pages, str):
            raise ValueError(f"Manifest item {i}: 'pdf_pages' must be a string when set")
        result.append(
            {
                "input": str(input_path),
                "caption": str(caption),
                "id": entry.get("id", f"item_{i + 1}"),
                "pdf_pages": pdf_pages,
            }
        )
    return result


def load_batch_report(batch_dir: Path) -> dict[str, Any]:
    """Load batch_report.json from a batch output directory.

    Args:
        batch_dir: Path to the batch run directory (e.g. outputs/batch_20250109_123456_abc).

    Returns:
        The report dict (batch_id, manifest, items, total_seconds).

    Raises:
        FileNotFoundError: If batch_dir or batch_report.json does not exist.
        ValueError: If the JSON is invalid or missing required keys.
    """
    batch_dir = Path(batch_dir).resolve()
    report_path = batch_dir / REPORT_FILENAME
    if not batch_dir.exists() or not batch_dir.is_dir():
        raise FileNotFoundError(f"Batch directory not found: {batch_dir}")
    if not report_path.exists():
        raise FileNotFoundError(f"No {REPORT_FILENAME} in {batch_dir}. Run a batch first.")
    raw = report_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict) or "items" not in data:
        raise ValueError(f"Invalid report: expected dict with 'items'. Got: {type(data)}")
    return data


def _report_summary(report: dict[str, Any]) -> tuple[int, int, float]:
    """Return (succeeded, total, total_seconds) from a batch report."""
    items = report.get("items", [])
    total = len(items)
    succeeded = sum(1 for x in items if x.get("output_path"))
    total_seconds = report.get("total_seconds") or 0.0
    return succeeded, total, float(total_seconds)


def generate_batch_report_md(report: dict[str, Any], batch_dir: Path) -> str:
    """Generate a Markdown report from a batch report dict."""
    batch_dir = Path(batch_dir).resolve()
    batch_id = report.get("batch_id", "batch")
    manifest = report.get("manifest", "")
    succeeded, total, total_seconds = _report_summary(report)
    lines = [
        f"# Batch Report: {batch_id}",
        "",
        f"- **Manifest:** `{manifest}`",
        f"- **Summary:** {succeeded}/{total} succeeded in {total_seconds:.1f}s",
        "",
        "| ID | Caption | Status | Output / Error | Iterations |",
        "|----|--------|--------|-----------------|------------|",
    ]
    for item in report.get("items", []):
        item_id = item.get("id", "—")
        caption = (item.get("caption") or "")[:60]
        if len(item.get("caption") or "") > 60:
            caption += "…"
        caption_escaped = caption.replace("|", "\\|")
        if item.get("output_path"):
            status = "✓ Success"
            out = item["output_path"]
            if Path(out).is_absolute() and out.startswith(str(batch_dir)):
                out = Path(out).relative_to(batch_dir).as_posix()
            out_escaped = str(out).replace("|", "\\|")
            iters = item.get("iterations", "—")
            lines.append(
                f"| {item_id} | {caption_escaped} | {status} | `{out_escaped}` | {iters} |"
            )
        else:
            status = "✗ Failed"
            err = (item.get("error") or "unknown").replace("|", "\\|")[:80]
            lines.append(f"| {item_id} | {caption_escaped} | {status} | {err} | — |")
    return "\n".join(lines)


def generate_batch_report_html(report: dict[str, Any], batch_dir: Path) -> str:
    """Generate an HTML report from a batch report dict."""
    batch_dir = Path(batch_dir).resolve()
    batch_id = report.get("batch_id", "batch")
    manifest = report.get("manifest", "")
    succeeded, total, total_seconds = _report_summary(report)

    def escape(s: str) -> str:
        return (
            s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        )

    rows = []
    for item in report.get("items", []):
        item_id = escape(str(item.get("id", "—")))
        caption = escape((item.get("caption") or "")[:80])
        if item.get("output_path"):
            status = '<span class="status success">Success</span>'
            out = item["output_path"]
            if Path(out).is_absolute() and out.startswith(str(batch_dir)):
                out = Path(out).relative_to(batch_dir).as_posix()
            out_cell = f'<a href="{escape(str(out))}">{escape(str(out))}</a>'
            iters = item.get("iterations", "—")
            rows.append(
                f"<tr><td>{item_id}</td><td>{caption}</td><td>{status}</td>"
                f"<td>{out_cell}</td><td>{iters}</td></tr>"
            )
        else:
            status = '<span class="status fail">Failed</span>'
            err = escape((item.get("error") or "unknown")[:200])
            rows.append(
                f"<tr><td>{item_id}</td><td>{caption}</td><td>{status}</td>"
                f'<td colspan="2">{err}</td></tr>'
            )

    body_rows = "\n".join(rows)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Batch Report — {escape(batch_id)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 1rem 2rem; max-width: 960px; }}
    h1 {{ font-size: 1.25rem; color: #333; }}
    .meta {{ color: #666; margin-bottom: 1rem; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 0.5rem 0.75rem; text-align: left; }}
    th {{ background: #f5f5f5; font-weight: 600; }}
    .status.success {{ color: #0a0; font-weight: 600; }}
    .status.fail {{ color: #c00; font-weight: 600; }}
    a {{ color: #06c; }}
  </style>
</head>
<body>
  <h1>Batch Report: {escape(batch_id)}</h1>
  <p class="meta">Manifest: <code>{escape(manifest)}</code><br>
  Summary: <strong>{succeeded}/{total}</strong> succeeded in
  <strong>{total_seconds:.1f}s</strong></p>
  <table>
    <thead><tr><th>ID</th><th>Caption</th><th>Status</th>
    <th>Output / Error</th><th>Iterations</th></tr></thead>
    <tbody>
{body_rows}
    </tbody>
  </table>
</body>
</html>
"""


def write_batch_report(
    batch_dir: Path,
    output_path: Path | None = None,
    format: Literal["markdown", "html", "md"] = "markdown",
) -> Path:
    """Load the batch report from batch_dir, generate a report, and write it to disk.

    Args:
        batch_dir: Path to the batch run directory.
        output_path: Where to write the report. If None, writes to batch_dir/batch_report.{md|html}.
        format: Report format: markdown, html, or md (alias for markdown).

    Returns:
        The path where the report was written.
    """
    batch_dir = Path(batch_dir).resolve()
    report = load_batch_report(batch_dir)
    ext = "html" if format == "html" else "md"
    if output_path is None:
        output_path = batch_dir / f"batch_report.{ext}"
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if format == "html":
        content = generate_batch_report_html(report, batch_dir)
    else:
        content = generate_batch_report_md(report, batch_dir)
    output_path.write_text(content, encoding="utf-8")
    logger.info("Wrote batch report", path=str(output_path), format=format)
    return output_path
