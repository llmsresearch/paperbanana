"""Render analytics summaries in JSON-friendly and markdown forms."""

from __future__ import annotations

from paperbanana.analytics.models import AnalyticsSummary


def summary_to_dict(summary: AnalyticsSummary) -> dict[str, object]:
    success_rate = (
        (summary.success_records / summary.total_records) if summary.total_records > 0 else 0.0
    )
    return {
        "total_records": summary.total_records,
        "success_records": summary.success_records,
        "failed_records": summary.failed_records,
        "success_rate": round(success_rate, 4),
        "total_seconds": round(summary.total_seconds, 3),
        "mean_seconds": round(summary.total_seconds / summary.total_records, 3)
        if summary.total_records
        else 0.0,
        "total_cost_usd": round(summary.total_cost_usd, 6),
        "cost_record_count": summary.cost_record_count,
        "source_type_counts": dict(sorted(summary.source_type_counts.items())),
        "vlm_provider_counts": dict(sorted(summary.vlm_provider_counts.items())),
        "image_provider_counts": dict(sorted(summary.image_provider_counts.items())),
    }


def render_markdown_summary(summary: AnalyticsSummary) -> str:
    """Render a compact markdown summary."""
    payload = summary_to_dict(summary)
    lines = [
        "# Run Analytics Summary",
        "",
        f"- Total records: **{payload['total_records']}**",
        f"- Success records: **{payload['success_records']}**",
        f"- Failed records: **{payload['failed_records']}**",
        f"- Success rate: **{payload['success_rate']:.2%}**",
        f"- Total seconds: **{payload['total_seconds']}**",
        f"- Mean seconds/record: **{payload['mean_seconds']}**",
        f"- Total cost (USD): **{payload['total_cost_usd']}**",
        f"- Records with explicit cost: **{payload['cost_record_count']}**",
        "",
        "## Source Types",
    ]
    source_counts: dict[str, int] = payload["source_type_counts"]  # type: ignore[assignment]
    if source_counts:
        for key, value in source_counts.items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("## Providers")
    vlm_counts: dict[str, int] = payload["vlm_provider_counts"]  # type: ignore[assignment]
    image_counts: dict[str, int] = payload["image_provider_counts"]  # type: ignore[assignment]
    lines.append("- VLM providers:")
    if vlm_counts:
        for key, value in vlm_counts.items():
            lines.append(f"  - {key}: {value}")
    else:
        lines.append("  - (none)")
    lines.append("- Image providers:")
    if image_counts:
        for key, value in image_counts.items():
            lines.append(f"  - {key}: {value}")
    else:
        lines.append("  - (none)")

    return "\n".join(lines) + "\n"
