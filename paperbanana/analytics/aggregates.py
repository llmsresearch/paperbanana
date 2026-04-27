"""Aggregate analytics records into KPI summaries."""

from __future__ import annotations

from paperbanana.analytics.models import AnalyticsRecord, AnalyticsSummary


def _bump(counter: dict[str, int], key: str | None) -> None:
    if not key:
        return
    counter[key] = counter.get(key, 0) + 1


def summarize_records(records: list[AnalyticsRecord]) -> AnalyticsSummary:
    """Compute summary metrics from analytics records."""
    summary = AnalyticsSummary()
    for record in records:
        summary.total_records += 1
        if record.status == "success":
            summary.success_records += 1
        elif record.status == "failed":
            summary.failed_records += 1

        summary.total_seconds += float(record.total_seconds)
        if record.cost_usd is not None:
            summary.total_cost_usd += float(record.cost_usd)
            summary.cost_record_count += 1

        _bump(summary.source_type_counts, record.source_type)
        _bump(summary.vlm_provider_counts, record.vlm_provider)
        _bump(summary.image_provider_counts, record.image_provider)

    summary.total_seconds = round(summary.total_seconds, 3)
    summary.total_cost_usd = round(summary.total_cost_usd, 6)
    return summary

