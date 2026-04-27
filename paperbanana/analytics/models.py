"""Typed models for analytics records and summaries."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class AnalyticsRecord:
    """Normalized analytics event extracted from run artifacts."""

    source_type: str
    source_path: str
    source_id: str
    status: str
    total_seconds: float
    cost_usd: float | None = None
    vlm_provider: str | None = None
    image_provider: str | None = None


@dataclass(slots=True)
class AnalyticsSummary:
    """Aggregated KPIs over a set of analytics records."""

    total_records: int = 0
    success_records: int = 0
    failed_records: int = 0
    total_seconds: float = 0.0
    total_cost_usd: float = 0.0
    cost_record_count: int = 0
    source_type_counts: dict[str, int] = field(default_factory=dict)
    vlm_provider_counts: dict[str, int] = field(default_factory=dict)
    image_provider_counts: dict[str, int] = field(default_factory=dict)
