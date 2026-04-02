"""Variant sweep planning and result summarization utilities."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from statistics import mean
from typing import Any

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
