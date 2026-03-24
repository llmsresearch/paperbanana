"""Dry-run cost estimation for PaperBanana pipeline runs."""

from __future__ import annotations

from paperbanana.core.config import Settings
from paperbanana.core.pricing import lookup_image_price, lookup_vlm_price
from paperbanana.core.types import DiagramType

# Average token counts per agent call (empirical estimates).
_AVG_TOKENS: dict[str, tuple[int, int]] = {
    # (input_tokens, output_tokens)
    "optimizer": (2000, 2000),
    "retriever": (3000, 500),
    "planner": (6000, 2000),
    "stylist": (3000, 2000),
    "visualizer_vlm": (2000, 2000),  # for statistical plots (matplotlib code gen)
    "critic": (4000, 1500),
}


def estimate_cost(
    settings: Settings,
    diagram_type: DiagramType = DiagramType.METHODOLOGY,
) -> dict:
    """Estimate the cost of a pipeline run without making API calls.

    Returns a dict with estimated_total_usd, vlm_calls, image_calls,
    breakdown_by_agent, and pricing_note.
    """
    vlm_provider = settings.vlm_provider
    vlm_model = settings.effective_vlm_model
    image_provider = settings.image_provider
    image_model = settings.effective_image_model

    vlm_pricing = lookup_vlm_price(vlm_provider, vlm_model)
    image_pricing = lookup_image_price(image_provider, image_model)

    if settings.auto_refine:
        iterations = settings.max_iterations
    else:
        iterations = settings.refinement_iterations

    # Count expected API calls
    vlm_calls = 0
    image_calls = 0
    breakdown: dict[str, float] = {}
    notes: list[str] = []

    def _vlm_cost(agent: str) -> float:
        nonlocal vlm_calls
        vlm_calls += 1
        if vlm_pricing is None:
            return 0.0
        inp, out = _AVG_TOKENS.get(agent, (3000, 1500))
        return inp * vlm_pricing["input_per_1k"] / 1000 + out * vlm_pricing["output_per_1k"] / 1000

    def _image_cost() -> float:
        nonlocal image_calls
        image_calls += 1
        if image_pricing is None:
            return 0.0
        return image_pricing

    # Phase 0: Optimizer (optional)
    if settings.optimize_inputs:
        breakdown["optimizer"] = _vlm_cost("optimizer")

    # Phase 1: Linear planning
    breakdown["retriever"] = _vlm_cost("retriever")
    breakdown["planner"] = _vlm_cost("planner")
    breakdown["stylist"] = _vlm_cost("stylist")

    # Phase 2: Iterative refinement
    vis_total = 0.0
    critic_total = 0.0
    for _ in range(iterations):
        if diagram_type == DiagramType.STATISTICAL_PLOT:
            vis_total += _vlm_cost("visualizer_vlm")
        else:
            vis_total += _image_cost()
        critic_total += _vlm_cost("critic")
    breakdown["visualizer"] = vis_total
    breakdown["critic"] = critic_total

    total = sum(breakdown.values())

    if vlm_pricing is None:
        notes.append(f"VLM pricing unknown for {vlm_provider}/{vlm_model}")
    if image_pricing is None:
        notes.append(f"Image pricing unknown for {image_provider}/{image_model}")
    if settings.auto_refine:
        notes.append(
            f"Auto-refine: estimated for max {iterations} iterations; "
            "actual cost may be lower if critic is satisfied early"
        )

    return {
        "estimated_total_usd": round(total, 6),
        "vlm_calls": vlm_calls,
        "image_calls": image_calls,
        "breakdown_by_agent": {k: round(v, 6) for k, v in breakdown.items()},
        "pricing_note": "; ".join(notes) if notes else None,
    }
