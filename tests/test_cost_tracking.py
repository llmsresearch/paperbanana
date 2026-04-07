"""Tests for cost tracking, pricing lookup, budget guard, and cost estimation."""

from __future__ import annotations

import pytest

from paperbanana.core.cost_estimator import estimate_cost
from paperbanana.core.cost_tracker import CostTracker
from paperbanana.core.pricing import lookup_image_price, lookup_vlm_price

# ── Pricing lookup ──────────────────────────────────────────────────


class TestPricingLookup:
    def test_exact_match_vlm(self):
        result = lookup_vlm_price("gemini", "gemini-2.0-flash")
        assert result is not None
        assert result["input_per_1k"] == 0.0
        assert result["output_per_1k"] == 0.0

    def test_prefix_match_vlm(self):
        # "gpt-5.2-turbo" should match "gpt-5.2" prefix
        result = lookup_vlm_price("openai", "gpt-5.2-turbo")
        assert result is not None
        assert result["input_per_1k"] > 0

    def test_unknown_vlm_returns_none(self):
        result = lookup_vlm_price("unknown_provider", "unknown_model")
        assert result is None

    def test_exact_match_image(self):
        result = lookup_image_price("google_imagen", "gemini-3-pro-image-preview")
        assert result is not None
        assert result == 0.0

    def test_openai_image_pricing(self):
        result = lookup_image_price("openai_imagen", "gpt-image-1.5")
        assert result is not None
        assert result > 0

    def test_unknown_image_returns_none(self):
        result = lookup_image_price("unknown", "unknown")
        assert result is None


# ── CostTracker ─────────────────────────────────────────────────────


class TestCostTracker:
    def test_empty_tracker(self):
        tracker = CostTracker()
        assert tracker.total_cost == 0.0
        assert tracker.entries == []
        summary = tracker.summary()
        assert summary["total_usd"] == 0.0
        assert summary["num_vlm_calls"] == 0
        assert summary["num_image_calls"] == 0

    def test_record_vlm_call(self):
        tracker = CostTracker()
        tracker.record_vlm_call(
            provider="openai",
            model="gpt-5.2",
            input_tokens=1000,
            output_tokens=500,
            agent="planner",
        )
        assert len(tracker.entries) == 1
        assert tracker.entries[0].agent == "planner"
        assert tracker.entries[0].call_type == "vlm"
        assert tracker.total_cost > 0

    def test_record_image_call(self):
        tracker = CostTracker()
        tracker.record_image_call(
            provider="openai_imagen",
            model="gpt-image-1.5",
            agent="visualizer",
        )
        assert len(tracker.entries) == 1
        assert tracker.entries[0].call_type == "image_gen"
        assert tracker.total_cost > 0

    def test_free_tier_cost_is_zero(self):
        tracker = CostTracker()
        tracker.record_vlm_call(
            provider="gemini",
            model="gemini-2.0-flash",
            input_tokens=5000,
            output_tokens=2000,
            agent="retriever",
        )
        tracker.record_image_call(
            provider="google_imagen",
            model="gemini-3-pro-image-preview",
            agent="visualizer",
        )
        assert tracker.total_cost == 0.0
        assert tracker.pricing_complete is True

    def test_set_agent_fallback(self):
        tracker = CostTracker()
        tracker.set_agent("stylist")
        tracker.record_vlm_call(
            provider="gemini",
            model="gemini-2.0-flash",
            input_tokens=100,
            output_tokens=50,
        )
        assert tracker.entries[0].agent == "stylist"

    def test_summary_by_agent(self):
        tracker = CostTracker()
        tracker.record_vlm_call(
            provider="openai",
            model="gpt-5.2",
            input_tokens=1000,
            output_tokens=500,
            agent="planner",
        )
        tracker.record_vlm_call(
            provider="openai",
            model="gpt-5.2",
            input_tokens=2000,
            output_tokens=800,
            agent="critic",
        )
        summary = tracker.summary()
        assert "planner" in summary["by_agent"]
        assert "critic" in summary["by_agent"]
        assert summary["num_vlm_calls"] == 2

    def test_vlm_and_image_cost_split(self):
        tracker = CostTracker()
        tracker.record_vlm_call(
            provider="openai",
            model="gpt-5.2",
            input_tokens=1000,
            output_tokens=500,
            agent="critic",
        )
        tracker.record_image_call(
            provider="openai_imagen",
            model="gpt-image-1.5",
            agent="visualizer",
        )
        assert tracker.vlm_cost > 0
        assert tracker.image_cost > 0
        assert tracker.total_cost == pytest.approx(tracker.vlm_cost + tracker.image_cost)

    def test_unknown_model_pricing_not_known(self):
        tracker = CostTracker()
        tracker.record_vlm_call(
            provider="unknown",
            model="mystery-model",
            input_tokens=1000,
            output_tokens=500,
            agent="test",
        )
        assert tracker.entries[0].pricing_known is False
        assert tracker.entries[0].cost_usd == 0.0
        assert tracker.pricing_complete is False


# ── Budget guard ────────────────────────────────────────────────────


class TestBudgetGuard:
    def test_no_budget_no_flag(self):
        tracker = CostTracker(budget=None)
        for _ in range(100):
            tracker.record_vlm_call(
                provider="openai",
                model="gpt-5.2",
                input_tokens=10000,
                output_tokens=5000,
                agent="test",
            )
        assert tracker.is_over_budget is False

    def test_is_over_budget_flag_set(self):
        """record_vlm_call should not raise; the pipeline checks is_over_budget at checkpoints."""
        tracker = CostTracker(budget=0.001)
        # Should NOT raise — just sets the over-budget flag
        tracker.record_vlm_call(
            provider="openai",
            model="gpt-5.2",
            input_tokens=10000,
            output_tokens=5000,
            agent="planner",
        )
        assert tracker.is_over_budget is True
        assert tracker.total_cost > 0.001

    def test_budget_not_exceeded_for_cheap_calls(self):
        tracker = CostTracker(budget=100.0)
        tracker.record_vlm_call(
            provider="gemini",
            model="gemini-2.0-flash",
            input_tokens=5000,
            output_tokens=2000,
            agent="retriever",
        )
        assert tracker.is_over_budget is False

    def test_image_call_sets_over_budget(self):
        """record_image_call should also set is_over_budget without raising."""
        tracker = CostTracker(budget=0.001)
        tracker.record_image_call(
            provider="openai_imagen",
            model="gpt-image-1.5",
            agent="visualizer",
        )
        assert tracker.is_over_budget is True

    def test_negative_budget_rejected(self):
        """Settings should reject negative or zero budget values."""
        from pydantic import ValidationError

        from paperbanana.core.config import Settings

        with pytest.raises(ValidationError):
            Settings(budget_usd=-5.0)
        with pytest.raises(ValidationError):
            Settings(budget_usd=0.0)


# ── Cost estimator ──────────────────────────────────────────────────


class TestCostEstimator:
    def test_basic_estimation(self):
        from paperbanana.core.config import Settings

        settings = Settings(
            vlm_provider="gemini",
            vlm_model="gemini-2.0-flash",
            image_provider="google_imagen",
            image_model="gemini-3-pro-image-preview",
            refinement_iterations=3,
        )
        result = estimate_cost(settings)
        assert "estimated_total_usd" in result
        assert "vlm_calls" in result
        assert "image_calls" in result
        assert result["vlm_calls"] >= 6  # retriever + planner + stylist + 3x critic
        assert result["image_calls"] == 3
        # Free tier — cost should be 0
        assert result["estimated_total_usd"] == 0.0

    def test_paid_provider_estimation(self):
        from paperbanana.core.config import Settings

        settings = Settings(
            vlm_provider="openai",
            vlm_model="gpt-5.2",
            image_provider="openai_imagen",
            image_model="gpt-image-1.5",
            refinement_iterations=3,
        )
        result = estimate_cost(settings)
        assert result["estimated_total_usd"] > 0
        assert result["vlm_calls"] >= 6
        assert result["image_calls"] == 3

    def test_auto_refine_estimation(self):
        from paperbanana.core.config import Settings

        settings = Settings(
            vlm_provider="openai",
            vlm_model="gpt-5.2",
            image_provider="openai_imagen",
            image_model="gpt-image-1.5",
            auto_refine=True,
            max_iterations=30,
        )
        result = estimate_cost(settings)
        assert result["image_calls"] == 30
        assert result["pricing_note"] is not None
        assert "auto" in result["pricing_note"].lower()

    def test_optimize_adds_vlm_call(self):
        from paperbanana.core.config import Settings

        settings_no_opt = Settings(
            vlm_provider="openai",
            vlm_model="gpt-5.2",
            image_provider="openai_imagen",
            image_model="gpt-image-1.5",
            refinement_iterations=1,
        )
        settings_opt = Settings(
            vlm_provider="openai",
            vlm_model="gpt-5.2",
            image_provider="openai_imagen",
            image_model="gpt-image-1.5",
            refinement_iterations=1,
            optimize_inputs=True,
        )
        no_opt = estimate_cost(settings_no_opt)
        with_opt = estimate_cost(settings_opt)
        assert with_opt["vlm_calls"] == no_opt["vlm_calls"] + 1
        assert with_opt["estimated_total_usd"] > no_opt["estimated_total_usd"]

    def test_unknown_model_note(self):
        from paperbanana.core.config import Settings

        settings = Settings(
            vlm_provider="unknown_provider",
            vlm_model="mystery",
            image_provider="unknown_img",
            image_model="mystery_img",
            refinement_iterations=1,
        )
        result = estimate_cost(settings)
        assert result["pricing_note"] is not None
        assert "unknown" in result["pricing_note"].lower()
