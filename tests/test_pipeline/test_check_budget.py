"""Unit tests for PaperBananaPipeline._check_budget helper (issue #153)."""

from __future__ import annotations

from types import SimpleNamespace

from paperbanana.core.pipeline import PaperBananaPipeline


def _call(cost_tracker, context: str, iteration: int | None = None) -> bool:
    """Invoke the unbound helper with a minimal stub self."""
    stub = SimpleNamespace(_cost_tracker=cost_tracker)
    return PaperBananaPipeline._check_budget(stub, context, iteration=iteration)


class TestCheckBudget:
    def test_no_tracker_returns_false(self):
        assert _call(None, "after planning phases") is False

    def test_under_budget_returns_false(self):
        tracker = SimpleNamespace(is_over_budget=False)
        assert _call(tracker, "between iterations", iteration=2) is False

    def test_over_budget_returns_true_without_iteration(self, capsys):
        tracker = SimpleNamespace(is_over_budget=True)
        assert _call(tracker, "after planning phases") is True
        out = capsys.readouterr().out
        assert "Budget exceeded after planning phases, skipping iterations" in out

    def test_over_budget_returns_true_with_iteration(self, capsys):
        tracker = SimpleNamespace(is_over_budget=True)
        assert _call(tracker, "between iterations", iteration=3) is True
        out = capsys.readouterr().out
        assert "Budget exceeded between iterations, stopping early" in out
        assert "iteration=3" in out
