"""Smoke tests for workflow_runner (no live API calls)."""

from __future__ import annotations

from pathlib import Path

import pytest

from paperbanana.core.workflow_runner import run_orchestration_package


def test_orchestration_dry_run_writes_plan(tmp_path: Path) -> None:
    paper = tmp_path / "paper.txt"
    paper.write_text(
        "A Short Paper Title\n\n1 Introduction\nWe motivate the problem.\n\n2 Method\nDetails.\n",
        encoding="utf-8",
    )
    result = run_orchestration_package(
        paper=str(paper),
        resume_orchestrate=None,
        output_dir=tmp_path,
        data_dir=None,
        max_method_figures=2,
        max_plot_figures=0,
        pdf_pages=None,
        dry_run=True,
        config=None,
        vlm_provider=None,
        vlm_model=None,
        image_provider=None,
        image_model=None,
        iterations=None,
        auto=False,
        max_iterations=None,
        optimize=False,
        format="png",
        save_prompts=None,
        venue=None,
        retry_failed=False,
        max_retries=0,
        concurrency=1,
    )
    assert result["dry_run"] is True
    assert result["strict_success"] is True
    plan_path = Path(result["orchestration_plan_path"])
    assert plan_path.is_file()


def test_methodology_batch_missing_manifest_raises() -> None:
    from paperbanana.core.workflow_runner import run_methodology_batch

    with pytest.raises(FileNotFoundError):
        run_methodology_batch(
            manifest_path=Path("/nonexistent/manifest.yaml"),
            output_dir=Path("."),
        )
