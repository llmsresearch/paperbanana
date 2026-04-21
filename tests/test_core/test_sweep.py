"""Tests for sweep variant planning helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paperbanana.core.sweep import (
    SWEEP_REPORT_FILENAME,
    build_sweep_variants,
    generate_sweep_report_html,
    generate_sweep_report_md,
    load_sweep_report,
    parse_csv_bools,
    parse_csv_ints,
    quality_proxy_score,
    rank_sweep_results,
    summarize_sweep,
    write_sweep_report,
)


def test_build_sweep_variants_cartesian_and_cap() -> None:
    variants = build_sweep_variants(
        vlm_providers=["gemini", "openai"],
        vlm_models=[],
        image_providers=["google_imagen"],
        image_models=[],
        refinement_iterations=[2, 3],
        optimize_inputs=[False, True],
        auto_refine=[False],
        max_variants=5,
    )
    assert len(variants) == 5
    assert variants[0].variant_id == "variant_001"
    assert variants[-1].variant_id == "variant_005"


def test_parse_csv_ints_validates_values() -> None:
    assert parse_csv_ints("1, 3,5", field_name="--iterations") == [1, 3, 5]
    with pytest.raises(ValueError, match="integers"):
        parse_csv_ints("x,2", field_name="--iterations")
    with pytest.raises(ValueError, match=">= 1"):
        parse_csv_ints("0,2", field_name="--iterations")


def test_quality_proxy_score_formula() -> None:
    assert quality_proxy_score(0) == 100.0
    assert quality_proxy_score(1) == 87.5
    assert quality_proxy_score(8) == 0.0
    assert quality_proxy_score(9) == 0.0


def test_parse_csv_bools_supports_common_forms() -> None:
    assert parse_csv_bools("on,off,true,0", field_name="--optimize-modes") == [
        True,
        False,
        True,
        False,
    ]
    with pytest.raises(ValueError, match="booleans"):
        parse_csv_bools("maybe", field_name="--optimize-modes")


def test_rank_and_summarize_sweep_results() -> None:
    results = [
        {
            "variant_id": "a",
            "status": "success",
            "quality_proxy_score": 80.0,
            "total_seconds": 20.0,
        },
        {"variant_id": "b", "status": "failed"},
        {
            "variant_id": "c",
            "status": "success",
            "quality_proxy_score": 90.0,
            "total_seconds": 25.0,
        },
    ]
    ranked = rank_sweep_results([x for x in results if x["status"] == "success"])
    assert [x["variant_id"] for x in ranked] == ["c", "a"]

    summary = summarize_sweep(results)
    assert summary["completed"] == 2
    assert summary["failed"] == 1
    assert summary["best_variant"] == "c"


# ---------------------------------------------------------------------------
# load_sweep_report
# ---------------------------------------------------------------------------


def _completed_report_payload(sweep_dir: Path) -> dict:
    return {
        "sweep_id": "sweep_test",
        "status": "completed",
        "input": "paper.pdf",
        "caption": "Figure 1",
        "total_seconds": 12.5,
        "summary": {
            "completed": 2,
            "failed": 1,
            "best_variant": "variant_002",
            "best_quality_proxy_score": 87.5,
            "mean_quality_proxy_score": 81.25,
            "mean_total_seconds": 6.0,
        },
        "results": [
            {
                "status": "success",
                "variant_id": "variant_001",
                "vlm_provider": "gemini",
                "vlm_model": "gemini-2.5-flash",
                "image_provider": "google_imagen",
                "image_model": None,
                "iterations_used": 2,
                "critic_suggestions": 2,
                "quality_proxy_score": 75.0,
                "total_seconds": 5.5,
                "output_path": str(sweep_dir / "variant_001" / "out.png"),
            },
            {
                "status": "success",
                "variant_id": "variant_002",
                "vlm_provider": "openai",
                "vlm_model": "gpt-4o",
                "image_provider": "openai_imagen",
                "image_model": None,
                "iterations_used": 3,
                "critic_suggestions": 1,
                "quality_proxy_score": 87.5,
                "total_seconds": 6.5,
                "output_path": str(sweep_dir / "variant_002" / "out.png"),
            },
            {
                "status": "failed",
                "variant_id": "variant_003",
                "vlm_provider": "gemini",
                "vlm_model": None,
                "image_provider": "google_imagen",
                "image_model": None,
                "error": "Provider timeout after 30s",
            },
        ],
        "ranked_results": [
            {
                "variant_id": "variant_002",
                "vlm_provider": "openai",
                "vlm_model": "gpt-4o",
                "image_provider": "openai_imagen",
                "image_model": None,
                "iterations_used": 3,
                "critic_suggestions": 1,
                "quality_proxy_score": 87.5,
                "total_seconds": 6.5,
            },
            {
                "variant_id": "variant_001",
                "vlm_provider": "gemini",
                "vlm_model": "gemini-2.5-flash",
                "image_provider": "google_imagen",
                "image_model": None,
                "iterations_used": 2,
                "critic_suggestions": 2,
                "quality_proxy_score": 75.0,
                "total_seconds": 5.5,
            },
        ],
        "quality_proxy_note": (
            "quality_proxy_score = max(0, 100 - 12.5 * N) where N is critic suggestion "
            "count on the final iteration"
        ),
    }


def _dry_run_payload() -> dict:
    return {
        "sweep_id": "sweep_dry",
        "status": "dry_run",
        "total_variants": 2,
        "preview": [
            {
                "variant_id": "variant_001",
                "vlm_provider": "gemini",
                "vlm_model": None,
                "image_provider": "google_imagen",
                "image_model": None,
                "refinement_iterations": 2,
                "optimize_inputs": False,
                "auto_refine": False,
            },
            {
                "variant_id": "variant_002",
                "vlm_provider": "openai",
                "vlm_model": "gpt-4o",
                "image_provider": "openai_imagen",
                "image_model": None,
                "refinement_iterations": 3,
                "optimize_inputs": True,
                "auto_refine": True,
            },
        ],
    }


def test_load_sweep_report_success(tmp_path: Path) -> None:
    payload = _completed_report_payload(tmp_path)
    (tmp_path / SWEEP_REPORT_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_sweep_report(tmp_path)
    assert loaded["sweep_id"] == "sweep_test"
    assert len(loaded["results"]) == 3


def test_load_sweep_report_dry_run(tmp_path: Path) -> None:
    (tmp_path / SWEEP_REPORT_FILENAME).write_text(json.dumps(_dry_run_payload()), encoding="utf-8")
    loaded = load_sweep_report(tmp_path)
    assert loaded["status"] == "dry_run"
    assert len(loaded["preview"]) == 2


def test_load_sweep_report_dir_not_found() -> None:
    with pytest.raises(FileNotFoundError, match="Sweep directory not found"):
        load_sweep_report(Path("/nonexistent/sweep_dir"))


def test_load_sweep_report_json_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No sweep_report.json"):
        load_sweep_report(tmp_path)


def test_load_sweep_report_invalid_json(tmp_path: Path) -> None:
    (tmp_path / SWEEP_REPORT_FILENAME).write_text("not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_sweep_report(tmp_path)


def test_load_sweep_report_missing_sweep_id(tmp_path: Path) -> None:
    (tmp_path / SWEEP_REPORT_FILENAME).write_text('{"status": "completed"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid report"):
        load_sweep_report(tmp_path)


def test_load_sweep_report_completed_missing_results(tmp_path: Path) -> None:
    (tmp_path / SWEEP_REPORT_FILENAME).write_text(
        '{"sweep_id": "x", "status": "completed"}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing 'results'"):
        load_sweep_report(tmp_path)


def test_load_sweep_report_dry_run_missing_preview(tmp_path: Path) -> None:
    (tmp_path / SWEEP_REPORT_FILENAME).write_text(
        '{"sweep_id": "x", "status": "dry_run"}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing 'preview'"):
        load_sweep_report(tmp_path)


# ---------------------------------------------------------------------------
# generate_sweep_report_md / html — completed
# ---------------------------------------------------------------------------


def test_generate_sweep_report_md_completed(tmp_path: Path) -> None:
    report = _completed_report_payload(tmp_path)
    md = generate_sweep_report_md(report, tmp_path)
    assert "# Sweep Report: sweep_test" in md
    assert "Figure 1" in md
    assert "Best variant" in md
    assert "variant_002" in md
    assert "Top Variants (ranked)" in md
    assert "All Variants" in md
    assert "✓ Success" in md
    assert "✗ Failed" in md
    assert "Provider timeout" in md
    assert "gemini / gemini-2.5-flash" in md
    assert "quality_proxy_score" in md
    assert "`variant_001/out.png`" in md


def test_generate_sweep_report_md_dry_run(tmp_path: Path) -> None:
    md = generate_sweep_report_md(_dry_run_payload(), tmp_path)
    assert "# Sweep Report: sweep_dry" in md
    assert "Planned variants" in md
    assert "variant_001" in md
    assert "variant_002" in md
    assert "Top Variants" not in md
    assert "All Variants" not in md


def test_generate_sweep_report_html_completed(tmp_path: Path) -> None:
    report = _completed_report_payload(tmp_path)
    html = generate_sweep_report_html(report, tmp_path)
    assert "<!DOCTYPE html>" in html
    assert "Sweep Report: sweep_test" in html
    assert "variant_002" in html
    assert "Top Variants (ranked)" in html
    assert "All Variants" in html
    assert "Success" in html
    assert "Failed" in html
    assert 'href="variant_001/out.png"' in html
    assert "quality_proxy_score" in html


def test_generate_sweep_report_html_escapes_caption(tmp_path: Path) -> None:
    report = _completed_report_payload(tmp_path)
    report["caption"] = "<script>alert('x')</script>"
    html = generate_sweep_report_html(report, tmp_path)
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html


def test_generate_sweep_report_html_dry_run(tmp_path: Path) -> None:
    html = generate_sweep_report_html(_dry_run_payload(), tmp_path)
    assert "Planned Variants (preview)" in html
    assert "Top Variants" not in html
    assert "All Variants" not in html


# ---------------------------------------------------------------------------
# write_sweep_report
# ---------------------------------------------------------------------------


def test_write_sweep_report_markdown(tmp_path: Path) -> None:
    payload = _completed_report_payload(tmp_path)
    (tmp_path / SWEEP_REPORT_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    out_path = tmp_path / "report.md"
    written = write_sweep_report(tmp_path, output_path=out_path, format="markdown")
    assert written == out_path
    assert out_path.exists()
    assert "Sweep Report: sweep_test" in out_path.read_text(encoding="utf-8")


def test_write_sweep_report_html_default_path(tmp_path: Path) -> None:
    payload = _completed_report_payload(tmp_path)
    (tmp_path / SWEEP_REPORT_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    written = write_sweep_report(tmp_path, format="html")
    assert written == tmp_path / "sweep_report.html"
    assert written.exists()
    assert "<!DOCTYPE html>" in written.read_text(encoding="utf-8")


def test_write_sweep_report_md_alias(tmp_path: Path) -> None:
    payload = _completed_report_payload(tmp_path)
    (tmp_path / SWEEP_REPORT_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    written = write_sweep_report(tmp_path, format="md")
    assert written == tmp_path / "sweep_report.md"
    assert written.exists()


# ---------------------------------------------------------------------------
# edge cases: empty ranked, sibling-dir paths, no quality note
# ---------------------------------------------------------------------------


def test_generate_sweep_report_md_skips_top_section_when_no_ranked(tmp_path: Path) -> None:
    report = _completed_report_payload(tmp_path)
    report["ranked_results"] = []
    md = generate_sweep_report_md(report, tmp_path)
    assert "Top Variants (ranked)" not in md
    assert "All Variants" in md


def test_generate_sweep_report_html_skips_top_section_when_no_ranked(tmp_path: Path) -> None:
    report = _completed_report_payload(tmp_path)
    report["ranked_results"] = []
    html = generate_sweep_report_html(report, tmp_path)
    assert "Top Variants (ranked)" not in html
    assert "All Variants" in html


def test_generate_sweep_report_output_path_outside_sweep_dir_stays_absolute(
    tmp_path: Path,
) -> None:
    report = _completed_report_payload(tmp_path)
    report["results"][0]["output_path"] = "/elsewhere/out.png"
    md = generate_sweep_report_md(report, tmp_path)
    assert "/elsewhere/out.png" in md


def test_generate_sweep_report_sibling_dir_path_not_collapsed(tmp_path: Path) -> None:
    """Path comparison must not collapse a sibling-dir match (startswith bug)."""
    sweep_dir = tmp_path / "sweep_abc"
    sibling = tmp_path / "sweep_abc_other" / "out.png"
    sweep_dir.mkdir()
    report = _completed_report_payload(sweep_dir)
    report["results"][0]["output_path"] = str(sibling)
    md = generate_sweep_report_md(report, sweep_dir)
    assert str(sibling) in md


def test_generate_sweep_report_md_without_quality_note(tmp_path: Path) -> None:
    report = _completed_report_payload(tmp_path)
    report.pop("quality_proxy_note")
    md = generate_sweep_report_md(report, tmp_path)
    assert "**Note:**" not in md


def test_generate_sweep_report_html_without_quality_note(tmp_path: Path) -> None:
    report = _completed_report_payload(tmp_path)
    report.pop("quality_proxy_note")
    html = generate_sweep_report_html(report, tmp_path)
    assert 'class="note"' not in html
