"""Tests for sweep variant planning helpers."""

from __future__ import annotations

import pytest

from paperbanana.core.sweep import (
    build_sweep_variants,
    parse_csv_bools,
    parse_csv_ints,
    quality_proxy_score,
    rank_sweep_results,
    summarize_sweep,
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
