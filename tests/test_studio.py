"""Tests for PaperBanana Studio (Gradio UI)."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "fn",
    [
        "list_run_ids",
        "list_batch_ids",
        "load_run_summary",
        "load_batch_summary",
    ],
)
def test_runs_helpers_smoke(fn: str, tmp_path):
    from paperbanana.studio import runs as runs_mod

    f = getattr(runs_mod, fn)
    if fn.startswith("load_"):
        out = f(str(tmp_path), "missing_id")
        assert isinstance(out, dict)
        assert out.get("exists") is False
    else:
        assert f(str(tmp_path)) == []


def test_build_settings_merge(tmp_path):
    from paperbanana.studio.runner import build_settings

    s = build_settings(
        config_path=None,
        output_dir=str(tmp_path / "out"),
        vlm_provider="gemini",
        vlm_model="gemini-2.0-flash",
        image_provider="google_imagen",
        image_model="gemini-3-pro-image-preview",
        output_format="png",
        refinement_iterations=2,
        auto_refine=False,
        max_iterations=10,
        optimize_inputs=True,
        save_prompts=False,
    )
    assert s.output_dir == str(tmp_path / "out")
    assert s.refinement_iterations == 2
    assert s.optimize_inputs is True


def test_build_studio_app():
    gradio = pytest.importorskip("gradio")
    from paperbanana.studio.app import build_studio_app

    _ = gradio
    demo = build_studio_app(default_output_dir="outputs", config_path=None)
    assert demo is not None
