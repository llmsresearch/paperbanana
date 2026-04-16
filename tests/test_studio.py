"""Tests for PaperBanana Studio (Gradio UI)."""

from __future__ import annotations

from pathlib import Path

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


def test_run_composite_smoke(tmp_path):
    from PIL import Image

    from paperbanana.studio.runner import run_composite

    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    Image.new("RGB", (100, 80), (255, 0, 0)).save(str(p1))
    Image.new("RGB", (100, 80), (0, 255, 0)).save(str(p2))

    out_dir = tmp_path / "out"
    log, output_path = run_composite(
        [str(p1), str(p2)],
        output_dir=str(out_dir),
        layout="1x2",
        output_filename="result.png",
    )
    assert output_path is not None
    assert (out_dir / "result.png").exists()
    assert "Done." in log


def test_run_composite_no_files_returns_error(tmp_path):
    from paperbanana.studio.runner import run_composite

    log, output_path = run_composite(
        [],
        output_dir=str(tmp_path),
    )
    assert output_path is None
    assert "No valid image" in log


def test_run_composite_invalid_label_position(tmp_path):
    from PIL import Image

    from paperbanana.studio.runner import run_composite

    p = tmp_path / "x.png"
    Image.new("RGB", (50, 50), (0, 0, 255)).save(str(p))
    log, output_path = run_composite(
        [str(p)],
        output_dir=str(tmp_path),
        label_position="left",
    )
    assert output_path is None
    assert "label_position" in log


def test_run_composite_explicit_labels(tmp_path):
    from PIL import Image

    from paperbanana.studio.runner import run_composite

    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    Image.new("RGB", (60, 60), (255, 0, 0)).save(str(p1))
    Image.new("RGB", (60, 60), (0, 255, 0)).save(str(p2))

    log, output_path = run_composite(
        [str(p1), str(p2)],
        output_dir=str(tmp_path / "out"),
        labels="Fig A, Fig B",
        layout="1x2",
    )
    assert output_path is not None
    assert "Done." in log


def test_run_composite_disable_labels(tmp_path):
    from PIL import Image

    from paperbanana.studio.runner import run_composite

    p = tmp_path / "x.png"
    Image.new("RGB", (60, 60), (0, 0, 255)).save(str(p))
    log, output_path = run_composite(
        [str(p)],
        output_dir=str(tmp_path / "out"),
        labels="none",
    )
    assert output_path is not None
    assert "Done." in log


def test_run_composite_zero_spacing_allowed(tmp_path):
    from PIL import Image

    from paperbanana.studio.runner import run_composite

    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    Image.new("RGB", (40, 40), (255, 0, 0)).save(str(p1))
    Image.new("RGB", (40, 40), (0, 255, 0)).save(str(p2))

    log, output_path = run_composite(
        [str(p1), str(p2)],
        output_dir=str(tmp_path / "out"),
        layout="1x2",
        spacing=0,
    )
    assert output_path is not None
    assert "Done." in log


def test_run_composite_negative_spacing_rejected(tmp_path):
    from PIL import Image

    from paperbanana.studio.runner import run_composite

    p = tmp_path / "x.png"
    Image.new("RGB", (40, 40), (255, 0, 0)).save(str(p))
    log, output_path = run_composite(
        [str(p)],
        output_dir=str(tmp_path / "out"),
        spacing=-5,
    )
    assert output_path is None
    assert "spacing" in log


def test_run_composite_invalid_font_size_rejected(tmp_path):
    from PIL import Image

    from paperbanana.studio.runner import run_composite

    p = tmp_path / "x.png"
    Image.new("RGB", (40, 40), (255, 0, 0)).save(str(p))
    log, output_path = run_composite(
        [str(p)],
        output_dir=str(tmp_path / "out"),
        label_font_size=0,
    )
    assert output_path is None
    assert "label_font_size" in log


def test_run_composite_path_traversal_sanitized(tmp_path):
    from PIL import Image

    from paperbanana.studio.runner import run_composite

    p = tmp_path / "x.png"
    Image.new("RGB", (40, 40), (255, 0, 0)).save(str(p))

    out_dir = tmp_path / "out"
    log, output_path = run_composite(
        [str(p)],
        output_dir=str(out_dir),
        output_filename="../escape.png",
    )
    assert output_path is not None
    # Output must stay inside the configured output_dir
    assert Path(output_path).parent.resolve() == out_dir.resolve()
    assert Path(output_path).name == "escape.png"
    assert not (tmp_path / "escape.png").exists()


def test_run_composite_dotdot_filename_falls_back(tmp_path):
    from PIL import Image

    from paperbanana.studio.runner import run_composite

    p = tmp_path / "x.png"
    Image.new("RGB", (40, 40), (255, 0, 0)).save(str(p))

    out_dir = tmp_path / "out"
    log, output_path = run_composite(
        [str(p)],
        output_dir=str(out_dir),
        output_filename="..",
    )
    assert output_path is not None
    assert Path(output_path).name == "composite.png"


def test_run_evaluate_plot_requires_data_file(tmp_path):
    """Plot evaluation mode validates data path before provider setup."""
    from paperbanana.core.config import Settings
    from paperbanana.core.types import DiagramType
    from paperbanana.studio.runner import run_evaluate

    generated = tmp_path / "g.png"
    reference = tmp_path / "r.png"
    generated.write_bytes(b"x")
    reference.write_bytes(b"y")

    log, result = run_evaluate(
        Settings(),
        generated_path=str(generated),
        reference_path=str(reference),
        source_context="",
        caption="Plot intent",
        evaluation_task=DiagramType.STATISTICAL_PLOT,
        plot_data_path=str(tmp_path / "missing.csv"),
    )
    assert "Plot data file not found" in log
    assert "Plot data file not found" in result
