"""Tests for _build_final_output() helper (issue #154)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from paperbanana.core.config import Settings
from paperbanana.core.pipeline import PaperBananaPipeline
from paperbanana.core.types import CritiqueResult, IterationRecord

# ── Helpers ─────────────────────────────────────────────────────────


class _StubVLM:
    name = "stub-vlm"
    model_name = "stub-model"

    async def generate(self, *args, **kwargs):
        return "stub"


class _StubImageGen:
    name = "stub-image-gen"
    model_name = "stub-image-model"

    async def generate(self, *args, **kwargs):
        return Image.new("RGB", (64, 64), color=(0, 0, 0))


def _make_pipeline(tmp_path: Path, output_format: str = "png") -> PaperBananaPipeline:
    settings = Settings(
        reference_set_path=str(tmp_path / "refs"),
        output_dir=str(tmp_path / "out"),
        refinement_iterations=1,
        save_iterations=False,
        output_format=output_format,
    )
    return PaperBananaPipeline(
        settings=settings,
        vlm_client=_StubVLM(),
        image_gen_fn=_StubImageGen(),
    )


def _create_source_image(path: Path) -> str:
    """Create a minimal PNG on disk and return its string path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=(128, 128, 128)).save(str(path))
    return str(path)


def _make_iteration(image_path: str, iteration: int = 1) -> IterationRecord:
    return IterationRecord(
        iteration=iteration,
        description="test description",
        image_path=image_path,
        critique=CritiqueResult(critic_suggestions=[], revised_description=None),
    )


# ── Tests: return value & file creation ─────────────────────────────


def test_returns_png_path_and_creates_file(tmp_path):
    """With iterations and default format, returns .png path and writes file."""
    pipeline = _make_pipeline(tmp_path, output_format="png")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    src = _create_source_image(run_dir / "iter_1.png")

    result = pipeline._build_final_output(
        [_make_iteration(src)],
        run_dir,
        "should not appear",
    )

    assert result == str(run_dir / "final_output.png")
    assert Path(result).exists()


def test_jpeg_format_uses_jpg_extension(tmp_path):
    """output_format='jpeg' produces a .jpg extension."""
    pipeline = _make_pipeline(tmp_path, output_format="jpeg")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    src = _create_source_image(run_dir / "iter_1.png")

    result = pipeline._build_final_output(
        [_make_iteration(src)],
        run_dir,
        "should not appear",
    )

    assert result.endswith(".jpg")
    assert Path(result).exists()


def test_webp_format(tmp_path):
    """output_format='webp' produces a .webp file."""
    pipeline = _make_pipeline(tmp_path, output_format="webp")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    src = _create_source_image(run_dir / "iter_1.png")

    result = pipeline._build_final_output(
        [_make_iteration(src)],
        run_dir,
        "should not appear",
    )

    assert result.endswith(".webp")
    assert Path(result).exists()


def test_uses_last_iteration_image(tmp_path):
    """When multiple iterations exist, the *last* one is used."""
    pipeline = _make_pipeline(tmp_path, output_format="png")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    # Two source images with different colours so we can distinguish them.
    img1_path = run_dir / "iter_1.png"
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(str(img1_path))
    img2_path = run_dir / "iter_2.png"
    Image.new("RGB", (32, 32), color=(0, 255, 0)).save(str(img2_path))

    result = pipeline._build_final_output(
        [
            _make_iteration(str(img1_path), iteration=1),
            _make_iteration(str(img2_path), iteration=2),
        ],
        run_dir,
        "should not appear",
    )

    # Verify the final output matches the second image's colour.
    final_img = Image.open(result)
    r, g, b = final_img.getpixel((0, 0))
    assert g > r  # green channel dominant — from iter_2


# ── Tests: empty iterations ─────────────────────────────────────────


def test_empty_iterations_returns_empty_string(tmp_path):
    """No iterations → returns empty string."""
    pipeline = _make_pipeline(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    result = pipeline._build_final_output([], run_dir, "budget warning")

    assert result == ""


def test_empty_iterations_logs_warning(tmp_path, capsys):
    """No iterations → warning message is logged."""
    pipeline = _make_pipeline(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    pipeline._build_final_output([], run_dir, "budget exceeded during test")

    captured = capsys.readouterr()
    assert "budget exceeded during test" in captured.out


# ── Tests: SVG format skips raster save ─────────────────────────────


def test_svg_format_skips_raster_save(tmp_path):
    """output_format='svg' returns the path but does NOT write a raster file."""
    pipeline = _make_pipeline(tmp_path, output_format="png")
    # SVG bypasses the Settings validator; set it post-construction like the
    # existing SVG tests in test_output_format.py do.
    pipeline.settings.output_format = "svg"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    src = _create_source_image(run_dir / "iter_1.png")

    result = pipeline._build_final_output(
        [_make_iteration(src)],
        run_dir,
        "should not appear",
    )

    assert result.endswith(".svg")
    # Helper must NOT write the file — SVG saving is caller's responsibility.
    assert not Path(result).exists()


# ── Tests: run_dir is respected ─────────────────────────────────────


def test_output_written_to_given_run_dir(tmp_path):
    """The file is created inside the provided run_dir, not elsewhere."""
    pipeline = _make_pipeline(tmp_path, output_format="png")
    custom_dir = tmp_path / "custom" / "dir"
    custom_dir.mkdir(parents=True)
    src = _create_source_image(custom_dir / "source.png")

    result = pipeline._build_final_output(
        [_make_iteration(src)],
        custom_dir,
        "should not appear",
    )

    assert Path(result).parent == custom_dir
    assert Path(result).exists()


# ── Tests: signature matches issue #154 ─────────────────────────────


def test_method_is_synchronous(tmp_path):
    """_build_final_output is a regular method, not async."""
    pipeline = _make_pipeline(tmp_path)
    import inspect

    assert not inspect.iscoroutinefunction(pipeline._build_final_output)


def test_return_type_is_str(tmp_path):
    """Return value is a plain str, not a tuple or other type."""
    pipeline = _make_pipeline(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    src = _create_source_image(run_dir / "iter_1.png")

    result = pipeline._build_final_output(
        [_make_iteration(src)],
        run_dir,
        "warn",
    )

    assert isinstance(result, str)
