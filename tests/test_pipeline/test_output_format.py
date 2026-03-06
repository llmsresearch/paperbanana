"""Tests for output format behavior in the pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from paperbanana.core.config import Settings
from paperbanana.core.pipeline import PaperBananaPipeline
from paperbanana.core.types import DiagramType, GenerationInput


class FakeVLM:
    """Minimal VLM stub for pipeline tests."""

    name = "fake-vlm"
    model_name = "fake-model"

    async def generate(self, *args, **kwargs):
        return "fake response"


class FakeImageGen:
    """Minimal image gen stub that returns a PIL Image."""

    async def generate(self, prompt=None, output_path=None, iteration=None, seed=None, **kwargs):
        iteration = iteration or 1
        img = Image.new("RGB", (256, 256), color=(iteration * 40 % 256, 100, 150))
        return img


@pytest.fixture
def empty_reference_dir(tmp_path):
    """Create a temp dir with no index.json (empty reference set)."""
    return tmp_path


@pytest.mark.asyncio
async def test_pipeline_default_output_is_png(empty_reference_dir):
    """Default behavior produces PNG output."""
    settings = Settings(
        reference_set_path=str(empty_reference_dir),
        output_dir=str(empty_reference_dir / "out"),
        refinement_iterations=1,
        save_iterations=False,
    )
    pipeline = PaperBananaPipeline(
        settings=settings,
        vlm_client=FakeVLM(),
        image_gen_fn=FakeImageGen(),
    )

    result = await pipeline.generate(
        GenerationInput(
            source_context="Test methodology.",
            communicative_intent="Test caption",
            diagram_type=DiagramType.METHODOLOGY,
        )
    )

    assert result.image_path.endswith(".png")
    assert Path(result.image_path).exists()


@pytest.mark.asyncio
async def test_pipeline_jpeg_output_extension(empty_reference_dir):
    """--format jpeg produces .jpg final output."""
    settings = Settings(
        reference_set_path=str(empty_reference_dir),
        output_dir=str(empty_reference_dir / "out"),
        output_format="jpeg",
        refinement_iterations=1,
        save_iterations=False,
    )
    pipeline = PaperBananaPipeline(
        settings=settings,
        vlm_client=FakeVLM(),
        image_gen_fn=FakeImageGen(),
    )

    result = await pipeline.generate(
        GenerationInput(
            source_context="Test methodology.",
            communicative_intent="Test caption",
            diagram_type=DiagramType.METHODOLOGY,
        )
    )

    assert result.image_path.endswith(".jpg")
    assert Path(result.image_path).exists()


@pytest.mark.asyncio
async def test_pipeline_webp_output_extension(empty_reference_dir):
    """--format webp produces .webp final output."""
    settings = Settings(
        reference_set_path=str(empty_reference_dir),
        output_dir=str(empty_reference_dir / "out"),
        output_format="webp",
        refinement_iterations=1,
        save_iterations=False,
    )
    pipeline = PaperBananaPipeline(
        settings=settings,
        vlm_client=FakeVLM(),
        image_gen_fn=FakeImageGen(),
    )

    result = await pipeline.generate(
        GenerationInput(
            source_context="Test methodology.",
            communicative_intent="Test caption",
            diagram_type=DiagramType.METHODOLOGY,
        )
    )

    assert result.image_path.endswith(".webp")
    assert Path(result.image_path).exists()


def test_cli_invalid_format_rejected():
    """Invalid format via CLI is rejected cleanly."""
    from typer.testing import CliRunner

    from paperbanana.cli import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "generate",
            "--input",
            "nonexistent.txt",  # Will fail on file check first
            "--caption",
            "Test",
            "--format",
            "gif",
        ],
    )
    # Either file-not-found or format validation - we want format to be validated
    # Format check runs before file load, so we should get format error
    assert result.exit_code != 0
    assert "png, jpeg, or webp" in result.output
