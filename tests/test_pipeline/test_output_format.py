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


# ── Vector export integration tests ──────────────────────────────

# Minimal matplotlib code the mock VLM returns for statistical plot generation.
# Uses Agg backend to avoid display issues in CI; saves to the injected OUTPUT_PATH.
_MOCK_PLOT_VLM_RESPONSE = """\
```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])
plt.savefig(OUTPUT_PATH, bbox_inches='tight')
```"""


class FakeVLMSequenced:
    """Returns pre-configured responses in order, cycling on the last one."""

    name = "fake-vlm"
    model_name = "fake-model"

    def __init__(self, responses: list[str]):
        self._responses = responses
        self._idx = 0

    async def generate(self, *args, **kwargs):
        idx = min(self._idx, len(self._responses) - 1)
        self._idx += 1
        return self._responses[idx]


@pytest.mark.asyncio
async def test_vector_export_produces_svg_and_pdf_for_statistical_plot(empty_reference_dir):
    """vector_export=True generates SVG and PDF alongside raster for statistical plots."""
    import json as json_mod

    critic_response = json_mod.dumps({"critic_suggestions": [], "revised_description": None})

    # VLM call order: planner → stylist → visualizer (matplotlib code) → critic
    vlm = FakeVLMSequenced(
        [
            "A bar chart comparing model accuracy across datasets",
            "A clean bar chart with publication-quality styling",
            _MOCK_PLOT_VLM_RESPONSE,
            critic_response,
        ]
    )

    settings = Settings(
        reference_set_path=str(empty_reference_dir),
        output_dir=str(empty_reference_dir / "out"),
        refinement_iterations=1,
        save_iterations=True,
        vector_export=True,
    )
    pipeline = PaperBananaPipeline(
        settings=settings,
        vlm_client=vlm,
        image_gen_fn=FakeImageGen(),
    )

    result = await pipeline.generate(
        GenerationInput(
            source_context="Comparison of accuracy across five benchmark datasets.",
            communicative_intent="Accuracy comparison bar chart",
            diagram_type=DiagramType.STATISTICAL_PLOT,
        )
    )

    # Raster output must exist
    assert Path(result.image_path).exists()

    # Vector paths must be recorded in metadata
    assert "vector_output_paths" in result.metadata
    vector_paths = result.metadata["vector_output_paths"]
    assert "svg" in vector_paths
    assert "pdf" in vector_paths
    assert Path(vector_paths["svg"]).exists()
    assert Path(vector_paths["pdf"]).exists()


@pytest.mark.asyncio
async def test_vector_export_not_in_metadata_when_disabled(empty_reference_dir):
    """Without vector_export, vector_output_paths is absent from metadata."""
    settings = Settings(
        reference_set_path=str(empty_reference_dir),
        output_dir=str(empty_reference_dir / "out"),
        refinement_iterations=1,
        save_iterations=False,
        vector_export=False,
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

    assert "vector_output_paths" not in result.metadata


@pytest.mark.asyncio
async def test_vector_export_not_in_metadata_for_methodology_diagram(empty_reference_dir):
    """vector_export=True has no effect on methodology diagrams — no vector paths in metadata."""
    settings = Settings(
        reference_set_path=str(empty_reference_dir),
        output_dir=str(empty_reference_dir / "out"),
        refinement_iterations=1,
        save_iterations=False,
        vector_export=True,
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

    # Methodology diagrams use image gen, not matplotlib — no vector output
    assert "vector_output_paths" not in result.metadata


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
