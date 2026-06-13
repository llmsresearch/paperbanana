"""Table parsing / re-setting / recharting tests, plus visualizer strict mode."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from paperbanana.agents.visualizer import PlotExecutionError, VisualizerAgent
from paperbanana.core.types import DiagramType
from paperbanana.poster.tables import (
    TableData,
    parse_table,
    render_table_matplotlib,
    table_to_plot_payload,
    verify_chart_numbers,
)

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"
PALETTE = {
    "primary": "#1A3A6B",
    "secondary": "#4A6FA5",
    "accent": "#E8A33D",
    "background": "#FFFFFF",
    "panel_bg": "#F5F7FA",
    "text": "#1A1A1A",
}


class _ScriptedVLM:
    name = "mock"
    model_name = "mock-model"
    cost_tracker = None

    def __init__(self, responses):
        self._responses = list(responses)

    async def generate(self, prompt, images=None, **kwargs):
        if not self._responses:
            raise AssertionError("scripted VLM exhausted")
        return self._responses.pop(0)


def _table() -> TableData:
    return TableData(
        columns=["Method", "PPL", "Bits"],
        rows=[["Baseline", "5.21", "16"], ["Ours", "5.18", "6.56"]],
        title="Main results",
        highlight_row=1,
    )


def test_numbers_extracts_every_numeric_token():
    assert _table().numbers() == ["5.21", "16", "5.18", "6.56"]


async def test_parse_table_valid():
    crop = Image.new("RGB", (400, 200), "white")
    vlm = _ScriptedVLM([json.dumps(_table().model_dump())])
    table = await parse_table(crop, vlm, PROMPT_DIR, caption="Table 1: results")
    assert table.columns == ["Method", "PPL", "Bits"]
    assert table.highlight_row == 1


async def test_parse_table_rejects_ragged_rows():
    crop = Image.new("RGB", (400, 200), "white")
    ragged = {"columns": ["A", "B"], "rows": [["1", "2"], ["only-one"]]}
    vlm = _ScriptedVLM([json.dumps(ragged)])
    with pytest.raises(ValueError, match="ragged"):
        await parse_table(crop, vlm, PROMPT_DIR)


def test_render_table_matplotlib_writes_print_dpi_png(tmp_path: Path):
    out = render_table_matplotlib(_table(), PALETTE, tmp_path / "t.png", width_mm=300.0)
    assert out.is_file()
    with Image.open(out) as img:
        assert img.width > 2000  # 300mm at 300 DPI (minus tight bbox) is print-scale


def test_table_to_plot_payload_carries_ground_truth():
    payload = table_to_plot_payload(_table(), "grouped_bar")
    assert payload["rows"] == _table().rows
    assert payload["requested_chart_kind"] == "grouped_bar"
    assert payload["highlight_row"] == 1


def test_verify_chart_numbers_reports_missing():
    assert verify_chart_numbers(_table(), ["5.21", "16", "5.18"]) == ["6.56"]
    assert verify_chart_numbers(_table(), _table().numbers()) == []


async def test_visualizer_strict_raises_instead_of_placeholder(tmp_path: Path):
    vlm = _ScriptedVLM(["```python\nimport nonexistent_module_xyz\n```"])
    visualizer = VisualizerAgent(
        image_gen=None, vlm_provider=vlm, prompt_dir=str(PROMPT_DIR), output_dir=str(tmp_path)
    )
    out = tmp_path / "chart.png"
    with pytest.raises(PlotExecutionError, match="strict mode forbids placeholder"):
        await visualizer.run(
            description="bar chart",
            diagram_type=DiagramType.STATISTICAL_PLOT,
            raw_data={"x": [1]},
            output_path=str(out),
            strict=True,
        )
    assert not out.is_file()  # no white-placeholder fallback


async def test_visualizer_default_still_emits_placeholder(tmp_path: Path):
    """The diagram pipeline's lenient default is unchanged."""
    vlm = _ScriptedVLM(["```python\nimport nonexistent_module_xyz\n```"])
    visualizer = VisualizerAgent(
        image_gen=None, vlm_provider=vlm, prompt_dir=str(PROMPT_DIR), output_dir=str(tmp_path)
    )
    out = tmp_path / "chart.png"
    result = await visualizer.run(
        description="bar chart",
        diagram_type=DiagramType.STATISTICAL_PLOT,
        raw_data={"x": [1]},
        output_path=str(out),
    )
    assert Path(result).is_file()


async def test_parse_table_passes_generation_controls():
    """Wide tables need token headroom; retries need temperature variation."""
    captured = {}

    class _CapturingVLM:
        name = "mock"
        model_name = "mock-model"
        cost_tracker = None

        async def generate(self, prompt, images=None, **kwargs):
            captured.update(kwargs)
            return json.dumps(_table().model_dump())

    await parse_table(
        Image.new("RGB", (400, 200), "white"), _CapturingVLM(), PROMPT_DIR, temperature=0.7
    )
    assert captured["max_tokens"] >= 8192
    assert captured["temperature"] == 0.7
