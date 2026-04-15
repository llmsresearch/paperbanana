"""Tests for VisualizerAgent code extraction edge cases and vector export."""

from __future__ import annotations

from pathlib import Path

from paperbanana.agents.visualizer import VisualizerAgent


class _DummyImageGen:
    async def generate(self, *args, **kwargs):
        return None


class _DummyVLM:
    async def generate(self, *args, **kwargs):
        return ""


def _make_agent(tmp_path):
    return VisualizerAgent(
        image_gen=_DummyImageGen(),
        vlm_provider=_DummyVLM(),
        prompt_dir=str(tmp_path),
        output_dir=str(tmp_path),
    )


def test_extract_code_handles_truncated_python_block(tmp_path):
    agent = _make_agent(tmp_path)
    response = "```python\nimport matplotlib.pyplot as plt\nplt.figure()\n"
    code = agent._extract_code(response)
    assert code == "import matplotlib.pyplot as plt\nplt.figure()"


def test_extract_code_handles_truncated_generic_block(tmp_path):
    agent = _make_agent(tmp_path)
    response = "```\nprint('hello')\n"
    code = agent._extract_code(response)
    assert code == "print('hello')"


def test_extract_code_handles_complete_python_block(tmp_path):
    agent = _make_agent(tmp_path)
    response = "```python\nprint('ok')\n```\nextra"
    code = agent._extract_code(response)
    assert code == "print('ok')"


def test_extract_code_handles_plain_code_response(tmp_path):
    agent = _make_agent(tmp_path)
    response = "import matplotlib.pyplot as plt\nplt.figure()"
    code = agent._extract_code(response)
    assert code == response


# ── Vector export tests ───────────────────────────────────────────────────────

_SIMPLE_PLOT_CODE = """\
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])
plt.savefig(OUTPUT_PATH, bbox_inches='tight')
"""


def test_execute_plot_code_produces_raster_only_by_default(tmp_path):
    agent = _make_agent(tmp_path)
    output_path = str(tmp_path / "plot.png")
    success = agent._execute_plot_code(_SIMPLE_PLOT_CODE, output_path)
    assert success
    assert Path(output_path).exists()
    assert agent._last_vector_paths == {}


def test_execute_plot_code_produces_svg_and_pdf_when_requested(tmp_path):
    agent = _make_agent(tmp_path)
    output_path = str(tmp_path / "plot.png")
    success = agent._execute_plot_code(
        _SIMPLE_PLOT_CODE, output_path, vector_formats=["svg", "pdf"]
    )
    assert success
    assert Path(output_path).exists()
    assert "svg" in agent._last_vector_paths
    assert "pdf" in agent._last_vector_paths
    assert Path(agent._last_vector_paths["svg"]).exists()
    assert Path(agent._last_vector_paths["pdf"]).exists()


def test_execute_plot_code_svg_path_has_correct_suffix(tmp_path):
    agent = _make_agent(tmp_path)
    output_path = str(tmp_path / "my_plot.png")
    agent._execute_plot_code(_SIMPLE_PLOT_CODE, output_path, vector_formats=["svg"])
    assert agent._last_vector_paths["svg"] == str(tmp_path / "my_plot.svg")


def test_execute_plot_code_strips_vlm_vector_path_assignments(tmp_path):
    """VLM-injected VECTOR_PATH_* assignments must be overridden by our header."""
    agent = _make_agent(tmp_path)
    output_path = str(tmp_path / "plot.png")
    code_with_stale_path = 'VECTOR_PATH_SVG = "/stale/path.svg"\n' + _SIMPLE_PLOT_CODE
    success = agent._execute_plot_code(code_with_stale_path, output_path, vector_formats=["svg"])
    assert success
    # Our injected path, not the stale one, should be used
    assert agent._last_vector_paths.get("svg") == str(tmp_path / "plot.svg")


def test_execute_plot_code_sets_empty_vector_paths_on_failure(tmp_path):
    agent = _make_agent(tmp_path)
    output_path = str(tmp_path / "plot.png")
    bad_code = "raise RuntimeError('intentional failure')"
    success = agent._execute_plot_code(bad_code, output_path, vector_formats=["svg", "pdf"])
    assert not success
    assert agent._last_vector_paths == {}


def test_last_vector_paths_reset_on_each_run_call(tmp_path):
    """_last_vector_paths from a previous run must not bleed into the next."""
    agent = _make_agent(tmp_path)
    output_path = str(tmp_path / "plot.png")
    # First call with vector export
    agent._execute_plot_code(_SIMPLE_PLOT_CODE, output_path, vector_formats=["svg"])
    assert "svg" in agent._last_vector_paths
    # Second call without vector export
    output_path2 = str(tmp_path / "plot2.png")
    agent._execute_plot_code(_SIMPLE_PLOT_CODE, output_path2)
    assert agent._last_vector_paths == {}
