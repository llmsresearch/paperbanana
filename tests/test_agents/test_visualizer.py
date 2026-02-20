"""Tests for VisualizerAgent code extraction edge cases."""

from __future__ import annotations

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
