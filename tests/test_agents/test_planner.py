"""Tests for planner agent formatting behavior."""

from __future__ import annotations

from io import BytesIO

from PIL import Image

from paperbanana.agents.planner import PlannerAgent
from paperbanana.core.types import ReferenceExample


class _MockVLM:
    name = "mock-vlm"
    model_name = "mock-model"

    async def generate(self, *args, **kwargs):
        return "ok"


def test_format_examples_includes_structure_hints():
    agent = PlannerAgent(_MockVLM())
    text = agent._format_examples(
        [
            ReferenceExample(
                id="ref_001",
                source_context="context",
                caption="caption",
                image_path="",
                structure_hints={"nodes": ["A"], "edges": ["A->B"]},
            )
        ]
    )

    assert "Structure Hints" in text
    assert "nodes" in text


def test_has_valid_image_accepts_safe_https_url():
    """_has_valid_image accepts safe https URLs."""
    agent = PlannerAgent(_MockVLM())
    ex = ReferenceExample(
        id="x",
        source_context="",
        caption="",
        image_path="https://example.com/diagram.png",
    )
    assert agent._has_valid_image(ex) is True


def test_has_valid_image_rejects_insecure_or_local_urls():
    """_has_valid_image rejects insecure schemes and localhost/private targets."""
    agent = PlannerAgent(_MockVLM())
    insecure = ReferenceExample(
        id="x",
        source_context="",
        caption="",
        image_path="http://example.com/fig.png",
    )
    localhost = ReferenceExample(
        id="x",
        source_context="",
        caption="",
        image_path="https://localhost/fig.png",
    )
    private_ip = ReferenceExample(
        id="x",
        source_context="",
        caption="",
        image_path="https://10.0.0.12/fig.png",
    )
    assert agent._has_valid_image(insecure) is False
    assert agent._has_valid_image(localhost) is False
    assert agent._has_valid_image(private_ip) is False


def test_load_example_images_loads_from_url(monkeypatch):
    """_load_example_images fetches and loads images from http(s) URLs."""
    agent = PlannerAgent(_MockVLM())
    # 1x1 red PNG bytes
    buf = BytesIO()
    Image.new("RGB", (1, 1), color=(255, 0, 0)).save(buf, format="PNG")
    image = Image.open(BytesIO(buf.getvalue())).convert("RGB")
    monkeypatch.setattr(agent, "_fetch_remote_image", lambda _url: image)
    examples = [
        ReferenceExample(
            id="ext_1",
            source_context="ctx",
            caption="cap",
            image_path="https://example.com/ref.png",
        )
    ]
    images = agent._load_example_images(examples)
    assert len(images) == 1
    assert images[0].size == (1, 1)
