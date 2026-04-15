"""Tests for paperbanana.core.composite — image composition and manifest parsing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from paperbanana.core.composite import (
    _auto_labels,
    _parse_layout,
    compose_images,
    parse_composite_config,
)

# ---------------------------------------------------------------------------
# _auto_labels
# ---------------------------------------------------------------------------


def test_auto_labels():
    assert _auto_labels(3) == ["(a)", "(b)", "(c)"]
    assert _auto_labels(1) == ["(a)"]
    assert _auto_labels(0) == []


# ---------------------------------------------------------------------------
# _parse_layout
# ---------------------------------------------------------------------------


def test_parse_layout_explicit():
    assert _parse_layout("1x3", 3) == (1, 3)
    assert _parse_layout("2x2", 4) == (2, 2)
    assert _parse_layout("3x1", 3) == (3, 1)


def test_parse_layout_auto():
    assert _parse_layout("auto", 2) == (1, 2)
    assert _parse_layout("auto", 3) == (1, 3)
    assert _parse_layout("auto", 4) == (2, 2)
    assert _parse_layout("auto", 6) == (2, 3)
    assert _parse_layout("auto", 9) == (3, 3)
    assert _parse_layout("auto", 10) == (3, 4)


def test_parse_layout_not_enough_cells():
    with pytest.raises(ValueError, match="cannot fit"):
        _parse_layout("1x2", 3)


def test_parse_layout_invalid_format():
    with pytest.raises(ValueError, match="RxC"):
        _parse_layout("abc", 2)


def test_parse_layout_zero():
    with pytest.raises(ValueError, match=">= 1"):
        _parse_layout("0x3", 1)


# ---------------------------------------------------------------------------
# compose_images
# ---------------------------------------------------------------------------


def _make_test_images(tmp_path: Path, count: int, size: tuple[int, int] = (200, 150)):
    """Create simple test images and return their paths."""
    paths = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    for i in range(count):
        img = Image.new("RGB", size, colors[i % len(colors)])
        p = tmp_path / f"panel_{i}.png"
        img.save(str(p))
        paths.append(str(p))
    return paths


def test_compose_images_basic(tmp_path: Path):
    paths = _make_test_images(tmp_path, 3)
    result = compose_images(paths, layout="1x3")
    assert isinstance(result, Image.Image)
    assert result.size[0] > 200  # wider than a single panel
    assert result.size[1] > 0


def test_compose_images_auto_layout(tmp_path: Path):
    paths = _make_test_images(tmp_path, 4)
    result = compose_images(paths, layout="auto")
    assert isinstance(result, Image.Image)


def test_compose_images_saves_to_file(tmp_path: Path):
    paths = _make_test_images(tmp_path, 2)
    out = tmp_path / "composite.png"
    result = compose_images(paths, layout="1x2", output_path=out)
    assert out.exists()
    reopened = Image.open(out)
    assert reopened.size == result.size


def test_compose_images_custom_labels(tmp_path: Path):
    paths = _make_test_images(tmp_path, 2)
    result = compose_images(paths, labels=["Fig A", "Fig B"])
    assert isinstance(result, Image.Image)


def test_compose_images_no_labels(tmp_path: Path):
    paths = _make_test_images(tmp_path, 2)
    result = compose_images(paths, auto_label=False)
    assert isinstance(result, Image.Image)


def test_compose_images_label_count_mismatch(tmp_path: Path):
    paths = _make_test_images(tmp_path, 2)
    with pytest.raises(ValueError, match="Expected 2 labels"):
        compose_images(paths, labels=["(a)"])


def test_compose_images_empty_raises():
    with pytest.raises(ValueError, match="At least one"):
        compose_images([])


def test_compose_images_top_labels(tmp_path: Path):
    paths = _make_test_images(tmp_path, 3)
    result = compose_images(paths, layout="1x3", label_position="top")
    assert isinstance(result, Image.Image)


def test_compose_images_different_sizes(tmp_path: Path):
    """Panels of different sizes should be scaled to equal height."""
    img1 = Image.new("RGB", (300, 200), (255, 0, 0))
    img2 = Image.new("RGB", (100, 400), (0, 255, 0))
    p1 = tmp_path / "wide.png"
    p2 = tmp_path / "tall.png"
    img1.save(str(p1))
    img2.save(str(p2))
    result = compose_images([str(p1), str(p2)], layout="1x2")
    assert isinstance(result, Image.Image)


def test_compose_images_2x2_grid(tmp_path: Path):
    paths = _make_test_images(tmp_path, 4)
    result = compose_images(paths, layout="2x2", spacing=10)
    assert isinstance(result, Image.Image)
    # 2x2 should be roughly square-ish
    w, h = result.size
    assert w > 0 and h > 0


# ---------------------------------------------------------------------------
# parse_composite_config
# ---------------------------------------------------------------------------


def test_parse_composite_config_none():
    assert parse_composite_config({"items": []}) is None


def test_parse_composite_config_auto_labels():
    config = parse_composite_config({"items": [], "composite": {"layout": "1x3", "labels": "auto"}})
    assert config is not None
    assert config["layout"] == "1x3"
    assert config["auto_label"] is True
    assert config["labels"] is None


def test_parse_composite_config_explicit_labels():
    config = parse_composite_config({"items": [], "composite": {"labels": ["(a)", "(b)"]}})
    assert config is not None
    assert config["auto_label"] is False
    assert config["labels"] == ["(a)", "(b)"]


def test_parse_composite_config_no_labels():
    config = parse_composite_config({"items": [], "composite": {"labels": None}})
    assert config is not None
    assert config["auto_label"] is False
    assert config["labels"] is None


def test_parse_composite_config_defaults():
    config = parse_composite_config({"items": [], "composite": {}})
    assert config is not None
    assert config["layout"] == "auto"
    assert config["auto_label"] is True
    assert config["spacing"] == 20
    assert config["label_position"] == "bottom"


def test_parse_composite_config_custom_spacing():
    config = parse_composite_config(
        {"items": [], "composite": {"spacing": 40, "label_position": "top"}}
    )
    assert config["spacing"] == 40
    assert config["label_position"] == "top"


def test_parse_composite_config_invalid_spacing():
    with pytest.raises(ValueError, match="spacing"):
        parse_composite_config({"items": [], "composite": {"spacing": -5}})


def test_parse_composite_config_invalid_label_position():
    with pytest.raises(ValueError, match="label_position"):
        parse_composite_config({"items": [], "composite": {"label_position": "left"}})


def test_parse_composite_config_output():
    config = parse_composite_config({"items": [], "composite": {"output": "figure2.png"}})
    assert config["output"] == "figure2.png"


# ---------------------------------------------------------------------------
# load_batch_manifest_with_composite
# ---------------------------------------------------------------------------


def test_load_batch_manifest_with_composite_no_composite(tmp_path: Path):
    from paperbanana.core.batch import load_batch_manifest_with_composite

    txt = tmp_path / "a.txt"
    txt.write_text("x", encoding="utf-8")
    m = tmp_path / "m.yaml"
    m.write_text(
        f"""items:
  - input: {txt.name}
    caption: "Fig 1"
""",
        encoding="utf-8",
    )
    items, comp = load_batch_manifest_with_composite(m)
    assert len(items) == 1
    assert comp is None


def test_load_batch_manifest_with_composite_has_composite(tmp_path: Path):
    from paperbanana.core.batch import load_batch_manifest_with_composite

    txt = tmp_path / "a.txt"
    txt.write_text("x", encoding="utf-8")
    m = tmp_path / "m.yaml"
    m.write_text(
        f"""composite:
  layout: "1x2"
  labels: auto
  spacing: 30
items:
  - input: {txt.name}
    caption: "Fig A"
  - input: {txt.name}
    caption: "Fig B"
""",
        encoding="utf-8",
    )
    items, comp = load_batch_manifest_with_composite(m)
    assert len(items) == 2
    assert comp is not None
    assert comp["layout"] == "1x2"
    assert comp["spacing"] == 30


def test_load_batch_manifest_with_composite_json(tmp_path: Path):
    from paperbanana.core.batch import load_batch_manifest_with_composite

    txt = tmp_path / "a.txt"
    txt.write_text("x", encoding="utf-8")
    m = tmp_path / "m.json"
    m.write_text(
        json.dumps(
            {
                "composite": {"layout": "auto", "output": "out.png"},
                "items": [
                    {"input": txt.name, "caption": "c1"},
                ],
            }
        ),
        encoding="utf-8",
    )
    items, comp = load_batch_manifest_with_composite(m)
    assert len(items) == 1
    assert comp is not None
    assert comp["output"] == "out.png"
