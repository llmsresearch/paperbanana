"""Figure-embedding tests: sentinel-slot detection, compositing, selection."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from paperbanana.poster.figure_embed import (
    SENTINEL_RGB,
    composite_into_slot,
    detect_slots,
    select_poster_figures,
    slot_spec,
)
from paperbanana.poster.types import PaperFigure


def _fig(tmp_path: Path, fid: str, kind: str, page: int, size=(800, 600)) -> PaperFigure:
    p = tmp_path / f"{fid}.png"
    Image.new("RGB", size, "gray").save(p)
    return PaperFigure(
        id=fid,
        kind=kind,
        page=page,
        bbox_norm=[0.1, 0.1, 0.9, 0.5],
        caption=f"{fid} caption",
        image_path=str(p),
        extract_dpi=300,
    )


def test_select_prefers_figures_then_order(tmp_path: Path):
    figs = [
        _fig(tmp_path, "tab1", "table", 2),
        _fig(tmp_path, "fig2", "figure", 3),
        _fig(tmp_path, "fig1", "figure", 1),
    ]
    picked = select_poster_figures(figs, max_figures=2)
    assert [f.id for f in picked] == ["fig1", "fig2"]  # figures first, by page


def test_slot_spec_counts_and_aspects(tmp_path: Path):
    figs = [_fig(tmp_path, "fig1", "figure", 1, (1600, 400))]  # wide
    spec = slot_spec(figs)
    assert "EXACTLY 1" in spec and "magenta" in spec.lower() and "wide" in spec


def test_detect_two_magenta_slots():
    poster = Image.new("RGB", (1000, 800), "white")
    # two magenta rectangles
    for box in [(50, 50, 400, 350), (550, 50, 900, 350)]:
        for x in range(box[0], box[2]):
            for y in range(box[1], box[3]):
                poster.putpixel((x, y), SENTINEL_RGB)
    slots = detect_slots(poster)
    assert len(slots) == 2
    # reading order: left box first
    assert slots[0].x < slots[1].x
    assert slots[0].w > 300 and slots[0].h > 250


def test_detect_ignores_tiny_specks():
    poster = Image.new("RGB", (1000, 800), "white")
    poster.putpixel((10, 10), SENTINEL_RGB)  # single stray magenta pixel
    assert detect_slots(poster) == []


def test_composite_fills_slot_and_replaces_magenta():
    from paperbanana.poster.figure_embed import SlotBox

    poster = Image.new("RGB", (1000, 800), "white")
    slot = SlotBox(x=100, y=100, w=400, h=300)
    fig = Image.new("RGB", (800, 600), (10, 120, 200))  # blue figure, 4:3
    composite_into_slot(poster, slot, fig)
    # center of the slot is now the figure colour, not white/magenta
    px = poster.getpixel((300, 250))
    assert px[2] > px[0]  # bluish
    # aspect preserved (4:3 fig in 4:3 slot fills it)
    assert poster.getpixel((110, 110)) != SENTINEL_RGB


def test_trim_whitespace_crops_margins():
    from paperbanana.poster.figure_embed import trim_whitespace

    img = Image.new("RGB", (1000, 600), "white")
    # a 200x100 dark content block offset from the corner
    for x in range(400, 600):
        for y in range(250, 350):
            img.putpixel((x, y), (20, 20, 20))
    trimmed = trim_whitespace(img, pad_frac=0.0)
    assert 190 <= trimmed.width <= 210 and 90 <= trimmed.height <= 110


def test_trim_whitespace_allwhite_noop():
    from paperbanana.poster.figure_embed import trim_whitespace

    img = Image.new("RGB", (300, 200), "white")
    assert trim_whitespace(img).size == (300, 200)
