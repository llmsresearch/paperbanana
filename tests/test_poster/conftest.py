"""Shared fixtures for poster tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from paperbanana.poster.types import (
    BBox,
    FigureAsset,
    FigureElement,
    FigureProvenance,
    Panel,
    PhysicalSize,
    PosterIR,
    StyleTokens,
    TextElement,
)


@pytest.fixture
def style_tokens() -> StyleTokens:
    return StyleTokens(
        palette={
            "primary": "#1A3A6B",
            "secondary": "#4A6FA5",
            "accent": "#E8A33D",
            "background": "#FFFFFF",
            "panel_bg": "#F5F7FA",
            "text": "#1A1A1A",
        },
        font_heading="Helvetica",
        font_body="Helvetica",
        type_scale_pt={
            "title": 72,
            "authors": 40,
            "affiliation": 30,
            "heading": 44,
            "body": 28,
            "caption": 20,
            "footnote": 16,
        },
    )


@pytest.fixture
def figure_asset(tmp_path: Path) -> FigureAsset:
    img_path = tmp_path / "fig1.png"
    Image.new("RGB", (2000, 1200), "white").save(img_path)
    return FigureAsset(
        id="fig1",
        path=str(img_path),
        width_px=2000,
        height_px=1200,
        provenance=FigureProvenance(
            origin="paper",
            paper_figure_id="fig1",
            source_page=3,
            source_bbox_norm=[0.1, 0.2, 0.9, 0.6],
            decision="reuse",
            decision_reason="vector-quality crop at sufficient DPI",
        ),
    )


def make_panel(
    panel_id: str,
    order: int,
    role: str = "method",
    bbox: BBox | None = None,
    elements: list | None = None,
) -> Panel:
    return Panel(
        id=panel_id,
        role=role,
        title=panel_id.title(),
        order=order,
        bbox=bbox,
        elements=elements or [TextElement(level="body", content=f"Content of {panel_id}")],
    )


@pytest.fixture
def poster_ir(style_tokens: StyleTokens, figure_asset: FigureAsset) -> PosterIR:
    """A small, fully-placed, valid landscape poster IR (NeurIPS default size)."""
    return PosterIR(
        venue="neurips",
        venue_spec_year=2025,
        size=PhysicalSize(width_mm=1219.2, height_mm=914.4),
        orientation="landscape",
        columns=3,
        style=style_tokens,
        assets={"fig1": figure_asset},
        paper_title="A Sample Paper Title",
        authors=["Ada Lovelace", "Alan Turing"],
        affiliations=["Analytical Engines Lab"],
        panels=[
            make_panel(
                "header",
                0,
                role="header",
                bbox=BBox(x_mm=20, y_mm=20, w_mm=1179.2, h_mm=120),
                elements=[
                    TextElement(level="title", content="A Sample Paper Title"),
                    TextElement(level="authors", content="Ada Lovelace, Alan Turing"),
                ],
            ),
            make_panel(
                "method",
                1,
                role="method",
                bbox=BBox(x_mm=20, y_mm=150, w_mm=380, h_mm=700),
                elements=[
                    TextElement(level="heading", content="Method"),
                    TextElement(level="body", content="- Step one\n- Step two"),
                    FigureElement(asset_id="fig1", caption="Figure 1: Overview."),
                ],
            ),
            make_panel(
                "results",
                2,
                role="results",
                bbox=BBox(x_mm=410, y_mm=150, w_mm=380, h_mm=700),
            ),
            make_panel(
                "conclusion",
                3,
                role="conclusion",
                bbox=BBox(x_mm=800, y_mm=150, w_mm=399.2, h_mm=700),
            ),
        ],
    )
