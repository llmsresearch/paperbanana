"""Figure curation execution tests with scripted agents/providers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from paperbanana.poster.agents.faithfulness import FaithfulnessAgent
from paperbanana.poster.agents.figure_curator import FigureCuratorAgent
from paperbanana.poster.figures import (
    PosterFigureError,
    curate_figures,
    estimate_placed_width_mm,
)
from paperbanana.poster.types import PaperAssets, PaperFigure, Storyboard, StoryboardPanel

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"
PALETTE = {
    "primary": "#1A3A6B",
    "secondary": "#4A6FA5",
    "accent": "#E8A33D",
    "background": "#FFFFFF",
    "panel_bg": "#F5F7FA",
    "text": "#1A1A1A",
}
REAUTHOR_TEMPLATE = (PROMPT_DIR / "poster" / "reauthor_edit.txt").read_text(encoding="utf-8")


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


class _GuidedImageGen:
    name = "mock-imagen"
    model_name = "mock-image-model"

    def __init__(self):
        self.calls = 0

    async def generate(
        self,
        prompt,
        negative_prompt=None,
        width=1024,
        height=1024,
        seed=None,
        aspect_ratio=None,
        quality=None,
        images=None,
    ):
        self.calls += 1
        return Image.new("RGB", (min(width, 4096), min(height, 4096)), "lightblue")


class _TextOnlyImageGen:
    name = "text-only"
    model_name = "no-edit-model"

    async def generate(
        self,
        prompt,
        negative_prompt=None,
        width=1024,
        height=1024,
        seed=None,
        aspect_ratio=None,
        quality=None,
    ):
        return Image.new("RGB", (width, height), "white")


def _paper_assets(tmp_path: Path) -> PaperAssets:
    figures = []
    for i, (w, h) in enumerate([(2400, 1400), (900, 700)], start=1):
        path = tmp_path / f"fig{i}.png"
        Image.new("RGB", (w, h), "white").save(path)
        figures.append(
            PaperFigure(
                id=f"fig{i}",
                kind="figure",
                page=i,
                bbox_norm=[0.1, 0.1, 0.9, 0.6],
                caption=f"Figure {i}: Something important.",
                image_path=str(path),
                extract_dpi=300,
                caption_anchored=True,
            )
        )
    return PaperAssets(
        pdf_path="paper.pdf",
        page_count=6,
        title="T",
        figures=figures,
        sections=[],
        abstract="A",
    )


def _storyboard(figure_ids: list[str], briefs: dict[str, str] | None = None) -> Storyboard:
    return Storyboard(
        panels=[
            StoryboardPanel(
                id="method",
                role="method",
                order=1,
                figure_ids=figure_ids,
                text_blocks=["- point"],
            )
        ],
        columns=3,
        new_figure_briefs=briefs or {},
    )


async def _diagram_generator_factory(tmp_path: Path):
    async def generate(slug: str, brief: str, context: str) -> Path:
        path = tmp_path / f"generated_{slug}.png"
        Image.new("RGB", (2000, 1200), "lavender").save(path)
        return path

    return generate


def test_estimate_placed_width():
    # NeurIPS default 1219.2mm, 3 cols, 20 margin, 10 gutter -> (1219.2-40-20)/3 - 20
    assert estimate_placed_width_mm(1219.2, 3, 20, 10) == pytest.approx(366.4, abs=0.1)


async def test_reuse_decision(tmp_path: Path):
    assets = _paper_assets(tmp_path)
    curator_vlm = _ScriptedVLM([json.dumps({"decision": "reuse", "reason": "crisp and large"})])
    asset_map, decisions = await curate_figures(
        assets,
        _storyboard(["fig1"]),
        FigureCuratorAgent(curator_vlm, prompt_dir=str(PROMPT_DIR)),
        FaithfulnessAgent(_ScriptedVLM([]), prompt_dir=str(PROMPT_DIR)),
        _GuidedImageGen(),
        await _diagram_generator_factory(tmp_path),
        REAUTHOR_TEMPLATE,
        PALETTE,
        placed_width_mm=366.4,
        min_dpi=100,
        out_dir=tmp_path / "out",
    )
    assert decisions[0].decision == "reuse"
    assert asset_map["fig1"].provenance.origin == "paper"
    assert asset_map["fig1"].path == assets.figures[0].image_path


async def test_reauthor_passes_faithfulness(tmp_path: Path):
    assets = _paper_assets(tmp_path)
    curator_vlm = _ScriptedVLM(
        [
            json.dumps(
                {
                    "decision": "reauthor",
                    "reason": "tiny labels",
                    "edit_instructions": "enlarge labels",
                }
            )
        ]
    )
    faith_vlm = _ScriptedVLM([json.dumps({"verdict": "pass", "differences": []})])
    gen = _GuidedImageGen()
    asset_map, decisions = await curate_figures(
        assets,
        _storyboard(["fig1"]),
        FigureCuratorAgent(curator_vlm, prompt_dir=str(PROMPT_DIR)),
        FaithfulnessAgent(faith_vlm, prompt_dir=str(PROMPT_DIR)),
        gen,
        await _diagram_generator_factory(tmp_path),
        REAUTHOR_TEMPLATE,
        PALETTE,
        placed_width_mm=366.4,
        min_dpi=100,
        out_dir=tmp_path / "out",
    )
    assert gen.calls == 1
    prov = asset_map["fig1"].provenance
    assert prov.decision == "reauthor" and prov.faithfulness == "verified"
    assert Path(asset_map["fig1"].path).name == "fig1_reauthored.png"


async def test_reauthor_fails_then_hard_error(tmp_path: Path):
    assets = _paper_assets(tmp_path)
    curator_vlm = _ScriptedVLM(
        [
            json.dumps(
                {
                    "decision": "reauthor",
                    "reason": "tiny labels",
                    "edit_instructions": "enlarge labels",
                }
            )
        ]
    )
    fail = json.dumps({"verdict": "fail", "differences": ["bar 3 value changed"]})
    faith_vlm = _ScriptedVLM([fail, fail])
    with pytest.raises(PosterFigureError, match="figure-decision fig1=reuse"):
        await curate_figures(
            assets,
            _storyboard(["fig1"]),
            FigureCuratorAgent(curator_vlm, prompt_dir=str(PROMPT_DIR)),
            FaithfulnessAgent(faith_vlm, prompt_dir=str(PROMPT_DIR)),
            _GuidedImageGen(),
            await _diagram_generator_factory(tmp_path),
            REAUTHOR_TEMPLATE,
            PALETTE,
            placed_width_mm=366.4,
            min_dpi=100,
            out_dir=tmp_path / "out",
            max_reauthor_attempts=2,
        )


async def test_reauthor_without_guided_edit_provider_errors(tmp_path: Path):
    assets = _paper_assets(tmp_path)
    curator_vlm = _ScriptedVLM(
        [json.dumps({"decision": "reauthor", "reason": "x", "edit_instructions": "y"})]
    )
    with pytest.raises(PosterFigureError, match="guided edits"):
        await curate_figures(
            assets,
            _storyboard(["fig1"]),
            FigureCuratorAgent(curator_vlm, prompt_dir=str(PROMPT_DIR)),
            FaithfulnessAgent(_ScriptedVLM([]), prompt_dir=str(PROMPT_DIR)),
            _TextOnlyImageGen(),
            await _diagram_generator_factory(tmp_path),
            REAUTHOR_TEMPLATE,
            PALETTE,
            placed_width_mm=366.4,
            min_dpi=100,
            out_dir=tmp_path / "out",
        )


async def test_new_figure_generation(tmp_path: Path):
    assets = _paper_assets(tmp_path)
    asset_map, decisions = await curate_figures(
        assets,
        _storyboard(["new:overview"], briefs={"overview": "pipeline diagram"}),
        FigureCuratorAgent(_ScriptedVLM([]), prompt_dir=str(PROMPT_DIR)),
        FaithfulnessAgent(_ScriptedVLM([]), prompt_dir=str(PROMPT_DIR)),
        _GuidedImageGen(),
        await _diagram_generator_factory(tmp_path),
        REAUTHOR_TEMPLATE,
        PALETTE,
        placed_width_mm=366.4,
        min_dpi=100,
        out_dir=tmp_path / "out",
    )
    assert asset_map["overview"].provenance.origin == "generated"
    assert decisions[0].decision == "generate"


async def test_user_override_skips_curator(tmp_path: Path):
    assets = _paper_assets(tmp_path)
    asset_map, decisions = await curate_figures(
        assets,
        _storyboard(["fig2"]),
        FigureCuratorAgent(_ScriptedVLM([]), prompt_dir=str(PROMPT_DIR)),  # never called
        FaithfulnessAgent(_ScriptedVLM([]), prompt_dir=str(PROMPT_DIR)),
        _GuidedImageGen(),
        await _diagram_generator_factory(tmp_path),
        REAUTHOR_TEMPLATE,
        PALETTE,
        placed_width_mm=366.4,
        min_dpi=100,
        out_dir=tmp_path / "out",
        overrides={"fig2": "reuse"},
    )
    assert decisions[0].reason.startswith("explicit user override")
    assert asset_map["fig2"].provenance.decision == "reuse"


async def test_unknown_figure_reference_raises(tmp_path: Path):
    assets = _paper_assets(tmp_path)
    with pytest.raises(PosterFigureError, match="unknown figure"):
        await curate_figures(
            assets,
            _storyboard(["fig99"]),
            FigureCuratorAgent(_ScriptedVLM([]), prompt_dir=str(PROMPT_DIR)),
            FaithfulnessAgent(_ScriptedVLM([]), prompt_dir=str(PROMPT_DIR)),
            _GuidedImageGen(),
            await _diagram_generator_factory(tmp_path),
            REAUTHOR_TEMPLATE,
            PALETTE,
            placed_width_mm=366.4,
            min_dpi=100,
            out_dir=tmp_path / "out",
        )
