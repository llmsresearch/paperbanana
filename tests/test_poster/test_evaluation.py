"""Poster evaluation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from paperbanana.poster.evaluation import evaluate_poster
from paperbanana.poster.types import PosterIR

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"
SPEC_DIR = Path(__file__).resolve().parents[2] / "data" / "venue_specs"


class _JudgeVLM:
    name = "mock"
    model_name = "mock-model"
    cost_tracker = None

    def __init__(self, response: str):
        self._response = response
        self.last_images = None

    async def generate(self, prompt, images=None, **kwargs):
        self.last_images = images
        return self._response


def _good_response() -> str:
    return json.dumps(
        {
            "content": {"score": 4, "rationale": "Captures the headline result."},
            "design": {"score": 3, "rationale": "Slightly sparse third column."},
            "coherence": {"score": 5, "rationale": "Clean left-to-right narrative."},
        }
    )


@pytest.fixture
def preview(tmp_path: Path) -> Path:
    path = tmp_path / "preview.png"
    Image.new("RGB", (1200, 900), "white").save(path)
    return path


async def test_evaluate_scores_and_overall(preview: Path):
    vlm = _JudgeVLM(_good_response())
    result = await evaluate_poster(
        vlm, preview_path=preview, paper_context="Some paper text", prompt_dir=PROMPT_DIR
    )
    assert {s.dimension for s in result.scores} == {"content", "design", "coherence"}
    assert result.overall == pytest.approx(4.0)
    assert result.compliance is None
    assert result.reference_used is False
    assert len(vlm.last_images) == 1


async def test_evaluate_with_reference_sends_two_images(preview: Path, tmp_path: Path):
    ref = tmp_path / "ref.png"
    Image.new("RGB", (800, 600), "gray").save(ref)
    vlm = _JudgeVLM(_good_response())
    result = await evaluate_poster(
        vlm,
        preview_path=preview,
        paper_context="ctx",
        prompt_dir=PROMPT_DIR,
        reference_path=ref,
    )
    assert result.reference_used is True
    assert len(vlm.last_images) == 2


async def test_evaluate_with_run_dir_compliance(
    preview: Path, tmp_path: Path, poster_ir: PosterIR, monkeypatch
):
    monkeypatch.chdir(Path(__file__).resolve().parents[2])  # builtin specs resolve relatively
    run_dir = tmp_path / "poster_run"
    run_dir.mkdir()
    (run_dir / "poster_ir.json").write_text(poster_ir.model_dump_json(), encoding="utf-8")
    vlm = _JudgeVLM(_good_response())
    result = await evaluate_poster(
        vlm,
        preview_path=preview,
        paper_context="ctx",
        prompt_dir=PROMPT_DIR,
        run_dir=run_dir,
    )
    assert result.compliance is not None
    assert result.compliance.passed


async def test_evaluate_rejects_missing_dimension(preview: Path):
    vlm = _JudgeVLM(json.dumps({"content": {"score": 4, "rationale": "x"}}))
    with pytest.raises(ValueError, match="missing dimension"):
        await evaluate_poster(vlm, preview_path=preview, paper_context="ctx", prompt_dir=PROMPT_DIR)


async def test_dual_judge_averages_dimensions(preview: Path):
    primary = _JudgeVLM(_good_response())  # 4 / 3 / 5
    secondary = _JudgeVLM(
        json.dumps(
            {
                "content": {"score": 2, "rationale": "Misses the ablation."},
                "design": {"score": 5, "rationale": "Striking hierarchy."},
                "coherence": {"score": 3, "rationale": "Results feel detached."},
            }
        )
    )
    result = await evaluate_poster(
        primary,
        preview_path=preview,
        paper_context="ctx",
        prompt_dir=PROMPT_DIR,
        secondary_vlm=secondary,
    )
    assert result.judges == 2
    by_dim = {s.dimension: s.score for s in result.scores}
    assert by_dim == {"content": 3.0, "design": 4.0, "coherence": 4.0}
    assert result.overall == pytest.approx(3.67, abs=0.01)
    assert "judge 2" in result.scores[0].rationale
    assert secondary.last_images is not None  # second judge saw the poster
