"""PaperQuiz agent tests: question generation and fresh-context grading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from paperbanana.poster.agents.quiz import QuizAgent, QuizQuestion
from paperbanana.poster.types import PaperAssets

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"


class _ScriptedVLM:
    name = "mock"
    model_name = "mock-model"
    cost_tracker = None

    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts: list[str] = []
        self.image_counts: list[int] = []

    async def generate(self, prompt, images=None, **kwargs):
        if not self._responses:
            raise AssertionError("scripted VLM exhausted")
        self.prompts.append(prompt)
        self.image_counts.append(len(images) if images else 0)
        return self._responses.pop(0)


def _assets() -> PaperAssets:
    return PaperAssets(
        pdf_path="paper.pdf",
        page_count=8,
        title="TurboAngle",
        abstract="We quantize rotations.",
        figures=[],
        sections=[],
    )


def _questions() -> list[QuizQuestion]:
    return [
        QuizQuestion(question="Total bits?", answer="6.56"),
        QuizQuestion(question="Needs calibration data?", answer="No"),
        QuizQuestion(question="Main baseline?", answer="GPTQ"),
    ]


async def test_generate_questions_caps_and_parses():
    payload = [q.model_dump() for q in _questions()] + [{"question": "extra", "answer": "dropped"}]
    vlm = _ScriptedVLM([json.dumps(payload)])
    agent = QuizAgent(vlm, prompt_dir=str(PROMPT_DIR))
    questions = await agent.generate_questions(_assets(), n_questions=3)
    assert len(questions) == 3
    assert questions[0].answer == "6.56"
    # The prompt template formatted fully (no stray placeholders).
    assert "{n_half}" not in vlm.prompts[0] and "{n}" not in vlm.prompts[0]


async def test_generate_questions_rejects_non_array():
    vlm = _ScriptedVLM([json.dumps({"oops": True})])
    agent = QuizAgent(vlm, prompt_dir=str(PROMPT_DIR))
    with pytest.raises(ValueError, match="no JSON array"):
        await agent.generate_questions(_assets(), n_questions=3)


async def test_grade_poster_marks_gaps():
    vlm = _ScriptedVLM(
        [
            json.dumps(["6.56", "NOT ON POSTER", "GPTQ"]),  # poster-only answers
            json.dumps([True, False, True]),  # grading
        ]
    )
    agent = QuizAgent(vlm, prompt_dir=str(PROMPT_DIR))
    poster = Image.new("RGB", (800, 600), "white")
    result = await agent.grade_poster(_questions(), poster)
    assert (result.total, result.correct) == (3, 2)
    assert result.gaps == ["Needs calibration data?"]
    assert result.score == pytest.approx(2 / 3)
    # Answering sees the poster; grading is text-only (fresh contexts).
    assert vlm.image_counts == [1, 0]


async def test_grade_poster_rejects_wrong_answer_count():
    vlm = _ScriptedVLM([json.dumps(["only-one"])])
    agent = QuizAgent(vlm, prompt_dir=str(PROMPT_DIR))
    poster = Image.new("RGB", (800, 600), "white")
    with pytest.raises(ValueError, match="wrong length"):
        await agent.grade_poster(_questions(), poster)
