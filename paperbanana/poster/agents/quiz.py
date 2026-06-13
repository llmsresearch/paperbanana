"""PaperQuiz-in-the-loop: does the poster actually communicate the paper?

Questions are generated once from the paper; a *fresh* VLM context then
answers them seeing only the rendered poster, and a grading pass marks
each answer. Failed questions become "comprehension gaps" the critic
must address — optimizing the thing a poster is for, not just looks.
"""

from __future__ import annotations

from typing import Any

import structlog
from PIL import Image
from pydantic import BaseModel, Field

from paperbanana.agents.base import BaseAgent
from paperbanana.core.utils import extract_json
from paperbanana.poster.types import PaperAssets

logger = structlog.get_logger()


class QuizQuestion(BaseModel):
    question: str
    answer: str = Field(description="Ground truth from the paper")


class QuizResult(BaseModel):
    total: int
    correct: int
    gaps: list[str] = Field(
        default_factory=list, description="Questions the poster failed to answer"
    )

    @property
    def score(self) -> float:
        return self.correct / self.total if self.total else 1.0


class QuizAgent(BaseAgent):
    """Generates paper-grounded questions and grades poster-only answers."""

    @property
    def agent_name(self) -> str:
        return "quiz"

    async def run(
        self, assets: PaperAssets, n_questions: int = 8, **kwargs: Any
    ) -> list[QuizQuestion]:
        return await self.generate_questions(assets, n_questions=n_questions)

    async def generate_questions(
        self, assets: PaperAssets, n_questions: int = 8, **kwargs: Any
    ) -> list[QuizQuestion]:
        template = self.load_prompt("poster")
        sections = "\n\n".join(f"## {s.heading}\n{s.text[:1500]}" for s in assets.sections[:8])
        prompt = template.format(
            title=assets.title,
            abstract=assets.abstract,
            sections=sections,
            n=n_questions,
            n_half=max(2, n_questions // 2),
        )
        raw = await self.vlm.generate(prompt=prompt, response_format="json", temperature=0.3)
        data = extract_json(raw)
        if not isinstance(data, list):
            raise ValueError(f"quiz generator returned no JSON array: {raw[:400]!r}")
        questions = [QuizQuestion(**q) for q in data][:n_questions]
        logger.info("Quiz generated", questions=len(questions))
        return questions

    async def grade_poster(
        self, questions: list[QuizQuestion], poster: Image.Image, **kwargs: Any
    ) -> QuizResult:
        """Fresh-context answering from the poster only, then grading."""
        numbered = "\n".join(f"{i + 1}. {q.question}" for i, q in enumerate(questions))
        answer_prompt = (
            "You can see ONLY the attached conference poster (you have not read the "
            "paper). Answer each question from the poster alone; if the poster does "
            "not contain the answer, reply exactly 'NOT ON POSTER'.\n\n"
            f"{numbered}\n\n"
            "Respond with ONLY a JSON array of answer strings, one per question."
        )
        raw = await self.vlm.generate(
            prompt=answer_prompt, images=[poster], response_format="json", temperature=0.1
        )
        answers = extract_json(raw)
        if not isinstance(answers, list) or len(answers) != len(questions):
            raise ValueError(
                f"quiz answering returned {type(answers)} of wrong length: {raw[:300]!r}"
            )
        grading_lines = "\n".join(
            f"{i + 1}. Q: {q.question}\n   GROUND TRUTH: {q.answer}\n   POSTER ANSWER: {a}"
            for i, (q, a) in enumerate(zip(questions, answers))
        )
        grade_prompt = (
            "Grade each poster answer against the ground truth. An answer is correct "
            "if it conveys the same fact (paraphrase fine, numbers must match); "
            "'NOT ON POSTER' is incorrect.\n\n"
            f"{grading_lines}\n\n"
            "Respond with ONLY a JSON array of booleans, one per question."
        )
        raw = await self.vlm.generate(prompt=grade_prompt, response_format="json", temperature=0.0)
        marks = extract_json(raw)
        if not isinstance(marks, list) or len(marks) != len(questions):
            raise ValueError(f"quiz grading returned wrong shape: {raw[:300]!r}")
        gaps = [q.question for q, ok in zip(questions, marks) if not bool(ok)]
        result = QuizResult(total=len(questions), correct=sum(bool(m) for m in marks), gaps=gaps)
        logger.info("Poster quizzed", score=f"{result.correct}/{result.total}", gaps=len(gaps))
        return result
