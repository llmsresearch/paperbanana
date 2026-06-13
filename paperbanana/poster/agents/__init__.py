"""Poster pipeline agents (generative path: paper extraction + verification)."""

from paperbanana.poster.agents.faithfulness import FaithfulnessAgent
from paperbanana.poster.agents.figure_curator import FigureCuratorAgent
from paperbanana.poster.agents.figure_detector import FigureDetectorAgent
from paperbanana.poster.agents.paper_metadata import PaperMetadataAgent
from paperbanana.poster.agents.quiz import QuizAgent

__all__ = [
    "FaithfulnessAgent",
    "FigureCuratorAgent",
    "FigureDetectorAgent",
    "PaperMetadataAgent",
    "QuizAgent",
]
