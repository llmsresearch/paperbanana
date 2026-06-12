"""Poster pipeline agents."""

from paperbanana.poster.agents.content_planner import PosterContentAgent
from paperbanana.poster.agents.critic import PosterCriticAgent
from paperbanana.poster.agents.faithfulness import FaithfulnessAgent
from paperbanana.poster.agents.figure_curator import FigureCuratorAgent
from paperbanana.poster.agents.figure_detector import FigureDetectorAgent
from paperbanana.poster.agents.paper_metadata import PaperMetadataAgent
from paperbanana.poster.agents.stylist import PosterStylistAgent

__all__ = [
    "FaithfulnessAgent",
    "FigureCuratorAgent",
    "FigureDetectorAgent",
    "PaperMetadataAgent",
    "PosterContentAgent",
    "PosterCriticAgent",
    "PosterStylistAgent",
]
