"""Agent implementations for the PaperBanana pipeline."""

from paperbanana.agents.base import BaseAgent
from paperbanana.agents.critic import CriticAgent
from paperbanana.agents.optimizer import InputOptimizerAgent
from paperbanana.agents.planner import PlannerAgent
from paperbanana.agents.polish import PolishAgent
from paperbanana.agents.retriever import RetrieverAgent
from paperbanana.agents.stylist import StylistAgent
from paperbanana.agents.vanilla import VanillaAgent
from paperbanana.agents.visualizer import VisualizerAgent

__all__ = [
    "BaseAgent",
    "InputOptimizerAgent",
    "RetrieverAgent",
    "PlannerAgent",
    "StylistAgent",
    "VisualizerAgent",
    "CriticAgent",
    "VanillaAgent",
    "PolishAgent",
]
