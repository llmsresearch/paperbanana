"""PaperBanana: Agentic framework for automated academic illustration generation."""

__version__ = "0.1.0"

from paperbanana.core.pipeline import PaperBananaPipeline
from paperbanana.core.types import DiagramType, GenerationInput, GenerationOutput

__all__ = [
    "PaperBananaPipeline",
    "DiagramType",
    "GenerationInput",
    "GenerationOutput",
]
