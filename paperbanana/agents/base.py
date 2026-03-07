"""Base agent class for all PaperBanana agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import structlog

from paperbanana.providers.base import VLMProvider

logger = structlog.get_logger()


class BaseAgent(ABC):
    """Base class for all agents in the PaperBanana pipeline.

    Each agent wraps a VLM provider and a prompt template to perform
    a specific role in the generation process.
    """

    def __init__(
        self,
        vlm_provider: VLMProvider,
        prompt_dir: str = "prompts",
        prompt_recorder: Any | None = None,
    ):
        self.vlm = vlm_provider
        self.prompt_dir = Path(prompt_dir)
        self._prompt_recorder = prompt_recorder

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Name of this agent (e.g., 'retriever', 'planner')."""
        ...

    @abstractmethod
    async def run(self, **kwargs: Any) -> Any:
        """Execute the agent's task and return results."""
        ...

    def load_prompt(self, diagram_type: str = "diagram") -> str:
        """Load the prompt template for this agent.

        Args:
            diagram_type: 'diagram' or 'plot'

        Returns:
            Prompt template string with {placeholders}.
        """
        path = self.prompt_dir / diagram_type / f"{self.agent_name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")
        return path.read_text(encoding="utf-8")

    def format_prompt(self, template: str, **kwargs: Any) -> str:
        """Format a prompt template with the given values.

        If a prompt recorder is configured, this method will write the formatted
        prompt to the active run directory.
        """
        # Reserved internal argument (not forwarded into template.format()).
        prompt_label = kwargs.pop("prompt_label", None)

        formatted = template.format(**kwargs)
        if self._prompt_recorder is not None:
            try:
                self._prompt_recorder.record(
                    agent_name=self.agent_name,
                    label=str(prompt_label) if prompt_label else None,
                    prompt=formatted,
                )
            except Exception:
                # Recording is best-effort; do not break generation on I/O issues.
                logger.warning("Prompt recording failed", agent=self.agent_name, label=prompt_label)
        return formatted
