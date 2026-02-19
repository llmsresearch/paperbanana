"""Input Optimizer Agent: Preprocesses source context and caption for better generation.

Runs two parallel sub-tasks:
  1. Context Enricher — structures raw methodology text into diagram-ready format
  2. Caption Sharpener — refines vague captions into precise visual specifications
"""

from __future__ import annotations

import asyncio

import structlog

from paperbanana.agents.base import BaseAgent
from paperbanana.core.types import DiagramType
from paperbanana.providers.base import VLMProvider

logger = structlog.get_logger()


class InputOptimizerAgent(BaseAgent):
    """Optimizes source_context and caption before the main pipeline.

    Runs two VLM calls in parallel:
    - Context enricher: extracts and structures components, flows, groupings
    - Caption sharpener: converts vague intent into precise visual specification
    """

    def __init__(self, vlm_provider: VLMProvider, prompt_dir: str = "prompts"):
        super().__init__(vlm_provider, prompt_dir)

    @property
    def agent_name(self) -> str:
        return "optimizer"

    async def run(
        self,
        source_context: str,
        caption: str,
        diagram_type: DiagramType = DiagramType.METHODOLOGY,
    ) -> dict:
        """Optimize inputs for downstream agents.

        Args:
            source_context: Raw methodology text.
            caption: Raw figure caption / communicative intent.
            diagram_type: Type of diagram.

        Returns:
            Dict with 'optimized_context' and 'optimized_caption'.
        """
        prompt_type = "diagram" if diagram_type == DiagramType.METHODOLOGY else "plot"

        # Load both prompt templates
        context_template = self._load_sub_prompt(prompt_type, "context_enricher")
        caption_template = self._load_sub_prompt(prompt_type, "caption_sharpener")

        context_prompt = self.format_prompt(
            context_template,
            source_context=source_context,
            caption=caption,
        )
        caption_prompt = self.format_prompt(
            caption_template,
            source_context=source_context,
            caption=caption,
        )

        logger.info(
            "Running input optimizer (parallel)",
            context_length=len(source_context),
            caption_length=len(caption),
        )

        # Run both optimizations in parallel
        enriched_context, sharpened_caption = await asyncio.gather(
            self.vlm.generate(
                prompt=context_prompt,
                temperature=0.4,
                max_tokens=4096,
            ),
            self.vlm.generate(
                prompt=caption_prompt,
                temperature=0.4,
                max_tokens=1024,
            ),
        )

        logger.info(
            "Input optimization complete",
            enriched_context_length=len(enriched_context),
            sharpened_caption_length=len(sharpened_caption),
        )

        return {
            "optimized_context": enriched_context.strip(),
            "optimized_caption": sharpened_caption.strip(),
        }

    def _load_sub_prompt(self, diagram_type: str, sub_name: str) -> str:
        """Load a sub-prompt template (context_enricher or caption_sharpener)."""
        path = self.prompt_dir / diagram_type / f"{sub_name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Optimizer prompt not found: {path}")
        return path.read_text(encoding="utf-8")
