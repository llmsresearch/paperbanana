"""Retriever Agent: Selects relevant reference examples for in-context learning."""

from __future__ import annotations

import json

import structlog

from paperbanana.agents.base import BaseAgent
from paperbanana.core.types import DiagramType, ReferenceExample
from paperbanana.providers.base import VLMProvider

logger = structlog.get_logger()


class RetrieverAgent(BaseAgent):
    """Retrieves the most relevant reference examples from the curated set.

    Given a source context and caption, uses the VLM to identify which
    reference examples are most useful for generating the target diagram.
    """

    def __init__(self, vlm_provider: VLMProvider, prompt_dir: str = "prompts"):
        super().__init__(vlm_provider, prompt_dir)

    @property
    def agent_name(self) -> str:
        return "retriever"

    async def run(
        self,
        source_context: str,
        caption: str,
        candidates: list[ReferenceExample],
        num_examples: int = 10,
        diagram_type: DiagramType = DiagramType.METHODOLOGY,
    ) -> list[ReferenceExample]:
        """Select the most relevant reference examples.

        Args:
            source_context: Methodology text from the paper.
            caption: Communicative intent / figure caption.
            candidates: All available reference examples.
            num_examples: Number of examples to retrieve.
            diagram_type: Type of diagram being generated.

        Returns:
            List of selected reference examples, ordered by relevance.
        """
        if not candidates:
            logger.warning("No reference candidates available, returning empty list")
            return []

        # If we have fewer candidates than requested, return all
        if len(candidates) <= num_examples:
            logger.info(
                "Fewer candidates than requested, returning all",
                candidates=len(candidates),
                requested=num_examples,
            )
            return candidates

        # Format candidates for the prompt
        candidates_text = self._format_candidates(candidates)

        # Load and format the retriever prompt
        prompt_type = "diagram" if diagram_type == DiagramType.METHODOLOGY else "plot"
        template = self.load_prompt(prompt_type)
        prompt = self.format_prompt(
            template,
            source_context=source_context,
            caption=caption,
            candidates=candidates_text,
            num_examples=num_examples,
        )

        # Call the VLM
        logger.info(
            "Running retriever agent",
            num_candidates=len(candidates),
            num_requested=num_examples,
        )
        response = await self.vlm.generate(
            prompt=prompt,
            temperature=0.3,  # Low temperature for consistent selection
            response_format="json",
        )

        # Parse response
        selected = self._parse_response(response, candidates)
        logger.info("Retriever selected examples", count=len(selected))
        return selected[:num_examples]

    def _format_candidates(self, candidates: list[ReferenceExample]) -> str:
        """Format candidate examples for the prompt.

        Matches paper's format: Paper ID, Caption, Methodology section.
        """
        lines = []
        for i, c in enumerate(candidates):
            lines.append(
                f"Candidate Paper {i + 1}:\n"
                f"- **Paper ID:** {c.id}\n"
                f"- **Caption:** {c.caption}\n"
                f"- **Methodology section:** {c.source_context[:300]}...\n"
            )
        return "\n".join(lines)

    def _parse_response(
        self, response: str, candidates: list[ReferenceExample]
    ) -> list[ReferenceExample]:
        """Parse the VLM response to extract selected example IDs.

        Handles both 'selected_ids' (our format) and 'top_10_papers'/'top_10_plots'
        (paper's format) JSON keys for robustness.
        """
        try:
            data = json.loads(response)
            selected_ids = (
                data.get("selected_ids")
                or data.get("top_10_papers")
                or data.get("top_10_plots")
                or []
            )
        except json.JSONDecodeError:
            logger.warning("Failed to parse retriever response as JSON, using fallback")
            # Fallback: return first N candidates
            return candidates

        # Map IDs back to ReferenceExample objects
        id_to_example = {c.id: c for c in candidates}
        selected = []
        for eid in selected_ids:
            if eid in id_to_example:
                selected.append(id_to_example[eid])
            else:
                logger.warning("Retriever selected unknown ID", id=eid)

        return selected
