"""Structurer Agent: textual diagram description -> Diagram IR JSON."""

from __future__ import annotations

import json

import structlog

from paperbanana.agents.base import BaseAgent
from paperbanana.core.diagram_ir import DiagramIR

logger = structlog.get_logger()
MAX_STRUCTURER_CONTEXT_CHARS = 8000


def _extract_json_blob(text: str) -> str:
    """Strip markdown fences and return JSON payload."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        inner: list[str] = []
        for line in lines[1:]:
            if line.strip().startswith("```"):
                break
            inner.append(line)
        return "\n".join(inner).strip()
    return t


class StructurerAgent(BaseAgent):
    """Produces validated Diagram IR from the stylist's description."""

    @property
    def agent_name(self) -> str:
        return "structurer"

    async def run(
        self,
        description: str,
        source_context: str,
        caption: str,
    ) -> DiagramIR:
        """Return Diagram IR from the final textual figure description (one automatic retry)."""
        last_err: str | None = None
        for attempt in range(2):
            repair_section = ""
            if attempt and last_err:
                repair_section = (
                    f"\n\n## Previous JSON was invalid\nFix the following: {last_err}\n"
                    "Output corrected JSON only.\n"
                )
            template = self.load_prompt("diagram")
            prompt = self.format_prompt(
                template,
                prompt_label=f"structurer_attempt_{attempt + 1}",
                description=description,
                source_context=source_context[:MAX_STRUCTURER_CONTEXT_CHARS],
                caption=caption,
                repair_section=repair_section,
            )
            raw = await self.vlm.generate(
                prompt=prompt,
                images=None,
                temperature=0.15 if attempt else 0.2,
                max_tokens=8192,
                response_format="json",
            )
            try:
                blob = _extract_json_blob(raw)
                data = json.loads(blob)
                return DiagramIR.model_validate(data)
            except (json.JSONDecodeError, ValueError) as e:
                last_err = str(e)
                logger.warning("Structurer validation failed", attempt=attempt + 1, error=last_err)

        raise ValueError(f"Structurer failed after retries: {last_err}")
