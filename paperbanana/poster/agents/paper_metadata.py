"""Paper metadata extraction (title, authors, affiliations, abstract)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from paperbanana.agents.base import BaseAgent
from paperbanana.core.utils import extract_json


class PaperMetadata(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    affiliations: list[str] = Field(default_factory=list)
    abstract: str = ""


class PaperMetadataAgent(BaseAgent):
    """Extracts bibliographic metadata from the paper's opening pages."""

    @property
    def agent_name(self) -> str:
        return "paper_metadata"

    async def run(self, first_pages_text: str, **kwargs: Any) -> PaperMetadata:
        template = self.load_prompt("poster")
        prompt = self.format_prompt(template, first_pages_text=first_pages_text)
        raw = await self.vlm.generate(prompt=prompt, response_format="json", temperature=0.1)
        data = extract_json(raw)
        if not isinstance(data, dict):
            raise ValueError(f"paper metadata agent returned no JSON object: {raw[:400]!r}")
        return PaperMetadata(**data)
