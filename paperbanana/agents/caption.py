"""Caption Agent: Generates publication-ready figure captions from the final pipeline output."""

from __future__ import annotations

import structlog

from paperbanana.agents.base import BaseAgent
from paperbanana.core.types import DiagramType
from paperbanana.core.utils import load_image
from paperbanana.providers.base import VLMProvider

logger = structlog.get_logger()


class CaptionAgent(BaseAgent):
    """Generates a publication-ready figure caption after the final iteration.

    Makes a single vision-language model call using the final image together
    with the source context, communicative intent, and styled description to
    produce a 1-3 sentence academic caption — the kind that appears beneath
    figures in conference and journal papers.
    """

    def __init__(
        self, vlm_provider: VLMProvider, prompt_dir: str = "prompts", prompt_recorder=None
    ):
        super().__init__(vlm_provider, prompt_dir, prompt_recorder=prompt_recorder)

    @property
    def agent_name(self) -> str:
        return "caption"

    async def run(
        self,
        image_path: str,
        source_context: str,
        intent: str,
        description: str,
        diagram_type: DiagramType = DiagramType.METHODOLOGY,
    ) -> str:
        """Generate a publication-ready caption for the final figure.

        Args:
            image_path: Path to the final generated image.
            source_context: Original methodology text or data context.
            intent: User-supplied communicative intent (raw caption input).
            description: Final styled description used to produce the image.
            diagram_type: Type of diagram (methodology or statistical_plot).

        Returns:
            A 1-3 sentence publication-ready figure caption string.
        """
        image = load_image(image_path)

        prompt_type = "diagram" if diagram_type == DiagramType.METHODOLOGY else "plot"
        template = self.load_prompt(prompt_type)
        prompt = self.format_prompt(
            template,
            source_context=source_context,
            intent=intent,
            description=description,
            prompt_label="caption",
        )

        logger.info("Running caption agent", image_path=image_path)

        response = await self.vlm.generate(
            prompt=prompt,
            images=[image],
            temperature=0.3,
            max_tokens=512,
        )

        caption = response.strip()
        # Strip any surrounding quotes the model may add
        if len(caption) >= 2 and caption[0] in ('"', "'") and caption[-1] == caption[0]:
            caption = caption[1:-1].strip()

        logger.info("Caption generated", length=len(caption))
        return caption
