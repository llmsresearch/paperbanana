"""Vanilla Agent: Direct generation without retrieval or planning.

Serves as a baseline — generates diagrams or plots directly from source
context and visual intent, bypassing the Retriever/Planner/Stylist pipeline.
Ported from PaperBanana-official/agents/vanilla_agent.py.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import structlog
from PIL import Image

from paperbanana.agents.base import BaseAgent
from paperbanana.core.types import DiagramType
from paperbanana.core.utils import save_image
from paperbanana.providers.base import ImageGenProvider, VLMProvider

logger = structlog.get_logger()

DIAGRAM_SYSTEM_PROMPT = """\
## ROLE
You are a Lead Visual Designer for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
You will be provided with a "Method Section" and a "Diagram Caption". Your task is to \
generate a high-quality scientific diagram that effectively illustrates the method \
described in the text, as the caption requires, and adhering strictly to modern \
academic visualization standards.

**CRITICAL INSTRUCTION ON CAPTION:**
The "Diagram Caption" is provided solely to describe the visual content and logic you \
need to draw. **DO NOT render, write, or include the caption text itself (e.g., \
"Figure 1: ...") inside the generated image.**

## OUTPUT
Generate a single, high-resolution image that visually explains the method and aligns \
well with the caption.\
"""

PLOT_SYSTEM_PROMPT = """\
## ROLE
You are an expert statistical plot illustrator for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
You will be provided with "Plot Raw Data" and a "Visual Intent of the Desired Plot". \
Your task is to write matplotlib code to generate a high-quality statistical plot that \
effectively visualizes the data according to the visual intent, adhering strictly to \
modern academic visualization standards.

## OUTPUT
Write Python matplotlib code to generate the plot. Only provide the code without any \
explanations.\
"""


class VanillaAgent(BaseAgent):
    """Direct generation agent — no retrieval, no planning.

    For diagrams: uses image generation provider directly.
    For plots: generates matplotlib code via VLM and executes it.
    """

    def __init__(
        self,
        vlm_provider: VLMProvider,
        image_gen: Optional[ImageGenProvider] = None,
        prompt_dir: str = "prompts",
        output_dir: str = "outputs",
    ):
        super().__init__(vlm_provider, prompt_dir)
        self.image_gen = image_gen
        self.output_dir = Path(output_dir)

    @property
    def agent_name(self) -> str:
        return "vanilla"

    async def run(
        self,
        source_context: str,
        visual_intent: str,
        diagram_type: DiagramType = DiagramType.METHODOLOGY,
        raw_data: Optional[dict] = None,
        output_path: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> str:
        """Generate an image directly from context + intent.

        Args:
            source_context: Method section text or raw data.
            visual_intent: Figure caption / visual intent description.
            diagram_type: METHODOLOGY (image gen) or STATISTICAL_PLOT (code gen).
            raw_data: Raw data dict for plot tasks.
            output_path: Where to save the generated image.
            seed: Random seed for reproducibility.

        Returns:
            Path to the generated image.
        """
        if diagram_type == DiagramType.STATISTICAL_PLOT:
            return await self._generate_plot(
                source_context, visual_intent, raw_data, output_path,
            )
        else:
            return await self._generate_diagram(
                source_context, visual_intent, output_path, seed,
            )

    async def _generate_diagram(
        self,
        source_context: str,
        visual_intent: str,
        output_path: Optional[str],
        seed: Optional[int],
    ) -> str:
        """Generate a diagram via image generation model."""
        if self.image_gen is None:
            raise RuntimeError("VanillaAgent requires an ImageGenProvider for diagram tasks.")

        prompt = (
            f"**Method Section**: {source_context}\n"
            f"**Diagram Caption**: {visual_intent}\n"
            "Note that do not include figure titles in the image.\n"
            "**Generated Diagram**: "
        )

        logger.info("Vanilla: generating diagram", prompt_length=len(prompt))

        image = await self.image_gen.generate(
            prompt=prompt,
            width=getattr(self, '_width', 1792),
            height=getattr(self, '_height', 1024),
            seed=seed,
        )

        if output_path is None:
            output_path = str(self.output_dir / "vanilla_diagram.png")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        save_image(image, output_path)
        logger.info("Vanilla diagram saved", path=output_path)
        return output_path

    async def _generate_plot(
        self,
        source_context: str,
        visual_intent: str,
        raw_data: Optional[dict],
        output_path: Optional[str],
    ) -> str:
        """Generate a plot via VLM code generation + subprocess execution."""
        content = json.dumps(raw_data) if isinstance(raw_data, (dict, list)) else source_context

        prompt = (
            f"**Plot Raw Data**: {content}\n"
            f"**Visual Intent of the Desired Plot**: {visual_intent}\n\n"
            "Use python matplotlib to generate a statistical plot based on the above "
            "information. Only provide the code without any explanations. Code:"
        )

        logger.info("Vanilla: generating plot code", prompt_length=len(prompt))

        code_response = await self.vlm.generate(
            prompt=prompt,
            system_prompt=PLOT_SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=16384,
        )

        code = _extract_code(code_response)

        if output_path is None:
            output_path = str(self.output_dir / "vanilla_plot.png")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        success = _execute_plot_code(code, output_path)
        if not success:
            logger.error("Vanilla plot code execution failed, creating placeholder")
            placeholder = Image.new("RGB", (1024, 768), color=(255, 255, 255))
            save_image(placeholder, output_path)

        return output_path


def _extract_code(response: str) -> str:
    """Extract Python code from a VLM response."""
    if "```python" in response:
        start = response.index("```python") + len("```python")
        try:
            end = response.index("```", start)
        except ValueError:
            end = len(response)
        return response[start:end].strip()
    elif "```" in response:
        start = response.index("```") + 3
        try:
            end = response.index("```", start)
        except ValueError:
            end = len(response)
        return response[start:end].strip()
    return response.strip()


def _execute_plot_code(code: str, output_path: str) -> bool:
    """Execute matplotlib code in a subprocess to generate a plot."""
    code = re.sub(r'^OUTPUT_PATH\s*=\s*.*$', '', code, flags=re.MULTILINE)
    safe_path = output_path.replace("\\", "/")
    full_code = f'OUTPUT_PATH = "{safe_path}"\n{code}'

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        temp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            logger.error("Plot code error", stderr=result.stderr[:2000])
            return False
        if not Path(output_path).exists():
            logger.error("Plot code ran but output file not found")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error("Plot code timed out")
        return False
    finally:
        Path(temp_path).unlink(missing_ok=True)
