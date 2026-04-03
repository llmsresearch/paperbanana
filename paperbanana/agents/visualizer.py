"""Visualizer Agent: Generates images from descriptions (diagram or code-based)."""

from __future__ import annotations

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


class VisualizerAgent(BaseAgent):
    """Generates images from descriptions.

    For methodology diagrams: Uses an image generation model.
    For statistical plots: Generates and executes matplotlib code.
    """

    def __init__(
        self,
        image_gen: ImageGenProvider,
        vlm_provider: VLMProvider,
        prompt_dir: str = "prompts",
        output_dir: str = "outputs",
        prompt_recorder=None,
    ):
        super().__init__(vlm_provider, prompt_dir, prompt_recorder=prompt_recorder)
        self.image_gen = image_gen
        self.output_dir = Path(output_dir)
        self._last_vector_paths: dict[str, str] = {}

    @property
    def agent_name(self) -> str:
        return "visualizer"

    async def run(
        self,
        description: str,
        diagram_type: DiagramType = DiagramType.METHODOLOGY,
        raw_data: Optional[dict] = None,
        output_path: Optional[str] = None,
        iteration: int = 0,
        seed: Optional[int] = None,
        aspect_ratio: Optional[str] = None,
        vector_formats: Optional[list[str]] = None,
    ) -> str:
        """Generate an image from a description.

        Args:
            description: Textual description of what to generate.
            diagram_type: Type of diagram.
            raw_data: Raw data for statistical plots.
            output_path: Where to save the generated image.
            iteration: Current iteration number (for naming).
            seed: Random seed for reproducibility.
            aspect_ratio: Target aspect ratio (e.g., '16:9', '1:1').
            vector_formats: Vector formats to export alongside raster (e.g., ['svg', 'pdf']).
                Only applies to statistical plots; ignored for methodology diagrams.

        Returns:
            Path to the generated raster image.
        """
        self._last_vector_paths = {}
        if diagram_type == DiagramType.STATISTICAL_PLOT:
            return await self._generate_plot(
                description, raw_data, output_path, iteration, aspect_ratio, vector_formats
            )
        else:
            return await self._generate_diagram(
                description,
                output_path,
                iteration,
                seed,
                aspect_ratio,
            )

    async def _generate_diagram(
        self,
        description: str,
        output_path: Optional[str],
        iteration: int,
        seed: Optional[int],
        aspect_ratio: Optional[str] = None,
    ) -> str:
        """Generate a methodology diagram using the image generation model."""
        template = self.load_prompt("diagram")
        prompt = self.format_prompt(
            template,
            prompt_label=f"visualizer_diagram_iter_{iteration}",
            description=description,
        )

        logger.info("Generating diagram image", iteration=iteration)

        # Determine dimensions from aspect ratio or use defaults
        w, h = self._ratio_to_dimensions(aspect_ratio) if aspect_ratio else (1792, 1024)

        image = await self.image_gen.generate(
            prompt=prompt,
            width=w,
            height=h,
            seed=seed,
            aspect_ratio=aspect_ratio,
        )

        if output_path is None:
            output_path = str(self.output_dir / f"diagram_iter_{iteration}.png")

        save_image(image, output_path)
        logger.info("Diagram saved", path=output_path)
        return output_path

    @staticmethod
    def _ratio_to_dimensions(ratio: str) -> tuple[int, int]:
        """Convert aspect ratio string to pixel dimensions."""
        mapping = {
            "21:9": (2016, 864),
            "16:9": (1792, 1024),
            "4:3": (1365, 1024),
            "3:2": (1536, 1024),
            "1:1": (1024, 1024),
            "2:3": (1024, 1536),
            "3:4": (1024, 1365),
            "9:16": (1024, 1792),
        }
        return mapping.get(ratio, (1792, 1024))

    async def _generate_plot(
        self,
        description: str,
        raw_data: Optional[dict],
        output_path: Optional[str],
        iteration: int,
        aspect_ratio: Optional[str] = None,
        vector_formats: Optional[list[str]] = None,
    ) -> str:
        """Generate a statistical plot by generating and executing matplotlib code."""
        # Build the description with raw data appended
        full_description = description
        if raw_data:
            import json

            full_description += f"\n\n## Raw Data\n```json\n{json.dumps(raw_data, indent=2)}\n```"

        # Load and format the plot visualizer prompt template
        template = self.load_prompt("plot")
        code_prompt = self.format_prompt(
            template,
            prompt_label=f"visualizer_plot_iter_{iteration}",
            description=full_description,
        )

        logger.info("Generating plot code", iteration=iteration)

        code_response = await self.vlm.generate(
            prompt=code_prompt,
            temperature=0.3,
            max_tokens=4096,
        )

        # Extract code from response
        code = self._extract_code(code_response)

        if output_path is None:
            output_path = str(self.output_dir / f"plot_iter_{iteration}.png")

        # Save generated code for inspection / manual editing
        code_path = Path(output_path).with_suffix(".py")
        code_path.parent.mkdir(parents=True, exist_ok=True)
        code_path.write_text(code)
        logger.info("Plot code saved", path=str(code_path))

        # Execute the code
        success = self._execute_plot_code(code, output_path, aspect_ratio, vector_formats)
        if not success:
            logger.error("Plot code execution failed, using placeholder")
            # Create a placeholder image
            placeholder = Image.new("RGB", (1024, 768), color=(255, 255, 255))
            save_image(placeholder, output_path)

        if self._last_vector_paths:
            logger.info("Vector outputs saved", paths=self._last_vector_paths)

        return output_path

    def _extract_code(self, response: str) -> str:
        """Extract Python code from a VLM response."""
        # Look for code blocks
        if "```python" in response:
            start = response.find("```python") + len("```python")
            end = response.find("```", start)
            if end == -1:
                logger.warning("Plot code block is missing closing fence; using remaining response")
                return response[start:].strip()
            return response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end == -1:
                logger.warning("Plot code block is missing closing fence; using remaining response")
                return response[start:].strip()
            return response[start:end].strip()
        return response.strip()

    def _execute_plot_code(
        self,
        code: str,
        output_path: str,
        aspect_ratio: Optional[str] = None,
        vector_formats: Optional[list[str]] = None,
    ) -> bool:
        """Execute matplotlib code in a subprocess to generate a plot.

        When *vector_formats* is provided (e.g. ``['svg', 'pdf']``), additional
        vector files are saved alongside the raster output using ``plt.savefig``.
        Paths are stored in ``self._last_vector_paths`` after a successful run.
        """
        # Strip any OUTPUT_PATH assignments from VLM-generated code so the
        # injected value below is authoritative (the VLM is prompted to set
        # OUTPUT_PATH itself, which would override the injected line).
        code = re.sub(r'^OUTPUT_PATH\s*=\s*["\'].*["\']\s*$', "", code, flags=re.MULTILINE)
        # Strip any VECTOR_PATH_* assignments the VLM may have generated.
        code = re.sub(r'^VECTOR_PATH_\w+\s*=\s*["\'].*["\']\s*$', "", code, flags=re.MULTILINE)

        # Build header: inject authoritative path variables.
        # Use forward slashes to avoid invalid unicode escapes on Windows
        # (e.g. C:\Users → \U is an invalid escape sequence in Python strings).
        safe_output = output_path.replace("\\", "/")
        header = f'OUTPUT_PATH = "{safe_output}"\n'

        # Map each requested vector format to its output path
        fmt_to_path: dict[str, str] = {}
        if vector_formats:
            for fmt in vector_formats:
                vec_path = str(Path(output_path).with_suffix(f".{fmt}"))
                fmt_to_path[fmt] = vec_path
                safe_vec = vec_path.replace("\\", "/")
                header += f'VECTOR_PATH_{fmt.upper()} = "{safe_vec}"\n'

        # Inject figure size from aspect ratio
        if aspect_ratio:
            w, h = self._ratio_to_dimensions(aspect_ratio)
            # Scale to reasonable matplotlib inches (assume 150 dpi)
            fig_w, fig_h = round(w / 150, 1), round(h / 150, 1)
            header += (
                f"import matplotlib\nmatplotlib.rcParams['figure.figsize'] = [{fig_w}, {fig_h}]\n"
            )

        # Append vector savefig calls after the user code
        vector_suffix = ""
        if fmt_to_path:
            vector_suffix = "\nimport matplotlib.pyplot as _pb_plt\n"
            for fmt in fmt_to_path:
                var = f"VECTOR_PATH_{fmt.upper()}"
                vector_suffix += f"_pb_plt.savefig({var}, format='{fmt}', bbox_inches='tight')\n"

        full_code = header + code + vector_suffix

        # Ensure output directory exists
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
                logger.error("Plot code error", stderr=result.stderr[:500])
                self._last_vector_paths = {}
                return False

            # Collect successfully written vector outputs
            self._last_vector_paths = {
                fmt: path for fmt, path in fmt_to_path.items() if Path(path).exists()
            }
            return Path(output_path).exists()
        except subprocess.TimeoutExpired:
            logger.error("Plot code timed out")
            self._last_vector_paths = {}
            return False
        finally:
            Path(temp_path).unlink(missing_ok=True)
