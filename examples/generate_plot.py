"""Example: Generate a statistical plot from data."""

import asyncio

from dotenv import load_dotenv

from paperbanana import DiagramType, GenerationInput, PaperBananaPipeline
from paperbanana.core.config import Settings

load_dotenv()


async def main():
    # Example: model performance comparison
    source_context = """
    Table 1: Performance comparison of different models on three benchmarks.

    | Model     | MMLU  | HellaSwag | ARC-C |
    |-----------|-------|-----------|-------|
    | GPT-4o    | 88.7  | 95.3      | 96.4  |
    | Claude 3  | 86.8  | 93.7      | 93.5  |
    | Gemini    | 85.0  | 87.8      | 89.8  |
    | Llama 3   | 79.2  | 82.0      | 83.4  |
    | Mistral   | 75.3  | 81.4      | 78.6  |
    """

    raw_data = {
        "models": ["GPT-4o", "Claude 3", "Gemini", "Llama 3", "Mistral"],
        "MMLU": [88.7, 86.8, 85.0, 79.2, 75.3],
        "HellaSwag": [95.3, 93.7, 87.8, 82.0, 81.4],
        "ARC-C": [96.4, 93.5, 89.8, 83.4, 78.6],
    }

    caption = "Performance comparison of frontier LLMs across three benchmarks."

    settings = Settings(
        vlm_provider="gemini",
        refinement_iterations=2,
    )

    pipeline = PaperBananaPipeline(settings=settings)

    result = await pipeline.generate(
        GenerationInput(
            source_context=source_context,
            communicative_intent=caption,
            diagram_type=DiagramType.STATISTICAL_PLOT,
            raw_data=raw_data,
        )
    )

    print(f"Generated plot: {result.image_path}")


if __name__ == "__main__":
    asyncio.run(main())
