"""Example: Generate a methodology diagram from text."""

import asyncio

from dotenv import load_dotenv

from paperbanana import DiagramType, GenerationInput, PaperBananaPipeline
from paperbanana.core.config import Settings

load_dotenv()


async def main():
    # Example methodology text
    source_context = """
    Our framework, PaperBanana, automates the generation of publication-quality
    academic illustrations through a multi-agent pipeline. The system takes as input
    a methodology section (S) and a figure caption (C).

    Phase 1 - Linear Planning:
    1. The Retriever agent selects the top-10 most relevant reference examples from
       a curated set of high-quality diagrams.
    2. The Planner agent uses in-context learning from these examples to generate
       a detailed textual description (P) of the target diagram.
    3. The Stylist agent refines the description to optimize visual aesthetics (P*).

    Phase 2 - Iterative Refinement:
    4. The Visualizer agent renders the description into an image using a
       text-to-image generation model.
    5. The Critic agent evaluates the image on faithfulness, conciseness,
       readability, and aesthetics, providing targeted revision feedback.
    6. Steps 4-5 repeat for up to 3 iterations until quality is satisfactory.
    """

    caption = (
        "Overview of the PaperBanana multi-agent framework for automated "
        "academic illustration generation."
    )

    settings = Settings(
        vlm_provider="gemini",
        vlm_model="gemini-2.0-flash",
        image_provider="google_imagen",
        image_model="gemini-3-pro-image-preview",
        refinement_iterations=2,
    )

    # Create pipeline and generate
    pipeline = PaperBananaPipeline(settings=settings)

    result = await pipeline.generate(
        GenerationInput(
            source_context=source_context,
            communicative_intent=caption,
            diagram_type=DiagramType.METHODOLOGY,
        )
    )

    print(f"Generated diagram: {result.image_path}")
    print(f"Total iterations: {len(result.iterations)}")
    print(f"Run ID: {result.metadata.get('run_id')}")


if __name__ == "__main__":
    asyncio.run(main())
