import asyncio

from PIL import Image

from paperbanana.core.pipeline import PaperBananaPipeline
from paperbanana.core.types import DiagramType, GenerationInput


# -------- Fake VLM --------
class FakeVLM:
    name = "fake-vlm"
    model_name = "fake-model"

    async def generate(self, *args, **kwargs):
        return "fake response"


# -------- Fake Image Generator --------
class FakeImageGen:
    async def generate(
        self, prompt=None, output_path=None, iteration=None, seed=None, **kwargs
    ):
        iteration = iteration or 1  # fallback to 1 if None
        img = Image.new("RGB", (256, 256), color=(iteration * 40 % 256, 100, 150))
        return img


async def main():
    pipeline = PaperBananaPipeline(vlm_client=FakeVLM(), image_gen_fn=FakeImageGen())

    inp = GenerationInput(
        source_context="A neural network with input, hidden and output layers.",
        communicative_intent="Explain feedforward architecture",
        diagram_type=DiagramType.METHODOLOGY,
        raw_data=None,
    )

    output = await pipeline.generate(inp)

    print("TIMING FOUND:")
    print(output.metadata["timing"])


asyncio.run(main())
