"""Tab 2: Slide Generation (single + batch)."""

from __future__ import annotations

import asyncio
import os
import tempfile
import zipfile
from pathlib import Path

import gradio as gr
import structlog

from paperbanana.app.components.config_panel import get_settings_from_config
from paperbanana.app.components.iteration_gallery import build_gallery_items
from paperbanana.app.components.style_picker import (
    get_style_choices,
    get_style_info,
)
from paperbanana.app.utils import settings_from_ui
from paperbanana.core.types import DiagramType, GenerationInput
from paperbanana.guidelines.slide_styles import get_style_prompt

logger = structlog.get_logger()


def create_slides_tab(config: dict):
    """Build the Slides tab UI and wire callbacks."""

    config_inputs = get_settings_from_config(config)

    with gr.Tabs():
        # ── Sub-tab: Single Slide ──
        with gr.Tab("Single Slide"):
            with gr.Row():
                with gr.Column(scale=2):
                    style = gr.Dropdown(
                        choices=get_style_choices(),
                        value="blueprint",
                        label="Style Preset",
                    )
                    style_info = gr.Markdown(value=get_style_info("blueprint"))
                    slide_prompt = gr.Textbox(
                        label="Slide Content Prompt",
                        placeholder="Describe the slide content (title, body, visual layout)...",
                        lines=8,
                    )
                    single_gen_btn = gr.Button("Generate Slide", variant="primary")

                with gr.Column(scale=3):
                    single_image = gr.Image(label="Generated Slide", type="filepath")
                    single_gallery = gr.Gallery(
                        label="Iteration History",
                        columns=4,
                        height=180,
                    )

            single_log = gr.Textbox(label="Progress", lines=2, interactive=False)

            style.change(
                fn=get_style_info,
                inputs=[style],
                outputs=[style_info],
            )

            async def on_single_generate(
                style_name, prompt_text,
                vlm_prov, vlm_mod, img_prov, img_mod, key1, key2,
                iters, res, auto_ref, opt_in,
            ):
                if not key1.strip():
                    raise gr.Error("API Key is required.")
                if not prompt_text.strip():
                    raise gr.Error("Slide prompt is required.")

                try:
                    settings = settings_from_ui(
                        vlm_prov, vlm_mod, img_prov, img_mod,
                        key1, key2, iters, res, auto_ref, opt_in,
                    )
                    style_prompt_text = get_style_prompt(style_name) if style_name else ""
                    full_context = (
                        f"{style_prompt_text}\n\n---\n\n{prompt_text}"
                        if style_prompt_text
                        else prompt_text
                    )

                    from paperbanana.core.pipeline import PaperBananaPipeline

                    pipeline = PaperBananaPipeline(settings=settings)
                    gen_input = GenerationInput(
                        source_context=full_context,
                        communicative_intent=f"Presentation slide: {prompt_text[:100]}",
                        diagram_type=DiagramType.SLIDE,
                        aspect_ratio="16:9",
                    )
                    result = await pipeline.generate(gen_input)
                    gallery_items = build_gallery_items(result)

                    return (
                        result.image_path,
                        gallery_items,
                        f"Done! {len(result.iterations)} iterations.",
                    )
                except Exception as e:
                    logger.exception("Slide generation failed")
                    raise gr.Error(f"Slide generation failed: {e}")

            single_gen_btn.click(
                fn=on_single_generate,
                inputs=[style, slide_prompt] + config_inputs,
                outputs=[single_image, single_gallery, single_log],
            )

        # ── Sub-tab: Batch Slides ──
        with gr.Tab("Batch Slides"):
            gr.Markdown("Upload multiple `.md` prompt files to generate a full slide deck.")

            with gr.Row():
                with gr.Column(scale=2):
                    batch_style = gr.Dropdown(
                        choices=get_style_choices(),
                        value="blueprint",
                        label="Style Preset",
                    )
                    batch_files = gr.File(
                        label="Upload Prompt Files (.md)",
                        file_types=[".md", ".txt"],
                        file_count="multiple",
                    )
                    batch_gen_btn = gr.Button("Generate All Slides", variant="primary")

                with gr.Column(scale=3):
                    batch_gallery = gr.Gallery(
                        label="Generated Slides",
                        columns=3,
                        height=400,
                    )
                    batch_download = gr.File(label="Download ZIP", visible=False)

            batch_log = gr.Textbox(label="Progress", lines=3, interactive=False)

            async def on_batch_generate(
                style_name, files,
                vlm_prov, vlm_mod, img_prov, img_mod, key1, key2,
                iters, res, auto_ref, opt_in,
            ):
                if not key1.strip():
                    raise gr.Error("API Key is required.")
                if not files:
                    raise gr.Error("Please upload at least one prompt file.")

                try:
                    settings = settings_from_ui(
                        vlm_prov, vlm_mod, img_prov, img_mod,
                        key1, key2, iters, res, auto_ref, opt_in,
                    )
                    style_prompt_text = get_style_prompt(style_name) if style_name else ""

                    from paperbanana.core.pipeline import PaperBananaPipeline

                    pipeline = PaperBananaPipeline(settings=settings)
                    sorted_files = sorted(files, key=lambda f: Path(f.name).name)

                    sem = asyncio.Semaphore(settings.batch_concurrent)

                    async def gen_one(idx, file):
                        async with sem:
                            prompt_text = Path(file.name).read_text(encoding="utf-8")
                            full_context = (
                                f"{style_prompt_text}\n\n---\n\n{prompt_text}"
                                if style_prompt_text
                                else prompt_text
                            )
                            gen_input = GenerationInput(
                                source_context=full_context,
                                communicative_intent=f"Slide {idx + 1}: {Path(file.name).stem}",
                                diagram_type=DiagramType.SLIDE,
                                aspect_ratio="16:9",
                            )
                            return await pipeline.generate(gen_input)

                    results = await asyncio.gather(
                        *[gen_one(i, f) for i, f in enumerate(sorted_files)]
                    )

                    gallery_items = []
                    output_paths = []
                    for idx, result in enumerate(results):
                        gallery_items.append(
                            (result.image_path, f"Slide {idx + 1}: {Path(sorted_files[idx].name).stem}")
                        )
                        output_paths.append(result.image_path)

                    # Create ZIP for download (use mkstemp for safety)
                    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".zip", prefix="paperbanana_slides_")
                    os.close(tmp_fd)
                    tmp_zip = Path(tmp_path)
                    with zipfile.ZipFile(tmp_zip, "w") as zf:
                        for p in output_paths:
                            zf.write(p, Path(p).name)

                    return (
                        gallery_items,
                        gr.update(value=str(tmp_zip), visible=True),
                        f"Done! Generated {len(output_paths)} slides.",
                    )
                except Exception as e:
                    logger.exception("Batch slide generation failed")
                    raise gr.Error(f"Batch generation failed: {e}")

            batch_gen_btn.click(
                fn=on_batch_generate,
                inputs=[batch_style, batch_files] + config_inputs,
                outputs=[batch_gallery, batch_download, batch_log],
            )
