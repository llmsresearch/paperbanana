"""Tab 1: Academic Diagram Generation."""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import structlog

from paperbanana.app.components.config_panel import get_settings_from_config
from paperbanana.app.components.iteration_gallery import (
    build_gallery_items,
    format_all_critiques,
)
from paperbanana.app.utils import settings_from_ui
from paperbanana.core.types import DiagramType, GenerationInput

logger = structlog.get_logger()

DIAGRAM_TYPES = [t.value for t in DiagramType if t != DiagramType.SLIDE]
ASPECT_RATIOS = ["Auto", "1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"]


def create_diagrams_tab(config: dict):
    """Build the Diagrams tab UI and wire callbacks."""

    with gr.Row():
        # ── Left: Inputs ──
        with gr.Column(scale=2):
            source_context = gr.Textbox(
                label="Source Context",
                placeholder="Paste your methodology section or paper excerpt here...",
                lines=10,
            )
            caption = gr.Textbox(
                label="Communicative Intent (Caption)",
                placeholder="e.g. Overview of the proposed three-stage framework",
                lines=2,
            )
            with gr.Row():
                diagram_type = gr.Dropdown(
                    choices=DIAGRAM_TYPES,
                    value="methodology",
                    label="Diagram Type",
                )
                aspect_ratio = gr.Dropdown(
                    choices=ASPECT_RATIOS,
                    value="Auto",
                    label="Aspect Ratio",
                )

            with gr.Row():
                generate_btn = gr.Button("Generate", variant="primary", scale=2)
                continue_btn = gr.Button("Continue / Refine", variant="secondary", scale=1)

            feedback = gr.Textbox(
                label="Feedback for Refinement",
                placeholder="Optional: describe what to change for the next iteration...",
                lines=2,
                visible=True,
            )

        # ── Right: Outputs ──
        with gr.Column(scale=3):
            result_image = gr.Image(label="Generated Diagram", type="filepath")
            gallery = gr.Gallery(
                label="Iteration History",
                columns=4,
                height=200,
            )
            description = gr.Textbox(
                label="Final Description",
                lines=4,
                interactive=False,
            )
            critiques = gr.Textbox(
                label="Critique Details",
                lines=6,
                interactive=False,
            )

    # Hidden state to store last run metadata for continue
    last_run_dir = gr.State(value=None)
    last_run_id = gr.State(value=None)

    # ── Progress log ──
    progress_log = gr.Textbox(
        label="Progress",
        lines=3,
        interactive=False,
    )

    # ── Generate callback ──
    config_inputs = get_settings_from_config(config)

    async def on_generate(
        context, intent, dtype, ratio,
        vlm_prov, vlm_mod, img_prov, img_mod, key1, key2,
        iters, res, auto_ref, opt_in,
    ):
        if not key1.strip():
            raise gr.Error("API Key is required. Please enter your key in the Settings panel.")
        if not context.strip():
            raise gr.Error("Source context is required.")
        if not intent.strip():
            raise gr.Error("Caption / communicative intent is required.")

        try:
            settings = settings_from_ui(
                vlm_prov, vlm_mod, img_prov, img_mod,
                key1, key2, iters, res, auto_ref, opt_in,
            )
            from paperbanana.core.pipeline import PaperBananaPipeline

            pipeline = PaperBananaPipeline(settings=settings)
            gen_input = GenerationInput(
                source_context=context,
                communicative_intent=intent,
                diagram_type=DiagramType(dtype),
                aspect_ratio=ratio if ratio != "Auto" else None,
            )
            result = await pipeline.generate(gen_input)
            gallery_items = build_gallery_items(result)
            critique_text = format_all_critiques(result)
            run_dir = result.metadata.get("run_dir", "")
            run_id = result.metadata.get("run_id", "")

            return (
                result.image_path,
                gallery_items,
                result.description,
                critique_text,
                run_dir,
                run_id,
                f"Done! {len(result.iterations)} iterations. Run: {run_id}",
            )
        except Exception as e:
            logger.exception("Diagram generation failed")
            raise gr.Error(f"Generation failed: {e}")

    generate_btn.click(
        fn=on_generate,
        inputs=[source_context, caption, diagram_type, aspect_ratio] + config_inputs,
        outputs=[result_image, gallery, description, critiques, last_run_dir, last_run_id, progress_log],
    )

    # ── Continue callback ──
    async def on_continue(
        run_dir_val, run_id_val, fb_text,
        vlm_prov, vlm_mod, img_prov, img_mod, key1, key2,
        iters, res, auto_ref, opt_in,
    ):
        if not run_id_val:
            raise gr.Error("No previous run to continue. Generate first.")

        try:
            settings = settings_from_ui(
                vlm_prov, vlm_mod, img_prov, img_mod,
                key1, key2, iters, res, auto_ref, opt_in,
            )
            from paperbanana.core.pipeline import PaperBananaPipeline
            from paperbanana.core.resume import load_resume_state

            output_dir = str(Path(run_dir_val).parent) if run_dir_val else settings.output_dir
            resume = load_resume_state(output_dir, run_id_val)
            pipeline = PaperBananaPipeline(settings=settings)
            result = await pipeline.continue_run(
                resume,
                additional_iterations=iters,
                user_feedback=fb_text if fb_text.strip() else None,
            )
            gallery_items = build_gallery_items(result)
            critique_text = format_all_critiques(result)

            return (
                result.image_path,
                gallery_items,
                result.description,
                critique_text,
                f"Continued! {len(result.iterations)} total iterations.",
            )
        except Exception as e:
            logger.exception("Continue failed")
            raise gr.Error(f"Continue failed: {e}")

    continue_btn.click(
        fn=on_continue,
        inputs=[last_run_dir, last_run_id, feedback] + config_inputs,
        outputs=[result_image, gallery, description, critiques, progress_log],
    )
