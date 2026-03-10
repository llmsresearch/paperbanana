"""Tab 3: Statistical Plot Generation."""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import structlog

from paperbanana.app.components.config_panel import get_settings_from_config
from paperbanana.app.components.iteration_gallery import build_gallery_items
from paperbanana.app.utils import settings_from_ui
from paperbanana.core.types import DiagramType, GenerationInput

logger = structlog.get_logger()


def create_plots_tab(config: dict):
    """Build the Plots tab UI and wire callbacks."""

    with gr.Row():
        # ── Left: Inputs ──
        with gr.Column(scale=2):
            data_file = gr.File(
                label="Upload Data (CSV or JSON)",
                file_types=[".csv", ".json"],
                file_count="single",
            )
            data_preview = gr.Dataframe(
                label="Data Preview (first 10 rows)",
                interactive=False,
                visible=False,
            )
            intent = gr.Textbox(
                label="Plot Intent",
                placeholder="e.g. Bar chart comparing model accuracy across 5 benchmarks",
                lines=3,
            )
            generate_btn = gr.Button("Generate Plot", variant="primary")

        # ── Right: Outputs ──
        with gr.Column(scale=3):
            result_image = gr.Image(label="Generated Plot", type="filepath")
            gallery = gr.Gallery(
                label="Iteration History",
                columns=4,
                height=180,
            )

    progress_log = gr.Textbox(label="Progress", lines=2, interactive=False)

    # Hidden state for data content
    data_text_state = gr.State(value="")
    data_dict_state = gr.State(value=None)

    # ── Data preview callback ──
    def on_data_upload(file):
        if file is None:
            return gr.update(visible=False), "", None

        import json
        import pandas as pd

        path = Path(file.name)
        try:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
                data_text = df.to_csv(index=False)
                data_dict = {"csv_content": data_text, "format": "csv"}
            else:
                with open(path) as f:
                    raw = json.load(f)
                df = pd.DataFrame(raw) if isinstance(raw, list) else pd.DataFrame([raw])
                data_text = json.dumps(raw, indent=2)
                data_dict = {"json_content": data_text, "format": "json"}

            preview = df.head(10)
            return gr.update(value=preview, visible=True), data_text, data_dict
        except Exception as e:
            raise gr.Error(f"Failed to read data file: {e}")

    data_file.change(
        fn=on_data_upload,
        inputs=[data_file],
        outputs=[data_preview, data_text_state, data_dict_state],
    )

    # ── Generate callback ──
    config_inputs = get_settings_from_config(config)

    async def on_generate(
        data_text, data_dict, intent_text,
        vlm_prov, vlm_mod, img_prov, img_mod, key1, key2,
        iters, res, auto_ref, opt_in,
    ):
        if not key1.strip():
            raise gr.Error("API Key is required.")
        if not data_text and not intent_text.strip():
            raise gr.Error("Please upload data and/or provide a plot intent.")

        try:
            settings = settings_from_ui(
                vlm_prov, vlm_mod, img_prov, img_mod,
                key1, key2, iters, res, auto_ref, opt_in,
            )

            from paperbanana.core.pipeline import PaperBananaPipeline

            pipeline = PaperBananaPipeline(settings=settings)

            context = data_text if data_text else intent_text
            gen_input = GenerationInput(
                source_context=context,
                communicative_intent=intent_text or "Statistical visualization of the provided data",
                diagram_type=DiagramType.STATISTICAL_PLOT,
                raw_data=data_dict,
            )
            result = await pipeline.generate(gen_input)
            gallery_items = build_gallery_items(result)

            return (
                result.image_path,
                gallery_items,
                f"Done! {len(result.iterations)} iterations.",
            )
        except Exception as e:
            logger.exception("Plot generation failed")
            raise gr.Error(f"Plot generation failed: {e}")

    generate_btn.click(
        fn=on_generate,
        inputs=[data_text_state, data_dict_state, intent] + config_inputs,
        outputs=[result_image, gallery, progress_log],
    )
