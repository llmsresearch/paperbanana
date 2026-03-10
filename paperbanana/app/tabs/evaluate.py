"""Tab 4: VLM-as-Judge Evaluation."""

from __future__ import annotations

import gradio as gr
import structlog

from paperbanana.app.components.config_panel import get_settings_from_config
from paperbanana.app.utils import settings_from_ui
from paperbanana.core.types import EvaluationScore

logger = structlog.get_logger()


def _format_score(score: EvaluationScore) -> str:
    """Format EvaluationScore into a readable markdown string."""
    lines = [
        f"## Overall: **{score.overall_winner}** (Score: {score.overall_score:.1f}/100)",
        "",
        "| Dimension | Winner | Score | Reasoning |",
        "|-----------|--------|-------|-----------|",
    ]
    for dim_name in ["faithfulness", "conciseness", "readability", "aesthetics"]:
        dim = getattr(score, dim_name)
        lines.append(
            f"| {dim_name.capitalize()} | {dim.winner} | {dim.score:.0f} | {dim.reasoning[:100]} |"
        )
    return "\n".join(lines)


def create_evaluate_tab(config: dict):
    """Build the Evaluate tab UI and wire callbacks."""

    with gr.Row():
        # ── Left: Inputs ──
        with gr.Column(scale=2):
            generated_img = gr.Image(
                label="Generated Image",
                type="filepath",
            )
            reference_img = gr.Image(
                label="Reference Image (Human-drawn)",
                type="filepath",
            )
            source_context = gr.Textbox(
                label="Source Context",
                placeholder="Original methodology text...",
                lines=5,
            )
            caption = gr.Textbox(
                label="Figure Caption",
                placeholder="e.g. Overview of the proposed framework",
                lines=2,
            )
            evaluate_btn = gr.Button("Evaluate", variant="primary")

        # ── Right: Results ──
        with gr.Column(scale=3):
            result_md = gr.Markdown(
                value="*Upload images and click Evaluate to see results.*",
            )

    progress_log = gr.Textbox(label="Progress", lines=2, interactive=False)

    config_inputs = get_settings_from_config(config)

    async def on_evaluate(
        gen_path, ref_path, context_text, caption_text,
        vlm_prov, vlm_mod, img_prov, img_mod, key1, key2,
        iters, res, auto_ref, opt_in,
    ):
        if not key1.strip():
            raise gr.Error("API Key is required.")
        if not gen_path:
            raise gr.Error("Please upload a generated image.")
        if not ref_path:
            raise gr.Error("Please upload a reference image.")
        if not context_text.strip():
            raise gr.Error("Source context is required.")
        if not caption_text.strip():
            raise gr.Error("Caption is required.")

        try:
            settings = settings_from_ui(
                vlm_prov, vlm_mod, img_prov, img_mod,
                key1, key2, iters, res, auto_ref, opt_in,
            )

            from paperbanana.core.utils import find_prompt_dir
            from paperbanana.evaluation.judge import VLMJudge
            from paperbanana.providers.registry import ProviderRegistry

            vlm = ProviderRegistry.create_vlm(settings)
            judge = VLMJudge(vlm, prompt_dir=find_prompt_dir())

            score = await judge.evaluate(
                image_path=gen_path,
                source_context=context_text,
                caption=caption_text,
                reference_path=ref_path,
            )
            result_text = _format_score(score)
            return result_text, "Evaluation complete."
        except Exception as e:
            logger.exception("Evaluation failed")
            raise gr.Error(f"Evaluation failed: {e}")

    evaluate_btn.click(
        fn=on_evaluate,
        inputs=[generated_img, reference_img, source_context, caption] + config_inputs,
        outputs=[result_md, progress_log],
    )
