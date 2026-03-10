"""PaperBanana Gradio Application — main entry point."""

from __future__ import annotations

import gradio as gr

from paperbanana.app.components.config_panel import create_config_panel
from paperbanana.app.tabs.diagrams import create_diagrams_tab
from paperbanana.app.tabs.evaluate import create_evaluate_tab
from paperbanana.app.tabs.plots import create_plots_tab
from paperbanana.app.tabs.slides import create_slides_tab

TITLE = "PaperBanana"
DESCRIPTION = (
    "Generate publication-quality academic illustrations, slides, and statistical plots "
    "using a multi-agent AI pipeline."
)

CSS = """
.config-panel {
    border-right: 1px solid #e0e0e0;
    padding-right: 16px;
}
"""


def create_app() -> gr.Blocks:
    """Create and return the Gradio Blocks application."""
    with gr.Blocks(
        title=TITLE,
        theme=gr.themes.Soft(),
        css=CSS,
    ) as demo:
        gr.Markdown(f"# {TITLE}")
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            # ── Sidebar: Config ──
            with gr.Column(scale=1, elem_classes=["config-panel"]):
                config = create_config_panel()

            # ── Main: Tabs ──
            with gr.Column(scale=4):
                with gr.Tabs():
                    with gr.Tab("Diagrams"):
                        create_diagrams_tab(config)

                    with gr.Tab("Slides"):
                        create_slides_tab(config)

                    with gr.Tab("Plots"):
                        create_plots_tab(config)

                    with gr.Tab("Evaluate"):
                        create_evaluate_tab(config)

    return demo


def launch(port: int = 7860, share: bool = False):
    """Create and launch the app."""
    demo = create_app()
    demo.launch(server_port=port, share=share)


if __name__ == "__main__":
    launch()
