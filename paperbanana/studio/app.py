"""Gradio-based PaperBanana Studio — local web UI."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

from paperbanana.studio import runs as runs_mod
from paperbanana.studio.runner import (
    ASPECT_RATIO_CHOICES,
    IMAGE_PROVIDER_CHOICES,
    VLM_PROVIDER_CHOICES,
    build_settings,
    merge_context,
    run_batch,
    run_continue,
    run_evaluate,
    run_methodology,
    run_plot,
)


def _dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def _upload_path(file_obj: Any) -> Optional[str]:
    """Normalize Gradio file upload to a filesystem path."""
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj if file_obj.strip() else None
    return getattr(file_obj, "name", None) or str(file_obj)


def build_studio_app(
    *,
    default_output_dir: str = "outputs",
    config_path: Optional[str] = None,
):
    """Construct the Gradio Blocks interface."""
    import gradio as gr

    _dotenv()

    default_cfg = config_path or ""

    def _settings(
        out_dir: str,
        cfg: str,
        vlm_p: str,
        vlm_m: str,
        img_p: str,
        img_m: str,
        fmt: str,
        iters: float,
        auto: bool,
        max_it: float,
        opt: bool,
        save_pr: bool,
        seed_val: Optional[float],
    ):
        seed_int: Optional[int] = None
        if seed_val is not None:
            if isinstance(seed_val, float) and math.isnan(seed_val):
                seed_int = None
            else:
                try:
                    seed_int = int(seed_val)
                except (TypeError, ValueError):
                    seed_int = None
        return build_settings(
            config_path=(cfg or "").strip() or None,
            output_dir=(out_dir or default_output_dir).strip() or default_output_dir,
            vlm_provider=vlm_p,
            vlm_model=vlm_m,
            image_provider=img_p,
            image_model=img_m,
            output_format=fmt,
            refinement_iterations=max(1, int(iters)),
            auto_refine=auto,
            max_iterations=max(1, int(max_it)),
            optimize_inputs=opt,
            save_prompts=save_pr,
            seed=seed_int,
        )

    # ── Shared settings column ─────────────────────────────────────────
    def settings_accordion():
        with gr.Accordion("Model & pipeline", open=False):
            out_dir = gr.Textbox(
                label="Output directory",
                value=default_output_dir,
                info="Run folders (run_*) and batch folders are created here.",
            )
            cfg = gr.Textbox(
                label="Config YAML (optional)",
                value=default_cfg,
                placeholder="configs/config.yaml",
            )
            with gr.Row():
                vlm_p = gr.Dropdown(
                    label="VLM provider",
                    choices=VLM_PROVIDER_CHOICES,
                    value=VLM_PROVIDER_CHOICES[0],
                )
                vlm_m = gr.Textbox(
                    label="VLM model",
                    value="",
                    placeholder="Leave blank for provider default",
                )
            with gr.Row():
                img_p = gr.Dropdown(
                    label="Image provider",
                    choices=IMAGE_PROVIDER_CHOICES,
                    value=IMAGE_PROVIDER_CHOICES[0],
                )
                img_m = gr.Textbox(
                    label="Image model",
                    value="",
                    placeholder="Leave blank for provider default",
                )
            with gr.Row():
                fmt = gr.Dropdown(
                    label="Output format",
                    choices=["png", "jpeg", "webp"],
                    value="png",
                )
                iters = gr.Number(label="Refinement iterations", value=3, precision=0, minimum=1)
                max_it = gr.Number(label="Max iterations (auto mode cap)", value=30, precision=0)
            with gr.Row():
                opt = gr.Checkbox(label="Optimize inputs", value=False)
                auto = gr.Checkbox(label="Auto-refine until critic satisfied", value=False)
                save_pr = gr.Checkbox(label="Save prompts to run dir", value=True)
            seed_val = gr.Number(
                label="Random seed (optional)",
                value=None,
                precision=0,
                info="Empty = provider default",
            )
        return (
            out_dir,
            cfg,
            vlm_p,
            vlm_m,
            img_p,
            img_m,
            fmt,
            iters,
            auto,
            max_it,
            opt,
            save_pr,
            seed_val,
        )

    with gr.Blocks(
        title="PaperBanana Studio",
        theme=gr.themes.Soft(
            primary_hue="amber",
            secondary_hue="slate",
        ),
    ) as demo:
        gr.Markdown(
            "# PaperBanana Studio\n"
            "Generate methodology diagrams, statistical plots, and run evaluations "
            "in the browser. API keys are read from your environment or `.env` "
            "(same as the CLI)."
        )

        shared = settings_accordion()
        (
            out_dir,
            cfg,
            vlm_p,
            vlm_m,
            img_p,
            img_m,
            fmt,
            iters,
            auto_ref,
            max_it,
            opt_in,
            save_pr,
            seed_val,
        ) = shared

        with gr.Tabs():
            # ── Diagram ─────────────────────────────────────────────────
            with gr.Tab("Diagram"):
                gr.Markdown(
                    "Paste methodology text or upload a `.txt` / `.md` file. "
                    "If both are set, the **uploaded file** takes precedence."
                )
                ctx_text = gr.Textbox(
                    label="Methodology / context",
                    lines=12,
                    placeholder="Describe your method, architecture, or paste a paper excerpt…",
                )
                ctx_file = gr.File(
                    label="Context file (optional)",
                    file_types=[".txt", ".md"],
                )
                cap = gr.Textbox(
                    label="Figure caption / communicative intent",
                    lines=2,
                    placeholder="e.g. Overview of our encoder–decoder with sparse routing",
                )
                ar = gr.Dropdown(
                    label="Aspect ratio",
                    choices=ASPECT_RATIO_CHOICES,
                    value="default",
                )
                d_log = gr.Textbox(label="Progress log", lines=18)
                d_img = gr.Image(label="Final diagram", type="filepath")
                d_gal = gr.Gallery(
                    label="Iteration images",
                    columns=4,
                    height=240,
                    object_fit="contain",
                )
                d_go = gr.Button("Generate diagram", variant="primary")

                def _do_diagram(
                    od,
                    c,
                    vp,
                    vm,
                    ip,
                    im,
                    fo,
                    it,
                    au,
                    mx,
                    op,
                    sp,
                    sd,
                    text,
                    file,
                    caption,
                    aspect,
                ):
                    _dotenv()
                    try:
                        st = _settings(od, c, vp, vm, ip, im, fo, it, au, mx, op, sp, sd)
                        ctx = merge_context(text, _upload_path(file))
                        if not ctx.strip():
                            return "Context is empty.", None, []
                        if not (caption or "").strip():
                            return "Caption is required.", None, []
                        log, img, gal, err = run_methodology(
                            st, ctx, caption, aspect, verbose_logging=False
                        )
                        if err:
                            return log, None, gal
                        return log, img, gal
                    except Exception as e:
                        return f"{type(e).__name__}: {e}", None, []

                d_go.click(
                    _do_diagram,
                    inputs=[
                        out_dir,
                        cfg,
                        vlm_p,
                        vlm_m,
                        img_p,
                        img_m,
                        fmt,
                        iters,
                        auto_ref,
                        max_it,
                        opt_in,
                        save_pr,
                        seed_val,
                        ctx_text,
                        ctx_file,
                        cap,
                        ar,
                    ],
                    outputs=[d_log, d_img, d_gal],
                )

            # ── Plot ────────────────────────────────────────────────────
            with gr.Tab("Plot"):
                gr.Markdown(
                    "Upload a **CSV** or **JSON** data file and describe the plot you want."
                )
                data_f = gr.File(label="Data file", file_types=[".csv", ".json"])
                intent = gr.Textbox(
                    label="Communicative intent",
                    lines=2,
                    placeholder="e.g. Bar chart comparing accuracy across benchmarks",
                )
                ar_p = gr.Dropdown(
                    label="Aspect ratio",
                    choices=ASPECT_RATIO_CHOICES,
                    value="default",
                )
                p_log = gr.Textbox(label="Progress log", lines=18)
                p_img = gr.Image(label="Final plot", type="filepath")
                p_gal = gr.Gallery(label="Iteration images", columns=4, height=240)
                p_go = gr.Button("Generate plot", variant="primary")

                def _do_plot(
                    od,
                    c,
                    vp,
                    vm,
                    ip,
                    im,
                    fo,
                    it,
                    au,
                    mx,
                    op,
                    sp,
                    sd,
                    dfile,
                    inten,
                    aspect,
                ):
                    _dotenv()
                    try:
                        st = _settings(od, c, vp, vm, ip, im, fo, it, au, mx, op, sp, sd)
                        path = _upload_path(dfile)
                        if not path:
                            return "Upload a data file.", None, []
                        if not (inten or "").strip():
                            return "Communicative intent is required.", None, []
                        log, img, gal, err = run_plot(
                            st, path, inten, aspect, verbose_logging=False
                        )
                        if err:
                            return log, None, gal
                        return log, img, gal
                    except Exception as e:
                        return f"{type(e).__name__}: {e}", None, []

                p_go.click(
                    _do_plot,
                    inputs=[
                        out_dir,
                        cfg,
                        vlm_p,
                        vlm_m,
                        img_p,
                        img_m,
                        fmt,
                        iters,
                        auto_ref,
                        max_it,
                        opt_in,
                        save_pr,
                        seed_val,
                        data_f,
                        intent,
                        ar_p,
                    ],
                    outputs=[p_log, p_img, p_gal],
                )

            # ── Evaluate ────────────────────────────────────────────────
            with gr.Tab("Evaluate"):
                gr.Markdown(
                    "Compare a **generated** image to a **human reference** using the "
                    "paper’s VLM-as-judge protocol (four dimensions + overall)."
                )
                g_img = gr.Image(label="Generated diagram", type="filepath")
                r_img = gr.Image(label="Human reference", type="filepath")
                ev_ctx = gr.Textbox(label="Source context", lines=8)
                ev_ctx_f = gr.File(
                    label="Context file (optional)",
                    file_types=[".txt", ".md"],
                )
                ev_cap = gr.Textbox(label="Figure caption", lines=2)
                ev_log = gr.Textbox(label="Log", lines=6)
                ev_out = gr.Markdown()
                ev_go = gr.Button("Run evaluation", variant="primary")

                def _do_eval(
                    od,
                    c,
                    vp,
                    vm,
                    ip,
                    im,
                    fo,
                    it,
                    au,
                    mx,
                    op,
                    sp,
                    sd,
                    gen,
                    ref,
                    etext,
                    efile,
                    ecap,
                ):
                    _dotenv()
                    try:
                        st = _settings(od, c, vp, vm, ip, im, fo, it, au, mx, op, sp, sd)
                        gp = _upload_path(gen) or ""
                        rp = _upload_path(ref) or ""
                        ctx = merge_context(etext, _upload_path(efile))
                        log, res = run_evaluate(st, gp, rp, ctx, ecap or "", verbose_logging=False)
                        return log, res
                    except Exception as e:
                        return f"{type(e).__name__}: {e}", str(e)

                ev_go.click(
                    _do_eval,
                    inputs=[
                        out_dir,
                        cfg,
                        vlm_p,
                        vlm_m,
                        img_p,
                        img_m,
                        fmt,
                        iters,
                        auto_ref,
                        max_it,
                        opt_in,
                        save_pr,
                        seed_val,
                        g_img,
                        r_img,
                        ev_ctx,
                        ev_ctx_f,
                        ev_cap,
                    ],
                    outputs=[ev_log, ev_out],
                )

            # ── Continue run ─────────────────────────────────────────────
            with gr.Tab("Continue"):
                gr.Markdown(
                    "Load state from a previous **run_*** folder under the output directory "
                    "and run more visualizer–critic iterations."
                )
                cr_id = gr.Textbox(
                    label="Run ID",
                    placeholder="run_20260218_125448_e7b876",
                )
                cr_fb = gr.Textbox(
                    label="Feedback for critic (optional)",
                    lines=3,
                    placeholder="e.g. Make arrows thicker and increase color contrast",
                )
                cr_extra = gr.Number(
                    label="Additional iterations (optional)",
                    value=None,
                    precision=0,
                    info="Empty = use pipeline default from settings",
                )
                cr_log = gr.Textbox(label="Progress log", lines=16)
                cr_img = gr.Image(label="Latest result", type="filepath")
                cr_gal = gr.Gallery(label="New iteration images", columns=4, height=200)
                cr_go = gr.Button("Continue run", variant="primary")

                def _do_continue(
                    od,
                    c,
                    vp,
                    vm,
                    ip,
                    im,
                    fo,
                    it,
                    au,
                    mx,
                    op,
                    sp,
                    sd,
                    rid,
                    fb,
                    extra,
                ):
                    _dotenv()
                    try:
                        st = _settings(od, c, vp, vm, ip, im, fo, it, au, mx, op, sp, sd)
                        if not (rid or "").strip():
                            return "Run ID is required.", None, []
                        ex: Optional[int] = None
                        if extra is not None and not (
                            isinstance(extra, float) and math.isnan(extra)
                        ):
                            try:
                                ex = int(extra)
                            except (TypeError, ValueError):
                                ex = None
                        log, img, gal, err = run_continue(
                            st,
                            (od or default_output_dir).strip() or default_output_dir,
                            rid,
                            fb or "",
                            ex,
                            verbose_logging=False,
                        )
                        if err:
                            return log, None, gal
                        return log, img, gal
                    except Exception as e:
                        return f"{type(e).__name__}: {e}", None, []

                cr_go.click(
                    _do_continue,
                    inputs=[
                        out_dir,
                        cfg,
                        vlm_p,
                        vlm_m,
                        img_p,
                        img_m,
                        fmt,
                        iters,
                        auto_ref,
                        max_it,
                        opt_in,
                        save_pr,
                        seed_val,
                        cr_id,
                        cr_fb,
                        cr_extra,
                    ],
                    outputs=[cr_log, cr_img, cr_gal],
                )

            # ── Batch ───────────────────────────────────────────────────
            with gr.Tab("Batch"):
                gr.Markdown(
                    "Upload a **YAML** or **JSON** manifest (same format as "
                    "`paperbanana batch --manifest`). Paths inside the file are "
                    "resolved relative to the manifest’s directory — prefer absolute "
                    "paths or keep inputs next to the manifest."
                )
                bf = gr.File(label="Manifest", file_types=[".yaml", ".yml", ".json"])
                b_log = gr.Textbox(label="Batch log", lines=22)
                b_dir = gr.Textbox(label="Batch output directory", lines=1)
                b_go = gr.Button("Run batch", variant="primary")

                def _do_batch(
                    od,
                    c,
                    vp,
                    vm,
                    ip,
                    im,
                    fo,
                    it,
                    au,
                    mx,
                    op,
                    sp,
                    sd,
                    mfile,
                ):
                    _dotenv()
                    try:
                        st0 = _settings(od, c, vp, vm, ip, im, fo, it, au, mx, op, sp, sd)
                        path = _upload_path(mfile)
                        if not path:
                            return "Upload a manifest file.", ""
                        log, bpath = run_batch(st0, path, verbose_logging=False)
                        return log, bpath
                    except Exception as e:
                        return f"{type(e).__name__}: {e}", ""

                b_go.click(
                    _do_batch,
                    inputs=[
                        out_dir,
                        cfg,
                        vlm_p,
                        vlm_m,
                        img_p,
                        img_m,
                        fmt,
                        iters,
                        auto_ref,
                        max_it,
                        opt_in,
                        save_pr,
                        seed_val,
                        bf,
                    ],
                    outputs=[b_log, b_dir],
                )

            # ── Runs browser ──────────────────────────────────────────────
            with gr.Tab("Runs"):
                gr.Markdown("Inspect previous **run_*** and **batch_*** directories.")
                rb_refresh = gr.Button("Refresh lists")
                with gr.Row():
                    run_pick = gr.Dropdown(
                        label="Runs",
                        choices=[],
                        allow_custom_value=True,
                    )
                    batch_pick = gr.Dropdown(
                        label="Batches",
                        choices=[],
                        allow_custom_value=True,
                    )
                rb_img = gr.Image(label="Final output (selected run)", type="filepath")
                rb_meta = gr.Textbox(label="metadata.json (preview)", lines=14)
                rb_inp = gr.Textbox(label="run_input.json (preview)", lines=10)
                rb_gal = gr.Gallery(label="Iteration thumbnails", columns=4, height=220)
                bb_report = gr.Textbox(label="batch_report.json (preview)", lines=14)

                def _refresh(od: str):
                    root = (od or default_output_dir).strip() or default_output_dir
                    r = runs_mod.list_run_ids(root)
                    b = runs_mod.list_batch_ids(root)
                    return (
                        gr.update(choices=r, value=r[-1] if r else None),
                        gr.update(choices=b, value=b[-1] if b else None),
                    )

                def _show_run(od: str, rid: Optional[str]):
                    if not rid:
                        return None, "", "", []
                    root = (od or default_output_dir).strip() or default_output_dir
                    s = runs_mod.load_run_summary(root, rid)
                    img = s.get("final_image")
                    meta = s.get("metadata_preview") or ""
                    inp = s.get("run_input_preview") or ""
                    gal = [(p, Path(p).name) for p in s.get("iteration_images") or []]
                    return img if img else None, meta, inp, gal

                def _show_batch(od: str, bid: Optional[str]):
                    if not bid:
                        return ""
                    root = (od or default_output_dir).strip() or default_output_dir
                    s = runs_mod.load_batch_summary(root, bid)
                    return s.get("report_preview") or ""

                rb_refresh.click(
                    _refresh,
                    inputs=[out_dir],
                    outputs=[run_pick, batch_pick],
                )
                run_pick.change(
                    _show_run,
                    inputs=[out_dir, run_pick],
                    outputs=[rb_img, rb_meta, rb_inp, rb_gal],
                )
                batch_pick.change(
                    _show_batch,
                    inputs=[out_dir, batch_pick],
                    outputs=[bb_report],
                )

        gr.Markdown(
            "---\n"
            "Tip: run `paperbanana data download` for the expanded reference set. "
            "Studio optional install: `pip install 'paperbanana[studio]'`."
        )

    return demo


def launch_studio(
    *,
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
    config_path: Optional[str] = None,
    default_output_dir: str = "outputs",
    root_path: Optional[str] = None,
) -> None:
    """Build and launch the Gradio server."""
    demo = build_studio_app(
        default_output_dir=default_output_dir,
        config_path=config_path,
    )
    demo.queue()
    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        root_path=root_path or None,
    )
