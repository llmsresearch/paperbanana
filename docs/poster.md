# Poster Generation

`paperbanana poster` turns a paper PDF into a venue-compliant conference
poster: an **editable .pptx**, a **press-ready PDF** at the venue's physical
size, a **preview PNG**, and a **preflight report** covering print fidelity
and venue compliance.

```bash
pip install -e ".[google,poster]"
brew install --cask libreoffice   # required: pptx -> PDF conversion

paperbanana poster --paper paper.pdf --venue neurips --qr-url https://arxiv.org/abs/XXXX.XXXXX
```

## What makes it different

1. **Figures are never just copied.** Every figure/table the storyboard
   places is judged against print physics (its computed DPI at the planned
   placement, legibility at 1.5–2 m viewing distance) and then:
   - **reuse** — placed as cropped from the paper (only if print-quality),
   - **reauthor** — re-rendered via image-conditioned generation (larger
     labels, simplified legends, palette harmonization) behind a strict
     **data-faithfulness gate**: a verifier compares original vs.
     re-authored and the run fails rather than ship a figure whose data
     changed, or
   - **generate** — replaced/added by a newly generated diagram via the
     PaperBanana diagram pipeline (e.g. an overview figure the paper lacks).

   Every figure carries provenance (source page/bbox, decision, reason,
   faithfulness status) in `poster_ir.json`. Override any decision with
   `--figure-decision fig3=reuse` (repeatable).

2. **Venue regulation bank.** Poster rules are versioned YAML specs with
   cited official sources: `data/venue_specs/<venue>/<year>.yaml`
   (built-ins: neurips, icml, cvpr, acl). List them with
   `paperbanana venues specs`. Add your own venue by dropping a YAML in
   `~/.config/paperbanana/venue_specs/<venue>/<year>.yaml` (or set
   `PAPERBANANA_VENUE_SPEC_DIR`). Specs drive both generation (page size,
   orientation) and the compliance checks.

3. **Deterministic geometry + preflight.** The VLM decides content, order,
   and emphasis; pure-Python code assigns physical coordinates (mm), so
   no-overlap/no-overflow/minimum-font invariants are enforced, not hoped
   for. Preflight checks: page size & orientation vs. spec, per-level font
   minima (venue rules merged with built-in legibility floors), each
   image's effective DPI at printed size, required elements, WCAG contrast,
   text overflow, PDF/PNG file rules. The CLI exits non-zero if preflight
   fails.

## Pipeline

```
ingest (text + VLM figure detection w/ caption anchoring to the PDF text layer)
  -> storyboard (panels, reading order, weights)
  -> style tokens (palette, typography >= venue/legibility minima)
  -> figure curation (reuse / reauthor+faithfulness gate / generate)
  -> deterministic layout -> pptx render -> LibreOffice PDF -> preview + per-panel crops
  -> preflight + VLM critic (panel zoom-ins) -> structured edit ops -> iterate
```

## CLI

```bash
paperbanana poster \
  --paper paper.pdf \
  --venue neurips \              # see: paperbanana venues specs
  --year 2025 \                  # default: latest spec
  --qr-url https://example.org \ # QR code on the poster
  --figure-decision fig2=reauthor --figure-decision tab1=generate \
  --iterations 2 \               # critic refinement rounds
  --resume outputs/poster_20260612_101500_ab12cd   # resume a previous run
```

Useful flags: `--budget` (USD cap), `--save-prompts`, `--vlm-model`,
`--image-model`, `--config`, `-v/--verbose`.

## Outputs (`outputs/poster_<ts>_<id>/`)

| File | What it is |
|---|---|
| `poster.pptx` | Editable deck at physical page size (open in PowerPoint/Keynote) |
| `poster.pdf` | Press-ready PDF — send this to the print shop |
| `preview.png` | Raster preview |
| `poster_ir.json` | Full IR: panels, geometry, style, per-figure provenance |
| `preflight_report.md` / `.json` | Print-fidelity + compliance report |
| `paper_assets/` | Extracted text/figures from the paper |
| `curated_figures/` | Re-authored figure outputs |
| `subruns/` | Nested diagram-generation runs (GENERATE decisions) |
| `iter_N/` | Per-iteration pptx/PDF/preview/panel crops/critique |

## Print scale

PowerPoint caps pages at 56 inches. Venues larger than that (e.g. CVPR's
84"x42") are designed at 1/2 scale (`print_scale: 2` in the IR, fonts
halved on the design page) — tell the print shop to **print at 200%**, the
standard large-format practice. The CLI prints a reminder when this
applies; preflight always evaluates the *printed* dimensions.

## Failure behavior (no fallbacks)

The poster head fails loudly instead of degrading: missing LibreOffice,
an unknown venue/year, text that cannot fit at the minimum font sizes
after bounded rewrite attempts, and re-authored figures that cannot pass
the faithfulness gate all stop the run with a specific error and the
override to use (`--figure-decision figN=reuse`).

## MCP

The MCP server exposes `generate_poster(paper_pdf, venue, year, qr_url,
figure_decisions, iterations, output_dir, config)` returning artifact
paths plus the preflight summary as JSON.
