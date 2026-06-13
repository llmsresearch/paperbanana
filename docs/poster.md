# Poster Generation

`paperbanana poster` turns a paper PDF into a venue-compliant conference
poster: a high-resolution **PNG** and a **print-ready PDF** at the venue's
exact physical size, plus the grounded facts and a faithfulness audit.

```bash
pip install -e ".[google,poster]"
paperbanana poster --paper paper.pdf --venue neurips --qr-url https://arxiv.org/abs/XXXX.XXXXX
```

No LibreOffice required — the PDF is written directly at the venue's
physical dimensions.

## Architecture: generate by design, verify against the paper

The poster is **designed by an image model at full power** — the same kind
of model PaperBanana already uses for figures — and then wrapped with the
verification a one-shot model call cannot do for itself:

1. **Ground.** A VLM extracts only the paper's *verified* facts (title,
   authors, takeaway, method, the exact result numbers). Only these reach
   the design prompt, with an explicit instruction to invent no model,
   dataset, baseline, or number that is not listed.
2. **Generate.** The image model produces the whole poster at the venue's
   aspect and the largest legal resolution — varied color, real typography,
   integrated figures, big numbers — the design quality a hand-coded layout
   engine cannot reach.
3. **Audit.** A VLM re-reads the *rendered poster* against the paper and
   lists every claim that contradicts or is unsupported by it (wrong
   numbers, invented names, fabricated metadata).
4. **Repair.** Findings are fed back and the poster is regenerated, bounded
   by `--repair-rounds`.
5. **Comply.** Deterministic checks on the rendered artifact: physical size
   and orientation vs. the venue spec, and print DPI. This is the only
   place determinism survives — physics and verification, never design.

This is the project's thesis applied to posters: **amplify the model, add
the grounding/verification it lacks.** The generator is a swappable
commodity; the moat is trust, and it sharpens as models improve.

## Figures (`--figures`)

`--figures generated` (default today): the model draws the figures; the
audit guards their numbers. `--figures real` / `--figures auto` (next
milestone) embed the paper's *real* extracted figures, deciding per figure
whether the paper's asset or a PaperBanana-generated faithful one is better
(`--figure figN=real|generate|reauthor` to override). A poster figure is
then always either the paper's real figure or a PaperBanana-faithful one —
never the image model's fabrication.

## Venue regulation bank

Poster rules are versioned YAML specs with cited official sources:
`data/venue_specs/<venue>/<year>.yaml` (built-ins: neurips, icml, cvpr,
acl, iclr, aaai). List them with `paperbanana venues specs`; add your own
under `~/.config/paperbanana/venue_specs/<venue>/<year>.yaml` (or set
`PAPERBANANA_VENUE_SPEC_DIR`). Specs drive both generation (physical size,
orientation) and the compliance checks.

## CLI

```bash
paperbanana poster \
  --paper paper.pdf \
  --venue neurips \              # see: paperbanana venues specs
  --year 2025 \                  # default: latest spec
  --qr-url https://example.org \ # scannable QR composited on the poster
  --figures generated \          # generated | real | auto
  --repair-rounds 1              # faithfulness repair regenerations
```

Useful flags: `--budget` (USD cap), `--vlm-provider/--vlm-model`,
`--image-provider/--image-model`, `--config`, `-v/--verbose`.

## Outputs (`outputs/poster_<ts>_<id>/`)

| File | What it is |
|---|---|
| `poster.png` | High-resolution poster image |
| `poster.pdf` | Print-ready PDF at the venue's physical size — send to the print shop |
| `grounding.txt` | The verified paper facts the design was built from |
| `audit.json` | Faithfulness findings + repair rounds |
| `poster_output.json` | Run metadata: venue, physical size, compliance |
| `poster_v1.png`, `poster_v2.png` | Pre- and post-repair drafts |

## Failure behavior (no fallbacks)

The poster head fails loudly instead of degrading: an unknown venue/year
stops the run with a specific error, and `--figures real|auto` raises a
clear "not yet available" until that milestone lands rather than silently
falling back to generated.

## MCP

The MCP server exposes
`generate_poster(paper_pdf, venue, year, qr_url, figures, figure_overrides,
repair_rounds, output_dir, config)` returning the PNG/PDF paths, venue and
physical size, the audit findings, and the compliance summary as JSON —
peer to the figure-side `generate_diagram` / `generate_plot` tools.

## Evaluation & benchmarking

```bash
paperbanana evaluate-poster --run-dir outputs/poster_<id> [--reference author_poster.png] [--dual-judge]
```

Two components, deliberately separated:

- **VLM judge** — Content / Design / Coherence on a 1-5 scale, using the
  PPTEval rubric so scores are directly comparable with Paper2Poster
  (PosterAgent-4o: 3.72 overall vs 3.77 for human posters) and successor
  baselines. `--reference` calibrates against the author's poster;
  `--dual-judge` averages a second judge model.
- **Deterministic compliance** — image-level venue checks (dimensions,
  orientation, DPI) recomputed from the rendered poster; never judged by a
  model.

Benchmark roadmap: run the harness over the Paper2Poster benchmark
(100 paper-poster pairs from NeurIPS/ICML/ICLR, on HuggingFace as
`Paper2Poster/Paper2Poster`) judging generated vs author posters, plus the
PaperQuiz comprehension test (a fresh VLM answers paper questions seeing
only the poster). The metrics PaperBanana adds on top: faithfulness
(audit findings vs the paper) and venue compliance — dimensions no
published system measures.

## Roadmap

- **Real-figure embedding** (`--figures real|auto`): extract the paper's
  figures, decide per figure (paper asset vs PaperBanana-generated), and
  composite into reserved slots in the generated design.
- **Print resolution**: tile/upscale beyond the single-image ~4K budget so
  large boards exceed the figure-DPI minimum (currently a compliance
  *warning* on big boards).
- **Design corpus**: condition generation on retrieved real venue posters
  (style exemplars), the same retrieval idea the figure pipeline uses.
