# PaperBanana GitHub Action — Overleaf Integration

Generate a publication-quality methodology diagram from your paper's LaTeX source on every push, and have it appear in your Overleaf project via Overleaf's built-in GitHub sync.

**The loop:**

1. Your paper lives in a GitHub repo linked to Overleaf (Menu → GitHub in Overleaf).
2. You edit your methodology section and push (or sync from Overleaf to GitHub).
3. This action extracts the section, runs the PaperBanana pipeline (Retriever → Planner → Stylist → Visualizer ↔ Critic), and commits back the figure plus a ready-to-`\input` LaTeX snippet.
4. In Overleaf, pull from GitHub — the figure is in your file tree.

## Quick start

Add `.github/workflows/paperbanana.yml` to your paper repo:

```yaml
name: PaperBanana figure

on:
  workflow_dispatch:        # run manually from the Actions tab
  push:
    paths:
      - "sections/method.tex"   # only regenerate when the methodology changes

permissions:
  contents: write           # the action commits the figure back

jobs:
  figure:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: llmsresearch/paperbanana/integrations/github-action@main
        with:
          tex-file: sections/method.tex
          caption: "Overview of our proposed framework"
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

Then add your provider API key under **Settings → Secrets and variables → Actions** in your repo, and put this in your document body:

```latex
\input{figures/method_overview}
```

That's it. Edit your method section, push, pull in Overleaf.

## Inputs

| Input | Default | Description |
|-------|---------|-------------|
| `tex-file` | *(required)* | Path to the `.tex` file containing the methodology |
| `caption` | *(required)* | Figure caption / communicative intent |
| `section` | `Method` | Section title to extract (case-insensitive substring match against `\section{...}` titles) |
| `output-path` | `figures/method_overview.png` | Where the image is written (`.png`, `.jpg`) |
| `snippet-path` | output-path with `.tex` | Where the `\begin{figure}` snippet is written |
| `figure-label` | `fig:method-overview` | `\label` used in the snippet |
| `figure-width` | `\columnwidth` | Width argument for `\includegraphics` |
| `vlm-provider` | config default | `openai`, `gemini`, `atlas`, `openrouter`, `anthropic`, `bedrock`, `ollama`, `litellm` |
| `image-provider` | config default | `openai_imagen`, `google_imagen`, `atlas_imagen`, `openrouter_imagen`, `bedrock_imagen` |
| `vlm-model` / `image-model` | provider default | Model overrides |
| `iterations` | config default | Visualizer ↔ Critic refinement iterations |
| `optimize` | `false` | Preprocess inputs for better generation |
| `budget` | none | USD cap per run — recommended for CI |
| `seed` | none | Reproducible generation |
| `paperbanana-version` | latest from `main` | Set a PyPI version to pin (use ≥ 0.1.3 — earlier releases predate the `budget`/`seed` flags) |
| `commit` | `true` | Commit and push the generated files |
| `commit-message` | `chore: update methodology figure via PaperBanana [skip ci]` | Keep `[skip ci]` to avoid retrigger loops |

API keys are passed as `env:` on the action step (`OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ATLASCLOUD_API_KEY`, `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, or AWS credentials for Bedrock) — always from repo secrets, never inline.

## Outputs

| Output | Description |
|--------|-------------|
| `image-path` | Repo-relative path of the generated image |
| `snippet-path` | Repo-relative path of the generated LaTeX snippet |

## Notes

- **Avoiding workflow loops:** the default commit message contains `[skip ci]`, and the quick-start workflow uses a `paths:` filter so the action's own commit (a `.png` + `.tex` under `figures/`) can't retrigger it. Keep one of the two in place if you customize.
- **Image paths in Overleaf:** the snippet references the image by its repo-relative path (e.g. `figures/method_overview.png`), which matches Overleaf's file tree after a pull. If your main `.tex` lives in a subdirectory, set `output-path`/`snippet-path` accordingly or use `\graphicspath`.
- **Cost control:** set `budget` (e.g. `"0.50"`) so a runaway refinement loop aborts gracefully instead of burning credits.
- **`\input`ed sections:** the extractor reads only the file you point it at — if your methodology is split across `\input` files, point `tex-file` at the file that holds the actual section body.
- **Overleaf GitHub sync** requires an Overleaf premium plan. The action itself works with any LaTeX repo, Overleaf or not.
