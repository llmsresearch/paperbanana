# PaperBanana Poster — GitHub Action

Generate a venue-compliant conference **poster** (high-resolution PNG + print-ready
PDF, with the paper's **real figures embedded**) from your paper PDF, and commit it
back to the repo. Pairs with [Overleaf's GitHub sync](https://www.overleaf.com/learn/how-to/Using_Git_and_GitHub):
push the paper, the action generates the poster, pull in Overleaf.

This is the poster counterpart of the [figure action](../github-action/).

## Usage

```yaml
name: Poster
on:
  workflow_dispatch:        # generate on demand
  push:
    paths: ["paper.pdf"]    # or whenever the paper changes
permissions:
  contents: write           # to commit the poster back
jobs:
  poster:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: llmsresearch/paperbanana/integrations/github-action-poster@main
        with:
          paper-file: paper.pdf
          venue: neurips
          qr-url: https://arxiv.org/abs/XXXX.XXXXX
          figures: auto
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          # or GOOGLE_API_KEY / ATLASCLOUD_API_KEY, matching your providers
```

Commits `poster/poster.png`, `poster/poster.pdf`, and a `poster/poster.tex`
`\includegraphics` snippet. Print or submit the PDF (it's at the venue's exact
physical size), or `\input{poster/poster}` to embed the image.

## Inputs

| Input | Required | Default | Description |
|-------|----------|---------|-------------|
| `paper-file` | ✓ | — | Path to the paper PDF |
| `venue` | ✓ | — | Venue spec (`neurips`, `cvpr`, `icml`, `iclr`, `acl`, `aaai`) |
| `year` | | latest | Venue spec year |
| `qr-url` | | — | URL composited as a scannable QR code |
| `figures` | | `auto` | `auto` · `real` · `generated` |
| `repair-rounds` | | 1 | Faithfulness repair regenerations |
| `output-path` | | `poster/poster.png` | Repo path for the poster PNG |
| `pdf-path` | | `<output>.pdf` | Repo path for the print-ready PDF |
| `snippet-path` | | `<output>.tex` | Repo path for the LaTeX snippet |
| `vlm-provider` / `vlm-model` | | provider default | VLM overrides |
| `image-provider` / `image-model` | | provider default | Image-gen overrides |
| `budget` | | — | USD cap |
| `paperbanana-version` | | latest (main) | Pin a PyPI version (≥ 0.4.0) |
| `commit` / `commit-message` | | `true` | Commit + push the generated files |

## Outputs

`image-path`, `pdf-path`, `snippet-path` — repo-relative paths of the generated files.

> Provide the API key for whichever providers you use as repository secrets. See
> [docs/overleaf.md](../../docs/overleaf.md) for the full Overleaf workflow.
