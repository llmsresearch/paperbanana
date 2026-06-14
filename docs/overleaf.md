# Using PaperBanana with Overleaf

PaperBanana produces standard artifacts — `.png` images, vector/LaTeX figure
snippets, and print-ready `.pdf` posters — so they drop straight into an Overleaf
project. There are two paths.

## 1. Figures, automatically (GitHub ↔ Overleaf sync)

Overleaf projects can be [synced with a GitHub repo](https://www.overleaf.com/learn/how-to/Using_Git_and_GitHub).
PaperBanana ships a **GitHub Action** that, on a push to your `.tex`, extracts the
methodology section, generates a diagram, and commits the image plus an
`\includegraphics` LaTeX snippet back to the repo — which then appears in Overleaf
on the next sync. See [`integrations/github-action`](../integrations/github-action/).

```yaml
# .github/workflows/figure.yml
- uses: llmsresearch/paperbanana/integrations/github-action@main
  with:
    tex-file: main.tex
    caption: "Overview of our method"
    section: Method
```

Include the generated snippet in your paper:

```latex
\input{figures/method_overview.tex}   % \begin{figure}...\includegraphics...\end{figure}
```

## 2. Posters and figures, manually (a few seconds)

Generate locally, then upload to Overleaf (drag-and-drop, or `git`):

```bash
# a figure
paperbanana generate --input method.txt --caption "Overview" --format png

# a full conference poster (PNG + print-ready PDF at the venue size)
paperbanana poster --paper paper.pdf --venue neurips --qr-url https://arxiv.org/abs/XXXX.XXXXX
```

Drop `poster.pdf` into Overleaf to include or print directly, or embed a figure:

```latex
\usepackage{graphicx}
\includegraphics[width=\columnwidth]{figures/method_overview.png}
```

> Tip: for a LaTeX-native poster you can also `\includegraphics` PaperBanana's
> poster PNG full-bleed in a `tikzposter`/`beamerposter` page, but most users
> just print or submit the generated `poster.pdf` (already at the venue's exact
> physical dimensions).

## Roadmap

- A poster mode for the GitHub Action (auto-generate the poster on a tagged
  release) and an "Open in Overleaf" template project.
