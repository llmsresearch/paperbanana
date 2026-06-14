# Contributing to PaperBanana

The most impactful contribution right now is improving the reference dataset. Output quality scales directly with reference quality, so even a single well-chosen example helps. 
However, we are open to improving distributions or adding new features. Support is always welcome.

> **Before you contribute:** PaperBanana is source-available under the [Elastic License 2.0](LICENSE), and contributions are accepted under the [Contributor License Agreement](CLA.md) (rights granted to LLMs Research Inc. so the project can be maintained and relicensed cleanly). By opening a pull request you agree to the CLA.

## Contributing Reference Examples

PaperBanana uses a curated set of methodology diagrams for in-context learning. We need more diverse, high-quality samples across different diagram styles and research domains.

### What makes a good reference example

- The diagram clearly illustrates system architecture, pipeline flow, or framework structure
- Landscape layout with aspect ratio roughly between 1.5 and 2.5 (width / height)
- The methodology section in the paper is self-contained enough to describe the approach
- The diagram is not a results plot, ablation table, t-SNE visualization, or data sample

### Option 1: Submit a paper recommendation (easiest)

Open a [Discussion post](https://github.com/llmsresearch/paperbanana/discussions) or an [Issue](https://github.com/llmsresearch/paperbanana/issues) with:

- arXiv link (or other public paper URL)
- Figure number of the methodology diagram
- (Optional) Which category it falls under: Agent & Reasoning, Vision & Perception, Generative & Learning, or Science & Applications

We'll handle extraction and curation from there.

### Option 2: Submit a parsed reference tuple (more involved)

Add a complete reference example via pull request. Each example is a directory under `data/reference_sets/` containing three files:

```
data/reference_sets/your_example_name/
├── methodology.txt    # Extracted methodology section text
├── diagram.png        # Methodology diagram image
└── metadata.json      # Caption and metadata
```

**metadata.json format:**

```json
{
  "paper_title": "Full paper title",
  "arxiv_id": "2601.23265",
  "figure_number": 2,
  "caption": "Original figure caption from the paper",
  "category": "agent_reasoning",
  "source_url": "https://arxiv.org/abs/2601.23265",
  "aspect_ratio": 1.85
}
```

Valid categories: `agent_reasoning`, `vision_perception`, `generative_learning`, `science_applications`

**Before submitting, verify:**

- [ ] The methodology text matches what the diagram actually depicts
- [ ] The diagram image is clean (no scan artifacts, readable at 800px width)
- [ ] The aspect ratio is between 1.5 and 2.5
- [ ] The paper is publicly available

### Categories we're short on

- **Science & Applications**: domain-specific architectures outside core ML
- **Vision & Perception**: detection, segmentation, multimodal pipelines

## Contributing Code

### Setup

```bash
git clone https://github.com/llmsresearch/paperbanana.git
cd paperbanana
pip install -e ".[dev,google]"
```

### Running tests

```bash
pytest tests/ -v
```

### Code style

We use `ruff` for linting and formatting:

```bash
ruff check paperbanana/ mcp_server/ tests/ scripts/
ruff format paperbanana/ mcp_server/ tests/ scripts/
```

### Pull request process

1. Fork the repo and create a branch from `main`
2. Make your changes with clear, descriptive commit messages
3. Add or update tests if applicable
4. Ensure `pytest` and `ruff check` pass
5. Open a PR with a brief description of what changed and why

### What we accept (and what we don't)

To keep the review queue healthy, a few ground rules:

- **CI must be green.** PRs with failing lint or tests won't be reviewed until they pass; `ruff check`, `ruff format --check`, and `pytest` locally before pushing saves everyone a round-trip.
- **No personal or institution-specific content.** Your university's thesis style, your paper's figures, or `.docx`/example files specific to your own project belong in a fork. Generic, configurable mechanisms (e.g. a venue/style system anyone can use) are very welcome — see issue #89.
- **New providers go through the generic routes first.** PaperBanana supports any OpenAI-compatible endpoint via `openai_local`, plus `litellm` for almost everything else. We only add first-class provider modules when there's a strong case (significant user demand or a sponsorship that funds its maintenance — see SPONSORS.md). A docs example showing your provider via `litellm` is the fastest way to get it supported.
- **Don't change project-wide defaults** (models, styles, providers) in a feature PR. Defaults are a maintainer decision with cost/quality implications for every user; propose the change in an issue instead.
- **Stale PRs may be finished by maintainers.** If a PR is approved-with-comments and the author doesn't respond within ~2 weeks, a maintainer may push the remaining fixes to the branch (with credit preserved) and merge it, so good work doesn't strand.

### Areas where code contributions are welcome

- **Provider support**: Adding backends beyond OpenAI and Gemini (Anthropic, local models via Ollama)
- **Reference set tooling**: Improving the automated extraction pipeline in `scripts/`
- **Evaluation**: Expanding the VLM-as-Judge evaluation with additional metrics or human correlation studies
- **MCP server**: Additional tools, better error handling, support for more IDE clients
- **Documentation**: Usage examples, tutorials, edge case documentation

## Reporting Issues

When opening an issue, include:

- What you were trying to do
- The input you used (methodology text and caption)
- The output you got (attach the generated image if possible)
- Python version and OS
- Any error messages or tracebacks

For diagram quality issues specifically, attaching both the generated output and the expected result (or a reference from the paper) helps us diagnose whether the issue is in retrieval, planning, or rendering.

## Questions

Use [GitHub Discussions](https://github.com/llmsresearch/paperbanana/discussions) for questions, ideas, and general conversation. Issues are for bugs and concrete feature requests.
