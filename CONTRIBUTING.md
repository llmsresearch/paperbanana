# Contributing to PaperBanana

The most impactful contribution right now is improving the reference dataset. Output quality scales directly with reference quality, so even a single well-chosen example helps. 
However, we are open to improving distributions or adding new features. Support is always welcome.

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
