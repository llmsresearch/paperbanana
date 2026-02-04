<p align="center">
  <img src="https://dwzhu-pku.github.io/PaperBanana/static/images/logo.jpg" alt="PaperBanana Logo" width="200"/>
</p>

<h1 align="center">PaperBanana</h1>

<p align="center">
  <strong>Automated Academic Illustration for AI Scientists</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+"/></a>
  <a href="https://arxiv.org/abs/2601.23265"><img src="https://img.shields.io/badge/arXiv-2601.23265-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?logo=opensourceinitiative&logoColor=white" alt="License: MIT"/></a>
  <br/>
  <a href="https://pydantic.dev"><img src="https://img.shields.io/badge/Pydantic-v2-e92063?logo=pydantic&logoColor=white" alt="Pydantic v2"/></a>
  <a href="https://typer.tiangolo.com"><img src="https://img.shields.io/badge/CLI-Typer-009688?logo=gnubash&logoColor=white" alt="Typer"/></a>
  <a href="https://ai.google.dev/"><img src="https://img.shields.io/badge/Gemini-Free%20Tier-4285F4?logo=google&logoColor=white" alt="Gemini Free Tier"/></a>
</p>

---

> **Disclaimer**: This is an **unofficial, community-driven open-source implementation** of the paper
> *"PaperBanana: Automating Academic Illustration for AI Scientists"* by Dawei Zhu, Rui Meng, Yale Song,
> Xiyu Wei, Sujian Li, Tomas Pfister, and Jinsung Yoon ([arXiv:2601.23265](https://arxiv.org/abs/2601.23265)).
> This project is **not affiliated with or endorsed by** the original authors or Google Research.
> The implementation is based on the publicly available paper and may differ from the original system.

An agentic framework for generating publication-quality academic diagrams and statistical plots from text descriptions. Uses Google Gemini for both VLM and image generation.

---

## Quick Start

### Prerequisites

- Python 3.10+
- A Google Gemini API key (available at no cost from [Google AI Studio](https://makersuite.google.com/app/apikey))

### Step 1: Install

```bash
git clone https://github.com/llmsresearch/paperbanana.git
cd paperbanana

pip install -e ".[google]"
```

### Step 2: Get Your API Key

Run the interactive setup wizard:

```bash
paperbanana setup
```

This opens your browser to get a Google Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey) and saves it to `.env`.

Or set it up manually:

```bash
cp .env.example .env
# Edit .env and add: GOOGLE_API_KEY=your-key-here
```

### Step 3: Generate a Diagram

```bash
# Using the included sample input
paperbanana generate \
  --input examples/sample_inputs/transformer_method.txt \
  --caption "Overview of our encoder-decoder architecture with sparse routing"
```

Or write your own methodology text:

```bash
cat > my_method.txt << 'EOF'
Our framework consists of an encoder that processes input sequences
through multi-head self-attention layers, followed by a decoder that
generates output tokens auto-regressively using cross-attention to
the encoder representations. We add a novel routing mechanism that
selects relevant encoder states for each decoder step.
EOF

paperbanana generate \
  --input my_method.txt \
  --caption "Overview of our encoder-decoder framework"
```

Output is saved to `outputs/run_<timestamp>/final_output.png` along with all intermediate iterations and metadata.

---

## How It Works

PaperBanana implements a two-phase multi-agent pipeline with 5 specialized agents:

**Phase 1 -- Linear Planning:**

1. **Retriever** selects the most relevant reference examples from a curated set of 13 methodology diagrams spanning agent/reasoning, vision/perception, generative/learning, and science/applications domains
2. **Planner** generates a detailed textual description of the target diagram via in-context learning from the retrieved examples
3. **Stylist** refines the description for visual aesthetics using NeurIPS-style guidelines (color palette, layout, typography)

**Phase 2 -- Iterative Refinement (3 rounds):**

4. **Visualizer** renders the description into an image (Gemini 3 Pro for diagrams, Matplotlib code for plots)
5. **Critic** evaluates the generated image against the source context and provides a revised description addressing any issues
6. Steps 4-5 repeat for up to 3 iterations

## Providers

| Component | Provider | Model |
|-----------|----------|-------|
| VLM (planning, critique) | Google Gemini | `gemini-2.0-flash` |
| Image Generation | Google Gemini | `gemini-3-pro-image-preview` |

---

## CLI Reference

### `paperbanana generate` -- Methodology Diagrams

```bash
paperbanana generate \
  --input method.txt \
  --caption "Overview of our framework" \
  --output diagram.png \
  --iterations 3
```

| Flag | Short | Description |
|------|-------|-------------|
| `--input` | `-i` | Path to methodology text file (required) |
| `--caption` | `-c` | Figure caption / communicative intent (required) |
| `--output` | `-o` | Output image path (default: auto-generated in `outputs/`) |
| `--iterations` | `-n` | Number of Visualizer-Critic refinement rounds |
| `--vlm-provider` | | VLM provider name (default: `gemini`) |
| `--vlm-model` | | VLM model name (default: `gemini-2.0-flash`) |
| `--image-provider` | | Image gen provider (default: `google_imagen`) |
| `--image-model` | | Image gen model (default: `gemini-3-pro-image-preview`) |
| `--config` | | Path to YAML config file (see `configs/config.yaml`) |

### `paperbanana plot` -- Statistical Plots

```bash
paperbanana plot \
  --data results.csv \
  --intent "Bar chart comparing model accuracy across benchmarks"
```

| Flag | Short | Description |
|------|-------|-------------|
| `--data` | `-d` | Path to data file, CSV or JSON (required) |
| `--intent` | | Communicative intent for the plot (required) |
| `--output` | `-o` | Output image path |
| `--iterations` | `-n` | Refinement iterations (default: 3) |

### `paperbanana evaluate` -- Quality Assessment

Comparative evaluation of a generated diagram against a human reference using VLM-as-a-Judge:

```bash
paperbanana evaluate \
  --generated diagram.png \
  --reference human_diagram.png \
  --context method.txt \
  --caption "Overview of our framework"
```

| Flag | Short | Description |
|------|-------|-------------|
| `--generated` | `-g` | Path to generated image (required) |
| `--reference` | `-r` | Path to human reference image (required) |
| `--context` | | Path to source context text file (required) |
| `--caption` | `-c` | Figure caption (required) |

Scores on 4 dimensions (hierarchical aggregation per the paper):
- **Primary**: Faithfulness, Readability
- **Secondary**: Conciseness, Aesthetics

### `paperbanana setup` -- First-Time Configuration

```bash
paperbanana setup
```

Interactive wizard that walks you through obtaining a Google Gemini API key and saving it to `.env`.

---

## Python API

```python
import asyncio
from paperbanana import PaperBananaPipeline, GenerationInput, DiagramType
from paperbanana.core.config import Settings

settings = Settings(
    vlm_provider="gemini",
    image_provider="google_imagen",
    refinement_iterations=3,
)

pipeline = PaperBananaPipeline(settings=settings)

result = asyncio.run(pipeline.generate(
    GenerationInput(
        source_context="Our framework consists of...",
        communicative_intent="Overview of the proposed method.",
        diagram_type=DiagramType.METHODOLOGY,
    )
))

print(f"Output: {result.image_path}")
```

See `examples/generate_diagram.py` and `examples/generate_plot.py` for complete working examples.

---

## Configuration

Default settings are in `configs/config.yaml`. Override via CLI flags or a custom YAML:

```bash
paperbanana generate \
  --input method.txt \
  --caption "Overview" \
  --config my_config.yaml
```

Key settings:

```yaml
vlm:
  provider: gemini
  model: gemini-2.0-flash

image:
  provider: google_imagen
  model: gemini-3-pro-image-preview

pipeline:
  num_retrieval_examples: 10
  refinement_iterations: 3
  output_resolution: "2k"

reference:
  path: data/reference_sets

output:
  dir: outputs
  save_iterations: true
  save_metadata: true
```

---

## Project Structure

```
paperbanana/
├── paperbanana/
│   ├── core/          # Pipeline orchestration, types, config, utilities
│   ├── agents/        # Retriever, Planner, Stylist, Visualizer, Critic
│   ├── providers/     # VLM and image gen provider implementations
│   │   ├── vlm/       # Gemini VLM provider
│   │   └── image_gen/ # Gemini 3 Pro Image provider
│   ├── reference/     # Reference set management (13 curated examples)
│   ├── guidelines/    # Style guidelines loader
│   └── evaluation/    # VLM-as-Judge evaluation system
├── configs/           # YAML configuration files
├── prompts/           # Prompt templates for all 5 agents + evaluation
│   ├── diagram/       # retriever, planner, stylist, visualizer, critic
│   ├── plot/          # plot-specific prompt variants
│   └── evaluation/    # faithfulness, conciseness, readability, aesthetics
├── data/
│   ├── reference_sets/  # 13 verified methodology diagrams
│   └── guidelines/              # NeurIPS-style aesthetic guidelines
├── examples/          # Working example scripts + sample inputs
├── scripts/           # Data curation and build scripts
├── tests/             # Test suite (34 tests)
└── mcp_server/        # MCP server for IDE integration
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev,google]"

# Run tests
pytest tests/ -v

# Lint
ruff check paperbanana/ tests/ scripts/

# Format
ruff format paperbanana/ tests/ scripts/
```

## Citation

This is an **unofficial** implementation. If you use this work, please cite the **original paper**:

```bibtex
@article{zhu2026paperbanana,
  title={PaperBanana: Automating Academic Illustration for AI Scientists},
  author={Zhu, Dawei and Meng, Rui and Song, Yale and Wei, Xiyu
          and Li, Sujian and Pfister, Tomas and Yoon, Jinsung},
  journal={arXiv preprint arXiv:2601.23265},
  year={2026}
}
```

**Original paper**: [https://arxiv.org/abs/2601.23265](https://arxiv.org/abs/2601.23265)

## Disclaimer

This project is an independent open-source reimplementation based on the publicly available paper.
It is not affiliated with, endorsed by, or connected to the original authors, Google Research, or
Peking University in any way. The implementation may differ from the original system described in the paper.
Use at your own discretion.

## License

MIT
