<!-- mcp-name: io.github.llmsresearch/paperbanana -->
<table align="center" width="100%" style="border: none; border-collapse: collapse;">
  <tr>
    <td width="220" align="left" valign="middle" style="border: none;">
      <img src="https://dwzhu-pku.github.io/PaperBanana/static/images/logo.jpg" alt="PaperBanana Logo" width="180"/>
    </td>
    <td align="left" valign="middle" style="border: none;">
      <h1>PaperBanana</h1>
      <p><strong>Automated Academic Illustration for AI Scientists</strong></p>
      <p>
        <a href="https://github.com/llmsresearch/paperbanana/actions/workflows/ci.yml"><img src="https://github.com/llmsresearch/paperbanana/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
        <a href="https://pypi.org/project/paperbanana/"><img src="https://img.shields.io/pypi/dm/paperbanana?label=PyPI%20downloads&logo=pypi&logoColor=white" alt="PyPI Downloads"/></a>
        <a href="https://huggingface.co/spaces/llmsresearch/paperbanana"><img src="https://img.shields.io/badge/Demo-HuggingFace-yellow?logo=huggingface&logoColor=white" alt="Demo"/></a>
        <br/>
        <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+"/></a>
        <a href="https://arxiv.org/abs/2601.23265"><img src="https://img.shields.io/badge/arXiv-2601.23265-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"/></a>
        <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?logo=opensourceinitiative&logoColor=white" alt="License: MIT"/></a>
        <br/>
        <a href="https://pydantic.dev"><img src="https://img.shields.io/badge/Pydantic-v2-e92063?logo=pydantic&logoColor=white" alt="Pydantic v2"/></a>
        <a href="https://typer.tiangolo.com"><img src="https://img.shields.io/badge/CLI-Typer-009688?logo=gnubash&logoColor=white" alt="Typer"/></a>
        <a href="https://ai.google.dev/"><img src="https://img.shields.io/badge/Gemini-Free%20Tier-4285F4?logo=google&logoColor=white" alt="Gemini Free Tier"/></a>
      </p>
    </td>
  </tr>
</table>

---

> **Disclaimer**: This is an **unofficial, community-driven open-source implementation** of the paper
> *"PaperBanana: Automating Academic Illustration for AI Scientists"* by Dawei Zhu, Rui Meng, Yale Song,
> Xiyu Wei, Sujian Li, Tomas Pfister, and Jinsung Yoon ([arXiv:2601.23265](https://arxiv.org/abs/2601.23265)).
> This project is **not affiliated with or endorsed by** the original authors or Google Research.
> The implementation is based on the publicly available paper and may differ from the original system.

An agentic framework for generating publication-quality academic diagrams and statistical plots from text descriptions. Supports OpenAI (GPT-5.2 + GPT-Image-1.5), Azure OpenAI / Foundry, and Google Gemini providers.

- Two-phase multi-agent pipeline with iterative refinement
- Multiple VLM and image generation providers (OpenAI, Azure, Gemini)
- Input optimization layer for better generation quality
- Auto-refine mode and run continuation with user feedback
- CLI, Python API, and MCP server for IDE integration
- Claude Code skills for `/generate-diagram`, `/generate-plot`, and `/evaluate-diagram`

<p align="center">
  <img src="assets/img/hero_image.png" alt="PaperBanana takes paper as input and provide diagram as output" style="max-width: 960px; width: 100%; height: auto;"/>
</p>

---

## Quick Start

### Prerequisites

- Python 3.10+
- An OpenAI API key ([platform.openai.com](https://platform.openai.com/api-keys)) or Azure OpenAI / Foundry endpoint
- Or a Google Gemini API key (free, [Google AI Studio](https://makersuite.google.com/app/apikey))

### Step 1: Install

```bash
pip install paperbanana
```

Or install from source for development:

```bash
git clone https://github.com/llmsresearch/paperbanana.git
cd paperbanana
pip install -e ".[dev,openai,google]"
```

### Step 2: Get Your API Key

```bash
cp .env.example .env
# Edit .env and add your API key:
#   OPENAI_API_KEY=your-key-here
#
# For Azure OpenAI / Foundry:
#   OPENAI_BASE_URL=https://<resource>.openai.azure.com/openai/v1
```

Or use the setup wizard for Gemini:

```bash
paperbanana setup
```

### Step 3: Generate a Diagram

```bash
paperbanana generate \
  --input examples/sample_inputs/transformer_method.txt \
  --caption "Overview of our encoder-decoder architecture with sparse routing"
```

With input optimization and auto-refine:

```bash
paperbanana generate \
  --input my_method.txt \
  --caption "Overview of our encoder-decoder framework" \
  --optimize --auto
```

Output is saved to `outputs/run_<timestamp>/final_output.png` along with all intermediate iterations and metadata.

---

## How It Works

PaperBanana implements a multi-agent pipeline with up to 7 specialized agents:

**Phase 0 -- Input Optimization (optional, `--optimize`):**

0. **Input Optimizer** runs two parallel VLM calls:
   - **Context Enricher** structures raw methodology text into diagram-ready format (components, flows, groupings, I/O)
   - **Caption Sharpener** transforms vague captions into precise visual specifications

**Phase 1 -- Linear Planning:**

1. **Retriever** selects the most relevant reference examples from a curated set of 13 methodology diagrams spanning agent/reasoning, vision/perception, generative/learning, and science/applications domains
2. **Planner** generates a detailed textual description of the target diagram via in-context learning from the retrieved examples
3. **Stylist** refines the description for visual aesthetics using NeurIPS-style guidelines (color palette, layout, typography)

**Phase 2 -- Iterative Refinement:**

4. **Visualizer** renders the description into an image
5. **Critic** evaluates the generated image against the source context and provides a revised description addressing any issues
6. Steps 4-5 repeat for a fixed number of iterations (default 3), or until the critic is satisfied (`--auto`)

## Providers

PaperBanana supports multiple VLM and image generation providers:

| Component | Provider | Model | Notes |
|-----------|----------|-------|-------|
| VLM (planning, critique) | OpenAI | `gpt-5.2` | Default |
| Image Generation | OpenAI | `gpt-image-1.5` | Default |
| VLM | Google Gemini | `gemini-2.5-flash` | Free tier |
| Image Generation | Google Gemini | `gemini-3-pro-image-preview` | Free tier |
| VLM / Image | OpenRouter | Any supported model | Flexible routing |

Azure OpenAI / Foundry endpoints are auto-detected — set `OPENAI_BASE_URL` to your endpoint.

---

## CLI Reference

### `paperbanana generate` -- Methodology Diagrams

```bash
# Basic generation
paperbanana generate \
  --input method.txt \
  --caption "Overview of our framework"

# With input optimization and auto-refine
paperbanana generate \
  --input method.txt \
  --caption "Overview of our framework" \
  --optimize --auto

# Continue the latest run with user feedback
paperbanana generate --continue \
  --feedback "Make arrows thicker and colors more distinct"

# Continue a specific run
paperbanana generate --continue-run run_20260218_125448_e7b876 \
  --iterations 3
```

| Flag | Short | Description |
|------|-------|-------------|
| `--input` | `-i` | Path to methodology text file (required for new runs) |
| `--caption` | `-c` | Figure caption / communicative intent (required for new runs) |
| `--output` | `-o` | Output image path (default: auto-generated in `outputs/`) |
| `--iterations` | `-n` | Number of Visualizer-Critic refinement rounds (default: 3) |
| `--auto` | | Loop until critic is satisfied (with `--max-iterations` safety cap) |
| `--max-iterations` | | Safety cap for `--auto` mode (default: 30) |
| `--optimize` | | Preprocess inputs with parallel context enrichment and caption sharpening |
| `--continue` | | Continue from the latest run in `outputs/` |
| `--continue-run` | | Continue from a specific run ID |
| `--feedback` | | User feedback for the critic when continuing a run |
| `--vlm-provider` | | VLM provider name (default: `openai`) |
| `--vlm-model` | | VLM model name (default: `gpt-5.2`) |
| `--image-provider` | | Image gen provider (default: `openai_imagen`) |
| `--image-model` | | Image gen model (default: `gpt-image-1.5`) |
| `--format` | `-f` | Output format: `png`, `jpeg`, or `webp` (default: `png`) |
| `--config` | | Path to YAML config file (see `configs/config.yaml`) |
| `--verbose` | `-v` | Show detailed agent progress and timing |

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
    vlm_provider="openai",
    vlm_model="gpt-5.2",
    image_provider="openai_imagen",
    image_model="gpt-image-1.5",
    optimize_inputs=True,   # Enable input optimization
    auto_refine=True,       # Loop until critic is satisfied
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

To continue a previous run:

```python
from paperbanana.core.resume import load_resume_state

state = load_resume_state("outputs", "run_20260218_125448_e7b876")
result = asyncio.run(pipeline.continue_run(
    resume_state=state,
    additional_iterations=3,
    user_feedback="Make the encoder block more prominent",
))
```

See `examples/generate_diagram.py` and `examples/generate_plot.py` for complete working examples.

---

## MCP Server

PaperBanana includes an MCP server for use with Claude Code, Cursor, or any MCP-compatible client. Add the following config to use it via `uvx` without a local clone:

```json
{
  "mcpServers": {
    "paperbanana": {
      "command": "uvx",
      "args": ["--from", "paperbanana[mcp]", "paperbanana-mcp"],
      "env": { "GOOGLE_API_KEY": "your-google-api-key" }
    }
  }
}
```

Three MCP tools are exposed: `generate_diagram`, `generate_plot`, and `evaluate_diagram`.

The repo also ships with 3 Claude Code skills:
- `/generate-diagram <file> [caption]` - generate a methodology diagram from a text file
- `/generate-plot <data-file> [intent]` - generate a statistical plot from CSV/JSON data
- `/evaluate-diagram <generated> <reference>` - evaluate a diagram against a human reference

See [`mcp_server/README.md`](mcp_server/README.md) for full setup details (Claude Code, Cursor, local development).

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
  provider: openai           # openai, gemini, or openrouter
  model: gpt-5.2

image:
  provider: openai_imagen    # openai_imagen, google_imagen, or openrouter_imagen
  model: gpt-image-1.5

pipeline:
  num_retrieval_examples: 10
  refinement_iterations: 3
  # auto_refine: true        # Loop until critic is satisfied
  # max_iterations: 30       # Safety cap for auto_refine mode
  # optimize_inputs: true    # Preprocess inputs for better generation
  output_resolution: "2k"

reference:
  path: data/reference_sets

output:
  dir: outputs
  save_iterations: true
  save_metadata: true
```

Environment variables (`.env`):

```bash
# OpenAI (default)
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=https://api.openai.com/v1    # or Azure endpoint
OPENAI_VLM_MODEL=gpt-5.2                      # override model
OPENAI_IMAGE_MODEL=gpt-image-1.5              # override model

# Google Gemini (alternative, free)
GOOGLE_API_KEY=your-key
```

---

## Project Structure

```
paperbanana/
├── paperbanana/
│   ├── core/          # Pipeline orchestration, types, config, resume, utilities
│   ├── agents/        # Optimizer, Retriever, Planner, Stylist, Visualizer, Critic
│   ├── providers/     # VLM and image gen provider implementations
│   │   ├── vlm/       # OpenAI, Gemini, OpenRouter VLM providers
│   │   └── image_gen/ # OpenAI, Gemini, OpenRouter image gen providers
│   ├── reference/     # Reference set management (13 curated examples)
│   ├── guidelines/    # Style guidelines loader
│   └── evaluation/    # VLM-as-Judge evaluation system
├── configs/           # YAML configuration files
├── prompts/           # Prompt templates for all agents + evaluation
│   ├── diagram/       # context_enricher, caption_sharpener, retriever, planner, stylist, visualizer, critic
│   ├── plot/          # plot-specific prompt variants
│   └── evaluation/    # faithfulness, conciseness, readability, aesthetics
├── data/
│   ├── reference_sets/  # 13 verified methodology diagrams
│   └── guidelines/      # NeurIPS-style aesthetic guidelines
├── examples/          # Working example scripts + sample inputs
├── scripts/           # Data curation and build scripts
├── tests/             # Test suite
├── mcp_server/        # MCP server for IDE integration
└── .claude/skills/    # Claude Code skills (generate-diagram, generate-plot, evaluate-diagram)
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev,openai,google]"

# Run tests
pytest tests/ -v

# Lint
ruff check paperbanana/ mcp_server/ tests/ scripts/

# Format
ruff format paperbanana/ mcp_server/ tests/ scripts/
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
