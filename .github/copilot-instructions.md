# Copilot Instructions for PaperBanana

## Build & Test

```bash
# Install for development
pip install -e ".[dev,openai,google]"

# Run full test suite
pytest tests/ -v

# Run a single test file
pytest tests/test_pipeline/test_types.py -v

# Run a single test by name
pytest tests/ -k "test_critique_result_needs_revision" -v

# Lint
ruff check paperbanana/ mcp_server/ tests/ scripts/

# Format
ruff format paperbanana/ mcp_server/ tests/ scripts/
```

CI runs lint, tests (Python 3.10–3.12 on Linux/macOS/Windows), and package build. Tests must pass without a `GOOGLE_API_KEY`—all tests mock external providers.

## Architecture

PaperBanana is an agentic framework that generates publication-quality academic diagrams from text. It implements a **two-phase multi-agent pipeline**:

**Phase 1 — Linear Planning:** Retriever → Planner → Stylist  
**Phase 2 — Iterative Refinement:** Visualizer ↔ Critic (up to N rounds)

Key architectural layers:

- **`paperbanana/core/`** — Pipeline orchestrator (`pipeline.py`), Pydantic data types (`types.py`), config via `pydantic-settings` (`config.py`). `Settings` loads from env vars, `.env` file, or YAML config.
- **`paperbanana/agents/`** — Seven agents (Optimizer, Retriever, Planner, Stylist, Visualizer, Critic, plus InputOptimizer with parallel sub-tasks), all inheriting from `BaseAgent`. Each agent wraps a VLM provider and a prompt template loaded from `prompts/`.
- **`paperbanana/providers/`** — Abstract `VLMProvider` and `ImageGenProvider` base classes in `base.py`. Concrete implementations in `vlm/` (OpenAI, Gemini, OpenRouter) and `image_gen/` (OpenAI, Google Imagen, OpenRouter). `ProviderRegistry` is the factory that creates providers from `Settings`.
- **`prompts/`** — Text prompt templates organized by type (`diagram/`, `plot/`, `evaluation/`). Templates use `{placeholder}` formatting, loaded by `BaseAgent.load_prompt()`.
- **`paperbanana/evaluation/`** — VLM-as-Judge system. Scores on 4 dimensions (Faithfulness, Readability, Conciseness, Aesthetics) with hierarchical aggregation.
- **`mcp_server/`** — FastMCP server exposing three tools: `generate_diagram`, `generate_plot`, `evaluate_diagram`.
- **`data/reference_sets/`** — 13 curated methodology diagram examples used for in-context learning by the Retriever agent.

## Conventions

- **Async everywhere**: The pipeline and all agents use `async/await`. Tests use `pytest-asyncio` with `asyncio_mode = "auto"`.
- **Pydantic models for all data types**: Inputs, outputs, configs, and intermediate results are Pydantic `BaseModel` subclasses. Use `model_dump()` for serialization.
- **Provider pattern**: To add a new provider, implement `VLMProvider` or `ImageGenProvider` from `providers/base.py`, then register it in `ProviderRegistry`.
- **Prompt templates live in `prompts/`, not in code**: Agent prompts are `.txt` files with `{placeholder}` substitution. Don't inline prompts in Python.
- **Config hierarchy**: `Settings` merges env vars → `.env` file → YAML config → CLI overrides. API keys come from environment only (`OPENAI_API_KEY`, `GOOGLE_API_KEY`, `OPENROUTER_API_KEY`).
- **Ruff for all linting/formatting**: Line length 100, target Python 3.10. Select rules: E, F, I, N, W.
- **structlog for logging**: Use `structlog.get_logger()` with keyword arguments, not f-strings in log calls.
- **Entry points**: CLI via Typer (`paperbanana.cli:app`), MCP server via FastMCP (`mcp_server.server:main`).
