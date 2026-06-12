# PaperBanana

In this repository:

- Keep changes small and focused. Do not bundle unrelated fixes, providers, docs, refactors, and features in one change.
- Do not change project-wide defaults such as default providers, default models, default venue/style, output format, or refinement behavior unless the task explicitly asks for it.
- Prefer generic, configurable mechanisms over one-off content. Do not add institution-, company-, thesis-, paper-, or user-specific assets/styles as core defaults.
- Preserve backward compatibility for CLI flags, config fields, manifest formats, output directory layout, metadata keys, MCP tool behavior, Studio flows, and resume/continue behavior.
- Put reusable implementation in shared modules, not only in `paperbanana/cli.py` or `mcp_server/server.py`.
  - CLI, MCP, Studio, and Python API should be thin wrappers around shared pipeline/workflow code.
  - Batch/orchestration-like behavior should generally go through `paperbanana/core/workflow_runner.py` or another shared core module.
- Reuse existing abstractions before adding new ones:
  - `paperbanana.core.config.Settings` for configuration.
  - Pydantic models in `paperbanana.core.types` or a focused new `types.py` for cross-module data.
  - `paperbanana.agents.base.BaseAgent` for prompt-template-driven agents.
  - `paperbanana.providers.base` interfaces and `ProviderRegistry` for model providers.
  - `PromptRecorder`, `CostTracker`, progress callbacks, and existing output metadata patterns for pipeline stages.
- Avoid growing central orchestration files unnecessarily. For substantial new concepts, add focused modules rather than making `pipeline.py`, `cli.py`, or `server.py` larger without need.
- For complex model-generated structures, prefer a typed intermediate representation over unstructured strings passed through many layers.
- Use deterministic checks for deterministic constraints; use VLM critique for subjective quality only.

## Code style

- Use Python 3.10+ idioms compatible with this package.
- Keep `from __future__ import annotations` immediately after the module docstring when present.
- Use `pathlib.Path` for filesystem paths where practical.
- Always specify UTF-8 for text I/O.
- Keep imports Ruff-clean and sorted.
- Avoid module-level side effects such as global console width changes or environment mutation.
- Do not add helper functions/classes that are only used once unless they clarify a complex boundary or make testing meaningfully easier.
- Replace important magic literals with named constants.
- Keep provider-specific quirks inside provider adapters, not in the core pipeline.
- Handle `None`, empty strings, malformed JSON, and empty model responses defensively.
- Do not assume any generation iteration completed; budget limits or provider failures may produce zero images.
- Do not assume external optional tools are installed. Check availability and degrade gracefully where existing code does so.

## Prompts and agents

- Prompt templates live under `prompts/<task>/<agent>.txt`; keep this convention.
- If code loads a prompt template, include the corresponding prompt file in the same change.
- When adding an agent, subclass `BaseAgent`, use existing prompt loading/recording, and add tests for parsing/fallback behavior.
- Do not silently change existing prompt semantics without updating tests that cover the affected behavior.
- If a pipeline stage formats prompts, ensure prompt recording still works when `save_prompts=True`.
- Keep diagram and plot prompt paths separate unless intentionally creating shared behavior.

## Provider changes

- Prefer existing generic routes (`openai_local`, OpenAI-compatible endpoints, or LiteLLM) unless first-class provider support is specifically needed.
- A first-class provider must:
  - implement the appropriate provider base interface;
  - be registered in `ProviderRegistry`;
  - validate required credentials with helpful errors;
  - support timeouts/retries consistently with similar providers;
  - integrate cost tracking when pricing is known;
  - include tests for registry creation, missing credentials, and mocked success/failure paths;
  - avoid leaking API keys in logs or metadata.
- Do not require an image provider for workflows that do not need image generation, such as code-rendered plot paths.

## Security and metadata

- Never write API keys, tokens, or secrets to `metadata.json`, logs, reports, prompt recordings, or test snapshots.
- When adding a new secret/config key, update metadata redaction tests.
- Validate user-provided file paths and image inputs before passing them into model/provider code.
- Treat remote image fetching carefully; preserve SSRF-style safety checks and global-address validation.
- Do not commit local generated artifacts, personal examples, private decks, `.env`, cache directories, or temporary files.

## Outputs and artifacts

- Keep run outputs inspectable and predictable.
- Single-generation runs should continue to use `outputs/run_*`-style directories unless explicitly changed.
- Batch and orchestration flows should preserve checkpoint/report behavior and resume semantics.
- If adding optional stages, record status/fallback/error details in metadata rather than failing silently.
- Preserve final output naming conventions such as `final_output.<format>` unless the task is explicitly about changing them.
- If generated code is executed, save the generated source beside the artifact when existing patterns do so.

## Tests

- New behavior needs tests. A feature without tests is usually incomplete.
- Prefer testing through the public/shared layer that owns the behavior rather than only testing a CLI wrapper.
- Add or update tests for:
  - new config fields and validation;
  - new CLI commands/options and invalid input handling;
  - new MCP tools or changed tool signatures;
  - Studio runner changes;
  - provider registry branches and credential errors;
  - pipeline branches, fallbacks, retries, rollback, and metadata;
  - batch checkpoint/resume/retry behavior when touched;
  - prompt parsing and missing/invalid model responses;
  - Windows/path escaping if generated Python code or path strings are involved.
- Prefer assertions on complete structured objects when practical instead of many unrelated field-by-field assertions.
- Do not add tests for static constants alone.
- Do not add negative tests for behavior that was removed.
- Avoid mutating global process environment in tests; use monkeypatch/scoped fixtures.

Run focused tests for the area you changed. For broad changes, run:

```bash
ruff check paperbanana/ mcp_server/ tests/ scripts/
ruff format --check paperbanana/ mcp_server/ tests/ scripts/
pytest tests/ -v
```

If formatting is needed, run:

```bash
ruff format paperbanana/ mcp_server/ tests/ scripts/
```

Do not claim tests passed unless you actually ran them.

## CLI, MCP, and Studio

- Keep CLI/MCP/Studio features consistent when they expose the same workflow.
- New user-facing CLI behavior should have validation and a smoke test.
- New MCP tools should have tool-surface tests and clear error messages.
- Avoid duplicating batch/generation logic separately across CLI, MCP, and Studio.
- Validate inputs early and return actionable errors.
- Do not make CLI display changes that globally affect unrelated commands.

## Data, references, and guidelines

- Reference examples are not just images; they include source context, captions, categories, aspect ratios, image paths, and optional structure hints.
- Keep reference IDs stable and meaningful. Do not introduce arbitrary IDs where existing conventions expect paper/arXiv-like IDs.
- If changing reference loading, test category filters, explicit reference IDs, missing IDs, and path resolution.
- If changing guideline/venue behavior, keep user-supplied packs separate from built-in defaults and validate invalid venues clearly.

## Common review blockers to avoid

- Failing Ruff or tests.
- Missing tests for new behavior.
- Missing prompt files referenced by code.
- Feature PRs that change defaults.
- One-off personal/institution-specific content in core.
- Overlapping or duplicate implementations instead of reusing existing paths.
- Provider code without retries, credential validation, or mocked tests.
- Secrets included in metadata snapshots.
- Path escaping bugs, especially Windows paths injected into generated Python.
- Empty-iteration or `None` response crashes.
- Large central-file changes where a new module would be clearer.
- CLI-only implementation of logic that MCP/Studio/Python API should share.
