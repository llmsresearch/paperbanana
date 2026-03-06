---
user-invocable: true
allowed-tools:
  - mcp__paperbanana__evaluate_diagram
  - Read
  - "Bash(paperbanana *)"
---

# Evaluate Diagram

Evaluate a generated diagram against a human reference using PaperBanana's VLM-as-Judge scoring.

## Instructions

1. `$ARGUMENTS[0]` is the path to the generated image.
2. `$ARGUMENTS[1]` is the path to the human reference image.
3. Ask the user for:
   - **Source context**: the methodology text (or a file path to read it from). If the user provides a file path, read that file to get the text.
   - **Figure caption**: a description of what the diagram communicates.
4. Call the MCP tool `evaluate_diagram` with:
   - `generated_path`: the generated image path
   - `reference_path`: the reference image path
   - `context`: the methodology text content
   - `caption`: the figure caption
5. Present the evaluation scores to the user. Scores cover 4 dimensions: Faithfulness, Conciseness, Readability, and Aesthetics.

## CLI Fallback

If the MCP tool is not available, fall back to the CLI:

```bash
paperbanana evaluate --generated <generated-img> --reference <reference-img> --context <context-file> --caption "<caption>"
```

## Example

```
/evaluate-diagram output.png reference.png
```
