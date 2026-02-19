# PaperBanana MCP Server

MCP server that exposes PaperBanana's diagram and plot generation as tools for Claude Code, Cursor, or any MCP-compatible client.

## Tools

| Tool | Description |
|------|-------------|
| `generate_diagram` | Generate a methodology diagram from text context + caption |
| `generate_plot` | Generate a statistical plot from JSON data + intent description |
| `evaluate_diagram` | Compare a generated diagram against a human reference (4 dimensions) |

## Installation

### Quick Install (via `uvx`)

No local clone needed. Add the config below to your MCP client.

### Claude Code

Add to `.claude/claude_code_config.json` (or project-level):

```json
{
  "mcpServers": {
    "paperbanana": {
      "command": "uvx",
      "args": ["--from", "paperbanana[mcp]", "paperbanana-mcp"],
      "env": { "OPENAI_API_KEY": "your-openai-api-key" }
    }
  }
}
```

### Cursor

Add to `.cursor/mcp.json` in your project:

```json
{
  "mcpServers": {
    "paperbanana": {
      "command": "uvx",
      "args": ["--from", "paperbanana[mcp]", "paperbanana-mcp"],
      "env": { "OPENAI_API_KEY": "your-openai-api-key" }
    }
  }
}
```

### Development / Local Install

For contributors or local development:

```bash
pip install -e ".[mcp]"
```

This installs `fastmcp` and registers the `paperbanana-mcp` console script. Then use the same MCP config as above but replace the `uvx` command with a direct call:

```json
{
  "mcpServers": {
    "paperbanana": {
      "command": "paperbanana-mcp",
      "env": { "OPENAI_API_KEY": "your-openai-api-key" }
    }
  }
}
```

## Skills (Claude Code)

This repo ships with 3 Claude Code skills in `.claude/skills/`:

| Skill | Description |
|-------|-------------|
| `/generate-diagram <file> [caption]` | Generate a methodology diagram from a text file |
| `/generate-plot <data-file> [intent]` | Generate a statistical plot from CSV or JSON data |
| `/evaluate-diagram <generated> <reference>` | Evaluate a diagram against a human reference |

Skills are available automatically when you clone the repo and use Claude Code.

## Usage Examples

### Generate a methodology diagram

```
User: Generate a diagram for this methodology:
      "Our framework uses a two-phase pipeline: first a linear planning
       phase with Retriever, Planner, and Stylist agents, followed by
       an iterative refinement phase with Visualizer and Critic agents."
      Caption: "Overview of the PaperBanana multi-agent framework"
```

### Generate a statistical plot

```
User: Create a bar chart from this data:
      {"models": ["GPT-4", "Claude", "Gemini"], "accuracy": [0.92, 0.94, 0.91]}
      Intent: "Bar chart comparing model accuracy on benchmark"
```

### Evaluate a diagram

```
User: Evaluate the diagram at ./output.png against the reference at ./reference.png
      Context: [methodology text]
      Caption: "System architecture overview"
```

## Configuration

The server reads configuration from environment variables and `.env` files.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (none) | OpenAI API key (default provider) |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI endpoint (or Azure OpenAI / Foundry URL) |
| `GOOGLE_API_KEY` | (none) | Google API key (for Gemini provider) |
| `SKIP_SSL_VERIFICATION` | `false` | Disable SSL verification for proxied environments |

## Listing on MCP Directories

After publishing to PyPI, you can submit PaperBanana to MCP directories for discoverability:

- [Official MCP Registry](https://registry.modelcontextprotocol.io) - uses the `mcp-publisher` CLI; see their docs for the current submission process
- [Smithery.ai](https://smithery.ai) - submit through their website
- [Glama.ai](https://glama.ai) - community listing submission
- [mcp.so](https://mcp.so) - community-driven, submit via their GitHub
