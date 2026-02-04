# PaperBanana MCP Server

MCP server that exposes PaperBanana's diagram and plot generation as tools for Claude Code, Cursor, or any MCP-compatible client.

## Tools

| Tool | Description |
|------|-------------|
| `generate_diagram` | Generate a methodology diagram from text context + caption |
| `generate_plot` | Generate a statistical plot from JSON data + intent description |
| `evaluate_diagram` | Compare a generated diagram against a human reference (4 dimensions) |

## Installation

```bash
pip install -e ".[mcp]"
```

This installs `fastmcp` and registers the `paperbanana-mcp` console script.

## Setup

### Claude Code

Add to your Claude Code MCP settings (`.claude/claude_code_config.json` or project-level):

```json
{
  "mcpServers": {
    "paperbanana": {
      "command": "paperbanana-mcp",
      "env": {
        "GOOGLE_API_KEY": "your-google-api-key"
      }
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
      "command": "paperbanana-mcp",
      "env": {
        "GOOGLE_API_KEY": "your-google-api-key"
      }
    }
  }
}
```

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
| `GOOGLE_API_KEY` | (none) | Google API key (required) |
| `SKIP_SSL_VERIFICATION` | `false` | Disable SSL verification for proxied environments |
