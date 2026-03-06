---
user-invocable: true
allowed-tools:
  - mcp__paperbanana__generate_plot
  - Read
  - "Bash(paperbanana *)"
---

# Generate Plot

Generate a publication-quality statistical plot from a data file using PaperBanana.

## Instructions

1. Read the data file at `$ARGUMENTS[0]`.
2. Prepare the data for the MCP tool:
   - If the file is **CSV**: parse it and convert to a column-keyed dictionary (keys = column names, values = arrays of column values), then serialize with `json.dumps()` to produce a JSON string.
   - If the file is **JSON**: use the raw file content as-is (it is already a JSON string).
3. If `$ARGUMENTS[1]` is provided, use it as the plot intent. Otherwise, ask the user for a description of the desired plot (e.g., "Bar chart comparing model accuracy across benchmarks").
4. Call the MCP tool `generate_plot` with:
   - `data_json`: the JSON string (not a parsed object)
   - `intent`: the plot description
   - `iterations`: 3 (default)
5. Present the generated plot to the user.

## CLI Fallback

If the MCP tool is not available, fall back to the CLI:

```bash
paperbanana plot --data <file> --intent "<intent>"
```

## Example

```
/generate-plot results.csv "Bar chart comparing model accuracy"
```
