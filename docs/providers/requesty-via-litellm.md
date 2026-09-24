# Using Requesty via LiteLLM

[Requesty](https://requesty.ai) is an OpenAI-compatible LLM gateway. PaperBanana can use it for the VLM agents (planning, styling, critique, plot code) through the existing `litellm` provider, with no extra code.

## Setup

```bash
pip install 'paperbanana[litellm]'
```

Create a key at [app.requesty.ai/api-keys](https://app.requesty.ai/api-keys) and add it to `.env`:

```bash
LITELLM_API_KEY=your-requesty-key
LITELLM_API_BASE=https://router.requesty.ai/v1   # EU: https://router.eu.requesty.ai/v1
```

## Usage

Requesty model ids look like `openai/gpt-4o-mini` or `anthropic/claude-sonnet-4-5`. Prefix them with `openai/` so LiteLLM sends the request to the OpenAI-compatible endpoint set in `LITELLM_API_BASE`:

```bash
paperbanana plot \
  --data examples/sample_data/benchmark_slice.csv \
  --intent "Bar chart comparing model accuracy" \
  --vlm-provider litellm \
  --vlm-model openai/openai/gpt-4o-mini
```

For `paperbanana generate`, keep your usual `--image-provider`. Only the VLM calls go through Requesty.

Notes:

- The cost summary shows `$0.0000` for these models because they are not in PaperBanana's pricing table. Usage and cost are visible in the Requesty dashboard.
- Model list: [docs.requesty.ai](https://docs.requesty.ai)
