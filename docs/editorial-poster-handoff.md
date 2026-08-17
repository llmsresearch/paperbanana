# Editorial Poster Compositor Handoff

## Mission

Create a genuinely conference-quality ERNIE 5.0 research poster that visibly holds up beside the supplied NeurIPS, ICML, and ICLR posters.

Do not call pipeline completion, test success, compliance, or relative improvement success. There is currently no approved poster. Release only after direct full-resolution and thumbnail review finds no design blockers and the calibrated Azure review is at least 7/10 with zero blockers.

## User Constraints

- Use Azure AI Foundry only. Do not use Atlas.
- VLM: `azure_foundry` with deployment `gpt-4o`.
- Image provider, when needed for faithful figure reauthoring only: `azure_foundry_image` with deployment `gpt-image-2`.
- Do not print, log, or save the Azure API key.
- Do not commit or push.
- Preserve unrelated dirty worktree changes.
- Do not create fallback mechanisms. If a required provider, renderer, or dependency fails, stop and report the exact blocker.
- Do not release the best available candidate merely because it is the best candidate.

## Current State After Cleanup

The failed hybrid/skeleton/card-renderer experiment has been removed:

- Removed `design_skeleton.py`, `skeleton_renderer.py`, `vector_renderer.py`, `harness.py`, `scene.py`, `hybrid.py`, `asset_harness.py`, and `quality_harness.py`.
- Removed their tests and prompts.
- Removed the abandoned `--engine hybrid`, `--candidates`, and `--render-candidates` CLI route.
- Removed all rejected and temporary `outputs/hybrid_samples` artifacts, including the misleading release directory.
- Retained Azure provider support, ingestion compatibility fixes, source papers, human references, and grounded ERNIE extraction assets.

Cleanup validation completed:

- CLI, poster, provider registry, and config tests: 164 passed.
- Ruff on the remaining poster and Azure surfaces: passed.
- No abandoned hybrid symbols remain in Python, prompt, or Markdown files.

Do not restore the deleted card/skeleton architecture from git history or old chat instructions.

## Retained Inputs

Paper:

- `research/source_papers/ernie5.pdf`

Grounded extraction bundle:

- `outputs/editorial_handoff/ernie5/paper_assets.json`
- `outputs/editorial_handoff/ernie5/hires/`

Prepared scientific figures:

- `outputs/editorial_handoff/ernie5/figures/fig1_architecture.png` (`3739x2147`)
- `outputs/editorial_handoff/ernie5/figures/fig4_elastic_training.png` (`3459x1831`)
- `outputs/editorial_handoff/ernie5/figures/fig8_expert_routing.png` (`2756x2869`)

Human poster references:

- `research/posters/NeurIPS_2022_52843.png` (`1498x999`)
- `research/posters/ICML_2022_16033.png` (`4608x2304`)
- `research/posters/ICML_2024_32793.png` (`720x960`, portrait reference)
- `research/posters/ICLR_2023_10708.png` (`4000x2000`)

Reference corpus statistics:

- `data/poster_schemas/schemas.json`
- For landscape three-column posters in the corpus: median 6 sections, median 9 text blocks, median visual share about 25%, median title band about 14%.
- Treat these as descriptive baselines, not quality guarantees.

## What Failed

### 1. Generated skeletons

Image-generated sentinel regions were converted into deterministic text and figure fields. The representation guaranteed a dashboard because every field received the same heading/image/text treatment. Angular geometry and overlapping fields did not solve this.

### 2. Sparse visual-thesis layouts

A dominant hero, attached evidence, two statistics, and a takeaway still produced excessive empty space, tiny scientific visuals, and generic callouts. Removing headings did not create editorial hierarchy.

### 3. Dense vector layouts

The dense layout filled more area but remained a title bar plus isolated rectangular sections. It had no integrated captions, figure annotations, visual explanation, editorial rhythm, or varied section treatment.

The renderer's core assumption was:

`rectangle -> heading -> figure -> bullets`

Changing rectangle proportions cannot reproduce the supplied references.

### 4. Content and figure mapping failures

- Figure 2 was placed next to the 15% routing-speed claim even though Figure 2 describes visual tokenization. This was semantically wrong.
- Multiple figures were stacked inside one evidence panel, shrinking one figure to approximately `43x23 mm`.
- Character-count text wrapping silently clipped final words.
- Mechanical `text_overflow=[]` and panel coverage metrics created false confidence.

### 5. Evaluation failures

The initial multi-image Azure judge could not reliably bind candidate and reference image identities. It scored a real NeurIPS poster at 1/10 and repeated canned blockers.

Independent single-image assessment calibrated correctly (the human poster scored about 8.6), but it later scored a visibly weak generated poster at 8.2. Therefore Azure is only a secondary signal. It cannot authorize release.

### 6. Full-poster image generation is not the answer

`paperbanana/poster/generative.py` can produce attractive bitmap composition, but generated text and scientific diagrams cannot be trusted. It may be used for visual ideation or background language only, never as the final source of text, numbers, or figures.

## Core Diagnosis

The target requires an editorial composition system, not a generic layout engine.

The reference posters achieve quality through:

- one unmistakable visual thesis
- figures that explain the paper rather than decorate sections
- captions, callouts, labels, and arrows integrated around figures
- varied visual treatments across the page
- deliberate reading order
- compact, grounded scientific detail
- consistent typography and page-level color logic
- purposeful whitespace, not empty allocation

The current repository has no existing renderer that provides this. Build a new reference-derived editorial SVG compositor.

## New Architecture

Create a new implementation rather than reviving removed files. Suggested boundary:

- `paperbanana/poster/editorial_types.py`
- `paperbanana/poster/editorial_renderer.py`
- `paperbanana/poster/editorial_ernie5.py` for the first authored composition
- focused tests under `tests/test_poster/`

Do not wire it into the CLI until one ERNIE poster passes direct review. Premature pipeline integration previously encouraged generic abstractions.

### Editorial primitives

The compositor should support primitives such as:

- title and author band
- section label without mandatory container
- paragraph and compact bullet list
- real figure with caption
- cropped figure detail
- numbered annotation marker
- annotation line or connector
- explanatory label adjacent to a figure region
- big statistic with source note
- comparison strip
- thin rule, color band, and reading-path cue
- conclusion statement

Do not make `Panel` the only composition primitive. A scientific figure must be able to span a large canvas region with annotations placed around or over it.

### Output format

Use SVG as the source of truth:

- deterministic vector text
- real embedded scientific figures
- exact geometry
- inspectable element bounds
- export SVG, PDF, and PNG

Use a structured SVG library or XML API. Do not assemble SVG by fragile string replacement when a structured interface is practical.

Use measured text metrics. Every text block must expose its final bounds and fail if it exceeds its allocation. PDF text extraction must recover every complete claim after whitespace normalization.

## ERNIE Scientific Narrative

The poster should tell one argument:

> ERNIE 5.0 unifies multimodal understanding and generation in one autoregressive MoE backbone, then uses elastic depth, width, and routing sparsity to serve different compute budgets from one training run.

Recommended evidence sequence:

1. **Problem:** late-fusion systems separate understanding and generation and rely on modality-specific components.
2. **Unified architecture:** Figure 1 is the dominant visual. Explain how text, vision, and audio enter one shared ultra-sparse MoE backbone.
3. **Shared expert behavior:** Figure 8 shows expert utilization across layers, modalities, and tasks. Explain task-shaped specialization rather than fixed modality boundaries.
4. **Elastic deployment:** Figure 4 explains elastic depth, width, and sparsity.
5. **Quantitative evidence:** use the exact grounded facts:
   - reducing routing top-k to 25% yields over 15% decoding speedup with minor accuracy loss
   - near-full performance is retained using 53.7% activated parameters and 35.8% total parameters
6. **Takeaway:** one checkpoint supports different memory, compute, and latency constraints.

Do not use Figure 2 for routing evidence. It describes vision tokenization and generation.

Table 12 in `paper_assets.json` is relevant quantitative evidence for elastic variants. Consider resetting only the necessary rows into a compact editorial comparison strip rather than embedding a full paper table screenshot.

## Reference Grammar Extraction

Before writing the compositor, analyze the four references and create a small design-spec artifact, for example:

- `outputs/editorial_handoff/reference_grammar.json`

Record:

- title-band fraction
- main visual fraction
- text and visual density
- section count
- dominant alignment lines
- reading path
- typography scale ratios
- caption treatment
- annotation style
- color roles
- whitespace distribution
- how the references distinguish primary evidence from secondary evidence

Do not record only panel rectangles. The purpose is to capture relationships and hierarchy.

Use at least two landscape references as direct composition teachers. The portrait ICML poster can inform typography and section treatment but should not dictate landscape geometry.

## Concept Stage

Create three meaningfully different ERNIE concepts:

### Concept A: Architecture-led editorial spread

- Figure 1 dominates the center-left or full upper body.
- Problem text appears as a compact entry point.
- Routing and elasticity are attached as explanatory evidence around the architecture.
- Strong for a three-minute walkthrough.

### Concept B: Narrative columns

- Left: problem and unified input representation.
- Center: large architecture with integrated annotations.
- Right: routing behavior and elastic deployment evidence.
- Avoid equal-weight columns and repeated card styling.

### Concept C: Annotated technical plate

- Figure 1 behaves like a large technical plate.
- Numbered annotation markers explain the shared backbone and routing.
- Figure 8 and Figure 4 become inset evidence plates connected to relevant annotations.
- Minimal prose outside the plate, but enough grounded detail for comprehension.

Render all three as full posters. They must differ in visual thesis and reading order, not just color or coordinates.

## Visual Direction

The previous dark-green title bar, orange statistic, white-card template is rejected. Establish a page-level visual language from the references and subject matter.

Requirements:

- expressive but professional typography
- at least three functional color roles, with color concentrated in figures and evidence cues
- no rounded-card dashboard
- no repeated equal-weight boxes
- no large decorative empty regions
- no oversized statistics that displace scientific explanation
- no tiny screenshots
- captions must explain why each figure matters
- annotations must point to real visual regions, not float as generic bullets

## Validation Gates

A candidate must pass all gates before it can be considered for release.

### Grounding

- Every factual claim is traceable to `paper_assets.json` or the source PDF.
- Every number matches the paper exactly.
- Every figure supports its adjacent claim.
- Figure captions and panel labels are semantically correct.

### Text

- Every complete authored claim round-trips through PDF extraction after whitespace normalization.
- No line is silently omitted.
- No horizontal or vertical clipping.
- At conference-distance thumbnail scale, section headings and central argument remain readable.

### Figures

- Use real or faithfulness-approved reauthored figures only.
- Minimum physical size must be checked, but physical size alone is insufficient.
- Labels inside each figure must be readable in a full-poster screenshot.
- No panel may contain multiple figures merely stacked to satisfy completeness.
- If a figure remains too dense, reauthor or create a faithful detail view rather than shrinking it.

### Composition

- Compare side by side with the human references.
- The main contribution must be identifiable in under five seconds.
- The three-minute walkthrough must have an obvious path.
- No generic dashboard or template appearance.
- No underdeveloped region.
- No repeated section treatment across the entire poster.

### Technical

- Venue dimensions and orientation pass.
- SVG, PDF, and PNG all render correctly.
- Text and figures do not overlap.
- Tests and Ruff pass for the touched implementation.

### Azure secondary review

Only after direct review passes:

- Evaluate candidate and references independently as single images.
- Require the human calibration poster to score at least 7 with zero blockers.
- Require the candidate to score at least 7 with zero blockers.
- Do not release solely because Azure passes.

## Azure Access

Verified Azure resource:

- Subscription: `LLMs grant`
- Subscription ID: `d2d73c65-4153-4a38-97a7-1ef22524f335`
- Resource group: `llm-newsletter-rg`
- Account: `dip-makl2p31-eastus2`
- Base URL: `https://dip-makl2p31-eastus2.cognitiveservices.azure.com/openai/v1/`
- VLM deployment: `gpt-4o`
- Image deployment: `gpt-image-2`

Retrieve the key directly into a shell environment variable. Never print it or save it:

```bash
export AZURE_FOUNDRY_API_KEY="$(az cognitiveservices account keys list \
  --subscription d2d73c65-4153-4a38-97a7-1ef22524f335 \
  --name dip-makl2p31-eastus2 \
  --resource-group llm-newsletter-rg \
  --query 'key1' --output tsv)"
export AZURE_FOUNDRY_BASE_URL='https://dip-makl2p31-eastus2.cognitiveservices.azure.com/openai/v1/'
export AZURE_FOUNDRY_VLM_DEPLOYMENT='gpt-4o'
export AZURE_FOUNDRY_IMAGE_DEPLOYMENT='gpt-image-2'
```

Unset the key after the operation.

If Azure CLI reports an SSL error, export `REQUESTS_CA_BUNDLE=/System/Library/Templates/Data/private/etc/ssl/cert.pem` once and retry normally. Do not disable certificate verification.

## First Actions In The New Conversation

1. Read:
   - `.github/copilot-instructions.md`
   - `CLAUDE.md`
   - this handoff document
2. Confirm the rejected hybrid files and `outputs/hybrid_samples` remain absent.
3. Inspect all four human references at full resolution.
4. Create `reference_grammar.json` with relationship-level design observations.
5. Read the ERNIE paper asset manifest around Figure 1, Figure 4, Figure 8, and Table 12.
6. Write the grounded poster copy and annotation plan before writing renderer code.
7. Build one reversible SVG proof containing:
   - title system
   - dominant Figure 1
   - two real annotations
   - one evidence inset
   - measured typography
8. Render and inspect that proof before expanding to a full poster.
9. Stop if the proof still resembles a dashboard. Do not proceed by adding more rectangles.

## Definition Of Done

Done means there is one artifact that a human reviewer can place beside the supplied conference posters without an obvious quality collapse.

It must be scientifically faithful, visually authored, readable, technically valid, and free of direct-review blockers. Until then, report progress honestly and keep the release status as rejected.
