# Presentation Generation Pipeline Plan

## 1. Goal

Extend PaperBanana from generating individual academic figures into generating complete, presentation-ready slide decks.

The core PaperBanana philosophy remains the same:

```text
input material
  → retrieve relevant examples
  → plan structured output
  → apply style/alignment
  → render artifact
  → critique/refine
  → export final deliverable
```

For presentations, the artifact changes from a single figure to a multi-slide deck. The additional requirement is **alignment**: generated slides must align with a company/team's existing visual style, branding, tone, layout habits, and storytelling patterns.

---

## 2. Core idea

PaperBanana already has these reusable primitives:

- provider abstraction
- prompt-template-driven agents
- reference/example retrieval
- planning/styling/visualization/critique loop
- guideline/venue packs
- artifact saving and metadata
- batch/orchestration patterns
- evaluation/critic infrastructure

Presentation generation should build on the same architecture instead of becoming a separate system.

The new direction can be summarized as:

```text
PaperBanana today:
  source context + caption → refined figure

PaperBanana presentations:
  source material + deck objective + style references → refined slide deck
```

---

## 3. Proposed high-level architecture

```text
User input
  ├─ source material: docs, notes, research paper, markdown, meeting notes, data
  ├─ deck objective: pitch, review, report, lecture, sales, strategy, etc.
  ├─ audience: executives, engineers, customers, researchers, students, etc.
  ├─ desired slide count / duration
  ├─ optional old decks / brand examples
  └─ optional datasets / images / diagrams
        │
        ▼
Source Loader + Deck Context Builder
        │
        ▼
Style / Brand Alignment Layer
  ├─ extract style from uploaded decks
  ├─ load existing company style profile
  ├─ retrieve similar slides/layouts
  └─ produce PresentationStyleProfile
        │
        ▼
Deck Planner Agent
        │
        ▼
Slide Planner Agent
        │
        ▼
Slide Stylist / Alignment Agent
        │
        ▼
Slide Renderer / Deck Assembler
        │
        ▼
Deck Critic + Alignment Critic
        │
        ▼
Refinement Loop
        │
        ▼
Final Export
  ├─ .pptx
  ├─ .pdf
  ├─ slide images
  ├─ metadata.json
  └─ style/alignment report
```

---

## 4. Why alignment is the key new layer

For figure generation, style mainly means:

- academic visual quality
- venue guidelines
- clean colors
- readable labels
- diagram/plot conventions

For presentation generation, style also includes:

- brand colors
- fonts
- title placement
- logo/footer behavior
- section divider format
- chart color rules
- icon style
- slide density
- text tone
- storytelling patterns
- agenda/progress indicator patterns
- executive-summary conventions
- appendix style
- common slide layouts

Therefore, the core pipeline can remain similar, but presentations need a dedicated **alignment layer** before and after slide generation.

Alignment appears in two places:

1. **Before generation**: extract/load style and guide planning/rendering.
2. **After generation**: validate slides against style/brand/narrative constraints.

---

## 5. New intermediate representations

To avoid brittle direct PPTX generation from free-form LLM output, introduce structured IRs.

### 5.1 `PresentationStyleProfile`

Represents learned or configured brand/style information.

```python
class PresentationStyleProfile(BaseModel):
    id: str
    name: str
    source_decks: list[str] = []

    slide_size: str | None = None  # e.g. widescreen 16:9
    fonts: dict[str, str] = {}     # title/body/mono/etc.
    colors: dict[str, str] = {}    # primary/secondary/accent/background/text
    logo_path: str | None = None

    title_style: dict = {}
    body_text_style: dict = {}
    footer_style: dict = {}
    chart_style: dict = {}
    icon_style: dict = {}

    common_layouts: list[dict] = []
    section_divider_style: dict = {}
    agenda_style: dict = {}
    appendix_style: dict = {}

    tone: str | None = None
    slide_density: str | None = None  # low / medium / high
    storytelling_patterns: list[str] = []

    constraints: list[str] = []
```

This is analogous to a venue/style pack, but presentation-specific.

### 5.2 `PresentationIR`

Represents a full deck before rendering.

```python
class PresentationIR(BaseModel):
    title: str
    objective: str
    audience: str | None = None
    narrative_summary: str | None = None
    style_profile_id: str | None = None
    slides: list[SlideIR]
    metadata: dict = {}
```

### 5.3 `SlideIR`

Represents one slide.

```python
class SlideIR(BaseModel):
    id: str
    index: int
    section: str | None = None
    title: str
    message: str
    layout_type: str
    elements: list[SlideElementIR]
    speaker_notes: str | None = None
    citations: list[str] = []
    alignment_requirements: list[str] = []
```

### 5.4 `SlideElementIR`

Represents objects inside a slide.

```python
class SlideElementIR(BaseModel):
    id: str
    type: Literal[
        "text", "image", "chart", "diagram", "table", "callout", "icon", "shape"
    ]
    content: dict
    position: dict | None = None
    style: dict = {}
    source_ref: str | None = None
```

This keeps LLM planning separate from rendering.

---

## 6. Proposed agents

The existing agent pattern should be reused. New agents can live under:

```text
paperbanana/agents/presentation/
```

or initially under:

```text
paperbanana/presentation/agents/
```

Recommended new agents:

### 6.1 `DeckContextAgent`

Purpose: convert raw input material into a compact, structured brief for deck generation.

Input:

- uploaded documents
- markdown/text notes
- research paper sections
- user objective
- audience
- desired slide count

Output:

```json
{
  "topic": "...",
  "objective": "...",
  "audience": "...",
  "key_points": [...],
  "supporting_evidence": [...],
  "data_assets": [...],
  "must_include": [...],
  "risks_or_caveats": [...]
}
```

This is similar to input optimization, but deck-level.

### 6.2 `DeckStyleAnalyzerAgent`

Purpose: analyze uploaded old decks and create/update `PresentationStyleProfile`.

Input:

- extracted slide metadata
- slide screenshots
- theme XML / pptx style data if available
- optional company brand guidelines

Output:

- `PresentationStyleProfile`
- extracted layout examples
- style confidence report

Important: not everything should rely on VLMs. Some style extraction should be deterministic:

- slide size
- fonts
- colors
- master/theme information
- shape positions
- title bounding boxes
- logo locations

VLM can summarize subjective parts:

- tone
- visual style
- storytelling pattern
- slide density
- common layout intent

### 6.3 `SlideRetrieverAgent`

Purpose: retrieve similar past slides/layouts to guide generation.

This is presentation equivalent of `RetrieverAgent`.

Retrieval candidates can include:

- old slides from uploaded decks
- layout examples
- slide screenshots
- slide metadata
- extracted text/title/message
- section type
- chart/diagram type

Output:

- relevant slide examples per planned slide
- relevant layout examples

### 6.4 `DeckPlannerAgent`

Purpose: create the overall presentation storyline.

Input:

- structured deck brief
- style profile
- audience/objective
- desired slide count
- similar deck examples if available

Output:

- `PresentationIR` skeleton
- section list
- slide list with title/message/purpose/layout hints

Responsibilities:

- narrative arc
- section ordering
- key message per slide
- slide count discipline
- appendix strategy
- speaker intent

### 6.5 `SlidePlannerAgent`

Purpose: expand each planned slide into a detailed `SlideIR`.

Input:

- deck plan
- slide brief
- source context/evidence
- style profile
- retrieved slide examples

Output:

- complete `SlideIR` with elements and content requirements

Responsibilities:

- choose layout type
- decide text blocks
- choose visual type: chart, diagram, table, image, callout
- attach data references
- propose speaker notes

### 6.6 `SlideStylistAgent`

Purpose: align `SlideIR` with the company style profile.

Input:

- `SlideIR`
- `PresentationStyleProfile`
- retrieved examples

Output:

- styled/aligned `SlideIR`

Responsibilities:

- apply tone
- enforce text density
- select approved colors
- choose layout variants
- specify visual hierarchy
- ensure title/message style matches profile

This is analogous to `StylistAgent` but slide-specific.

### 6.7 `SlideAssetAgent`

Purpose: generate or prepare assets needed by slide elements.

For each element:

- `diagram`: call existing PaperBanana diagram pipeline or a lighter diagram generation function
- `chart`: call existing statistical plot pipeline or generate chart directly from data
- `image`: use image provider or place uploaded image
- `table`: format structured data
- `icon`: select from icon set or generate simple SVG

Output:

- asset files with paths
- updated `SlideIR` references

This is the strongest reuse point for existing PaperBanana figure/plot generation.

### 6.8 `DeckAssembler`

Purpose: render `PresentationIR` into a real deck.

Likely MVP implementation:

```text
python-pptx → .pptx
```

Possible future renderers:

- HTML/CSS slides
- reveal.js
- PDF exporter
- Google Slides API
- PowerPoint XML/template renderer

Input:

- `PresentationIR`
- `PresentationStyleProfile`
- generated assets

Output:

- `.pptx`
- optional slide preview images

### 6.9 `DeckCriticAgent`

Purpose: evaluate the whole deck.

Checks:

- narrative coherence
- missing slides
- duplicate slides
- weak transitions
- audience fit
- logical flow
- executive readability
- source faithfulness

Output:

```json
{
  "deck_level_issues": [...],
  "slide_level_issues": [...],
  "revised_deck_plan": {...},
  "slides_to_regenerate": ["slide_03", "slide_07"]
}
```

### 6.10 `AlignmentCriticAgent`

Purpose: validate brand/style adherence.

Checks:

- colors match style profile
- fonts match style profile
- title placement consistency
- logo/footer consistency
- chart style consistency
- slide density
- layout similarity to examples
- tone consistency

This should combine deterministic validators and VLM-based visual critique.

---

## 7. Proposed pipeline phases

### Phase A: Source preparation

```text
load documents / notes / data
  → normalize text
  → extract sections/tables/assets
  → build deck brief
```

Implementation candidates:

- reuse `source_loader.py`
- reuse `pdf_text.py`
- extend plot data loading for presentation datasets
- add support for `.pptx`, `.docx`, `.md`, `.txt`, `.csv`, `.json`

### Phase B: Style and branding alignment setup

```text
old decks / brand kit / style profile
  → deterministic extraction
  → VLM style summary
  → PresentationStyleProfile
  → slide/layout reference index
```

This can be cached so companies upload old decks once.

Artifacts:

```text
style_profile.json
slide_examples/index.json
slide_screenshots/
layout_clusters.json
```

### Phase C: Deck planning

```text
deck brief + style profile + audience/objective
  → DeckPlannerAgent
  → PresentationIR skeleton
```

Output example:

```text
Slide 1: Title / context
Slide 2: Executive summary
Slide 3: Problem framing
Slide 4: Evidence / data
Slide 5: Recommendation
Slide 6: Roadmap
Slide 7: Risks
Slide 8: Next steps
```

### Phase D: Slide planning

For each slide:

```text
slide brief + source evidence + style profile + retrieved examples
  → SlidePlannerAgent
  → SlideIR
```

### Phase E: Slide alignment/styling

For each slide:

```text
SlideIR + style profile + retrieved examples
  → SlideStylistAgent
  → aligned SlideIR
```

### Phase F: Asset generation

For each slide element requiring an asset:

```text
chart element → existing plot pipeline / chart renderer
diagram element → existing diagram pipeline
image/icon element → provider or asset library
table element → deterministic table renderer
```

Outputs are stored under the presentation run directory.

### Phase G: Deck assembly

```text
PresentationIR + style profile + assets
  → DeckAssembler
  → .pptx + previews
```

### Phase H: Critique and refinement

```text
assembled deck + previews + source brief + style profile
  → DeckCriticAgent + AlignmentCriticAgent
  → issues + slides_to_regenerate
  → revise SlideIR / regenerate assets / reassemble
```

This mirrors PaperBanana's existing iterative refinement pattern.

### Phase I: Export

Final artifacts:

```text
final_deck.pptx
final_deck.pdf        optional
slide_previews/       optional
presentation_ir.json
style_profile.json
metadata.json
alignment_report.json
prompts/
assets/
```

---

## 8. Alignment layer design

The alignment layer should be treated as a first-class subsystem.

### 8.1 Style extraction from old decks

Suggested deterministic extraction:

- slide dimensions
- theme colors
- fonts
- master layouts
- placeholder positions
- title box coordinates
- footer/logo positions
- recurring shapes
- chart colors where extractable
- image usage patterns
- text density statistics

Suggested VLM extraction from slide screenshots:

- layout intent
- perceived style
- tone
- visual hierarchy
- common storytelling format
- design adjectives
- slide archetypes

### 8.2 Slide/layout indexing

Uploaded old decks should be converted into retrievable examples:

```text
SlideExample
  - id
  - deck_id
  - slide_index
  - title
  - extracted_text
  - screenshot_path
  - layout_type
  - section_type
  - visual_types
  - colors/fonts/positions summary
  - speaker_notes if available
```

This gives the presentation pipeline the same retrieval advantage PaperBanana has for figures.

### 8.3 Alignment validation

Use deterministic checks where possible:

- title within expected region
- approved fonts used
- colors from palette
- logo present if required
- footer present if required
- max words per slide
- max bullet count
- chart colors from allowed sequence
- slide size correct

Use VLM checks for subjective issues:

- does this feel like the reference decks?
- is visual hierarchy consistent?
- is this too dense?
- is the tone company-like?
- is the slide suitable for the target audience?

---

## 9. Suggested file/module structure

A clean implementation can add a new package:

```text
paperbanana/presentation/
  __init__.py
  types.py                 # PresentationIR, SlideIR, StyleProfile, etc.
  pipeline.py              # PresentationPipeline
  source.py                # deck/source loading helpers
  style_extractor.py       # pptx/theme/style extraction
  slide_index.py           # indexes old slides as examples
  retrieval.py             # slide/layout retrieval
  renderer.py              # PPTX assembly
  validators.py            # deterministic alignment checks
  assets.py                # chart/diagram/image asset generation
  agents/
    __init__.py
    context.py
    style_analyzer.py
    deck_planner.py
    slide_planner.py
    slide_stylist.py
    deck_critic.py
    alignment_critic.py
```

Add prompts:

```text
prompts/presentation/
  context.txt
  style_analyzer.txt
  deck_planner.txt
  slide_planner.txt
  slide_stylist.txt
  deck_critic.txt
  alignment_critic.txt
```

Add configs:

```text
configs/pipeline/presentation.yaml
```

Add tests:

```text
tests/test_presentation/
  test_types.py
  test_style_extractor.py
  test_renderer.py
  test_pipeline.py
  test_validators.py
```

---

## 10. CLI/API proposal

### 10.1 CLI commands

Possible commands:

```bash
paperbanana presentation generate \
  --input report.md \
  --objective "Board update on Q3 product performance" \
  --audience executives \
  --slides 10 \
  --style-profile company_style.json \
  --output-dir outputs
```

```bash
paperbanana presentation learn-style \
  --deck examples/company_old_deck.pptx \
  --output company_style.json
```

```bash
paperbanana presentation generate \
  --input strategy_notes.txt \
  --reference-deck old_company_deck.pptx \
  --slides 8
```

```bash
paperbanana presentation critique \
  --deck outputs/presentation_run/final_deck.pptx \
  --style-profile company_style.json
```

### 10.2 Python API

```python
from paperbanana.presentation.pipeline import PresentationPipeline
from paperbanana.presentation.types import PresentationInput

pipeline = PresentationPipeline(settings=settings)
result = await pipeline.generate(
    PresentationInput(
        source_paths=["report.md", "metrics.csv"],
        objective="Create an executive strategy update",
        audience="executives",
        slide_count=10,
        style_profile_path="company_style.json",
    )
)

print(result.pptx_path)
```

### 10.3 MCP tools

Potential MCP tools:

- `generate_presentation`
- `learn_presentation_style`
- `critique_presentation`
- `continue_presentation`

---

## 11. MVP plan

### MVP 1: Presentation generation with predefined style profile

Goal: generate a simple `.pptx` from text/markdown without learning style from old decks yet.

Scope:

- `PresentationIR`, `SlideIR`, `SlideElementIR`
- `PresentationPipeline`
- `DeckPlannerAgent`
- `SlidePlannerAgent`
- basic `SlideStylistAgent`
- `python-pptx` renderer
- predefined style profiles: `consulting`, `academic`, `startup`, `executive`
- final `.pptx` export

Avoid initially:

- old deck style learning
- complex PowerPoint master editing
- highly accurate brand matching
- direct image-edit refinement

### MVP 2: Style extraction from old decks

Goal: upload old decks and generate a reusable style profile.

Scope:

- `.pptx` parser
- theme/font/color extraction
- slide screenshot extraction if feasible
- layout metadata extraction
- `DeckStyleAnalyzerAgent`
- `style_profile.json`
- style profile reuse in generation

### MVP 3: Slide example retrieval

Goal: retrieve old slides/layouts as in-context examples.

Scope:

- `SlideExample` index
- screenshot storage
- slide metadata extraction
- `SlideRetrieverAgent`
- slide-level retrieval during planning/styling

### MVP 4: Critique/refinement loop

Goal: improve deck quality automatically.

Scope:

- render slide previews
- `DeckCriticAgent`
- `AlignmentCriticAgent`
- deterministic validators
- regenerate selected weak slides
- `alignment_report.json`

### MVP 5: Deep PaperBanana integration

Goal: use existing figure/plot generation inside slides.

Scope:

- slide element type `diagram` calls methodology figure pipeline
- slide element type `chart` calls statistical plot pipeline or chart renderer
- generated assets are inserted into slides
- deck-level metadata references nested figure runs

---

## 12. Reuse from existing PaperBanana

### Reuse directly

- `Settings`
- provider registry
- VLM/image provider abstractions
- `BaseAgent`
- prompt recorder
- cost tracker
- source loading/PDF extraction
- plot data loading
- progress callback conventions
- output directory pattern
- metadata saving

### Adapt

- `RetrieverAgent` concept → `SlideRetrieverAgent`
- `StylistAgent` concept → `SlideStylistAgent`
- `CriticAgent` concept → `DeckCriticAgent` and `AlignmentCriticAgent`
- `orchestrate.py` concept → deck-level orchestration
- `ReferenceStore` concept → slide/deck example store
- venue packs → presentation style profiles

### Add new

- presentation IR models
- PPTX parser
- PPTX renderer
- style extractor
- slide layout index
- deterministic alignment validators
- deck-specific prompts

---

## 13. Technical risks and mitigations

### Risk 1: PPTX extraction is messy

PowerPoint files contain masters, layouts, placeholders, manual overrides, embedded images, and theme XML.

Mitigation:

- start with best-effort extraction
- prioritize colors/fonts/slide size/title/footer positions
- store screenshots as visual examples
- do not aim for perfect reconstruction in MVP

### Risk 2: Generated slides become too text-heavy

Mitigation:

- enforce text density limits in `SlideIR`
- deterministic validators for word count and bullet count
- critic checks for readability
- style profile includes slide density preference

### Risk 3: Brand fidelity is hard to measure

Mitigation:

- deterministic checks for colors/fonts/logo/footer/title position
- VLM visual comparison against old slide examples
- alignment report with explicit pass/fail checks

### Risk 4: `python-pptx` rendering limitations

Mitigation:

- keep MVP layouts simple
- use image-based assets for complex diagrams/charts
- later add HTML/PDF or template-based renderers

### Risk 5: Deck narrative may be weak

Mitigation:

- separate deck planning from slide planning
- add deck-level critic
- require one key message per slide
- use audience/objective in every planning prompt

---

## 14. Open design questions

1. Should presentation support live inside `paperbanana/presentation/` or as a sibling package?
2. Should style profiles be generalized into the existing guideline/venue system?
3. Should generated slides be editable PowerPoint shapes or mostly image-based for visual fidelity?
4. Should chart/diagram generation inside slides call full PaperBanana pipelines or lighter internal renderers?
5. Should old deck upload be a one-time style-learning command or part of every generation run?
6. What should be the first target use case: business decks, academic talks, pitch decks, or internal reports?
7. Should speaker notes be generated by default?
8. Should generated decks include citations/source traceability per slide?

---

## 15. Recommended starting direction

Start small but architecturally clean:

1. Add `paperbanana/presentation/types.py` with `PresentationIR`, `SlideIR`, `SlideElementIR`, and `PresentationStyleProfile`.
2. Add a basic `PresentationPipeline` that creates a deck from text using predefined style profiles.
3. Implement a simple `python-pptx` renderer with a small layout library.
4. Add deck/slide planning prompts.
5. Add deterministic validators for text density, title presence, and style colors.
6. Only then add style extraction from uploaded old decks.

This preserves the core PaperBanana architecture while introducing the minimum new pieces needed for deck generation.

---

## 16. Final mental model

Presentation generation should be treated as a higher-level orchestration problem over the existing figure-generation architecture.

```text
PaperBanana figure pipeline:
  one context + one caption
    → one refined figure

PaperBanana presentation pipeline:
  many source materials + one deck objective + style profile
    → deck plan
    → many slide plans
    → many aligned slide artifacts
    → one refined presentation deck
```

The biggest new concept is not generation itself. It is **alignment**: extracting, applying, and validating that generated slides match the organization's established visual and storytelling style.
