# PaperBanana Slide-Deck Integration Design

**Date:** 2026-03-07
**Status:** Approved
**Decision:** Integrate PPT creation capability into PaperBanana via Nexus RDIV orchestration + baoyu asset referencing (S1 strategy)

## Problem

PaperBanana can generate individual slide images (`slide` / `slide-batch`), but lacks:
1. End-to-end slide deck workflow (content analysis -> outline -> prompts -> images -> PPTX)
2. Interactive style selection with Claude presenting recommendations
3. Rich style presets (limited to hardcoded Stylist agent defaults)
4. PPTX/PDF output

## Architecture

### Layer Separation

| Layer | Responsibility | Owns |
|-------|---------------|------|
| PaperBanana CLI (Python) | Image generation, style preset fallback | `slide`, `slide-batch`, `slide_styles.py` (23 presets) |
| Nexus RDIV Skill (Claude) | Content analysis, interactive selection, outline/prompt generation, orchestration | Workflow state, user interaction |
| Baoyu Assets (runtime ref) | Style definitions, dimension system, PPTX/PDF merge scripts | `references/styles/*.md`, `scripts/merge-to-*.ts` |

### Sync Strategy: S1 (Reference, Don't Copy)

- Nexus skill reads baoyu files at runtime via `Glob` + `Read`
- Baoyu plugin cache path: `~/.claude/plugins/cache/baoyu-skills/content-skills/*/skills/baoyu-slide-deck/`
- Fallback: `slide_styles.py` built-in 23 presets when baoyu unavailable

### Merge Strategy: M2 (Call Baoyu Scripts)

- PPTX: `${BUN_X} ${BAOYU_DIR}/scripts/merge-to-pptx.ts <dir>`
- PDF: `${BUN_X} ${BAOYU_DIR}/scripts/merge-to-pdf.ts <dir>`
- No python-pptx dependency needed

## RDIV Workflow

### R (Research) - Content Analysis & Style Selection

1. Read user input (file or pasted text), save as `source.md`
2. Glob baoyu `references/styles/*.md` to discover latest style list
3. Analyze content signals (discipline, audience, tone) -> match 2-3 recommended styles
4. AskUserQuestion: style, audience, slide count
5. Output: `analysis.md`

### D (Design) - Outline & Prompts

1. Read selected baoyu style file for full visual spec
2. Generate `outline.md` (per-slide: title, layout type, content points)
3. Optional: user reviews outline
4. Generate image prompts for each slide -> `prompts/*.md`
5. Each prompt auto-injected with style instructions

### I (Implement) - Generation & Merge

1. `paperbanana slide-batch --prompts-dir prompts/ --style {name}`
2. `${BUN_X} merge-to-pptx.ts <output-dir>`
3. `${BUN_X} merge-to-pdf.ts <output-dir>`

### V (Verify) - Review & Iterate

1. Read generated images for user review
2. User feedback: satisfied / modify specific slides / redo with different style
3. On modify: update prompt -> `paperbanana slide --input prompts/NN-slide.md --style {name}` -> re-merge

## Output Structure

```
slide-deck/{topic-slug}/
  source.md
  analysis.md
  outline.md
  prompts/
    01-slide-cover.md
    02-slide-xxx.md
    ...
  01-slide-cover.png
  02-slide-xxx.png
  {topic-slug}.pptx
  {topic-slug}.pdf
  nexus_state.json
```

## Already Completed

- [x] `slide_styles.py` - 23 style presets (baoyu 16 + elite-powerpoint 3 + scientific-slides 4)
- [x] `--style` parameter on `slide` command
- [x] `--style` parameter on `slide-batch` command
- [x] `--list-styles` flag for style discovery

## Remaining Implementation

1. Baoyu path discovery utility
2. PaperBanana slide-deck skill (Nexus RDIV)
3. Update PaperBanana SKILL.md with new style presets documentation

## YAGNI - Explicitly Not Doing

- No EXTEND.md preference system (PaperBanana has `config.yaml`)
- No Round 2 custom dimension interaction (23 presets sufficient)
- No python-pptx dependency (use baoyu merge scripts)
- No full copy of baoyu's 9-step workflow (Nexus RDIV replaces it)

## Impact Assessment

| Dimension | Impact |
|-----------|--------|
| Context consumption | Medium - RDIV phases are bounded, nexus_state enables resume |
| Tool call volume | High during I phase (Bash for CLI + merge), manageable |
| Session continuity | Good - nexus_state.json persists across sessions |
| Cache friendliness | Good - SKILL.md stable, baoyu files read once in R phase |
| Risk | Low - baoyu cache path change requires path discovery update |
