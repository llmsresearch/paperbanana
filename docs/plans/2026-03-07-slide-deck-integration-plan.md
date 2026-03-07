# PaperBanana Slide-Deck Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Give PaperBanana end-to-end slide deck creation capability via Nexus RDIV orchestration, referencing baoyu assets at runtime.

**Architecture:** PaperBanana CLI handles image generation + style presets (already done). A new utility discovers baoyu's plugin cache path for runtime asset access. A Nexus RDIV skill orchestrates the full workflow: content analysis, interactive style selection, outline/prompt generation, image batch generation, PPTX/PDF merge via baoyu scripts.

**Tech Stack:** Python (PaperBanana CLI), Claude skill (Nexus RDIV), bun (baoyu merge scripts), Glob/Read (S1 runtime referencing)

---

## Pre-flight: Already Completed

- [x] `paperbanana/guidelines/slide_styles.py` — 23 style presets with `get_style_prompt()`, `list_styles()`, `match_style()` API
- [x] `--style` / `--list-styles` on `slide` command
- [x] `--style` on `slide-batch` command

---

### Task 1: Baoyu Path Discovery Utility

**Files:**
- Create: `paperbanana/utils/baoyu_discovery.py`
- Test: `tests/test_baoyu_discovery.py`

**Context:** baoyu-slide-deck lives in `~/.claude/plugins/cache/baoyu-skills/content-skills/*/skills/baoyu-slide-deck/`. Multiple cache versions may exist (seen 11 copies). We need a function that finds the latest one.

**Step 1: Write the failing test**

```python
"""Tests for baoyu path discovery."""

from __future__ import annotations

from unittest.mock import patch
from pathlib import Path

from paperbanana.utils.baoyu_discovery import discover_baoyu_slide_deck


def test_discover_returns_none_when_no_cache():
    """Returns None when baoyu plugin cache doesn't exist."""
    with patch("paperbanana.utils.baoyu_discovery.BAOYU_CACHE_GLOB", "/nonexistent/*/skills/baoyu-slide-deck"):
        result = discover_baoyu_slide_deck()
    assert result is None


def test_discover_returns_path_with_skill_md():
    """Returns a path that contains SKILL.md when cache exists."""
    result = discover_baoyu_slide_deck()
    if result is not None:
        assert (result / "SKILL.md").exists()
        assert (result / "references" / "styles").is_dir()


def test_discover_styles_dir():
    """Discovered path has references/styles/ with .md files."""
    result = discover_baoyu_slide_deck()
    if result is not None:
        styles = list((result / "references" / "styles").glob("*.md"))
        assert len(styles) >= 14  # baoyu has 16 style files
```

**Step 2: Run test to verify it fails**

Run: `cd E:/VSCode_Project/paperbanana && python -m pytest tests/test_baoyu_discovery.py -v`
Expected: FAIL with "No module named 'paperbanana.utils.baoyu_discovery'"

**Step 3: Write minimal implementation**

```python
"""Discover baoyu-slide-deck plugin cache path at runtime."""

from __future__ import annotations

import os
from pathlib import Path

# Glob pattern for baoyu slide-deck cache (cross-platform)
_HOME = Path.home()
BAOYU_CACHE_GLOB = str(
    _HOME / ".claude" / "plugins" / "cache" / "baoyu-skills"
    / "content-skills" / "*" / "skills" / "baoyu-slide-deck"
)


def discover_baoyu_slide_deck() -> Path | None:
    """Find the latest baoyu-slide-deck plugin cache directory.

    Multiple cache versions may exist. Returns the most recently modified
    one that contains a valid SKILL.md, or None if not found.
    """
    from glob import glob

    candidates = [Path(p) for p in glob(BAOYU_CACHE_GLOB)]
    # Filter to valid installations (must have SKILL.md)
    valid = [p for p in candidates if (p / "SKILL.md").exists()]
    if not valid:
        return None
    # Return most recently modified
    return max(valid, key=lambda p: p.stat().st_mtime)


def get_baoyu_styles_dir() -> Path | None:
    """Return path to baoyu's references/styles/ directory, or None."""
    base = discover_baoyu_slide_deck()
    if base is None:
        return None
    styles_dir = base / "references" / "styles"
    return styles_dir if styles_dir.is_dir() else None


def get_baoyu_scripts_dir() -> Path | None:
    """Return path to baoyu's scripts/ directory (merge-to-pptx.ts etc), or None."""
    base = discover_baoyu_slide_deck()
    if base is None:
        return None
    scripts_dir = base / "scripts"
    return scripts_dir if scripts_dir.is_dir() else None


def get_bun_executable() -> str:
    """Return bun or npx fallback for running TypeScript scripts."""
    # Check bun first
    from shutil import which
    if which("bun"):
        return "bun"
    if which("npx"):
        return "npx -y bun"
    raise RuntimeError("Neither bun nor npx found. Install bun: https://bun.sh")
```

**Step 4: Run test to verify it passes**

Run: `cd E:/VSCode_Project/paperbanana && python -m pytest tests/test_baoyu_discovery.py -v`
Expected: PASS (all 3 tests)

**Step 5: Commit**

```bash
cd E:/VSCode_Project/paperbanana
git add paperbanana/utils/baoyu_discovery.py tests/test_baoyu_discovery.py
git commit -m "feat(slide-deck): add baoyu plugin cache path discovery utility"
```

---

### Task 2: Test slide_styles.py (already created, needs tests)

**Files:**
- Test: `tests/test_slide_styles.py`

**Step 1: Write tests**

```python
"""Tests for slide style presets."""

from __future__ import annotations

import pytest

from paperbanana.guidelines.slide_styles import (
    get_style_prompt,
    get_style_info,
    list_styles,
    match_style,
)


def test_list_styles_returns_23_presets():
    styles = list_styles()
    assert len(styles) == 23
    assert "blueprint" in styles
    assert "tech-keynote" in styles
    assert "scientific" in styles


def test_get_style_prompt_returns_string():
    prompt = get_style_prompt("blueprint")
    assert "Blueprint" in prompt
    assert "Color Palette" in prompt


def test_get_style_prompt_case_insensitive():
    assert get_style_prompt("Blueprint") == get_style_prompt("blueprint")


def test_get_style_prompt_unknown_raises():
    with pytest.raises(KeyError, match="Unknown slide style"):
        get_style_prompt("nonexistent-style")


def test_get_style_info_has_required_keys():
    info = get_style_info("scientific")
    assert "source" in info
    assert "feel" in info
    assert "prompt" in info
    assert info["source"] == "baoyu-slide-deck"


def test_match_style_finds_scientific():
    result = match_style("biology and medical imaging research")
    assert result == "scientific"


def test_match_style_finds_ml_ai():
    result = match_style("deep learning transformer neural network")
    assert result == "ml-ai"


def test_match_style_returns_none_for_garbage():
    result = match_style("xyzzy foobar")
    assert result is None
```

**Step 2: Run tests**

Run: `cd E:/VSCode_Project/paperbanana && python -m pytest tests/test_slide_styles.py -v`
Expected: PASS

**Step 3: Commit**

```bash
cd E:/VSCode_Project/paperbanana
git add tests/test_slide_styles.py
git commit -m "test(slide-deck): add tests for slide style presets"
```

---

### Task 3: CLI --list-styles Test

**Files:**
- Modify: `tests/test_cli.py` (append)

**Step 1: Write test**

```python
def test_slide_list_styles():
    """paperbanana slide --list-styles shows all presets."""
    result = runner.invoke(
        app,
        ["slide", "--input", "dummy", "--list-styles"],
    )
    assert result.exit_code == 0
    assert "blueprint" in result.output
    assert "tech-keynote" in result.output
    assert "23 styles" in result.output
```

**Step 2: Run test**

Run: `cd E:/VSCode_Project/paperbanana && python -m pytest tests/test_cli.py::test_slide_list_styles -v`
Expected: PASS (already implemented)

**Step 3: Commit**

```bash
cd E:/VSCode_Project/paperbanana
git add tests/test_cli.py
git commit -m "test(cli): add --list-styles test for slide command"
```

---

### Task 4: Update SKILL.md with Style Presets Documentation

**Files:**
- Modify: `~/.claude/skills/paperbanana/SKILL.md`

**Step 1: Add style preset section**

After the `slide` command's parameter table, add:

```markdown
### Style Presets (23 available)

Use `--style <name>` with `slide` or `slide-batch` commands. Use `--list-styles` to see all.

| Style | Source | Best For |
|-------|--------|----------|
| `blueprint` | baoyu | Architecture, system design, technical |
| `chalkboard` | baoyu | Classroom, teaching, education |
| `corporate` | baoyu | Business, investor, quarterly reports |
| `minimal` | baoyu | Executive briefings, clean/simple |
| `sketch-notes` | baoyu | Tutorials, guides, beginner content |
| `watercolor` | baoyu | Lifestyle, wellness, artistic |
| `dark-atmospheric` | baoyu | Entertainment, gaming, cinematic |
| `notion` | baoyu | SaaS, product, dashboards |
| `bold-editorial` | baoyu | Product launches, keynotes, marketing |
| `editorial-infographic` | baoyu | Science communication, explainers |
| `fantasy-animation` | baoyu | Storytelling, magical, children |
| `intuition-machine` | baoyu | Academic research, bilingual |
| `pixel-art` | baoyu | Gaming, retro, developer culture |
| `scientific` | baoyu | Biology, chemistry, medical |
| `vector-illustration` | baoyu | Creative, children, flat design |
| `vintage` | baoyu | Historical, heritage, expedition |
| `tech-keynote` | elite-ppt | Apple/Tesla premium minimalism |
| `creative-bold` | elite-ppt | Google/Airbnb energetic innovation |
| `financial-elite` | elite-ppt | Goldman Sachs/McKinsey sophistication |
| `biotech` | sci-slides | Life sciences, genomics |
| `neuroscience` | sci-slides | Brain research, cognitive science |
| `ml-ai` | sci-slides | Machine learning, deep learning |
| `environmental` | sci-slides | Ecology, climate, sustainability |
```

**Step 2: Verify skill loads**

Invoke `paperbanana` skill in Claude Code to verify the new section renders.

**Step 3: Commit**

```bash
cd ~/.claude/skills/paperbanana
git add SKILL.md  # if tracked
```

---

### Task 5: PaperBanana Slide-Deck Nexus Skill

**Files:**
- Create: `E:/VSCode_Project/.claude/skills/paperbanana-slide-deck/SKILL.md`

**Context:** This is a Claude skill (not Python code) that orchestrates the end-to-end slide deck workflow using Nexus RDIV. It references baoyu assets at runtime (S1) and calls PaperBanana CLI for image generation.

**Step 1: Create the skill**

The skill file should contain:
1. Nexus RDIV metadata (phases, routing)
2. R phase: content analysis, baoyu style discovery via Glob/Read, interactive style selection via AskUserQuestion
3. D phase: outline generation, prompt writing with style injection
4. I phase: `paperbanana slide-batch` + `bun merge-to-pptx.ts`
5. V phase: image review, selective regeneration, re-merge
6. Fallback logic: use `slide_styles.py` presets when baoyu unavailable
7. Output directory structure specification

Key implementation details:
- Baoyu styles discovery: `Glob ~/.claude/plugins/cache/baoyu-skills/content-skills/*/skills/baoyu-slide-deck/references/styles/*.md`
- Take the path with latest mtime
- Read style files for style recommendation
- Use `match_style()` from `slide_styles.py` for auto-selection as fallback
- Merge command: `bun ${BAOYU_DIR}/scripts/merge-to-pptx.ts <slide-deck-dir>`

**Step 2: Test skill loads in Claude Code**

Invoke `/paperbanana-slide-deck` or reference it — verify it loads and shows the workflow.

**Step 3: Commit**

```bash
cd E:/VSCode_Project
git add .claude/skills/paperbanana-slide-deck/SKILL.md
git commit -m "feat(skill): add paperbanana-slide-deck Nexus RDIV skill"
```

---

### Task 6: Update Self-Built Skills Registry

**Files:**
- Modify: `E:/VSCode_Project/雷宇轩的自建skills.md`

**Step 1: Add new skill entry**

Add `paperbanana-slide-deck` to the self-built skills list with:
- Name, version, scope (project-level)
- Description: PaperBanana slide deck orchestrator via Nexus RDIV
- Dependencies: baoyu-slide-deck (runtime S1), PaperBanana CLI

**Step 2: Update MEMORY.md skill count**

Update count from 13 to 14 self-built skills.

**Step 3: Commit**

```bash
cd E:/VSCode_Project
git add "雷宇轩的自建skills.md"
git commit -m "docs: register paperbanana-slide-deck in self-built skills"
```

---

## Execution Order & Dependencies

```
Task 1 (baoyu discovery) ─┐
Task 2 (style tests)      ├─> Task 4 (SKILL.md update) ─> Task 5 (Nexus skill) ─> Task 6 (registry)
Task 3 (CLI test)         ─┘
```

Tasks 1, 2, 3 are independent and can run in parallel.
Task 4 depends on Tasks 1-3 being committed.
Task 5 depends on Task 4 (needs to reference style list in skill).
Task 6 is always last.
