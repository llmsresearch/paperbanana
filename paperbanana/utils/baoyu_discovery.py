"""Discover baoyu-slide-deck plugin cache path at runtime."""

from __future__ import annotations

from pathlib import Path

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
    valid = [p for p in candidates if (p / "SKILL.md").exists()]
    if not valid:
        return None
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
    from shutil import which
    if which("bun"):
        return "bun"
    if which("npx"):
        return "npx -y bun"
    raise RuntimeError("Neither bun nor npx found. Install bun: https://bun.sh")
