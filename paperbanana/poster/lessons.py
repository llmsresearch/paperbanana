"""Cross-run lessons: each poster run teaches the next ones.

Every run records its observed failure signals (blocking critiques,
rejected edit ops, preflight failures, overflow rewrites) as *lessons* in
a per-user JSONL store. Subsequent runs inject the most recent distinct
lessons into the planner and stylist prompts, so run N starts where run
N-1 ended — the cheapest form of self-improvement: no training, fully
inspectable, trivially resettable (delete the file).
"""

from __future__ import annotations

import datetime
import json
import os
import re
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger()

LESSONS_DIR_ENV_VAR = "PAPERBANANA_LESSONS_DIR"
DEFAULT_LESSONS_DIR = Path.home() / ".config" / "paperbanana"
LESSONS_FILENAME = "poster_lessons.jsonl"

#: Maximum lessons injected into a prompt.
MAX_INJECTED_LESSONS = 12
#: Maximum lessons kept on disk (oldest dropped beyond this).
MAX_STORED_LESSONS = 400


def _lessons_path(lessons_dir: Optional[str | Path] = None) -> Path:
    if lessons_dir:
        base = Path(lessons_dir).expanduser()
    else:
        env = os.environ.get(LESSONS_DIR_ENV_VAR)
        base = Path(env).expanduser() if env else DEFAULT_LESSONS_DIR
    return base / LESSONS_FILENAME


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()[:240]


def record_lessons(
    run_id: str,
    venue: str,
    lessons: list[str],
    lessons_dir: Optional[str | Path] = None,
) -> int:
    """Append non-empty lessons for a finished run. Returns count written."""
    entries = [
        {
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "run_id": run_id,
            "venue": venue,
            "lesson": lesson.strip(),
        }
        for lesson in lessons
        if lesson and lesson.strip()
    ]
    if not entries:
        return 0
    path = _lessons_path(lessons_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if path.is_file():
        existing = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    combined = (existing + entries)[-MAX_STORED_LESSONS:]
    path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in combined) + "\n",
        encoding="utf-8",
    )
    logger.info("Recorded poster lessons", count=len(entries), store=str(path))
    return len(entries)


def load_lessons(
    venue: Optional[str] = None,
    limit: int = MAX_INJECTED_LESSONS,
    lessons_dir: Optional[str | Path] = None,
) -> list[str]:
    """Most recent distinct lessons, same-venue entries first."""
    path = _lessons_path(lessons_dir)
    if not path.is_file():
        return []
    entries = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    entries.reverse()  # newest first
    if venue:
        entries.sort(key=lambda e: 0 if e.get("venue") == venue else 1)
    seen: set[str] = set()
    lessons: list[str] = []
    for entry in entries:
        lesson = str(entry.get("lesson", "")).strip()
        key = _normalize(lesson)
        if not key or key in seen:
            continue
        seen.add(key)
        lessons.append(lesson)
        if len(lessons) >= limit:
            break
    return lessons


def format_lessons_block(lessons: list[str]) -> str:
    """Prompt block for the planner/stylist; empty string when no lessons."""
    if not lessons:
        return ""
    bullets = "\n".join(f"- {lesson}" for lesson in lessons)
    return (
        "\n\nLESSONS FROM PREVIOUS POSTER RUNS (recurring mistakes this system "
        "made before — actively avoid repeating them):\n" + bullets
    )
