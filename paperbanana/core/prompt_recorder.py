"""Prompt recording utilities.

PaperBanana is often debugged by inspecting the *exact* prompts sent to each agent.
This module provides a small helper that writes formatted prompts into each run's
output directory when enabled via Settings.save_prompts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import structlog

from paperbanana.core.utils import ensure_dir

logger = structlog.get_logger()


def _sanitize_filename(stem: str) -> str:
    stem = (stem or "").strip()
    if not stem:
        return "prompt"
    # Keep filenames readable and filesystem-safe.
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", stem)
    stem = stem.strip("._-")
    return stem or "prompt"


@dataclass
class PromptRecorder:
    """Write formatted prompts into the current run directory."""

    run_dir_provider: Callable[[], Path]
    subdir: str = "prompts"
    _collision_counters: dict[str, int] = field(default_factory=dict)

    def record(self, *, agent_name: str, label: Optional[str], prompt: str) -> Path:
        run_dir = self.run_dir_provider()
        prompt_dir = ensure_dir(run_dir / self.subdir)

        base = _sanitize_filename(label or agent_name)
        agent = _sanitize_filename(agent_name)
        if not base.startswith(agent):
            base = f"{agent}_{base}"

        # Avoid overwriting if called multiple times with same label.
        candidate = prompt_dir / f"{base}.txt"
        if candidate.exists():
            n = self._collision_counters.get(base, 1) + 1
            self._collision_counters[base] = n
            candidate = prompt_dir / f"{base}__{n}.txt"

        try:
            candidate.write_text(prompt, encoding="utf-8")
        except OSError as e:
            logger.warning("Failed to record prompt", agent=agent_name, label=label, error=str(e))
            raise

        return candidate

