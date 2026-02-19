"""Resume support — load state from a previous run to continue iterating."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import structlog
from pydantic import BaseModel

from paperbanana.core.types import DiagramType

logger = structlog.get_logger()


def find_latest_run(output_dir: str) -> str:
    """Find the most recent run ID in the output directory.

    Args:
        output_dir: Base output directory (e.g. "outputs").

    Returns:
        The run ID (directory name) of the latest run.

    Raises:
        FileNotFoundError: If no runs exist.
    """
    out = Path(output_dir)
    if not out.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    runs = sorted(
        [d.name for d in out.iterdir() if d.is_dir() and d.name.startswith("run_")],
    )
    if not runs:
        raise FileNotFoundError(f"No runs found in {output_dir}")

    return runs[-1]


class ResumeState(BaseModel):
    """State loaded from a previous run, sufficient to continue iterating."""

    run_dir: str
    run_id: str
    source_context: str
    communicative_intent: str
    diagram_type: DiagramType
    raw_data: Optional[dict[str, Any]] = None
    last_description: str
    last_iteration: int
    last_image_path: Optional[str] = None


def load_resume_state(output_dir: str, run_id: str) -> ResumeState:
    """Load state from a previous run directory.

    Args:
        output_dir: Base output directory (e.g. "outputs").
        run_id: The run ID (directory name under output_dir).

    Returns:
        ResumeState with all info needed to continue.

    Raises:
        FileNotFoundError: If run directory or required files don't exist.
        ValueError: If run state is incomplete.
    """
    run_dir = Path(output_dir) / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # Load original input
    input_path = run_dir / "run_input.json"
    if not input_path.exists():
        raise FileNotFoundError(
            f"run_input.json not found in {run_dir}. "
            f"Only runs created with PaperBanana >= 0.2.0 can be continued."
        )

    with open(input_path) as f:
        run_input = json.load(f)

    # Find the latest iteration
    iter_dirs = sorted(
        [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("iter_")],
        key=lambda d: int(d.name.split("_")[1]),
    )

    if not iter_dirs:
        # No iterations yet — use planning output
        planning_path = run_dir / "planning.json"
        if not planning_path.exists():
            raise ValueError(f"No iterations or planning.json found in {run_dir}")
        with open(planning_path) as f:
            planning = json.load(f)
        last_description = planning.get("optimized_description", "")
        last_iteration = 0
        last_image_path = None
    else:
        last_iter_dir = iter_dirs[-1]
        last_iteration = int(last_iter_dir.name.split("_")[1])
        details_path = last_iter_dir / "details.json"

        with open(details_path) as f:
            details = json.load(f)

        critique = details.get("critique", {})
        last_description = critique.get("revised_description") or details.get("description", "")

        # Find the last image
        last_image_path = None
        for candidate in [
            run_dir / f"diagram_iter_{last_iteration}.png",
            run_dir / f"diagram_iter_{last_iteration}.jpg",
        ]:
            if candidate.exists():
                last_image_path = str(candidate)
                break

    logger.info(
        "Loaded resume state",
        run_id=run_id,
        last_iteration=last_iteration,
        description_len=len(last_description),
    )

    return ResumeState(
        run_dir=str(run_dir),
        run_id=run_id,
        source_context=run_input["source_context"],
        communicative_intent=run_input["communicative_intent"],
        diagram_type=DiagramType(run_input["diagram_type"]),
        raw_data=run_input.get("raw_data"),
        last_description=last_description,
        last_iteration=last_iteration,
        last_image_path=last_image_path,
    )
