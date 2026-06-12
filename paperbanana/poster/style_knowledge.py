"""Poster design knowledge loading.

Follows the venue style pack convention of
:mod:`paperbanana.guidelines.venues`: a flat default guide ships at
``data/guidelines/poster_style_guide.md``; a venue pack may override it
with its own ``poster_style_guide.md`` (built-in pack dir or user pack
dir). This is the hook where a future poster exemplar corpus plugs in —
synthesized per-venue design guidance lands in the same file.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from paperbanana.guidelines.venues import UnknownVenueError, resolve_venue

logger = structlog.get_logger()

POSTER_GUIDE_FILENAME = "poster_style_guide.md"


def load_poster_style_guide(
    guidelines_path: str | Path = "data/guidelines",
    venue: str | None = None,
    venue_dir: str | Path | None = None,
) -> str:
    """Load the poster design guide, preferring a venue pack override.

    Resolution order: ``<venue pack dir>/poster_style_guide.md`` (when the
    venue has a style pack) > flat ``<guidelines_path>/poster_style_guide.md``.

    Raises:
        FileNotFoundError: If no guide exists at either location — the
            default guide ships with the package, so this indicates a
            broken installation rather than a normal condition.
    """
    base = Path(guidelines_path)
    candidates: list[Path] = []
    if venue and venue != "custom":
        try:
            pack = resolve_venue(venue, builtin_dir=base, extra_dir=venue_dir)
            candidates.append(pack.dir / POSTER_GUIDE_FILENAME)
        except UnknownVenueError:
            logger.info("No style pack for venue; using flat poster guide", venue=venue)
    candidates.append(base / POSTER_GUIDE_FILENAME)

    for path in candidates:
        if path.is_file():
            logger.info("Loaded poster style guide", path=str(path))
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"poster style guide not found; looked at: {', '.join(str(c) for c in candidates)}. "
        "The default guide ships at data/guidelines/poster_style_guide.md — "
        "check the installation or the --config guidelines path."
    )
