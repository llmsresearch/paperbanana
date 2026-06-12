"""Poster generation head: paper PDF -> venue-compliant conference poster.

Public surface:

- :class:`paperbanana.poster.types.PosterIR` — the editable intermediate
  representation of a poster (single source of truth for rendering).
- :class:`paperbanana.poster.venue_spec.VenueSpec` — versioned venue
  poster regulations loaded from ``data/venue_specs/``.
- :class:`paperbanana.poster.pipeline.PosterPipeline` — the end-to-end
  generation pipeline.
"""

from paperbanana.poster.types import PosterIR, PosterOutput
from paperbanana.poster.venue_spec import VenueSpec, load_venue_spec

__all__ = ["PosterIR", "PosterOutput", "VenueSpec", "load_venue_spec"]
