"""Versioned venue poster regulation bank.

A *venue spec* is one YAML file describing the poster regulations of a
venue for a given year: physical dimensions, orientation, text/DPI
minima, required elements, and file rules — each spec citing the
official source URLs it was derived from.

Layout (one file per venue per year; adding a venue = dropping a file)::

    data/venue_specs/<venue>/<year>.yaml          # built-in
    ~/.config/paperbanana/venue_specs/<venue>/<year>.yaml   # user

User dir precedence mirrors :mod:`paperbanana.guidelines.venues`:
explicit argument > ``PAPERBANANA_VENUE_SPEC_DIR`` env var > default.
On a venue/year clash the built-in spec wins.

This bank is intentionally separate from venue *style packs*
(:mod:`paperbanana.guidelines.venues`): style packs are unversioned
aesthetic guidance, specs are versioned hard regulations. Both are keyed
by the same venue name so ``--venue neurips`` resolves both.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import structlog
import yaml
from pydantic import BaseModel, Field, model_validator

from paperbanana.poster.types import Orientation, TextLevel

logger = structlog.get_logger()

DEFAULT_BUILTIN_SPEC_DIR = "data/venue_specs"
VENUE_SPEC_DIR_ENV_VAR = "PAPERBANANA_VENUE_SPEC_DIR"
DEFAULT_USER_SPEC_DIR = Path.home() / ".config" / "paperbanana" / "venue_specs"

#: Rule names preflight knows how to check; specs may only use these.
KNOWN_ELEMENT_RULES = frozenset(
    {
        "text_level_present",
        "element_kind_present",
        "text_contains",
        "panel_role_present",
    }
)


class UnknownVenueSpecError(ValueError):
    """Raised when a venue/year does not resolve to any poster spec."""

    def __init__(self, venue: str, year: Optional[int], available: dict[str, list[int]]):
        self.venue = venue
        self.year = year
        self.available = available
        listing = (
            "; ".join(
                f"{name}: {', '.join(str(y) for y in years)}"
                for name, years in sorted(available.items())
            )
            or "(none)"
        )
        wanted = f"'{venue}'" if year is None else f"'{venue}' year {year}"
        super().__init__(
            f"No poster spec for venue {wanted}. Available specs: {listing}. "
            "Add one as data/venue_specs/<venue>/<year>.yaml (built-in) or "
            f"under {DEFAULT_USER_SPEC_DIR} (user), or set {VENUE_SPEC_DIR_ENV_VAR}."
        )


class DimensionRule(BaseModel):
    """Physical size regulation.

    - ``exact``: poster must be exactly width x height
    - ``max``: poster must fit within width x height (board size)
    """

    mode: Literal["exact", "max"]
    width_mm: float = Field(gt=0)
    height_mm: float = Field(gt=0)
    orientation: Literal["portrait", "landscape", "any"] = "any"
    default_width_mm: Optional[float] = Field(
        default=None,
        gt=0,
        description="Generation target width when mode='max' (must fit the bound)",
    )
    default_height_mm: Optional[float] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_defaults(self) -> "DimensionRule":
        has_w, has_h = self.default_width_mm is not None, self.default_height_mm is not None
        if has_w != has_h:
            raise ValueError("default_width_mm and default_height_mm must be set together")
        if self.mode == "max" and not has_w:
            raise ValueError("mode='max' requires default_width_mm/default_height_mm")
        if has_w:
            assert self.default_width_mm is not None and self.default_height_mm is not None
            if self.default_width_mm > self.width_mm or self.default_height_mm > self.height_mm:
                raise ValueError(
                    f"default size {self.default_width_mm}x{self.default_height_mm}mm "
                    f"exceeds the {self.mode} bound {self.width_mm}x{self.height_mm}mm"
                )
        return self

    def generation_size_mm(self) -> tuple[float, float]:
        """The (width, height) a generated poster should target."""
        if self.default_width_mm is not None and self.default_height_mm is not None:
            return self.default_width_mm, self.default_height_mm
        return self.width_mm, self.height_mm

    def generation_orientation(self) -> Orientation:
        w, h = self.generation_size_mm()
        return "landscape" if w >= h else "portrait"


class TextRules(BaseModel):
    """Legibility minima for text and embedded images."""

    min_pt: dict[TextLevel, float] = Field(default_factory=dict)
    min_image_dpi: int = Field(default=100, gt=0)


class RequiredElementRule(BaseModel):
    """A machine-checkable required element, dispatched by rule name."""

    id: str
    description: str
    rule: str
    params: dict = Field(default_factory=dict)
    severity: Literal["fail", "warn"] = "fail"

    @model_validator(mode="after")
    def validate_rule(self) -> "RequiredElementRule":
        if self.rule not in KNOWN_ELEMENT_RULES:
            raise ValueError(
                f"required element '{self.id}' uses unknown rule '{self.rule}'. "
                f"Known rules: {sorted(KNOWN_ELEMENT_RULES)}"
            )
        return self


class FileRules(BaseModel):
    """Output file constraints."""

    pdf_max_mb: Optional[float] = Field(default=None, gt=0)
    png_max_px: Optional[int] = Field(default=None, gt=0)


class VenueSpec(BaseModel):
    """Poster regulations for one venue and year."""

    spec_version: int = 1
    venue: str
    year: int = Field(ge=2000, le=2100)
    display_name: str
    sources: list[str] = Field(min_length=1, description="Official source URLs (citable)")
    dimensions: DimensionRule
    text_rules: TextRules = TextRules()
    required_elements: list[RequiredElementRule] = Field(default_factory=list)
    file_rules: FileRules = FileRules()
    notes: Optional[str] = None

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_spec(self) -> "VenueSpec":
        rule_ids = [r.id for r in self.required_elements]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError(f"duplicate required_elements ids: {rule_ids}")
        return self


def resolve_user_spec_dir(extra_dir: str | Path | None = None) -> Path:
    """Resolve the user venue-spec directory.

    Precedence: explicit ``extra_dir`` > ``PAPERBANANA_VENUE_SPEC_DIR`` env
    var > ``~/.config/paperbanana/venue_specs``.
    """
    if extra_dir:
        return Path(extra_dir).expanduser()
    env_dir = os.environ.get(VENUE_SPEC_DIR_ENV_VAR)
    if env_dir:
        return Path(env_dir).expanduser()
    return DEFAULT_USER_SPEC_DIR


def _scan_spec_files(root: Path) -> dict[str, dict[int, Path]]:
    """Map venue name -> {year -> yaml path} for all specs under root."""
    specs: dict[str, dict[int, Path]] = {}
    if not root.is_dir():
        return specs
    for venue_dir in sorted(root.iterdir()):
        if not venue_dir.is_dir():
            continue
        name = venue_dir.name.lower()
        for yaml_path in sorted(venue_dir.glob("*.yaml")):
            try:
                year = int(yaml_path.stem)
            except ValueError:
                raise ValueError(
                    f"Venue spec filename must be a year (e.g. 2025.yaml): {yaml_path}"
                ) from None
            specs.setdefault(name, {})[year] = yaml_path
    return specs


def _all_specs(
    builtin_dir: str | Path | None = None,
    extra_dir: str | Path | None = None,
) -> dict[str, dict[int, Path]]:
    """Merged spec map; built-in wins on venue/year clash."""
    merged = _scan_spec_files(resolve_user_spec_dir(extra_dir))
    base = Path(builtin_dir) if builtin_dir else Path(DEFAULT_BUILTIN_SPEC_DIR)
    for name, years in _scan_spec_files(base).items():
        for year, path in years.items():
            if year in merged.get(name, {}):
                logger.warning(
                    "User venue spec is shadowed by a built-in spec",
                    venue=name,
                    year=year,
                    user_spec=str(merged[name][year]),
                )
            merged.setdefault(name, {})[year] = path
    return merged


def list_venue_specs(
    builtin_dir: str | Path | None = None,
    extra_dir: str | Path | None = None,
) -> dict[str, list[int]]:
    """List available specs as venue name -> sorted years."""
    return {
        name: sorted(years) for name, years in sorted(_all_specs(builtin_dir, extra_dir).items())
    }


def load_venue_spec(
    venue: str,
    year: Optional[int] = None,
    builtin_dir: str | Path | None = None,
    extra_dir: str | Path | None = None,
) -> VenueSpec:
    """Load the poster spec for a venue.

    Args:
        venue: Venue name (case-insensitive), e.g. ``neurips``.
        year: Spec year; ``None`` selects the latest available year.
        builtin_dir: Built-in spec directory (default ``data/venue_specs``).
        extra_dir: User spec directory override.

    Raises:
        UnknownVenueSpecError: If the venue (or venue/year) has no spec.
        ValueError: If the YAML is malformed or fails schema validation.
    """
    normalized = venue.strip().lower()
    specs = _all_specs(builtin_dir, extra_dir)
    available = {name: sorted(years) for name, years in specs.items()}
    if normalized not in specs:
        raise UnknownVenueSpecError(normalized, year, available)
    years = specs[normalized]
    selected = max(years) if year is None else year
    if selected not in years:
        raise UnknownVenueSpecError(normalized, year, available)
    path = years[selected]

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid venue spec (expected a mapping at top level): {path}")
    spec = VenueSpec(**raw)
    if spec.venue.lower() != normalized or spec.year != selected:
        raise ValueError(
            f"Venue spec {path} declares venue='{spec.venue}' year={spec.year}, "
            f"but its location implies venue='{normalized}' year={selected}"
        )
    return spec
