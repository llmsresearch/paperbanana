"""Venue spec bank tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from paperbanana.poster.venue_spec import (
    DimensionRule,
    UnknownVenueSpecError,
    VenueSpec,
    list_venue_specs,
    load_venue_spec,
)

BUILTIN_DIR = Path(__file__).resolve().parents[2] / "data" / "venue_specs"


def test_builtin_bank_loads_all_specs():
    specs = list_venue_specs(builtin_dir=BUILTIN_DIR, extra_dir="/nonexistent")
    assert {"neurips", "icml", "cvpr", "acl"} <= set(specs)
    for venue, years in specs.items():
        for year in years:
            spec = load_venue_spec(venue, year, builtin_dir=BUILTIN_DIR, extra_dir="/nonexistent")
            assert spec.sources, f"{venue}/{year} must cite sources"
            w, h = spec.dimensions.generation_size_mm()
            assert w > 0 and h > 0


def test_neurips_spec_values():
    spec = load_venue_spec("neurips", builtin_dir=BUILTIN_DIR, extra_dir="/nonexistent")
    assert spec.year == 2025
    assert spec.dimensions.mode == "max"
    assert spec.dimensions.width_mm == pytest.approx(2438.4)  # 96 in board
    assert spec.dimensions.generation_size_mm() == (1219.2, 914.4)
    assert spec.dimensions.generation_orientation() == "landscape"


def test_cvpr_exact_dimensions():
    spec = load_venue_spec("cvpr", builtin_dir=BUILTIN_DIR, extra_dir="/nonexistent")
    assert spec.dimensions.mode == "exact"
    assert spec.dimensions.generation_size_mm() == (2133.6, 1066.8)


def test_acl_portrait():
    spec = load_venue_spec("acl", builtin_dir=BUILTIN_DIR, extra_dir="/nonexistent")
    assert spec.dimensions.generation_orientation() == "portrait"
    assert spec.dimensions.width_mm == 841.0


def test_latest_year_selected_when_year_omitted(tmp_path: Path):
    venue_dir = tmp_path / "testvenue"
    venue_dir.mkdir()
    for year in (2023, 2025):
        (venue_dir / f"{year}.yaml").write_text(_spec_yaml("testvenue", year), encoding="utf-8")
    spec = load_venue_spec("testvenue", builtin_dir=tmp_path, extra_dir="/nonexistent")
    assert spec.year == 2025
    spec_2023 = load_venue_spec("testvenue", 2023, builtin_dir=tmp_path, extra_dir="/nonexistent")
    assert spec_2023.year == 2023


def test_unknown_venue_lists_available():
    with pytest.raises(UnknownVenueSpecError, match="neurips"):
        load_venue_spec("ispor", builtin_dir=BUILTIN_DIR, extra_dir="/nonexistent")


def test_unknown_year_raises():
    with pytest.raises(UnknownVenueSpecError):
        load_venue_spec("neurips", 1999, builtin_dir=BUILTIN_DIR, extra_dir="/nonexistent")


def test_user_dir_specs_are_picked_up(tmp_path: Path):
    user_dir = tmp_path / "user_specs"
    (user_dir / "myconf").mkdir(parents=True)
    (user_dir / "myconf" / "2026.yaml").write_text(_spec_yaml("myconf", 2026), encoding="utf-8")
    spec = load_venue_spec("myconf", builtin_dir=BUILTIN_DIR, extra_dir=user_dir)
    assert spec.display_name == "Myconf 2026"


def test_spec_location_mismatch_raises(tmp_path: Path):
    venue_dir = tmp_path / "alpha"
    venue_dir.mkdir()
    (venue_dir / "2025.yaml").write_text(_spec_yaml("beta", 2025), encoding="utf-8")
    with pytest.raises(ValueError, match="declares venue"):
        load_venue_spec("alpha", builtin_dir=tmp_path, extra_dir="/nonexistent")


def test_spec_rejects_missing_sources():
    with pytest.raises(ValueError):
        VenueSpec(
            venue="x",
            year=2025,
            display_name="X",
            sources=[],
            dimensions=DimensionRule(mode="exact", width_mm=841, height_mm=1189),
        )


def test_spec_rejects_unknown_element_rule(tmp_path: Path):
    venue_dir = tmp_path / "badrule"
    venue_dir.mkdir()
    bad = _spec_yaml("badrule", 2025).replace("rule: text_level_present", "rule: not_a_rule")
    (venue_dir / "2025.yaml").write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="unknown rule"):
        load_venue_spec("badrule", builtin_dir=tmp_path, extra_dir="/nonexistent")


def test_max_mode_requires_default_size():
    with pytest.raises(ValueError, match="default_width_mm"):
        DimensionRule(mode="max", width_mm=2438.4, height_mm=1219.2)


def test_default_size_must_fit_bound():
    with pytest.raises(ValueError, match="exceeds"):
        DimensionRule(
            mode="max",
            width_mm=1000,
            height_mm=900,
            default_width_mm=1200,
            default_height_mm=900,
        )


def test_bad_year_filename_raises(tmp_path: Path):
    venue_dir = tmp_path / "oops"
    venue_dir.mkdir()
    (venue_dir / "latest.yaml").write_text(_spec_yaml("oops", 2025), encoding="utf-8")
    with pytest.raises(ValueError, match="year"):
        list_venue_specs(builtin_dir=tmp_path, extra_dir="/nonexistent")


def _spec_yaml(venue: str, year: int) -> str:
    return f"""
spec_version: 1
venue: {venue}
year: {year}
display_name: "{venue.title()} {year}"
sources:
  - "https://example.org/{venue}/{year}/posters"
dimensions:
  mode: exact
  width_mm: 841.0
  height_mm: 1189.0
  orientation: portrait
text_rules:
  min_pt: {{}}
  min_image_dpi: 100
required_elements:
  - id: paper_title
    description: "Title present"
    rule: text_level_present
    params: {{ level: title }}
file_rules: {{}}
"""
