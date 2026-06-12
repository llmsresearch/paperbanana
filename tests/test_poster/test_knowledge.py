"""Tests for the learning layers: schema retrieval and the lessons loop."""

from __future__ import annotations

from pathlib import Path

import pytest

from paperbanana.poster.lessons import (
    format_lessons_block,
    load_lessons,
    record_lessons,
)
from paperbanana.poster.schemas import (
    format_layout_priors,
    load_schema_library,
    select_schema,
)

REPO = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO / "data" / "poster_schemas" / "schemas.json"


def test_committed_schema_library_loads():
    library = load_schema_library(SCHEMA_PATH)
    assert library.n_posters > 7000
    assert {s.orientation for s in library.schemas} == {"landscape", "portrait"}
    # The corpus's strongest finding: portrait posters are 2-column by default.
    portrait = select_schema(library, "portrait")
    assert portrait.columns == 2
    landscape = select_schema(library, "landscape", min_columns=3)
    assert landscape.columns == 3


def test_schema_priors_block_mentions_data():
    library = load_schema_library(SCHEMA_PATH)
    schema = select_schema(library, "landscape", min_columns=3)
    block = format_layout_priors(library, schema)
    assert "REAL LANDSCAPE POSTERS" in block
    assert str(library.n_posters) in block
    assert "columns" in block


def test_missing_library_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="schema library"):
        load_schema_library(tmp_path / "nope.json")


def test_lessons_roundtrip(tmp_path: Path):
    n = record_lessons(
        "run_a", "neurips", ["Critic flagged: tables too dense", ""], lessons_dir=tmp_path
    )
    assert n == 1
    record_lessons(
        "run_b", "acl", ["Portrait posters need bigger title band"], lessons_dir=tmp_path
    )
    lessons = load_lessons(venue="neurips", lessons_dir=tmp_path)
    assert lessons[0].startswith("Critic flagged")  # same-venue first
    assert len(lessons) == 2


def test_lessons_dedupe_and_limit(tmp_path: Path):
    for i in range(20):
        record_lessons(f"run_{i}", "neurips", ["Same lesson  every   time"], lessons_dir=tmp_path)
    record_lessons("run_x", "neurips", ["A different lesson"], lessons_dir=tmp_path)
    lessons = load_lessons(venue="neurips", lessons_dir=tmp_path)
    assert len(lessons) == 2  # deduped


def test_lessons_block_formatting(tmp_path: Path):
    assert format_lessons_block([]) == ""
    block = format_lessons_block(["Do not shrink type below minima"])
    assert "LESSONS FROM PREVIOUS POSTER RUNS" in block
    assert "- Do not shrink type" in block


def test_no_lessons_file_is_empty(tmp_path: Path):
    assert load_lessons(lessons_dir=tmp_path) == []
