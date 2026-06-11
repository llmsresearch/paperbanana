"""Tests for the reference store."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from paperbanana.core.types import ReferenceExample
from paperbanana.reference.store import ReferenceStore


def test_load_from_directory():
    """Test loading references from a directory with index.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create index.json
        data = {
            "metadata": {"name": "test"},
            "examples": [
                {
                    "id": "ref_001",
                    "source_context": "Test context",
                    "caption": "Test caption",
                    "image_path": "images/test.png",
                    "category": "test",
                }
            ],
        }
        Path(tmpdir, "index.json").write_text(json.dumps(data))

        store = ReferenceStore(tmpdir)
        examples = store.get_all()

        assert len(examples) == 1
        assert examples[0].id == "ref_001"
        assert store.count == 1


def test_load_missing_directory():
    """Test loading from a nonexistent directory."""
    store = ReferenceStore("/nonexistent/path")
    assert store.get_all() == []
    assert store.count == 0


def test_get_by_category():
    """Test filtering by category."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data = {
            "examples": [
                {
                    "id": "r1",
                    "source_context": "c1",
                    "caption": "c1",
                    "image_path": "i1",
                    "category": "agent",
                },
                {
                    "id": "r2",
                    "source_context": "c2",
                    "caption": "c2",
                    "image_path": "i2",
                    "category": "vision",
                },
                {
                    "id": "r3",
                    "source_context": "c3",
                    "caption": "c3",
                    "image_path": "i3",
                    "category": "agent",
                },
            ],
        }
        Path(tmpdir, "index.json").write_text(json.dumps(data))

        store = ReferenceStore(tmpdir)
        agents = store.get_by_category("agent")
        assert len(agents) == 2


def test_get_by_categories_multiple():
    """get_by_categories returns examples matching any of the supplied categories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data = {
            "examples": [
                {
                    "id": "r1",
                    "source_context": "c1",
                    "caption": "c1",
                    "image_path": "i1",
                    "category": "agent_reasoning",
                },
                {
                    "id": "r2",
                    "source_context": "c2",
                    "caption": "c2",
                    "image_path": "i2",
                    "category": "vision_perception",
                },
                {
                    "id": "r3",
                    "source_context": "c3",
                    "caption": "c3",
                    "image_path": "i3",
                    "category": "nlp_language",
                },
            ],
        }
        Path(tmpdir, "index.json").write_text(json.dumps(data))

        store = ReferenceStore(tmpdir)
        selected = store.get_by_categories(["agent_reasoning", "nlp_language"])
        assert {e.id for e in selected} == {"r1", "r3"}


def test_get_by_categories_empty_returns_empty():
    """get_by_categories([]) yields no matches because the filter excludes every category."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data = {
            "examples": [
                {
                    "id": "r1",
                    "source_context": "c1",
                    "caption": "c1",
                    "image_path": "i1",
                    "category": "agent_reasoning",
                }
            ],
        }
        Path(tmpdir, "index.json").write_text(json.dumps(data))

        store = ReferenceStore(tmpdir)
        assert store.get_by_categories([]) == []


def test_get_by_categories_unknown_category():
    """Unknown categories are silently ignored and produce no matches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data = {
            "examples": [
                {
                    "id": "r1",
                    "source_context": "c1",
                    "caption": "c1",
                    "image_path": "i1",
                    "category": "agent_reasoning",
                }
            ],
        }
        Path(tmpdir, "index.json").write_text(json.dumps(data))

        store = ReferenceStore(tmpdir)
        assert store.get_by_categories(["not_a_real_category"]) == []


def test_get_by_categories_skips_examples_without_category():
    """Examples whose category is None must not be matched by any filter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data = {
            "examples": [
                {
                    "id": "r1",
                    "source_context": "c1",
                    "caption": "c1",
                    "image_path": "i1",
                },  # no category field
                {
                    "id": "r2",
                    "source_context": "c2",
                    "caption": "c2",
                    "image_path": "i2",
                    "category": "vision_perception",
                },
            ],
        }
        Path(tmpdir, "index.json").write_text(json.dumps(data))

        store = ReferenceStore(tmpdir)
        selected = store.get_by_categories(["vision_perception"])
        assert [e.id for e in selected] == ["r2"]


def test_available_categories_returns_sorted_unique():
    """available_categories returns a sorted, de-duplicated list of non-null categories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data = {
            "examples": [
                {
                    "id": "r1",
                    "source_context": "c1",
                    "caption": "c1",
                    "image_path": "i1",
                    "category": "vision_perception",
                },
                {
                    "id": "r2",
                    "source_context": "c2",
                    "caption": "c2",
                    "image_path": "i2",
                    "category": "agent_reasoning",
                },
                {
                    "id": "r3",
                    "source_context": "c3",
                    "caption": "c3",
                    "image_path": "i3",
                    "category": "vision_perception",
                },
                {
                    "id": "r4",
                    "source_context": "c4",
                    "caption": "c4",
                    "image_path": "i4",
                },  # no category
            ],
        }
        Path(tmpdir, "index.json").write_text(json.dumps(data))

        store = ReferenceStore(tmpdir)
        assert store.available_categories() == ["agent_reasoning", "vision_perception"]


def test_available_categories_empty_store():
    """available_categories on an empty store returns an empty list."""
    store = ReferenceStore("/nonexistent/path")
    assert store.available_categories() == []


def test_get_by_id():
    """Test getting a specific example by ID."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data = {
            "examples": [
                {
                    "id": "r1",
                    "source_context": "c1",
                    "caption": "c1",
                    "image_path": "i1",
                },
            ],
        }
        Path(tmpdir, "index.json").write_text(json.dumps(data))

        store = ReferenceStore(tmpdir)
        assert store.get_by_id("r1") is not None
        assert store.get_by_id("nonexistent") is None


def test_create_store():
    """Test creating a new reference store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        examples = [
            ReferenceExample(
                id="new_001",
                source_context="New context",
                caption="New caption",
                image_path="images/new.png",
            )
        ]

        store = ReferenceStore.create(tmpdir, examples, metadata={"name": "new"})
        assert store.count == 1

        # Verify file was created
        assert Path(tmpdir, "index.json").exists()
