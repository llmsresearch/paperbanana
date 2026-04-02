from __future__ import annotations

import json
from pathlib import Path

import pytest

from paperbanana.core.types import ReferenceExample
from paperbanana.reference.store import ReferenceStore

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = REPO_ROOT / "data" / "reference_sets"
INDEX_PATH = INDEX_DIR / "index.json"

EXPECTED_TOTAL = 38
EXPECTED_CATEGORIES = {
    "agent_reasoning",
    "generative_learning",
    "healthcare_medical",
    "multimodal_fusion",
    "nlp_language",
    "optimization_theory",
    "robotics_control",
    "science_applications",
    "systems_networking",
    "vision_perception",
}


@pytest.fixture()
def index_data() -> dict:
    with open(INDEX_PATH, encoding="utf-8") as f:
        return json.load(f)


class TestIndexJson:
    def test_index_file_exists(self):
        assert INDEX_PATH.exists(), f"{INDEX_PATH} is missing"

    def test_total_examples_matches(self, index_data: dict):
        examples = index_data["examples"]
        declared = index_data["metadata"]["total_examples"]
        assert len(examples) == declared
        assert len(examples) == EXPECTED_TOTAL

    def test_categories_complete(self, index_data: dict):
        declared = set(index_data["metadata"]["categories"])
        assert declared == EXPECTED_CATEGORIES

    def test_no_duplicate_ids(self, index_data: dict):
        ids = [e["id"] for e in index_data["examples"]]
        assert len(ids) == len(set(ids)), f"Duplicate IDs: {[x for x in ids if ids.count(x) > 1]}"

    def test_every_entry_has_source_paper(self, index_data: dict):
        for e in index_data["examples"]:
            assert "source_paper" in e and e["source_paper"], (
                f"Entry {e['id']} missing source_paper"
            )

    def test_every_category_assigned(self, index_data: dict):
        assigned = {e["category"] for e in index_data["examples"] if e.get("category")}
        missing = EXPECTED_CATEGORIES - assigned
        assert not missing, f"Categories with zero entries: {missing}"

    def test_image_files_exist(self, index_data: dict):
        missing = []
        for e in index_data["examples"]:
            img = INDEX_DIR / e["image_path"]
            if not img.exists():
                missing.append(e["id"])
        assert not missing, f"Missing images for: {missing}"

    def test_entries_parse_as_reference_example(self, index_data: dict):
        for e in index_data["examples"]:
            ReferenceExample(**e)

    def test_reference_store_loads_all(self):
        store = ReferenceStore(str(INDEX_DIR))
        assert store.count == EXPECTED_TOTAL

    def test_reference_store_category_counts(self):
        store = ReferenceStore(str(INDEX_DIR))
        for cat in EXPECTED_CATEGORIES:
            entries = store.get_by_category(cat)
            assert len(entries) > 0, f"No entries for category: {cat}"
