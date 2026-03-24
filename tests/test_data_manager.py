"""Tests for DatasetManager lazy-download and curated expansion support."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from paperbanana.data.manager import (
    DatasetManager,
    _merge_index,
    resolve_reference_path,
)


@pytest.fixture()
def tmp_cache(tmp_path):
    """Provide a temporary cache directory with a DatasetManager."""
    return DatasetManager(cache_dir=tmp_path)


# ── _merge_index ──────────────────────────────────────────────────────


class TestMergeIndex:
    def test_merge_into_empty(self, tmp_path):
        idx = tmp_path / "index.json"
        examples = [
            {"id": "a", "category": "cat1"},
            {"id": "b", "category": "cat2"},
        ]
        count = _merge_index(idx, examples)
        assert count == 2
        data = json.loads(idx.read_text())
        assert data["metadata"]["total_examples"] == 2
        assert {e["id"] for e in data["examples"]} == {"a", "b"}

    def test_merge_deduplicates_by_id(self, tmp_path):
        idx = tmp_path / "index.json"
        _merge_index(idx, [{"id": "a", "category": "old"}, {"id": "b", "category": "keep"}])
        count = _merge_index(idx, [{"id": "a", "category": "new"}, {"id": "c", "category": "c"}])
        assert count == 3
        data = json.loads(idx.read_text())
        ids = {e["id"] for e in data["examples"]}
        assert ids == {"a", "b", "c"}
        a_entry = next(e for e in data["examples"] if e["id"] == "a")
        assert a_entry["category"] == "new"

    def test_merge_updates_categories(self, tmp_path):
        idx = tmp_path / "index.json"
        _merge_index(idx, [{"id": "x", "category": "alpha"}])
        _merge_index(idx, [{"id": "y", "category": "beta"}])
        data = json.loads(idx.read_text())
        assert sorted(data["metadata"]["categories"]) == ["alpha", "beta"]

    def test_merge_handles_corrupt_existing(self, tmp_path):
        idx = tmp_path / "index.json"
        idx.write_text("not json")
        count = _merge_index(idx, [{"id": "a", "category": "c"}])
        assert count == 1


# ── DatasetManager.is_downloaded ──────────────────────────────────────


class TestIsDownloaded:
    def test_false_when_no_info_file(self, tmp_cache):
        assert not tmp_cache.is_downloaded()

    def test_true_with_datasets_marker(self, tmp_cache):
        tmp_cache.reference_dir.mkdir(parents=True)
        tmp_cache.info_path.write_text(json.dumps({
            "datasets": ["curated"],
            "version": "1.0.0",
        }))
        assert tmp_cache.is_downloaded()
        assert tmp_cache.is_downloaded(dataset="curated")
        assert not tmp_cache.is_downloaded(dataset="full_bench")

    def test_true_with_both_datasets(self, tmp_cache):
        tmp_cache.reference_dir.mkdir(parents=True)
        tmp_cache.info_path.write_text(json.dumps({
            "datasets": ["curated", "full_bench"],
            "version": "1.0.0",
        }))
        assert tmp_cache.is_downloaded()
        assert tmp_cache.is_downloaded(dataset="curated")
        assert tmp_cache.is_downloaded(dataset="full_bench")

    def test_back_compat_old_format(self, tmp_cache):
        """Old dataset_info.json without 'datasets' key but with DATASET_URL source."""
        from paperbanana.data.manager import DATASET_URL

        tmp_cache.reference_dir.mkdir(parents=True)
        tmp_cache.info_path.write_text(json.dumps({
            "version": "1.0.0",
            "source": DATASET_URL,
        }))
        assert tmp_cache.is_downloaded()
        assert tmp_cache.is_downloaded(dataset="full_bench")
        assert not tmp_cache.is_downloaded(dataset="curated")

    def test_false_with_corrupt_info(self, tmp_cache):
        tmp_cache.reference_dir.mkdir(parents=True)
        tmp_cache.info_path.write_text("not json")
        assert not tmp_cache.is_downloaded()


# ── DatasetManager._record_dataset ────────────────────────────────────


class TestRecordDataset:
    def test_records_new_dataset(self, tmp_cache):
        tmp_cache.reference_dir.mkdir(parents=True)
        tmp_cache._record_dataset("curated", "1.0.0", "https://example.com", 25)
        info = json.loads(tmp_cache.info_path.read_text())
        assert "curated" in info["datasets"]
        assert info["example_count"] == 25

    def test_preserves_existing_datasets(self, tmp_cache):
        tmp_cache.reference_dir.mkdir(parents=True)
        tmp_cache._record_dataset("curated", "1.0.0", "https://example.com", 25)
        tmp_cache._record_dataset("full_bench", "1.0.0", "https://example.com", 294)
        info = json.loads(tmp_cache.info_path.read_text())
        assert sorted(info["datasets"]) == ["curated", "full_bench"]

    def test_back_compat_upgrade(self, tmp_cache):
        """Recording a new dataset upgrades old-format info to include datasets list."""
        from paperbanana.data.manager import DATASET_URL

        tmp_cache.reference_dir.mkdir(parents=True)
        tmp_cache.info_path.write_text(json.dumps({
            "version": "1.0.0",
            "source": DATASET_URL,
        }))
        tmp_cache._record_dataset("curated", "1.0.0", "https://example.com", 38)
        info = json.loads(tmp_cache.info_path.read_text())
        assert sorted(info["datasets"]) == ["curated", "full_bench"]


# ── DatasetManager.download (skip check) ─────────────────────────────


class TestDownloadSkip:
    def test_skip_when_already_cached(self, tmp_cache):
        tmp_cache.reference_dir.mkdir(parents=True)
        # Write a cached index with 2 examples
        tmp_cache.index_path.write_text(json.dumps({
            "metadata": {},
            "examples": [{"id": "a"}, {"id": "b"}],
        }))
        tmp_cache.info_path.write_text(json.dumps({
            "datasets": ["curated"],
            "version": "1.0.0",
        }))
        count = tmp_cache.download(dataset="curated")
        assert count == 2  # returns cached count, no download


# ── DatasetManager._download_curated ─────────────────────────────────


class TestDownloadCurated:
    def _make_curated_zip(self, zip_dest: Path, examples: list[dict], images: dict[str, bytes]):
        """Helper to create a fake CuratedExpansion.zip."""
        import zipfile

        with tempfile.TemporaryDirectory() as staging:
            staging_path = Path(staging)
            expansion = staging_path / "CuratedExpansion"
            expansion.mkdir()
            images_dir = expansion / "images"
            images_dir.mkdir()
            expansion_index = {
                "metadata": {"name": "curated_expansion"},
                "examples": examples,
            }
            (expansion / "index.json").write_text(json.dumps(expansion_index))
            for name, data in images.items():
                (images_dir / name).write_bytes(data)

            with zipfile.ZipFile(zip_dest, "w") as zf:
                for f in expansion.rglob("*"):
                    zf.write(f, f.relative_to(staging_path))

    def test_download_curated_merges(self, tmp_cache):
        examples = [
            {"id": "new1", "category": "cat1", "image_path": "images/new1.jpg"},
            {"id": "new2", "category": "cat2", "image_path": "images/new2.jpg"},
        ]
        images = {"new1.jpg": b"\xff\xd8fake", "new2.jpg": b"\xff\xd8fake"}

        # Seed cache with one existing example
        tmp_cache.reference_dir.mkdir(parents=True)
        (tmp_cache.reference_dir / "images").mkdir()
        tmp_cache.index_path.write_text(json.dumps({
            "metadata": {},
            "examples": [{"id": "existing", "category": "cat0"}],
        }))

        def fake_download(url, dest):
            self._make_curated_zip(dest, examples, images)

        with patch("paperbanana.data.manager._download_file", side_effect=fake_download):
            count = tmp_cache.download(dataset="curated", force=True)

        assert count == 3  # 1 existing + 2 new
        data = json.loads(tmp_cache.index_path.read_text())
        ids = {e["id"] for e in data["examples"]}
        assert ids == {"existing", "new1", "new2"}

        info = json.loads(tmp_cache.info_path.read_text())
        assert "curated" in info["datasets"]

    def test_download_curated_with_progress(self, tmp_cache):
        examples = [{"id": "p1", "category": "c", "image_path": "images/p1.jpg"}]
        images = {"p1.jpg": b"\xff\xd8fake"}
        messages = []

        def fake_download(url, dest):
            self._make_curated_zip(dest, examples, images)

        with patch("paperbanana.data.manager._download_file", side_effect=fake_download):
            tmp_cache.download(
                dataset="curated",
                force=True,
                progress_callback=lambda msg: messages.append(msg),
            )

        assert any("curated" in m.lower() for m in messages)


# ── resolve_reference_path ────────────────────────────────────────────


class TestResolveReferencePath:
    def test_explicit_settings_path_wins(self, tmp_path):
        result = resolve_reference_path("/custom/path", cache_dir=str(tmp_path))
        assert result == "/custom/path"

    def test_uses_cache_when_downloaded(self, tmp_path):
        ref_dir = tmp_path / "reference_sets"
        ref_dir.mkdir(parents=True)
        (ref_dir / "index.json").write_text(json.dumps({
            "metadata": {},
            "examples": [{"id": "a"}],
        }))
        (ref_dir / "dataset_info.json").write_text(json.dumps({
            "datasets": ["curated"],
            "version": "1.0.0",
        }))
        result = resolve_reference_path("data/reference_sets", cache_dir=str(tmp_path))
        assert result == str(ref_dir)

    def test_falls_back_to_builtin(self, tmp_path):
        result = resolve_reference_path("data/reference_sets", cache_dir=str(tmp_path))
        assert result == "data/reference_sets"


# ── DatasetManager.clear ──────────────────────────────────────────────


class TestClear:
    def test_clear_removes_cache(self, tmp_cache):
        tmp_cache.reference_dir.mkdir(parents=True)
        tmp_cache.info_path.write_text("{}")
        tmp_cache.clear()
        assert not tmp_cache.reference_dir.exists()

    def test_clear_noop_when_empty(self, tmp_cache):
        tmp_cache.clear()  # should not raise
