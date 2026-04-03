"""Dataset management — download and cache official PaperBananaBench reference sets.

Cache layout:
    ~/.cache/paperbanana/              (or PAPERBANANA_CACHE_DIR)
    └── reference_sets/
        ├── index.json
        ├── dataset_info.json          (version + revision tracking)
        └── images/
            ├── ref_001.jpg
            └── ...
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Callable, Literal, Optional

import structlog

logger = structlog.get_logger()

# Pin dataset revision for reproducibility
DATASET_REVISION = "main"
DATASET_URL = (
    "https://huggingface.co/datasets/dwzhu/PaperBananaBench"
    f"/resolve/{DATASET_REVISION}/PaperBananaBench.zip"
)
DATASET_VERSION = "1.0.0"

# Curated expansion — lightweight set of 20–35 reference images
CURATED_EXPANSION_URL = (
    "https://huggingface.co/datasets/dwzhu/PaperBananaBench"
    f"/resolve/{DATASET_REVISION}/CuratedExpansion.zip"
)
CURATED_EXPANSION_VERSION = "1.0.0"


def default_cache_dir() -> Path:
    """Get the default cache directory using platformdirs."""
    from platformdirs import user_cache_dir

    return Path(user_cache_dir("paperbanana"))


def resolve_cache_dir(override: Optional[str] = None) -> Path:
    """Resolve cache directory from override → env var → platformdirs default.

    Args:
        override: Explicit cache dir path (highest priority).

    Returns:
        Resolved cache directory path.
    """
    if override:
        return Path(override)
    env_dir = os.environ.get("PAPERBANANA_CACHE_DIR")
    if env_dir:
        return Path(env_dir)
    return default_cache_dir()


class DatasetManager:
    """Manages downloading and caching of the official PaperBananaBench dataset.

    Provides a clean API for:
    - Downloading the dataset from HuggingFace
    - Converting to PaperBanana's index.json format
    - Caching in a user-local directory (~/.cache/paperbanana/)
    - Checking availability and version info
    """

    def __init__(self, cache_dir: Optional[str | Path] = None):
        """Initialize DatasetManager.

        Args:
            cache_dir: Override cache directory. Defaults to PAPERBANANA_CACHE_DIR
                       env var or ~/.cache/paperbanana/.
        """
        self._cache_dir = resolve_cache_dir(str(cache_dir) if cache_dir else None)

    @property
    def cache_dir(self) -> Path:
        """Root cache directory."""
        return self._cache_dir

    @property
    def reference_dir(self) -> Path:
        """Directory containing expanded reference set."""
        return self._cache_dir / "reference_sets"

    @property
    def index_path(self) -> Path:
        """Path to reference set index.json in cache."""
        return self.reference_dir / "index.json"

    @property
    def info_path(self) -> Path:
        """Path to dataset version info."""
        return self.reference_dir / "dataset_info.json"

    def is_downloaded(self, dataset: str | None = None) -> bool:
        """Check if an expanded reference set is available in cache.

        Args:
            dataset: Check for a specific dataset ('curated' or 'full_bench').
                     If None, returns True if *any* expansion is cached.

        Note:
            Caches that pre-date ``dataset_info.json`` (only ``index.json``
            present) are treated as an untracked expansion when *dataset* is
            None, but will not match a specific dataset name.
        """
        info = self.get_info()
        if info is None:
            # Fallback: legacy caches may only have index.json with no
            # dataset_info.json.  Treat them as "something is downloaded"
            # when no specific dataset is requested.
            if dataset is None and self.index_path.exists():
                return True
            return False
        downloaded = info.get("datasets", [])
        # Back-compat: old dataset_info.json without "datasets" key
        # was written by the full_bench downloader
        if not downloaded and info.get("source") == DATASET_URL:
            downloaded = ["full_bench"]
        if dataset is not None:
            return dataset in downloaded
        return len(downloaded) > 0

    def get_info(self) -> Optional[dict]:
        """Get cached dataset info (version, revision, count).

        Returns:
            Dataset info dict or None if not downloaded.
        """
        if not self.info_path.exists():
            return None
        try:
            with open(self.info_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def get_example_count(self) -> int:
        """Get the number of examples in the cached dataset."""
        if not self.index_path.exists():
            return 0
        try:
            with open(self.index_path) as f:
                data = json.load(f)
            return len(data.get("examples", []))
        except (json.JSONDecodeError, OSError):
            return 0

    def download(
        self,
        *,
        dataset: Literal["full_bench", "curated"] = "full_bench",
        task: str = "diagram",
        force: bool = False,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> int:
        """Download and cache a reference dataset.

        Args:
            dataset: Which dataset to fetch — 'full_bench' for the full
                     PaperBananaBench (~257MB) or 'curated' for the lightweight
                     curated expansion (~20–35 images).
            task: Which references to import ('diagram', 'plot', or 'both').
                  Only used for 'full_bench'.
            force: Re-download even if already cached.
            progress_callback: Optional callback(message) for progress updates.

        Returns:
            Number of examples in the cache after import.

        Raises:
            RuntimeError: If download or extraction fails.
        """
        if self.is_downloaded(dataset=dataset) and not force:
            count = self.get_example_count()
            logger.info(
                "Dataset already cached", dataset=dataset, count=count, path=str(self.reference_dir)
            )
            return count

        if dataset == "curated":
            return self._download_curated(force=force, progress_callback=progress_callback)
        return self._download_full_bench(
            task=task, force=force, progress_callback=progress_callback
        )

    def _download_curated(
        self,
        *,
        force: bool = False,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> int:
        """Download the curated expansion and merge into cache."""

        def _log(msg: str):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        with tempfile.TemporaryDirectory(prefix="paperbanana_curated_") as tmp:
            tmp_dir = Path(tmp)
            zip_path = tmp_dir / "CuratedExpansion.zip"

            _log("Downloading curated expansion set...")
            try:
                _download_file(CURATED_EXPANSION_URL, zip_path)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to download curated expansion from {CURATED_EXPANSION_URL} "
                    f"— the artifact may not be published yet. Original error: {exc}"
                ) from exc

            _log("Extracting curated expansion...")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp_dir)

            # Expect the zip to contain index.json + images/
            expansion_dir = tmp_dir / "CuratedExpansion"
            if not expansion_dir.exists():
                candidates = list(tmp_dir.glob("*/index.json"))
                if candidates:
                    expansion_dir = candidates[0].parent
                else:
                    raise RuntimeError("Could not find index.json in curated expansion archive.")

            expansion_index = expansion_dir / "index.json"
            if not expansion_index.exists():
                raise RuntimeError("Curated expansion archive missing index.json.")

            with open(expansion_index, encoding="utf-8") as f:
                expansion_data = json.load(f)

            new_examples = expansion_data.get("examples", [])
            if not new_examples:
                raise RuntimeError("Curated expansion contains no examples.")

            # Copy images into cache
            self.reference_dir.mkdir(parents=True, exist_ok=True)
            images_dir = self.reference_dir / "images"
            images_dir.mkdir(exist_ok=True)

            src_images = expansion_dir / "images"
            for ex in new_examples:
                img_rel = ex.get("image_path", "")
                if img_rel and src_images.exists():
                    src = expansion_dir / img_rel
                    if src.exists():
                        dest = self.reference_dir / img_rel
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        if not dest.exists() or force:
                            shutil.copy2(src, dest)

            # Merge into existing cache index
            count = _merge_index(self.index_path, new_examples)

            # Update dataset_info.json
            self._record_dataset("curated", CURATED_EXPANSION_VERSION, CURATED_EXPANSION_URL, count)

            _log(f"Merged {len(new_examples)} curated examples → {count} total in cache")
            return count

    def _download_full_bench(
        self,
        *,
        task: str = "diagram",
        force: bool = False,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> int:
        """Download the full PaperBananaBench dataset."""

        def _log(msg: str):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        with tempfile.TemporaryDirectory(prefix="paperbanana_") as tmp:
            tmp_dir = Path(tmp)
            zip_path = tmp_dir / "PaperBananaBench.zip"

            # Download
            _log(f"Downloading PaperBananaBench ({DATASET_REVISION})...")
            _download_file(DATASET_URL, zip_path)

            # Extract
            _log("Extracting dataset...")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp_dir)

            bench_dir = tmp_dir / "PaperBananaBench"
            if not bench_dir.exists():
                candidates = list(tmp_dir.glob("*/PaperBananaBench"))
                if candidates:
                    bench_dir = candidates[0]
                else:
                    raise RuntimeError(
                        "Could not find PaperBananaBench directory in extracted archive."
                    )

            # Convert and cache
            _log("Converting to PaperBanana format...")
            self.reference_dir.mkdir(parents=True, exist_ok=True)
            images_dir = self.reference_dir / "images"
            images_dir.mkdir(exist_ok=True)

            bench_examples = _import_from_bench(bench_dir, task, images_dir)
            count = _merge_index(self.index_path, bench_examples)

            # Update dataset_info.json
            self._record_dataset(
                "full_bench",
                DATASET_VERSION,
                DATASET_URL,
                count,
                extra={"revision": DATASET_REVISION, "task": task},
            )

            _log(f"Cached {count} reference examples to {self.reference_dir}")
            return count

    def _record_dataset(
        self,
        dataset: str,
        version: str,
        source: str,
        example_count: int,
        extra: dict | None = None,
    ) -> None:
        """Update dataset_info.json, preserving the list of downloaded datasets."""
        info = self.get_info() or {}
        downloaded = set(info.get("datasets", []))
        if not downloaded and info.get("source") == DATASET_URL:
            downloaded.add("full_bench")
        downloaded.add(dataset)

        dataset_meta: dict = info.get("dataset_meta", {})
        meta_entry: dict = {"version": version, "source": source}
        if extra:
            meta_entry.update(extra)
        dataset_meta[dataset] = meta_entry

        info.update(
            {
                "datasets": sorted(downloaded),
                "dataset_meta": dataset_meta,
                "example_count": example_count,
            }
        )
        info.pop("version", None)
        info.pop("source", None)
        info.pop("revision", None)

        with open(self.info_path, "w") as f:
            json.dump(info, f, indent=2)

    def clear(self) -> None:
        """Remove cached dataset."""
        if self.reference_dir.exists():
            shutil.rmtree(self.reference_dir)
            logger.info("Cleared cached dataset", path=str(self.reference_dir))


def _download_file(url: str, dest: Path) -> None:
    """Download a file using httpx (already a project dependency)."""
    import httpx

    with httpx.stream("GET", url, follow_redirects=True, timeout=300) as response:
        response.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in response.iter_bytes(chunk_size=8192):
                f.write(chunk)


def _merge_index(index_path: Path, new_examples: list[dict]) -> int:
    """Merge new examples into an existing cached index.json.

    Deduplicates by example ID — new entries overwrite existing ones with the
    same ID. Returns the total example count after merge.
    """
    existing_examples: list[dict] = []
    if index_path.exists():
        try:
            with open(index_path, encoding="utf-8") as f:
                data = json.load(f)
            existing_examples = data.get("examples", [])
        except (json.JSONDecodeError, OSError):
            pass

    by_id: dict[str, dict] = {}
    no_id: list[dict] = []
    for ex in existing_examples:
        ex_id = ex.get("id", "")
        if ex_id:
            by_id[ex_id] = ex
        else:
            no_id.append(ex)
    for ex in new_examples:
        ex_id = ex.get("id", "")
        if ex_id:
            by_id[ex_id] = ex
        else:
            no_id.append(ex)

    merged = list(by_id.values()) + no_id
    categories = sorted(set(e.get("category", "") for e in merged if e.get("category")))

    index_data = {
        "metadata": {
            "name": "paperbanana_combined",
            "description": (
                f"Combined reference set. "
                f"{len(merged)} examples across {len(categories)} categories."
            ),
            "version": "3.0.0",
            "categories": categories,
            "total_examples": len(merged),
        },
        "examples": merged,
    }

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)

    return len(merged)


def _import_from_bench(
    bench_dir: Path,
    task: str,
    images_dir: Path,
) -> list[dict]:
    """Convert official dataset format to a list of example dicts.

    Images are copied into *images_dir*; the caller is responsible for
    merging the returned examples into the cached ``index.json`` (via
    ``_merge_index``) so that previously-downloaded datasets (e.g. the
    curated expansion) are preserved.

    Args:
        bench_dir: Extracted PaperBananaBench directory.
        task: 'diagram', 'plot', or 'both'.
        images_dir: Destination directory for reference images.

    Returns:
        List of imported example dicts.
    """
    from PIL import Image

    tasks = ["diagram", "plot"] if task == "both" else [task]
    all_examples: list[dict] = []

    for t in tasks:
        task_dir = bench_dir / t
        ref_file = task_dir / "ref.json"

        if not ref_file.exists():
            logger.warning("Task ref.json not found, skipping", task=t, path=str(ref_file))
            continue

        with open(ref_file, encoding="utf-8") as f:
            entries = json.load(f)

        source_images_dir = task_dir / "images"
        count = 0

        for entry in entries:
            entry_id = entry.get("id", "")
            if task == "both":
                entry_id = f"{t}_{entry_id}"

            # Map fields from official → community format
            source_context = entry.get("content", "")
            if isinstance(source_context, (dict, list)):
                source_context = json.dumps(source_context, indent=2)

            example: dict = {
                "id": entry_id,
                "source_context": source_context,
                "caption": entry.get("visual_intent", ""),
                "category": entry.get("category", ""),
                "source_paper": entry_id,
            }

            # Copy image
            gt_image_rel = entry.get("path_to_gt_image", "")
            if not gt_image_rel:
                continue

            source_image = source_images_dir / gt_image_rel
            if not source_image.exists():
                source_image = source_images_dir.parent / gt_image_rel
            if not source_image.exists():
                logger.warning("Image not found, skipping", id=entry_id, path=str(source_image))
                continue

            dest_filename = f"{entry_id}.jpg"
            dest_image = images_dir / dest_filename
            if not dest_image.exists():
                shutil.copy2(source_image, dest_image)

            example["image_path"] = f"images/{dest_filename}"

            # Compute aspect ratio
            try:
                with Image.open(dest_image) as img:
                    w, h = img.size
                    example["aspect_ratio"] = round(w / h, 2) if h > 0 else None
            except Exception:
                example["aspect_ratio"] = None

            all_examples.append(example)
            count += 1

        logger.info("Imported task references", task=t, count=count, total=len(entries))

    if not all_examples:
        raise RuntimeError("No examples could be imported from the dataset.")

    return all_examples


def resolve_reference_path(
    settings_path: str,
    cache_dir: Optional[str] = None,
) -> str:
    """Resolve reference set path with fallback chain.

    Priority:
    1. Explicit settings path (non-default, from config/env/YAML)
    2. Cached expanded dataset (~/.cache/paperbanana/reference_sets/)
    3. Built-in reference set (data/reference_sets/)

    Args:
        settings_path: The reference_set_path from Settings (may be default or user-set).
        cache_dir: Optional cache dir override.

    Returns:
        Resolved path to the reference set directory.
    """
    default_path = "data/reference_sets"

    # If settings_path differs from the default, the user explicitly configured it
    # (via env var REFERENCE_SET_PATH, YAML config, or CLI). Honor it unconditionally.
    if settings_path != default_path:
        logger.info("Using explicitly configured reference set", path=settings_path)
        return settings_path

    # Check if any expanded dataset is cached (uses dataset_info.json marker)
    manager = DatasetManager(cache_dir=cache_dir)
    if manager.is_downloaded():
        logger.info(
            "Using cached expanded reference set",
            path=str(manager.reference_dir),
            count=manager.get_example_count(),
        )
        return str(manager.reference_dir)

    # Fallback to built-in
    return settings_path
