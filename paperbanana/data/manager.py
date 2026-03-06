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
from typing import Callable, Optional

import structlog

logger = structlog.get_logger()

# Pin dataset revision for reproducibility
DATASET_REVISION = "main"
DATASET_URL = (
    "https://huggingface.co/datasets/dwzhu/PaperBananaBench"
    f"/resolve/{DATASET_REVISION}/PaperBananaBench.zip"
)
DATASET_VERSION = "1.0.0"

# Built-in reference set ships with the package (13 curated examples)
_BUILTIN_THRESHOLD = 50  # If cache has > this many examples, it's the expanded set


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

    def is_downloaded(self) -> bool:
        """Check if the expanded reference set is available in cache."""
        if not self.index_path.exists():
            return False
        try:
            with open(self.index_path) as f:
                data = json.load(f)
            return len(data.get("examples", [])) > _BUILTIN_THRESHOLD
        except (json.JSONDecodeError, OSError):
            return False

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
        task: str = "diagram",
        force: bool = False,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> int:
        """Download and cache the official PaperBananaBench dataset.

        Args:
            task: Which references to import ('diagram', 'plot', or 'both').
            force: Re-download even if already cached.
            progress_callback: Optional callback(message) for progress updates.

        Returns:
            Number of examples imported.

        Raises:
            RuntimeError: If download or extraction fails.
        """
        if self.is_downloaded() and not force:
            count = self.get_example_count()
            logger.info("Dataset already cached", count=count, path=str(self.reference_dir))
            return count

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

            count = _import_from_bench(bench_dir, task, images_dir, self.index_path)

            # Write dataset info for version tracking
            info = {
                "version": DATASET_VERSION,
                "revision": DATASET_REVISION,
                "source": DATASET_URL,
                "task": task,
                "example_count": count,
            }
            with open(self.info_path, "w") as f:
                json.dump(info, f, indent=2)

            _log(f"Cached {count} reference examples to {self.reference_dir}")
            return count

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


def _import_from_bench(
    bench_dir: Path,
    task: str,
    images_dir: Path,
    index_path: Path,
) -> int:
    """Convert official dataset format to community index.json format.

    Args:
        bench_dir: Extracted PaperBananaBench directory.
        task: 'diagram', 'plot', or 'both'.
        images_dir: Destination directory for reference images.
        index_path: Path to write the generated index.json.

    Returns:
        Number of examples imported.
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

    # Write index.json
    categories = sorted(set(e.get("category", "") for e in all_examples if e.get("category")))

    index_data = {
        "metadata": {
            "name": "paperbanana_bench",
            "description": (
                f"Reference set from official PaperBananaBench dataset. "
                f"{len(all_examples)} examples across {len(categories)} categories."
            ),
            "version": "3.0.0",
            "source": "https://huggingface.co/datasets/dwzhu/PaperBananaBench",
            "categories": categories,
            "total_examples": len(all_examples),
        },
        "examples": all_examples,
    }

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)

    return len(all_examples)


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

    # Check if expanded dataset is cached
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
