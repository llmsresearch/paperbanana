"""Reference set management for PaperBanana."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import structlog

from paperbanana.core.types import ReferenceExample

logger = structlog.get_logger()


class ReferenceStore:
    """Manages curated reference sets of academic illustrations.

    Reference sets are stored as JSON files with associated images.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._examples: list[ReferenceExample] = []
        self._loaded = False

    def _load(self) -> None:
        """Load reference examples from the store directory."""
        if self._loaded:
            return

        index_file = self.path / "index.json"
        if not index_file.exists():
            logger.warning("No reference index found", path=str(self.path))
            self._loaded = True
            return

        with open(index_file, encoding="utf-8") as f:
            data = json.load(f)

        for item in data.get("examples", []):
            # Resolve image path relative to store directory
            image_path = item.get("image_path", "")
            if image_path and not Path(image_path).is_absolute():
                image_path = str(self.path / image_path)

            self._examples.append(
                ReferenceExample(
                    id=item["id"],
                    source_context=item["source_context"],
                    caption=item["caption"],
                    image_path=image_path,
                    category=item.get("category"),
                )
            )

        logger.info("Loaded reference examples", count=len(self._examples))
        self._loaded = True

    def get_all(self) -> list[ReferenceExample]:
        """Get all reference examples."""
        self._load()
        return self._examples

    def get_by_category(self, category: str) -> list[ReferenceExample]:
        """Get reference examples filtered by category."""
        self._load()
        return [e for e in self._examples if e.category == category]

    def get_by_id(self, example_id: str) -> Optional[ReferenceExample]:
        """Get a specific reference example by ID."""
        self._load()
        for e in self._examples:
            if e.id == example_id:
                return e
        return None

    @property
    def count(self) -> int:
        """Number of reference examples in the store."""
        self._load()
        return len(self._examples)

    @staticmethod
    def create(
        path: str | Path,
        examples: list[ReferenceExample],
        metadata: Optional[dict] = None,
    ) -> ReferenceStore:
        """Create a new reference store from examples.

        Args:
            path: Directory to create the store in.
            examples: List of reference examples to include.
            metadata: Optional metadata about the set.

        Returns:
            The created ReferenceStore.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": metadata or {},
            "examples": [e.model_dump() for e in examples],
        }

        with open(path / "index.json", "w") as f:
            json.dump(data, f, indent=2)

        logger.info("Created reference store", path=str(path), count=len(examples))
        store = ReferenceStore(path)
        store._examples = examples
        store._loaded = True
        return store
