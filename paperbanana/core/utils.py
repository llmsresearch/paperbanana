"""Shared utility functions for PaperBanana."""

from __future__ import annotations

import base64
import datetime
import hashlib
import json
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import structlog
from PIL import Image

logger = structlog.get_logger()


def generate_run_id() -> str:
    """Generate a unique run ID based on timestamp."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    return f"run_{ts}_{short_uuid}"


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert a PIL Image to a base64-encoded string."""
    buffer = BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_image(b64_string: str) -> Image.Image:
    """Convert a base64-encoded string to a PIL Image."""
    data = base64.b64decode(b64_string)
    return Image.open(BytesIO(data))


def load_image(path: str | Path) -> Image.Image:
    """Load an image from a file path."""
    return Image.open(path).convert("RGB")


def save_image(
    image: Image.Image,
    path: str | Path,
    format: str | None = None,
) -> Path:
    """Save a PIL Image to a file path."""
    path = Path(path)
    ensure_dir(path.parent)

    if format is not None:
        if format == "jpeg" and image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGB")
        image.save(path, format=format.upper())
    else:
        image.save(path)
    return path


def load_text(path: str | Path) -> str:
    """Load text content from a file."""
    return Path(path).read_text(encoding="utf-8")


def save_json(data: Any, path: str | Path) -> None:
    """Save data as JSON to a file."""
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    """Load JSON data from a file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def truncate_text(text: str, max_chars: int = 2000) -> str:
    """Truncate text to a maximum number of characters."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def hash_content(content: str) -> str:
    """Generate a short hash of content for deduplication."""
    return hashlib.sha256(content.encode()).hexdigest()[:12]
