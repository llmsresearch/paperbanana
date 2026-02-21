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
    """Save a PIL Image to a file path.

    When *format* is not given explicitly, the target format is inferred from
    the file extension so that the on-disk bytes always match the extension.
    Without this, PIL may fall back to the image's original format (e.g. JPEG
    data written to a ``.png`` file) when the Image object was opened from a
    byte stream whose format differs from the extension.
    """
    path = Path(path)
    ensure_dir(path.parent)

    if format is None:
        # Infer the target format from the file extension.
        ext_to_format: dict[str, str] = {
            ".png": "PNG",
            ".jpg": "JPEG",
            ".jpeg": "JPEG",
            ".webp": "WEBP",
            ".bmp": "BMP",
            ".gif": "GIF",
            ".tiff": "TIFF",
            ".tif": "TIFF",
        }
        format = ext_to_format.get(path.suffix.lower())

    if format is not None:
        fmt = format.upper()
        # JPEG does not support alpha channels.
        if fmt == "JPEG" and image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGB")
        image.save(path, format=fmt)
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


def detect_image_mime_type(path: str | Path) -> str:
    """Detect the actual image MIME type from file header bytes.

    Uses magic-byte detection rather than file extension, so the result
    reflects the true encoding of the file on disk.
    """
    with open(path, "rb") as f:
        header = f.read(12)
    if header[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if header[:2] == b"\xff\xd8":
        return "image/jpeg"
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    if header[:4] == b"GIF8":
        return "image/gif"
    if header[:2] in (b"BM",):
        return "image/bmp"
    if header[:4] in (b"II\x2a\x00", b"MM\x00\x2a"):
        return "image/tiff"
    # Fall back to extension-based guess.
    mime, _ = __import__("mimetypes").guess_type(str(path))
    return mime or "application/octet-stream"
