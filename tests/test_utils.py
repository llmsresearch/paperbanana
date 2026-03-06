"""Tests for paperbanana.core.utils — image save/detect helpers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from paperbanana.core.utils import (
    _ensure_pil_image,
    detect_image_mime_type,
    save_image,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rgb_image() -> Image.Image:
    """A tiny 4×4 RGB image."""
    return Image.new("RGB", (4, 4), color=(255, 0, 0))


@pytest.fixture()
def rgba_image() -> Image.Image:
    """A tiny 4×4 RGBA image (has alpha channel)."""
    return Image.new("RGBA", (4, 4), color=(0, 255, 0, 128))


@pytest.fixture()
def jpeg_sourced_image() -> Image.Image:
    """An RGB image whose *format* attribute is JPEG (simulating a
    PIL Image opened from a JPEG byte stream)."""
    img = Image.new("RGB", (4, 4), color=(0, 0, 255))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return Image.open(buf)  # .format == "JPEG"


# ---------------------------------------------------------------------------
# _ensure_pil_image
# ---------------------------------------------------------------------------


class TestEnsurePilImage:
    def test_pil_image_passthrough(self, rgb_image: Image.Image):
        assert _ensure_pil_image(rgb_image) is rgb_image

    def test_wrapper_with_image_bytes(self, rgb_image: Image.Image):
        """Objects exposing ``image_bytes`` are transparently converted."""
        buf = BytesIO()
        rgb_image.save(buf, format="PNG")

        class _FakeWrapper:
            image_bytes = buf.getvalue()

        result = _ensure_pil_image(_FakeWrapper())
        assert isinstance(result, Image.Image)

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Expected a PIL Image"):
            _ensure_pil_image("not an image")


# ---------------------------------------------------------------------------
# save_image — format inference from extension
# ---------------------------------------------------------------------------


class TestSaveImage:
    def test_png_extension_writes_png(self, tmp_path: Path, rgb_image: Image.Image):
        out = save_image(rgb_image, tmp_path / "out.png")
        with open(out, "rb") as f:
            assert f.read(8) == b"\x89PNG\r\n\x1a\n"

    def test_jpg_extension_writes_jpeg(self, tmp_path: Path, rgb_image: Image.Image):
        out = save_image(rgb_image, tmp_path / "out.jpg")
        with open(out, "rb") as f:
            assert f.read(2) == b"\xff\xd8"

    def test_jpeg_extension_writes_jpeg(self, tmp_path: Path, rgb_image: Image.Image):
        out = save_image(rgb_image, tmp_path / "out.jpeg")
        with open(out, "rb") as f:
            assert f.read(2) == b"\xff\xd8"

    def test_explicit_format_overrides_extension(self, tmp_path: Path, rgb_image: Image.Image):
        """When *format* is given explicitly, extension doesn't matter."""
        out = save_image(rgb_image, tmp_path / "out.png", format="jpeg")
        with open(out, "rb") as f:
            assert f.read(2) == b"\xff\xd8"

    def test_jpeg_sourced_image_saved_as_png(self, tmp_path: Path, jpeg_sourced_image: Image.Image):
        """Core bug scenario: JPEG-sourced PIL Image saved to .png must
        produce actual PNG bytes, not JPEG bytes in a .png file."""
        assert jpeg_sourced_image.format == "JPEG"
        out = save_image(jpeg_sourced_image, tmp_path / "output.png")
        with open(out, "rb") as f:
            assert f.read(8) == b"\x89PNG\r\n\x1a\n"

    def test_rgba_to_jpeg_converts_mode(self, tmp_path: Path, rgba_image: Image.Image):
        """RGBA images are auto-converted to RGB when saving as JPEG."""
        out = save_image(rgba_image, tmp_path / "out.jpg")
        assert out.exists()
        reopened = Image.open(out)
        assert reopened.mode == "RGB"

    def test_provider_wrapper_accepted(self, tmp_path: Path, rgb_image: Image.Image):
        """save_image accepts non-PIL objects with ``image_bytes``."""
        buf = BytesIO()
        rgb_image.save(buf, format="PNG")

        class _FakeWrapper:
            image_bytes = buf.getvalue()

        out = save_image(_FakeWrapper(), tmp_path / "out.png")
        assert out.exists()
        with open(out, "rb") as f:
            assert f.read(8) == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# detect_image_mime_type
# ---------------------------------------------------------------------------


def _write_png(path: Path) -> None:
    Image.new("RGB", (2, 2)).save(path, format="PNG")


def _write_jpeg(path: Path) -> None:
    Image.new("RGB", (2, 2)).save(path, format="JPEG")


def _write_bmp(path: Path) -> None:
    Image.new("RGB", (2, 2)).save(path, format="BMP")


def _write_gif(path: Path) -> None:
    Image.new("RGB", (2, 2)).save(path, format="GIF")


def _write_webp(path: Path) -> None:
    Image.new("RGB", (2, 2)).save(path, format="WEBP")


def _write_tiff(path: Path) -> None:
    Image.new("RGB", (2, 2)).save(path, format="TIFF")


class TestDetectImageMimeType:
    def test_png(self, tmp_path: Path):
        p = tmp_path / "img.png"
        _write_png(p)
        assert detect_image_mime_type(p) == "image/png"

    def test_jpeg(self, tmp_path: Path):
        p = tmp_path / "img.jpg"
        _write_jpeg(p)
        assert detect_image_mime_type(p) == "image/jpeg"

    def test_bmp(self, tmp_path: Path):
        p = tmp_path / "img.bmp"
        _write_bmp(p)
        assert detect_image_mime_type(p) == "image/bmp"

    def test_gif(self, tmp_path: Path):
        p = tmp_path / "img.gif"
        _write_gif(p)
        assert detect_image_mime_type(p) == "image/gif"

    def test_webp(self, tmp_path: Path):
        p = tmp_path / "img.webp"
        _write_webp(p)
        assert detect_image_mime_type(p) == "image/webp"

    def test_tiff(self, tmp_path: Path):
        p = tmp_path / "img.tiff"
        _write_tiff(p)
        assert detect_image_mime_type(p) == "image/tiff"

    def test_jpeg_in_png_extension(self, tmp_path: Path):
        """The MIME mismatch scenario: JPEG bytes inside a .png file."""
        p = tmp_path / "fake.png"
        _write_jpeg(p)
        assert detect_image_mime_type(p) == "image/jpeg"

    def test_fallback_to_extension(self, tmp_path: Path):
        """Unknown magic bytes → falls back to extension-based guess."""
        p = tmp_path / "mystery.png"
        p.write_bytes(b"\x00" * 12)
        assert detect_image_mime_type(p) == "image/png"

    def test_unknown_falls_to_octet_stream(self, tmp_path: Path):
        """Unknown magic + unknown extension → application/octet-stream."""
        p = tmp_path / "mystery.qqq"
        p.write_bytes(b"\x00" * 12)
        assert detect_image_mime_type(p) == "application/octet-stream"


# ---------------------------------------------------------------------------
# _compress_for_api (MCP server helper)
# ---------------------------------------------------------------------------


_has_fastmcp = True
try:
    import fastmcp  # noqa: F401
except ImportError:
    _has_fastmcp = False


@pytest.mark.skipif(not _has_fastmcp, reason="fastmcp not installed")
class TestCompressForApi:
    """Test the MCP server's _compress_for_api helper."""

    def test_small_image_passthrough(self, tmp_path: Path):
        from mcp_server.server import _compress_for_api

        p = tmp_path / "small.png"
        _write_png(p)
        path, fmt = _compress_for_api(str(p))
        assert path == str(p)
        assert fmt == "png"

    def test_large_image_compressed(self, tmp_path: Path, monkeypatch):
        from mcp_server import server
        from mcp_server.server import _compress_for_api

        # Set a very low limit so our test image exceeds it.
        monkeypatch.setattr(server, "_MAX_IMAGE_BYTES", 100)

        p = tmp_path / "big.png"
        Image.new("RGB", (200, 200), color=(128, 64, 32)).save(p, format="PNG")
        assert p.stat().st_size > 100

        path, fmt = _compress_for_api(str(p))
        assert fmt == "jpeg"
        assert path.endswith(".mcp.jpg")
        # Compressed file should actually exist
        assert Path(path).exists()

    def test_uncompressible_raises(self, tmp_path: Path, monkeypatch):
        from mcp_server import server
        from mcp_server.server import _compress_for_api

        # Set an impossibly low limit.
        monkeypatch.setattr(server, "_MAX_IMAGE_BYTES", 1)

        p = tmp_path / "huge.png"
        Image.new("RGB", (200, 200)).save(p, format="PNG")

        with pytest.raises(ValueError, match="could not be compressed"):
            _compress_for_api(str(p))
