"""Fetch SciPostLayout annotation JSONs without downloading the 32GB zip.

The HuggingFace dataset (omron-sinicx/scipostlayout_v2, CC-BY-4.0) ships a
single ``scipostlayout.zip``. Zip files keep a central directory at the
end, so with HTTP Range requests we can list the archive and extract just
the (small) COCO annotation files.

Usage:
    python scripts/fetch_scipostlayout.py list
    python scripts/fetch_scipostlayout.py fetch <out_dir>
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import httpx

URL = "https://huggingface.co/datasets/omron-sinicx/scipostlayout_v2/resolve/main/scipostlayout.zip"


class RangedFile:
    """Read-only file-like over HTTP using Range requests with chunk cache."""

    CHUNK = 4 * 1024 * 1024

    def __init__(self, url: str):
        self._client = httpx.Client(follow_redirects=True, timeout=120)
        head = self._client.head(url)
        head.raise_for_status()
        self._url = str(head.url)
        self._size = int(head.headers["content-length"])
        self._pos = 0
        self._cache: dict[int, bytes] = {}
        self.bytes_fetched = 0

    def _chunk(self, index: int) -> bytes:
        if index not in self._cache:
            start = index * self.CHUNK
            end = min(start + self.CHUNK, self._size) - 1
            resp = self._client.get(self._url, headers={"Range": f"bytes={start}-{end}"})
            resp.raise_for_status()
            self._cache[index] = resp.content
            self.bytes_fetched += len(resp.content)
        return self._cache[index]

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self._size - self._pos
        out = bytearray()
        while n > 0 and self._pos < self._size:
            index, offset = divmod(self._pos, self.CHUNK)
            chunk = self._chunk(index)
            piece = chunk[offset : offset + n]
            out.extend(piece)
            self._pos += len(piece)
            n -= len(piece)
        return bytes(out)

    def seek(self, pos: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = pos
        elif whence == 1:
            self._pos += pos
        elif whence == 2:
            self._pos = self._size + pos
        return self._pos

    def tell(self) -> int:
        return self._pos

    def seekable(self) -> bool:
        return True


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    rf = RangedFile(URL)
    print(f"archive size: {rf._size / 1e9:.1f} GB", file=sys.stderr)
    zf = zipfile.ZipFile(rf)  # reads only the central directory
    names = zf.namelist()
    annotations = [n for n in names if n.lower().endswith(".json") and "__macosx" not in n.lower()]
    if cmd == "list":
        print(f"entries: {len(names)}; json files: {len(annotations)}")
        for n in annotations[:60]:
            info = zf.getinfo(n)
            print(f"  {n}  ({info.file_size / 1e6:.1f} MB)")
        print(f"(central directory cost: {rf.bytes_fetched / 1e6:.1f} MB fetched)")
        return
    if cmd == "fetch":
        out_dir = Path(sys.argv[2])
        out_dir.mkdir(parents=True, exist_ok=True)
        for n in annotations:
            info = zf.getinfo(n)
            target = out_dir / Path(n).name
            print(f"extracting {n} ({info.file_size / 1e6:.1f} MB) -> {target}")
            with zf.open(n) as src, open(target, "wb") as dst:
                dst.write(src.read())
        print(f"done; total fetched over HTTP: {rf.bytes_fetched / 1e6:.1f} MB")
        return
    raise SystemExit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
