"""Induce poster layout schemas from SciPostLayout annotations.

Reads the COCO annotation files fetched by ``fetch_scipostlayout.py`` and
deterministically computes, per poster: orientation, column count,
section count, figure/table area share, and title-band height. Posters
are grouped into (orientation, columns) schemas with distribution
statistics, written to ``data/poster_schemas/schemas.json`` — the
committed artifact the content planner retrieves from at generation time.

No model calls: the annotations are human-made; the induction is pure
geometry, so rebuilding is free and reproducible.

Usage:
    python scripts/build_poster_schemas.py [annotations_dir] [out_path]
"""

from __future__ import annotations

import datetime
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_ANN_DIR = Path.home() / ".cache" / "paperbanana" / "scipostlayout"
DEFAULT_OUT = Path("data/poster_schemas/schemas.json")

CONTENT_CLASSES = {"Text", "List", "Figure", "Table"}
VISUAL_CLASSES = {"Figure", "Table"}
HEADER_CLASSES = {"Title", "Author Info"}

#: Horizontal gap (page-width fraction) separating column clusters.
COLUMN_GAP = 0.12
MIN_GROUP_SIZE = 50


def _columns_from_centers(centers: list[float]) -> int:
    if not centers:
        return 1
    centers = sorted(centers)
    clusters = [[centers[0]]]
    for c in centers[1:]:
        if c - clusters[-1][-1] > COLUMN_GAP:
            clusters.append([c])
        else:
            clusters[-1].append(c)
    real = [c for c in clusters if len(c) >= 2] or clusters
    return max(1, min(6, len(real)))


def analyze_poster(width: float, height: float, anns: list[dict], cat_names: dict) -> dict | None:
    if width <= 0 or height <= 0:
        return None
    page_area = width * height
    blocks = []
    for a in anns:
        name = cat_names.get(a["category_id"])
        if name is None:
            continue
        x, y, w, h = a["bbox"]
        blocks.append(
            {
                "class": name,
                "x0": x / width,
                "y0": y / height,
                "w": w / width,
                "h": h / height,
                "area": (w * h) / page_area,
            }
        )
    if not blocks:
        return None
    visual_share = min(1.0, sum(b["area"] for b in blocks if b["class"] in VISUAL_CLASSES))
    n_sections = sum(1 for b in blocks if b["class"] == "Section")
    content = [b for b in blocks if b["class"] in CONTENT_CLASSES and b["w"] <= 0.7]
    columns = _columns_from_centers([b["x0"] + b["w"] / 2 for b in content])
    header_bottoms = [b["y0"] + b["h"] for b in blocks if b["class"] in HEADER_CLASSES]
    title_band = min(0.4, max(header_bottoms)) if header_bottoms else None
    return {
        "orientation": "landscape" if width >= height else "portrait",
        "columns": columns,
        "n_sections": n_sections,
        "visual_share": visual_share,
        "title_band": title_band,
        "n_text_blocks": sum(1 for b in blocks if b["class"] in ("Text", "List")),
    }


def _dist(values: list[float]) -> dict:
    values = sorted(values)
    n = len(values)
    return {
        "p25": round(values[n // 4], 3),
        "median": round(statistics.median(values), 3),
        "p75": round(values[(3 * n) // 4], 3),
    }


def main() -> None:
    ann_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ANN_DIR
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT

    posters: list[dict] = []
    for split in ("train", "dev", "test"):
        path = ann_dir / f"{split}.json"
        if not path.is_file():
            raise SystemExit(
                f"missing {path}; run scripts/fetch_scipostlayout.py fetch {ann_dir} first"
            )
        coco = json.loads(path.read_text(encoding="utf-8"))
        cat_names = {c["id"]: c["name"] for c in coco["categories"]}
        by_image = defaultdict(list)
        for a in coco["annotations"]:
            by_image[a["image_id"]].append(a)
        for img in coco["images"]:
            result = analyze_poster(
                img["width"], img["height"], by_image.get(img["id"], []), cat_names
            )
            if result is not None:
                posters.append(result)
    print(f"analyzed {len(posters)} posters")

    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for p in posters:
        groups[(p["orientation"], p["columns"])].append(p)

    schemas = []
    for (orientation, columns), members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(members) < MIN_GROUP_SIZE:
            continue
        sections = [m["n_sections"] for m in members if m["n_sections"] > 0]
        bands = [m["title_band"] for m in members if m["title_band"] is not None]
        schemas.append(
            {
                "id": f"{orientation}-{columns}col",
                "orientation": orientation,
                "columns": columns,
                "n_posters": len(members),
                "share_of_corpus": round(len(members) / len(posters), 3),
                "sections": _dist(sections) if sections else None,
                "sections_per_column": (
                    round(statistics.median(sections) / columns, 1) if sections else None
                ),
                "visual_share": _dist([m["visual_share"] for m in members]),
                "title_band_frac": _dist(bands) if bands else None,
                "text_blocks": _dist([float(m["n_text_blocks"]) for m in members]),
            }
        )

    payload = {
        "source": (
            "SciPostLayout (omron-sinicx, CC-BY-4.0; posters from F1000Research, "
            "per-poster CC licenses) — https://huggingface.co/datasets/omron-sinicx/scipostlayout_v2"
        ),
        "built": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_posters": len(posters),
        "induction": "deterministic geometry over human layout annotations (no model calls)",
        "global": {
            "orientation_share": {
                o: round(sum(1 for p in posters if p["orientation"] == o) / len(posters), 3)
                for o in ("landscape", "portrait")
            },
            "visual_share": _dist([p["visual_share"] for p in posters]),
        },
        "schemas": schemas,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out_path} with {len(schemas)} schemas")
    for s in schemas:
        print(
            f"  {s['id']:16} n={s['n_posters']:5}  sections={s['sections']}  "
            f"visual={s['visual_share']['median']}"
        )


if __name__ == "__main__":
    main()
