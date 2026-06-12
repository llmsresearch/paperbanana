"""Build the PosterMemory seed from SciPostLayout annotations.

Programmatic skeleton extraction — no model calls: the COCO layout boxes
(Title, Author Info, Section, Text, List, Figure, Table) of each of the
7,855 human-made posters are converted into band/column/panel skeletons.
A stratified sample (closest-to-median members of each
orientation x columns group) is written as the shipped seed store.

Usage:
    python scripts/build_poster_exemplars.py [annotations_dir] [out_path] [per_group]

ML-venue exemplars (Paper2Poster author posters) are added separately via
``paperbanana posters ingest`` (VLM parsing) — see docs/poster.md.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paperbanana.poster.memory import (  # noqa: E402
    LayoutSkeleton,
    PosterExemplar,
    SkeletonBand,
    SkeletonPanel,
)

DEFAULT_ANN_DIR = Path.home() / ".cache" / "paperbanana" / "scipostlayout"
DEFAULT_OUT = Path("data/reference_sets/posters/exemplars.jsonl")
DEFAULT_PER_GROUP = 30

VISUAL_CLASSES = {"Figure", "Table"}
CONTENT_CLASSES = {"Text", "List", "Figure", "Table"}
HEADER_CLASSES = {"Title", "Author Info"}
COLUMN_GAP = 0.12


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


def extract_skeleton(width: float, height: float, anns: list[dict], cat_names: dict) -> dict | None:
    """One poster's COCO boxes -> skeleton + metadata, or None if unusable."""
    if width <= 0 or height <= 0 or not anns:
        return None
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
                "x1": (x + w) / width,
                "y1": (y + h) / height,
                "area": (w * h) / (width * height),
            }
        )
    if not blocks:
        return None

    header_bottoms = [b["y1"] for b in blocks if b["class"] in HEADER_CLASSES]
    title_band = min(0.30, max(0.05, max(header_bottoms))) if header_bottoms else 0.10
    body_frac = 1.0 - title_band

    content = [b for b in blocks if b["class"] in CONTENT_CLASSES and (b["x1"] - b["x0"]) <= 0.7]
    columns = _columns_from_centers([(b["x0"] + b["x1"]) / 2 for b in content])
    col_w = 1.0 / columns

    def column_of(b: dict) -> int:
        return max(0, min(columns - 1, int(((b["x0"] + b["x1"]) / 2) // col_w)))

    sections = sorted(
        (b for b in blocks if b["class"] == "Section" and b["y0"] >= title_band - 0.05),
        key=lambda b: (column_of(b), b["y0"]),
    )
    figures = [b for b in blocks if b["class"] in VISUAL_CLASSES]

    panels: list[SkeletonPanel] = []
    # Wide figures become spanning hero panels.
    hero_ys: list[tuple[float, float, int]] = []
    for fig in figures:
        span = (fig["x1"] - fig["x0"]) / col_w
        if span >= 1.6:
            col = max(0, min(columns - 1, int(fig["x0"] // col_w)))
            col_span = max(2, min(columns - col, round(span)))
            panels.append(
                SkeletonPanel(
                    band_index=1,
                    column=col,
                    col_span=col_span,
                    height_frac=max(0.05, min(0.9, (fig["y1"] - fig["y0"]) / body_frac)),
                    has_figure=True,
                )
            )
            hero_ys.append((fig["y0"], fig["y1"], col))

    by_column: dict[int, list[dict]] = defaultdict(list)
    for section in sections:
        by_column[column_of(section)].append(section)
    for col in range(columns):
        anchors = by_column.get(col, [])
        if not anchors:
            panels.append(
                SkeletonPanel(
                    band_index=1,
                    column=col,
                    col_span=1,
                    height_frac=0.9,
                    has_figure=any(
                        column_of(f) == col and (f["x1"] - f["x0"]) / col_w < 1.6 for f in figures
                    ),
                )
            )
            continue
        for i, anchor in enumerate(anchors):
            y0 = anchor["y0"]
            y1 = anchors[i + 1]["y0"] if i + 1 < len(anchors) else 1.0
            frac = max(0.03, min(0.95, (y1 - y0) / body_frac))
            has_fig = any(
                column_of(f) == col
                and (f["x1"] - f["x0"]) / col_w < 1.6
                and y0 <= (f["y0"] + f["y1"]) / 2 < y1
                for f in figures
            )
            panels.append(
                SkeletonPanel(
                    band_index=1, column=col, col_span=1, height_frac=frac, has_figure=has_fig
                )
            )
    if not panels:
        return None

    skeleton = LayoutSkeleton(
        bands=[
            SkeletonBand(kind="header", columns=1, height_frac=round(title_band, 3)),
            SkeletonBand(kind="body", columns=columns, height_frac=round(body_frac, 3)),
        ],
        panels=panels[:16],
    )
    return {
        "skeleton": skeleton,
        "orientation": "landscape" if width >= height else "portrait",
        "aspect_ratio": round(width / height, 3),
        "columns": columns,
        "n_panels": len(skeleton.panels),
        "n_figures": len(figures),
        "visual_share": round(min(1.0, sum(b["area"] for b in figures)), 3),
        "n_sections": len(sections),
    }


def main() -> None:
    ann_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ANN_DIR
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT
    per_group = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_PER_GROUP

    records: list[dict] = []
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
            record = extract_skeleton(
                img["width"], img["height"], by_image.get(img["id"], []), cat_names
            )
            if record is not None:
                record["image_id"] = f"{split}_{img['id']}"
                records.append(record)
    print(f"extracted {len(records)} skeletons")

    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["orientation"], r["columns"])].append(r)

    selected: list[dict] = []
    for (orientation, columns), members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(members) < 25:
            continue
        med_visual = statistics.median(m["visual_share"] for m in members)
        med_sections = statistics.median(m["n_sections"] for m in members)

        def representativeness(m: dict) -> float:
            return abs(m["visual_share"] - med_visual) + 0.1 * abs(m["n_sections"] - med_sections)

        members = sorted(members, key=representativeness)
        selected.extend(members[:per_group])
        print(f"  {orientation}-{columns}col: {len(members)} -> {min(per_group, len(members))}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in selected:
            exemplar = PosterExemplar(
                id=f"spl_{r['image_id']}",
                source="scipostlayout",
                venue=None,
                orientation=r["orientation"],
                aspect_ratio=r["aspect_ratio"],
                n_panels=r["n_panels"],
                n_figures=r["n_figures"],
                visual_share=r["visual_share"],
                skeleton=r["skeleton"],
                created="seed",
                tags=["seed"],
            )
            fh.write(json.dumps(json.loads(exemplar.model_dump_json())) + "\n")
    print(f"wrote {len(selected)} exemplars to {out_path}")


if __name__ == "__main__":
    main()
