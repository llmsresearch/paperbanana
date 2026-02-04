"""Build a reference set from MinerU local output.

Reads MinerU local output directories (content_list.json + images/),
extracts methodology sections and figures, filters by aspect ratio,
copies images, and outputs to PaperBanana's reference set format.

Usage:
    # Process all papers in MinerU output directory:
    python scripts/build_reference_set.py \
        --input data/reference_sets/output \
        --output data/reference_sets

    # Process a single paper:
    python scripts/build_reference_set.py \
        --input data/reference_sets/output/2601.15165v2/hybrid_auto \
        --output data/reference_sets

    # Append to existing reference set:
    python scripts/build_reference_set.py \
        --input data/reference_sets/output \
        --output data/reference_sets \
        --append
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

# Section title patterns that indicate methodology content
METHOD_PATTERNS = [
    r"^\d*\.?\s*method(ology)?",
    r"^\d*\.?\s*approach",
    r"^\d*\.?\s*proposed\s+(method|framework|approach|model|system)",
    r"^\d*\.?\s*our\s+(method|framework|approach|model)",
    r"^\d*\.?\s*framework",
    r"^\d*\.?\s*model(\s+architecture)?$",
    r"^\d*\.?\s*architecture",
    r"^\d*\.?\s*technical\s+(approach|details|design)",
    r"^\d*\.?\s*system\s+(overview|design)",
]

# Patterns for sections to stop collecting (after methodology ends)
STOP_PATTERNS = [
    r"^\d*\.?\s*experiment",
    r"^\d*\.?\s*evaluation",
    r"^\d*\.?\s*result",
    r"^\d*\.?\s*conclusion",
    r"^\d*\.?\s*discussion",
    r"^\d*\.?\s*related\s+work",
    r"^\d*\.?\s*acknowledgment",
    r"^\d*\.?\s*reference",
    r"^\d*\.?\s*appendix",
    r"^\d*\.?\s*limitation",
    r"^\d*\.?\s*broader\s+impact",
]

CATEGORIES = [
    "agent_reasoning",
    "vision_perception",
    "generative_learning",
    "science_applications",
]


def is_method_heading(text: str) -> bool:
    """Check if heading text indicates a methodology section."""
    lower = text.lower().strip()
    return any(re.match(p, lower) for p in METHOD_PATTERNS)


def is_stop_heading(text: str) -> bool:
    """Check if heading text indicates the method section has ended."""
    lower = text.lower().strip()
    return any(re.match(p, lower) for p in STOP_PATTERNS)


def get_section_number(text: str) -> int | None:
    """Extract top-level section number from heading (e.g., '3' from '3. Method')."""
    match = re.match(r"^(\d+)", text.strip())
    return int(match.group(1)) if match else None


def compute_aspect_ratio(bbox: list[float]) -> float:
    """Compute width/height aspect ratio from [x0, y0, x1, y1] bbox."""
    if len(bbox) < 4:
        return 0.0
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if height <= 0:
        return 0.0
    return width / height


def find_content_list_json(paper_dir: Path) -> Path | None:
    """Find the content_list.json file within a MinerU output directory."""
    # Direct: paper_dir is the hybrid_auto dir containing content_list.json
    candidates = list(paper_dir.glob("*_content_list.json"))
    if candidates:
        return candidates[0]
    # One level deeper: paper_dir/{paper_id}/hybrid_auto/
    for sub in paper_dir.iterdir():
        if sub.is_dir():
            for backend in sub.iterdir():
                if backend.is_dir():
                    candidates = list(backend.glob("*_content_list.json"))
                    if candidates:
                        return candidates[0]
    return None


def discover_paper_dirs(input_path: Path) -> list[Path]:
    """Discover MinerU output directories containing content_list.json.

    Handles both:
    - input_path is a single hybrid_auto directory
    - input_path is a parent containing {paper_id}/{backend}/ subdirectories
    """
    # Check if input_path itself has a content_list.json
    if list(input_path.glob("*_content_list.json")):
        return [input_path]

    # Search subdirectories: output/{paper_id}/{backend}/
    dirs = []
    for paper_dir in sorted(input_path.iterdir()):
        if not paper_dir.is_dir():
            continue
        for backend_dir in sorted(paper_dir.iterdir()):
            if not backend_dir.is_dir():
                continue
            if list(backend_dir.glob("*_content_list.json")):
                dirs.append(backend_dir)
    return dirs


def _find_method_sections_by_position(headings: list[dict]) -> list[int]:
    """Identify method section numbers by position between intro and experiments.

    Many papers name method sections after their system (e.g., "3 HERMES",
    "4 Just GRPO") rather than using "Methodology". This heuristic finds
    top-level sections between intro/prelim and experiments/results.
    """
    # Patterns for sections that come BEFORE the method
    pre_patterns = [
        r"^\d*\.?\s*introduction",
        r"^\d*\.?\s*preliminar",
        r"^\d*\.?\s*background",
        r"^\d*\.?\s*problem\s+(statement|formulation|setup|definition)",
        r"^\d*\.?\s*notation",
        r"^\d*\.?\s*setup",
    ]

    # Find the last pre-method section number and first post-method section
    last_pre_num = None
    first_post_num = None

    for h in headings:
        sec_num = h["sec_num"]
        if sec_num is None:
            continue
        text_lower = h["text"].lower().strip()

        # Check pre-method patterns
        for p in pre_patterns:
            if re.match(p, text_lower):
                if last_pre_num is None or sec_num > last_pre_num:
                    last_pre_num = sec_num
                break

        # Check post-method (stop) patterns
        if is_stop_heading(h["text"]):
            if first_post_num is None or sec_num < first_post_num:
                first_post_num = sec_num

    if last_pre_num is None or first_post_num is None:
        return []

    # Method sections are between last_pre and first_post
    return list(range(last_pre_num + 1, first_post_num))


def parse_content_list(content_list_path: Path) -> dict:
    """Parse a MinerU content_list.json and extract title, methodology, figures.

    The content_list.json is a flat array of items with types:
    - text (with optional text_level for headings)
    - image (with img_path and image_caption)
    - equation, table, list, header, etc.
    """
    base_dir = content_list_path.parent

    with open(content_list_path) as f:
        items = json.load(f)

    # 1. Extract paper title
    skip_titles = {
        "abstract", "contents", "table of contents", "references",
        "acknowledgment", "acknowledgments", "acknowledgement",
        "appendix", "supplementary material", "supplementary",
        "introduction", "conclusion", "conclusions",
        "related work", "limitations", "broader impact",
    }
    paper_title = ""

    # First try: text items with text_level that aren't numbered sections
    for item in items:
        if item.get("type") == "text" and item.get("text_level") == 1:
            text = item.get("text", "").strip()
            # Skip numbered sections ("1.", "1 ", "A.", "A.1") and common headings
            lower = text.lower()
            is_section = re.match(r"^(\d+|[A-Z]\.?\d*)[\.\s]", text)
            is_skip = any(lower.startswith(s) for s in skip_titles)
            if not is_section and not is_skip:
                paper_title = text
                break

    # Fallback: check header items for the paper title (some parsers put it there)
    if not paper_title:
        for item in items:
            if item.get("type") == "header":
                text = item.get("text", "").strip()
                # Paper titles are typically >20 chars and not just logos/dates
                if len(text) > 20 and not re.match(r"^\d+\.", text):
                    paper_title = text
                    break

    # 2. Collect all top-level headings for positional analysis
    headings = []
    for item in items:
        if item.get("text_level") and item.get("text", "").strip():
            text = item["text"].strip()
            sec_num = get_section_number(text)
            headings.append({"text": text, "sec_num": sec_num})

    # 3. Determine which sections are methodology
    # First try explicit METHOD_PATTERNS, then fall back to positional
    explicit_method_nums = set()
    for h in headings:
        if h["sec_num"] is not None and is_method_heading(h["text"]):
            explicit_method_nums.add(h["sec_num"])

    if explicit_method_nums:
        method_section_nums = explicit_method_nums
    else:
        # Positional fallback: sections between intro and experiments
        method_section_nums = set(
            _find_method_sections_by_position(headings)
        )

    # 4. Extract methodology section text
    in_method = False
    methodology_parts = []

    for item in items:
        item_type = item.get("type", "")
        text = item.get("text", "").strip()
        text_level = item.get("text_level")

        # Headings in content_list.json are text items with text_level
        if text_level and text:
            sec_num = get_section_number(text)

            if sec_num is not None and sec_num in method_section_nums:
                in_method = True
                methodology_parts.append(text)
                continue

            # Sub-section: check if parent section is a method section
            if sec_num is not None and in_method:
                # "4.1" → parent is 4; check via string prefix
                parent_nums = {
                    int(str(sec_num)[0])
                    for s in method_section_nums
                    if str(sec_num).startswith(str(s))
                }
                if parent_nums:
                    methodology_parts.append(text)
                    continue
                else:
                    in_method = False
                    continue

            if in_method and is_stop_heading(text):
                in_method = False
                continue

            if in_method:
                methodology_parts.append(text)
                continue

        if in_method:
            if item_type == "text" and text:
                methodology_parts.append(text)
            elif item_type == "equation" and text:
                methodology_parts.append(f"[Equation: {text}]")
            elif item_type == "list":
                list_items = item.get("list_items", [])
                for li in list_items:
                    if isinstance(li, str):
                        methodology_parts.append(f"- {li.strip()}")

    methodology_text = "\n\n".join(methodology_parts)

    # 4. Extract all figures with captions and aspect ratios
    figures = []
    for item in items:
        if item.get("type") != "image":
            continue

        img_path = item.get("img_path", "")
        captions = item.get("image_caption", [])
        caption = captions[0].strip() if captions else ""
        bbox = item.get("bbox", [])

        if not img_path:
            continue

        # Resolve to absolute path
        abs_img_path = base_dir / img_path
        if not abs_img_path.exists():
            continue

        ratio = compute_aspect_ratio(bbox)

        figures.append({
            "caption": caption,
            "local_path": abs_img_path,
            "img_path": img_path,
            "aspect_ratio": ratio,
            "bbox": bbox,
        })

    return {
        "title": paper_title,
        "methodology_text": methodology_text,
        "figures": figures,
        "source_dir": str(base_dir),
    }


def identify_methodology_figures(
    figures: list[dict],
    min_ratio: float = 1.5,
    max_ratio: float = 3.5,
) -> list[dict]:
    """Filter figures to find likely methodology diagrams.

    Criteria:
    - Has a caption (figures without captions are usually sub-figures)
    - Aspect ratio within [min_ratio, max_ratio]
    - Caption suggests methodology/architecture (not results/plots)
    """
    result_keywords = [
        "performance", "accuracy", "comparison", "ablation",
        "training curve", "loss curve", "convergence",
        "visualization of", "t-sne", "tsne", "qualitative",
        "pass@", "wall-clock", "efficiency",
    ]

    method_keywords = [
        "overview", "architecture", "framework", "pipeline",
        "model", "method", "proposed", "approach", "system",
        "structure", "design", "workflow", "diagram",
        "illustration", "confronting", "mechanism",
    ]

    candidates = []
    for fig in figures:
        # Skip figures without captions (usually sub-figures or logos)
        if not fig["caption"]:
            continue

        ratio = fig["aspect_ratio"]
        if ratio < min_ratio or ratio > max_ratio:
            continue

        caption_lower = fig["caption"].lower()

        # Skip figures that look like results/plots
        if any(kw in caption_lower for kw in result_keywords):
            continue

        is_method = any(kw in caption_lower for kw in method_keywords)
        candidates.append({**fig, "is_method_figure": is_method})

    # Sort: methodology figures first, then by aspect ratio closeness to 2.0
    candidates.sort(
        key=lambda x: (not x["is_method_figure"], abs(x["aspect_ratio"] - 2.0))
    )

    return candidates


def generate_paper_id(title: str, dir_name: str) -> str:
    """Generate a clean ID from paper title, falling back to directory name."""
    if title:
        words = re.sub(r"[^a-zA-Z0-9\s]", "", title).lower().split()[:5]
        slug = "_".join(words)
        if slug:
            return slug
    # Fall back to directory name (e.g., "2601.15165v2")
    return re.sub(r"[^a-zA-Z0-9]", "_", dir_name).strip("_")


def guess_category(title: str, methodology_text: str) -> str:
    """Guess the paper category based on title and methodology keywords."""
    combined = (title + " " + methodology_text).lower()

    category_keywords = {
        "agent_reasoning": [
            "agent", "llm", "language model", "retrieval", "reasoning",
            "reinforcement learning", "planning", "rag",
            "multi-agent", "dialogue", "chatbot", "instruction",
            "chain-of-thought", "code generation", "tool use",
        ],
        "vision_perception": [
            "vision", "image", "object detection", "segmentation",
            "visual", "point cloud", "3d", "video", "camera",
            "optical", "lidar", "depth", "perception", "reconstruction",
        ],
        "generative_learning": [
            "diffusion", "generative", "vae", "autoencoder", "gan",
            "generation", "synthesis", "denoising", "latent",
            "flow matching", "score-based",
        ],
        "science_applications": [
            "graph", "molecule", "protein", "drug", "chemical",
            "physics", "material", "biology", "genome",
            "gnn", "scientific", "neural network",
        ],
    }

    scores = {
        cat: sum(1 for kw in kws if kw in combined)
        for cat, kws in category_keywords.items()
    }

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "science_applications"


def process_paper(
    paper_dir: Path,
    output_dir: Path,
    min_ratio: float,
    max_ratio: float,
) -> list[dict]:
    """Process a single MinerU local output directory into reference examples."""
    content_list = find_content_list_json(paper_dir)
    if not content_list:
        print(f"\n  SKIP: No content_list.json found in {paper_dir}")
        return []

    # Derive paper ID from directory name
    # Structure: output/{paper_id}/hybrid_auto/ → paper_id is grandparent
    dir_name = paper_dir.parent.name if paper_dir.name.endswith("auto") else paper_dir.name

    print(f"\nProcessing: {dir_name}")
    print(f"  Source: {content_list}")

    parsed = parse_content_list(content_list)
    title = parsed["title"]
    methodology_text = parsed["methodology_text"]
    figures = parsed["figures"]

    print(f"  Title: {title[:80]}..." if len(title) > 80 else f"  Title: {title}")
    print(f"  Methodology text: {len(methodology_text)} chars")
    captioned = sum(1 for f in figures if f["caption"])
    print(f"  Total figures: {len(figures)} ({captioned} with captions)")

    if not methodology_text:
        print("  WARNING: No methodology section found. Skipping.")
        return []

    # Filter for methodology diagrams
    candidates = identify_methodology_figures(figures, min_ratio, max_ratio)
    print(f"  Methodology diagram candidates: {len(candidates)}")

    if not candidates:
        # Take the first captioned figure within aspect ratio range
        in_range = [
            f for f in figures
            if f["caption"] and min_ratio <= f["aspect_ratio"] <= max_ratio
        ]
        if in_range:
            print("  Falling back to first captioned figure in ratio range")
            candidates = in_range[:1]
        else:
            print("  WARNING: No suitable figures found. Skipping.")
            return []

    paper_id = generate_paper_id(title, dir_name)
    category = guess_category(title, methodology_text)
    print(f"  Paper ID: {paper_id}")
    print(f"  Category: {category}")

    examples = []
    images_dir = output_dir / "images"

    # Take the best candidate (first after sorting)
    fig = candidates[0]
    fig_id = paper_id

    # Copy local image to curated images directory
    src_path = fig["local_path"]
    ext = src_path.suffix or ".jpg"
    image_filename = f"{fig_id}{ext}"
    dst_path = images_dir / image_filename

    print(f"  Selected: {fig['caption'][:70]}...")
    print(f"    Aspect ratio: {fig['aspect_ratio']:.2f}")
    print(f"    Copying: {src_path.name} → {image_filename}")

    shutil.copy2(str(src_path), str(dst_path))

    example = {
        "id": fig_id,
        "source_context": methodology_text,
        "caption": fig["caption"],
        "image_path": f"images/{image_filename}",
        "category": category,
        "aspect_ratio": round(fig["aspect_ratio"], 2),
        "source_paper": dir_name,
    }
    examples.append(example)
    print(f"  Added: {fig_id}")

    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Build PaperBanana reference set from MinerU local output"
    )
    parser.add_argument(
        "--input", required=True,
        help="MinerU output directory (or single paper's hybrid_auto dir)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for reference set",
    )
    parser.add_argument(
        "--min-ratio", type=float, default=1.5,
        help="Minimum aspect ratio (default: 1.5)",
    )
    parser.add_argument(
        "--max-ratio", type=float, default=3.5,
        help="Maximum aspect ratio (default: 3.5)",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append to existing index.json instead of overwriting",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)

    # Discover paper directories
    paper_dirs = discover_paper_dirs(input_path)

    if not paper_dirs:
        print(f"No MinerU output directories found in {input_path}")
        return

    print(f"Found {len(paper_dirs)} MinerU output(s)")
    print(f"Aspect ratio range: [{args.min_ratio}, {args.max_ratio}]")
    print(f"Output: {output_dir}")

    # Load existing examples if appending
    index_path = output_dir / "index.json"
    existing_examples = []
    existing_ids = set()

    if args.append and index_path.exists():
        with open(index_path) as f:
            existing_data = json.load(f)
        existing_examples = existing_data.get("examples", [])
        existing_ids = {e["id"] for e in existing_examples}
        print(f"Appending to {len(existing_examples)} existing examples")

    # Process each paper
    all_new_examples = []
    for paper_dir in paper_dirs:
        examples = process_paper(paper_dir, output_dir, args.min_ratio, args.max_ratio)
        for ex in examples:
            if ex["id"] in existing_ids:
                print(f"  Skipping duplicate: {ex['id']}")
                continue
            all_new_examples.append(ex)
            existing_ids.add(ex["id"])

    # Write index.json
    all_examples = existing_examples + all_new_examples

    index_data = {
        "metadata": {
            "name": "curated",
            "description": "Curated reference set from MinerU-parsed academic papers.",
            "version": "1.0.0",
            "source": "MinerU local PDF extraction",
            "aspect_ratio_range": [args.min_ratio, args.max_ratio],
            "categories": CATEGORIES,
            "total_examples": len(all_examples),
        },
        "examples": all_examples,
    }

    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Done! {len(all_new_examples)} new examples added.")
    print(f"Total examples: {len(all_examples)}")
    print(f"Index written to: {index_path}")


if __name__ == "__main__":
    main()
