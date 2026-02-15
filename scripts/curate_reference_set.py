"""Curate reference set from visually verified paper selections.

Each paper's methodology figure was manually inspected to confirm it shows
a methodology/architecture diagram (not results plots or sub-figures).
Section numbers for methodology text extraction were also verified.

Usage:
    python scripts/curate_reference_set.py
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

# Base directories
REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = REPO_ROOT / "data" / "reference_sets" / "output"
OUTPUT_DIR = REPO_ROOT / "data" / "reference_sets"

CATEGORIES = [
    "agent_reasoning",
    "vision_perception",
    "generative_learning",
    "science_applications",
]

# Visually verified selections for all 13 papers.
# Each entry specifies:
#   - title: paper title
#   - fig: image hash (resolved to images/{hash}.jpg)
#   - caption: figure caption (from visual inspection)
#   - category: one of the 4 categories
#   - method_sections: section numbers containing methodology text


def _fig(h: str) -> str:
    """Build image path from hash, keeping lines short."""
    return f"images/{h}.jpg"


PAPER_SELECTIONS = {
    "2404.15806v1": {
        "title": (
            "Where to Mask: Structure-Guided Masking " "for Graph Masked Autoencoders"
        ),
        "selected_figure": _fig(
            "9da89b3f6897112272256431dcad41239" "009c826b5da14efdb95deaeb0e38199"
        ),
        "caption": (
            "Overview of StructMAE. (a) The overall pipeline: Input Graph -> SBS (Structure-based "
            "Scoring) -> SGM (Structure-guided Masking) -> Enc -> Dec -> Reconstructed Nodes. "
            "(b) SBS module with pre-defined/learnable scoring and TopK selection. "
            "(c) SGM module showing progressive masking across training epochs."
        ),
        "category": "science_applications",
        "method_sections": [4],
    },
    "2601.03570v1": {
        "title": (
            "How Do Large Language Models Learn Concepts "
            "During Continual Pre-Training?"
        ),
        "selected_figure": _fig(
            "6167e9ed1ed7500b7a76ba5fb813cabf" "b897d526c1f6d1c18981d3d41ba07a2e"
        ),
        "caption": (
            "Concept Circuits: A transformer block "
            "(Attention, FFN, Add&Norm layers) maps to "
            "a concept circuit graph. Graph metrics "
            "(Node Importance, Info Flow Efficiency, "
            "Redundancy, Robustness) characterize "
            "internal concept representations."
        ),
        "category": "agent_reasoning",
        "method_sections": [2, 3],
    },
    "2601.05110v1": {
        "title": (
            "GlimpRouter: Efficient Collaborative "
            "Inference by Glimpsing One Token of Thoughts"
        ),
        "selected_figure": _fig(
            "7b79ff05f01bf6cc8e5fdaeb95c92e3b" "c667deae618b1f2a821c482827489943"
        ),
        "caption": (
            "Overview of GlimpRouter. The framework "
            "coordinates three layers: LLM (generates "
            "full steps and final answers), GlimpRouter "
            "(entropy-based routing checking initial "
            "token entropy H_init), and SLM (generates "
            "first tokens and complete steps). "
            "High H_init routes to LLM; low H_init "
            "stays with SLM."
        ),
        "category": "agent_reasoning",
        "method_sections": [3],
    },
    "2601.05144v1": {
        "title": (
            "Distilling the Thought, Watermarking "
            "the Answer: A Principle Semantic Guided "
            "Watermark for Large Reasoning LLMs"
        ),
        "selected_figure": _fig(
            "222b8921a57bd4baf5ca6f251ff2081e" "fb7d3b764b17015fe5a02e427dd28c23"
        ),
        "caption": (
            "ReasonMark pipeline: A prompt is processed "
            "by RLLM with <think> reasoning. "
            "Criticality Score CS(w) = GCC(w) + "
            "log(1 + CPS(w)) identifies Top-K Critical "
            "Tokens. The PSV R_i is initialized with "
            "PCA and guides semantically-aligned "
            "watermark embedding."
        ),
        "category": "generative_learning",
        "method_sections": [3],
    },
    "2601.06411v1": {
        "title": "Structured Episodic Event Memory",
        "selected_figure": _fig(
            "d72791f8835f57030107137b3aa96990" "8fa785393b58d0d4d2afb91a7dce0204"
        ),
        "caption": (
            "SEEM architecture overview. Sequential "
            "passages are processed by Memory Generation "
            "Agents into two layers: (1) Graph Memory "
            "with timestamped facts, fine-grained "
            "semantic edges, and schema-agnostic "
            "knowledge; (2) Episodic Memory with "
            "structured narrative synthesis, cognitive "
            "frame instantiation, and granular semantic "
            "decomposition of events."
        ),
        "category": "agent_reasoning",
        "method_sections": [3],
    },
    "2601.06953v2": {
        "title": ("X-Coder: Fully Synthetic Training " "for Competitive Programming"),
        "selected_figure": _fig(
            "5a0097931425808d877b1ccdb7cabc36" "8545e988513d5344f181d168b61d21d5"
        ),
        "caption": (
            "Task Generation pipeline for X-Coder. "
            "Code Snippets undergo Feature Extraction "
            "(Sorting, Math, Traversal), Evolve and "
            "Merge, Select and Thinking (selected "
            "subtree + scenario generation), producing "
            "competition-level programming Tasks."
        ),
        "category": "generative_learning",
        "method_sections": [3],
    },
    "2601.07033v1": {
        "title": ("Codified Foreshadowing-Payoff " "Text Generation"),
        "selected_figure": _fig(
            "4c72a4be8084b767765f79246c6615dd" "2788558e43724d1ed7da56ac04503e68"
        ),
        "caption": (
            "The Codified Foreshadow-Payoff Generation "
            "loop. Story Prefix X_t feeds into Codified "
            "Casual State (Foreshadow Pool C_t), "
            "Eligibility Selection checks triggers, "
            "Conditional Generation (LM) produces "
            "continuation y, and State Update codifies "
            "new foreshadows and resolves commitments."
        ),
        "category": "generative_learning",
        "method_sections": [3],
    },
    "2601.07055v1": {
        "title": ("Dr. Zero: Self-Evolving Search Agents " "without Training Data"),
        "selected_figure": _fig(
            "1852b8683cae159e96fea0bc3e3f4736" "71975b10aea5a021abda1ffd88c60b1e"
        ),
        "caption": (
            "Self-Evolution Feedback Loop in Dr. Zero. "
            "The Proposer generates QA Pairs via "
            "Reason & Search, which the Solver attempts "
            "to solve. Predictions yield Difficulty "
            "Rewards and Outcome Rewards. Step 1 updates "
            "the Proposer via HRPO; Step 2 updates "
            "the Solver via GRPO."
        ),
        "category": "agent_reasoning",
        "method_sections": [3],
    },
    "2601.09259v1": {
        "title": ("MAXS: Meta-Adaptive Exploration " "with LLM Agents"),
        "selected_figure": _fig(
            "812e3007cb785627d953fdd256f5396e" "d2a3cdd921805d0783e913f899f4d1c1"
        ),
        "caption": (
            "MAXS architecture. An LLM Agent with "
            "Code and Search tools processes inputs "
            "through (a) Rollout & Lookahead, "
            "(b) Value Estimation combining Advantage, "
            "Step-level Variance, and Trend Slope, "
            "and (c) Integration weighting to select "
            "the best expansion."
        ),
        "category": "agent_reasoning",
        "method_sections": [2],
    },
    "2601.09708v1": {
        "title": (
            "Fast-ThinkAct: Efficient VLA Reasoning " "via Verbalizable Latent Planning"
        ),
        "selected_figure": _fig(
            "9daba4ee7c2e20f2e82e35d6a390779a" "cfe09bbf703f9f30e11ab4167f1a0a73"
        ),
        "caption": (
            "Fast-ThinkAct training framework. "
            "(a) A Textual Teacher generates verbose "
            "reasoning, distilled into a Latent Student "
            "producing compact latent tokens z. "
            "GRPO Rollouts with reward provide the "
            "Verbalizer LLM loss. (b) The Latent "
            "Student Spatial KV feeds into an Action "
            "Model for robotic manipulation."
        ),
        "category": "vision_perception",
        "method_sections": [3],
    },
    "2601.14724v2": {
        "title": (
            "HERMES: KV Cache as Hierarchical Memory "
            "for Streaming Video Understanding"
        ),
        "selected_figure": _fig(
            "813dda147e6b461cfaca5a3b170fe31f" "f051726459d57ba73017edb93890cfb4"
        ),
        "caption": (
            "HERMES architecture for streaming video "
            "understanding. Video chunks processed by "
            "Vision Encoder into hierarchical KV Cache: "
            "Shallow (Sensory Memory), Middle (Working "
            "Memory), Deep (Long-term Memory). "
            "Cross-Layer Smoothing and Position "
            "Re-Indexing enable real-time Streaming QA."
        ),
        "category": "vision_perception",
        "method_sections": [3],
    },
    "2601.15165v2": {
        "title": (
            "The Flexibility Trap: Why Arbitrary Order "
            "Limits Reasoning Potential in "
            "Diffusion Language Models"
        ),
        "selected_figure": _fig(
            "d54598fd7978964c341d00e87ed6afcb" "1d373e831716075f3dd82542bd05679e"
        ),
        "caption": (
            "Confronting vs. bypassing uncertainty in "
            "token generation. (a) AR Order confronts "
            "uncertain (forking) tokens sequentially. "
            "(b) Arbitrary Order bypasses uncertain "
            "tokens, skipping hard decisions, which "
            "limits reasoning potential."
        ),
        "category": "generative_learning",
        "method_sections": [3, 4],
    },
    "2601.15892v2": {
        "title": (
            "Stable-DiffCoder: Pushing the Frontier "
            "of Code Diffusion Large Language Model"
        ),
        "selected_figure": _fig(
            "449ee807689da1a7aa8d341c5d9b078d" "466858e306b8a8bb413498ebf7c601d0"
        ),
        "caption": (
            "Stable-DiffCoder training pipeline. "
            "AR Mode Pretraining -> Code Continuous "
            "Pretraining -> Small Block Diffusion "
            "(DLLM Mode) -> Big Block or Bidirectional "
            "Diffusion, producing AR and DLLM Base "
            "Models with four key training features."
        ),
        "category": "generative_learning",
        "method_sections": [3],
    },
}


def get_section_number(text: str) -> int | None:
    """Extract top-level section number from heading text."""
    match = re.match(r"^(\d+)", text.strip())
    return int(match.group(1)) if match else None


def extract_methodology_text(content_list: list[dict], section_nums: list[int]) -> str:
    """Extract text from specified section numbers."""
    section_set = set(section_nums)
    in_method = False
    parts: list[str] = []

    for item in content_list:
        item_type = item.get("type", "")
        text = item.get("text", "").strip()
        text_level = item.get("text_level")

        # Check headings
        if text_level and text:
            sec_num = get_section_number(text)

            if sec_num is not None:
                if sec_num in section_set:
                    in_method = True
                    parts.append(text)
                    continue
                # Check if it's a subsection (e.g., 3.1 when 3 is in section_set)
                parent_match = any(str(sec_num).startswith(str(s)) for s in section_set)
                if parent_match and in_method:
                    parts.append(text)
                    continue
                # Different top-level section
                in_method = False
                continue

            # Heading without section number while in method
            if in_method:
                parts.append(text)
                continue

        if in_method:
            if item_type == "text" and text:
                parts.append(text)
            elif item_type == "equation" and text:
                parts.append(f"[Equation: {text}]")
            elif item_type == "list":
                for li in item.get("list_items", []):
                    if isinstance(li, str):
                        parts.append(f"- {li.strip()}")

    return "\n\n".join(parts)


def extract_title(content_list: list[dict]) -> str:
    """Extract paper title from content list."""
    skip_titles = {
        "abstract",
        "contents",
        "table of contents",
        "references",
        "acknowledgment",
        "acknowledgments",
        "introduction",
        "conclusion",
        "related work",
    }
    for item in content_list:
        if item.get("type") == "text" and item.get("text_level") == 1:
            text = item.get("text", "").strip()
            lower = text.lower()
            is_section = re.match(r"^(\d+|[A-Z]\.?\d*)[\.\s]", text)
            is_skip = any(lower.startswith(s) for s in skip_titles)
            if not is_section and not is_skip:
                return text
    return ""


def main():
    print(f"Curating reference set from {len(PAPER_SELECTIONS)} verified papers")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images_dir = OUTPUT_DIR / "images"
    images_dir.mkdir(exist_ok=True)

    examples = []

    for paper_id, sel in sorted(PAPER_SELECTIONS.items()):
        print(f"\n--- {paper_id} ---")

        # Load content list
        content_path = (
            INPUT_DIR / paper_id / "hybrid_auto" / f"{paper_id}_content_list.json"
        )
        if not content_path.exists():
            print(f"  ERROR: Content list not found: {content_path}")
            continue

        with open(content_path) as f:
            content_list = json.load(f)

        # Extract methodology text
        method_text = extract_methodology_text(content_list, sel["method_sections"])
        if not method_text:
            print(
                f"  WARNING: No methodology text extracted for sections {sel['method_sections']}"
            )
            # Try to show what sections exist
            continue

        print(f"  Title: {sel['title'][:70]}")
        print(f"  Methodology text: {len(method_text)} chars")
        print(f"  Category: {sel['category']}")

        # Copy selected figure
        src_img = INPUT_DIR / paper_id / "hybrid_auto" / sel["selected_figure"]
        if not src_img.exists():
            print(f"  ERROR: Image not found: {src_img}")
            continue

        ext = src_img.suffix or ".jpg"
        dst_filename = f"{paper_id}{ext}"
        dst_img = images_dir / dst_filename
        shutil.copy2(str(src_img), str(dst_img))
        print(f"  Image: {sel['selected_figure']} -> {dst_filename}")

        # Compute aspect ratio from actual image
        try:
            from PIL import Image

            im = Image.open(dst_img)
            w, h = im.size
            aspect_ratio = round(w / h, 2)
        except ImportError:
            aspect_ratio = 0.0

        examples.append(
            {
                "id": paper_id,
                "source_context": method_text,
                "caption": sel["caption"],
                "image_path": f"images/{dst_filename}",
                "category": sel["category"],
                "aspect_ratio": aspect_ratio,
                "source_paper": paper_id,
            }
        )
        print(f"  Added: {paper_id} (ratio={aspect_ratio})")

    # Write index.json
    index_data = {
        "metadata": {
            "name": "curated",
            "description": "Curated reference set of methodology diagrams from 13 academic papers. "
            "Each figure was visually verified to be a methodology/architecture diagram.",
            "version": "2.0.0",
            "source": "Manual visual inspection of MinerU-parsed PDFs",
            "categories": CATEGORIES,
            "total_examples": len(examples),
        },
        "examples": examples,
    }

    index_path = OUTPUT_DIR / "index.json"
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Done! {len(examples)} examples curated.")
    print(f"Index: {index_path}")

    # Category distribution
    from collections import Counter

    cat_counts = Counter(e["category"] for e in examples)
    print("\nCategory distribution:")
    for cat in CATEGORIES:
        print(f"  {cat}: {cat_counts.get(cat, 0)}")


if __name__ == "__main__":
    main()
