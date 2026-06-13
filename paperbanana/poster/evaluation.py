"""Poster quality evaluation: VLM judge + deterministic compliance.

The judged dimensions (Content, Design, Coherence, 1-5) are aligned with
PPTEval — the rubric used by Paper2Poster and successors — so scores are
directly comparable with published baselines. Compliance is NOT judged:
it is recomputed deterministically from the rendered poster image and the
venue spec (dimensions / orientation / DPI), which is the part of poster
quality a VLM judge is provably bad at.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import structlog
from PIL import Image
from pydantic import BaseModel, Field

from paperbanana.core.utils import extract_json
from paperbanana.poster.compliance import ComplianceReport, check_poster_compliance
from paperbanana.poster.venue_spec import load_venue_spec

logger = structlog.get_logger()

JUDGE_DIMENSIONS = ("content", "design", "coherence")


class PosterDimensionScore(BaseModel):
    dimension: str
    score: float = Field(ge=1, le=5)
    rationale: str


class PosterEvaluation(BaseModel):
    """Judge scores plus deterministic compliance for one poster."""

    scores: list[PosterDimensionScore]
    overall: float = Field(ge=1, le=5)
    compliance: Optional[ComplianceReport] = None
    reference_used: bool = False
    judges: int = 1

    @classmethod
    def from_scores(
        cls,
        scores: list[PosterDimensionScore],
        compliance: Optional[ComplianceReport],
        reference_used: bool,
        judges: int = 1,
    ) -> "PosterEvaluation":
        overall = sum(s.score for s in scores) / len(scores)
        return cls(
            scores=scores,
            overall=round(overall, 2),
            compliance=compliance,
            reference_used=reference_used,
            judges=judges,
        )


async def _judge_once(vlm, prompt: str, images: list) -> list[PosterDimensionScore]:
    """One judge pass: prompt + poster image(s) -> per-dimension scores."""
    raw = await vlm.generate(prompt=prompt, images=images, response_format="json", temperature=0.2)
    data = extract_json(raw)
    if not isinstance(data, dict):
        raise ValueError(f"poster judge returned no JSON object: {raw[:400]!r}")
    scores = []
    for dim in JUDGE_DIMENSIONS:
        entry = data.get(dim)
        if not isinstance(entry, dict) or "score" not in entry:
            raise ValueError(f"poster judge response missing dimension '{dim}': {raw[:400]!r}")
        scores.append(
            PosterDimensionScore(
                dimension=dim,
                score=float(entry["score"]),
                rationale=str(entry.get("rationale", "")),
            )
        )
    return scores


async def evaluate_poster(
    vlm,
    poster_path: Path,
    paper_context: str,
    prompt_dir: Path,
    reference_path: Optional[Path] = None,
    run_dir: Optional[Path] = None,
    venue_spec_dir: Optional[str] = None,
    secondary_vlm=None,
) -> PosterEvaluation:
    """Evaluate a generated poster.

    Args:
        vlm: VLM provider used as judge.
        poster_path: Rendered poster image (PNG).
        paper_context: Paper abstract/method text grounding the content
            dimension.
        prompt_dir: Prompts root (expects ``poster/evaluate.txt``).
        reference_path: Optional author/reference poster image for
            comparative judging.
        run_dir: Optional poster run directory; when given, compliance is
            recomputed from its ``poster_output.json`` (venue + physical
            size) and the rendered image, against the venue spec.
        venue_spec_dir: Optional user venue-spec directory.
        secondary_vlm: Optional second judge; per-dimension scores are
            averaged across both judges (variance reduction, not a vote).
    """
    template = (Path(prompt_dir) / "poster" / "evaluate.txt").read_text(encoding="utf-8")
    images = [Image.open(poster_path).convert("RGB")]
    reference_note = "Only the generated poster is attached."
    if reference_path is not None:
        images.append(Image.open(reference_path).convert("RGB"))
        reference_note = (
            "Image 1 is the GENERATED poster; image 2 is the AUTHOR-MADE reference "
            "poster for the same paper. Judge the generated poster on its own merits, "
            "using the reference only as a calibration point for what was achievable."
        )
    prompt = template.format(
        paper_context=paper_context[:8000],
        reference_note=reference_note,
    )
    scores = await _judge_once(vlm, prompt, images)
    n_judges = 1
    if secondary_vlm is not None:
        second = await _judge_once(secondary_vlm, prompt, images)
        by_dim = {s.dimension: s for s in second}
        scores = [
            PosterDimensionScore(
                dimension=s.dimension,
                score=round((s.score + by_dim[s.dimension].score) / 2, 2),
                rationale=(
                    f"judge 1 ({s.score:.0f}): {s.rationale} "
                    f"| judge 2 ({by_dim[s.dimension].score:.0f}): "
                    f"{by_dim[s.dimension].rationale}"
                ),
            )
            for s in scores
        ]
        n_judges = 2

    compliance: Optional[ComplianceReport] = None
    if run_dir is not None:
        meta_path = Path(run_dir) / "poster_output.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"no poster_output.json in {run_dir}; cannot check compliance")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        spec = load_venue_spec(meta["venue"], meta["venue_spec_year"], extra_dir=venue_spec_dir)
        w_mm, h_mm = meta["size_mm"]
        with Image.open(poster_path) as img:
            w_px, h_px = img.width, img.height
        pdf_path = Path(run_dir) / "poster.pdf"
        compliance = check_poster_compliance(
            spec, w_px, h_px, w_mm, h_mm, pdf_path=pdf_path if pdf_path.is_file() else None
        )

    evaluation = PosterEvaluation.from_scores(
        scores, compliance, reference_used=reference_path is not None, judges=n_judges
    )
    logger.info(
        "Poster evaluated",
        overall=evaluation.overall,
        scores={s.dimension: s.score for s in scores},
        judges=n_judges,
        compliance_passed=compliance.passed if compliance else None,
    )
    return evaluation
