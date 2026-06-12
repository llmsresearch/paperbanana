"""Poster quality evaluation: VLM judge + deterministic compliance.

The judged dimensions (Content, Design, Coherence, 1-5) are aligned with
PPTEval — the rubric used by Paper2Poster and successors — so scores are
directly comparable with published baselines. Compliance is NOT judged:
it is recomputed deterministically from the IR and the venue spec, which
is the part of poster quality a VLM judge is provably bad at.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import structlog
from PIL import Image
from pydantic import BaseModel, Field

from paperbanana.core.utils import extract_json
from paperbanana.poster.migrate import load_poster_ir
from paperbanana.poster.preflight import run_preflight
from paperbanana.poster.types import PreflightReport
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
    compliance: Optional[PreflightReport] = None
    reference_used: bool = False

    @classmethod
    def from_scores(
        cls,
        scores: list[PosterDimensionScore],
        compliance: Optional[PreflightReport],
        reference_used: bool,
    ) -> "PosterEvaluation":
        overall = sum(s.score for s in scores) / len(scores)
        return cls(
            scores=scores,
            overall=round(overall, 2),
            compliance=compliance,
            reference_used=reference_used,
        )


async def evaluate_poster(
    vlm,
    preview_path: Path,
    paper_context: str,
    prompt_dir: Path,
    reference_path: Optional[Path] = None,
    run_dir: Optional[Path] = None,
    venue_spec_dir: Optional[str] = None,
) -> PosterEvaluation:
    """Evaluate a generated poster.

    Args:
        vlm: VLM provider used as judge.
        preview_path: Rendered poster preview image.
        paper_context: Paper abstract/method text grounding the content
            dimension.
        prompt_dir: Prompts root (expects ``poster/evaluate.txt``).
        reference_path: Optional author/reference poster image for
            comparative judging.
        run_dir: Optional poster run directory; when given, compliance is
            recomputed from its ``poster_ir.json`` and venue spec.
        venue_spec_dir: Optional user venue-spec directory.
    """
    template = (Path(prompt_dir) / "poster" / "evaluate.txt").read_text(encoding="utf-8")
    images = [Image.open(preview_path).convert("RGB")]
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

    compliance: Optional[PreflightReport] = None
    if run_dir is not None:
        ir_path = Path(run_dir) / "poster_ir.json"
        if not ir_path.is_file():
            raise FileNotFoundError(f"no poster_ir.json in {run_dir}; cannot check compliance")
        ir = load_poster_ir(json.loads(ir_path.read_text(encoding="utf-8")))
        spec = load_venue_spec(ir.venue, ir.venue_spec_year, extra_dir=venue_spec_dir)
        pdf_path = Path(run_dir) / "poster.pdf"
        compliance = run_preflight(
            ir,
            spec,
            pdf_path=pdf_path if pdf_path.is_file() else None,
            png_path=Path(preview_path),
        )

    evaluation = PosterEvaluation.from_scores(
        scores, compliance, reference_used=reference_path is not None
    )
    logger.info(
        "Poster evaluated",
        overall=evaluation.overall,
        scores={s.dimension: s.score for s in scores},
        compliance_passed=compliance.passed if compliance else None,
    )
    return evaluation
