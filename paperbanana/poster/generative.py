"""Generative poster pipeline: amplify the model, verify against the paper.

The poster is produced by a design-capable image model at full power, then
wrapped with PaperBanana's own power: grounding (only the paper's verified
facts reach the prompt), a faithfulness audit (a VLM checks the rendered
poster against the paper and flags hallucinations), and bounded repair.
Determinism survives only as verification — venue dimensions / DPI
compliance and the audit. Output is PNG + a print-ready PDF at the venue's
exact physical size (no pptx, no LibreOffice).
"""

from __future__ import annotations

import datetime
import json
import uuid
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import structlog
from PIL import Image
from pydantic import BaseModel, Field

from paperbanana.core.config import Settings
from paperbanana.core.cost_tracker import CostTracker
from paperbanana.core.pdf_text import extract_text_from_pdf
from paperbanana.poster.compliance import ComplianceReport, check_poster_compliance
from paperbanana.poster.figure_embed import slot_spec
from paperbanana.poster.venue_spec import load_venue_spec
from paperbanana.providers.image_gen.openai_imagen import legal_gpt_image_2_dims

logger = structlog.get_logger()

FigurePolicy = Literal["auto", "real", "generated"]
#: Largest edge (px) we ask the image provider for before its budget clamp.
_TARGET_LONG_EDGE_PX = 4000

GROUND_PROMPT = """Extract ONLY facts present in this paper, to build a FAITHFUL poster.
Output plain text with these labeled fields (invent nothing; if unknown write 'unknown'):
TITLE: exact title
AUTHORS: author names (+ affiliation if present)
VENUE: venue/year if stated, else 'unknown'
ONE-LINE TAKEAWAY: the single headline contribution in one sentence, using the paper's own numbers
PROBLEM: 2 short bullets on why it matters
METHOD: 3-5 short bullets on the approach (concrete, named components)
RESULTS: 3-5 short bullets with the EXACT key numbers/metrics from the paper
KEY NUMBERS: the 3-4 most important numbers a visitor must remember, each with its meaning
FIGURES: short list of what the paper's real figures show

=== PAPER ===
{paper}"""

POSTER_PROMPT = """A complete, professionally designed ACADEMIC CONFERENCE POSTER, {orientation},
physical size {width_mm:.0f}mm x {height_mm:.0f}mm, print quality, that a researcher would be proud
to present at {venue}. Confident, varied color scheme appropriate to the topic (NOT a generic
navy+orange template). Bold title band, clear section headers, strong visual hierarchy, LARGE
readable text and LARGE dominant figures/charts, emphasized big result numbers. Fill the whole
canvas - no large empty areas. Must read clearly from 2 meters.{qr_note}{venue_notes}
{slots}
Use ONLY these verified facts. Do NOT invent any model name, dataset, baseline, shot-count, or
number that is not listed here. Every number on the poster must match these exactly:

{grounding}
{repair}"""

AUDIT_PROMPT = """You are a strict FAITHFULNESS AUDITOR for a conference poster. Below is the source
paper (ground truth); the attached image is a generated poster for it. Report ONLY claims VISIBLE
ON THE POSTER that are CONTRADICTED BY or NOT SUPPORTED BY the paper — wrong numbers, invented
model/dataset/baseline names, invented details, mislabeled quantities. Read footers/metadata
carefully.

Scope — flag ONLY substantive factual claims: numbers/metrics, model/dataset/baseline names,
method facts, quantitative results. IGNORE section headings, panel labels, and poster design text
(venue/size specifications, QR captions, "Contact"), and IGNORE wording/section-title differences
from the paper — those are not factual errors.

Output ONLY genuine discrepancies, one per line:
POSTER SAYS "<quote>" | PAPER SAYS "<the real fact, or 'not in paper'>"
Do NOT list a claim you judge correct or supported — omit it entirely (never write "correct",
"no contradiction", or "not contradictory"). If the poster has no discrepancies, output: NONE

=== PAPER ===
{paper}
=== END PAPER ==="""

#: Phrases the auditor uses when it lists a claim it actually judged fine.
_AUDIT_OK_MARKERS = (
    "(correct",
    "no contradiction",
    "not contradicted",
    "correctly identifies",
    "correctly states",
    "correctly lists",
    "is correct",
    "is supported",
    "supported by the paper",
    "not a discrepancy",
    "this is not a difference",
    "matches the paper",
    "consistent with the paper",
)


def _real_findings(audit_raw: str) -> list[str]:
    """Keep only genuine contradiction lines.

    VLM auditors over-report: they tag some lines as actually-correct, and
    they list lines whose POSTER and PAPER quotes in fact agree. Drop both
    (markers + quote-agreement). Residual false positives are expected;
    per-finding verification is a later precision upgrade.
    """
    import re

    out = []
    for line in audit_raw.splitlines():
        if "POSTER SAYS" not in line:
            continue
        low = line.lower()
        if any(marker in low for marker in _AUDIT_OK_MARKERS):
            continue
        m = re.search(r'poster says\s*"(.*?)"\s*\|\s*paper says\s*"(.*?)"', line, re.IGNORECASE)
        if m:
            pq, paq = m.group(1).strip().lower(), m.group(2).strip().lower()
            # Identical or one-contains-the-other quotes = the auditor quoted a
            # matching fact, not a contradiction.
            if pq and paq and (pq == paq or pq in paq or paq in pq):
                continue
        out.append(line.strip())
    return out


class GenerativePosterOutput(BaseModel):
    """Artifacts of a generative poster run."""

    run_dir: str
    png_path: str
    pdf_path: str
    venue: str
    venue_spec_year: int
    size_mm: tuple[float, float]
    figures_policy: FigurePolicy
    grounding: str
    audit_findings: list[str] = Field(default_factory=list)
    repair_rounds: int = 0
    figure_decisions: list[dict] = Field(default_factory=list)
    compliance: ComplianceReport
    cost_usd: Optional[float] = None


def _run_id() -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"poster_{ts}_{uuid.uuid4().hex[:6]}"


class GenerativePosterPipeline:
    """Generates a venue-compliant poster image (+PDF) from a paper PDF."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        progress_callback: Optional[Callable[[str, dict], None]] = None,
        vlm_client=None,
        image_gen_client=None,
    ):
        self.settings = settings or Settings()
        self.run_id = _run_id()
        self._progress = progress_callback
        self._run_dir = Path(self.settings.output_dir) / self.run_id

        if vlm_client is not None and image_gen_client is not None:
            self._vlm = vlm_client
            self._image_gen = image_gen_client
        else:
            from paperbanana.providers.registry import ProviderRegistry

            self._vlm = ProviderRegistry.create_vlm(self.settings)
            self._image_gen = ProviderRegistry.create_image_gen(self.settings)
        self._cost = CostTracker(budget=self.settings.budget_usd)
        for p in (self._vlm, self._image_gen):
            if hasattr(p, "cost_tracker"):
                p.cost_tracker = self._cost
        from paperbanana.core.utils import find_prompt_dir

        self._prompt_dir = self.settings.prompt_dir or find_prompt_dir()

    def _emit(self, event: str, **payload: Any) -> None:
        logger.info("poster_progress", progress_event=event, **payload)
        if self._progress is not None:
            try:
                self._progress(event, payload)
            except Exception:
                logger.warning("progress callback failed", progress_event=event)

    def _poster_pixels(self, width_mm: float, height_mm: float) -> tuple[int, int]:
        """Legal provider pixel dims at the venue aspect, near the budget."""
        scale = _TARGET_LONG_EDGE_PX / max(width_mm, height_mm)
        return legal_gpt_image_2_dims(round(width_mm * scale), round(height_mm * scale))

    async def generate(
        self,
        paper_pdf: Path,
        venue: Optional[str] = None,
        year: Optional[int] = None,
        qr_url: Optional[str] = None,
        figures: FigurePolicy = "generated",
        figure_overrides: Optional[dict[str, str]] = None,
        repair_rounds: int = 1,
    ) -> GenerativePosterOutput:
        if figures not in ("generated", "real", "auto"):
            raise ValueError(f"figures must be generated|real|auto, got {figures!r}")
        venue_name = (venue or self.settings.venue).strip().lower()
        spec = load_venue_spec(venue_name, year, extra_dir=self.settings.venue_spec_dir)
        w_mm, h_mm = spec.dimensions.generation_size_mm()
        orientation = spec.dimensions.generation_orientation()
        self._run_dir.mkdir(parents=True, exist_ok=True)

        paper_text = extract_text_from_pdf(Path(paper_pdf))[:16000]
        self._emit("ingest_complete", chars=len(paper_text))

        # Real-figure embedding (auto/real): extract the paper's figures and
        # have the design reserve magenta slots we composite into.
        selected: list = []
        slot_block = ""
        if figures in ("auto", "real"):
            selected = await self._extract_figures(paper_pdf)
            if selected:
                slot_block = "\n" + slot_spec(selected) + "\n"
                self._emit("figures_extracted", count=len(selected))

        # Ground: only the paper's verified facts reach the design prompt.
        grounding = await self._vlm.generate(
            prompt=GROUND_PROMPT.format(paper=paper_text), temperature=0.0, max_tokens=1500
        )
        (self._run_dir / "grounding.txt").write_text(grounding, encoding="utf-8")
        self._emit("grounding_complete")

        qr_note = ""
        if qr_url:
            qr_note = (
                " Reserve a small clear square in a top corner for a QR code labeled "
                "'Scan for paper & code' (a real QR is composited there afterward)."
            )
        venue_notes = f" Venue notes: {spec.notes}" if spec.notes else ""
        w_px, h_px = self._poster_pixels(w_mm, h_mm)

        audit_findings: list[str] = []
        repair_block = ""
        image: Optional[Image.Image] = None
        rounds = 0
        for attempt in range(repair_rounds + 1):
            prompt = POSTER_PROMPT.format(
                orientation=orientation,
                width_mm=w_mm,
                height_mm=h_mm,
                venue=spec.display_name,
                qr_note=qr_note,
                venue_notes=venue_notes,
                slots=slot_block,
                grounding=grounding,
                repair=repair_block,
            )
            self._emit("generating", attempt=attempt, size_px=(w_px, h_px))
            image = await self._image_gen.generate(
                prompt=prompt, width=w_px, height=h_px, quality="high"
            )
            image.save(self._run_dir / f"poster_v{attempt + 1}.png")

            audit_raw = await self._vlm.generate(
                prompt=AUDIT_PROMPT.format(paper=paper_text),
                images=[image],
                temperature=0.0,
                max_tokens=1500,
            )
            audit_findings = _real_findings(audit_raw)
            self._emit("audit_complete", attempt=attempt, findings=len(audit_findings))
            if not audit_findings or attempt == repair_rounds:
                break
            repair_block = (
                "\n\nThe previous draft had these FACTUAL ERRORS - fix every one and include "
                f"nothing that is not in the verified facts above:\n{audit_raw}"
            )
            rounds += 1

        assert image is not None
        # Composite the real/reauthored figures into the magenta slots.
        figure_decisions: list[dict] = []
        if selected:
            image, figure_decisions = await self._embed_figures(
                image,
                selected,
                figures,
                w_mm,
                w_px,
                h_px,
                grounding,
                slot_block,
                qr_note,
                venue_notes,
                orientation,
                spec,
            )
        if qr_url:
            image = self._composite_qr(image, qr_url)

        png_path = self._run_dir / "poster.png"
        image.save(png_path)
        pdf_path = self._run_dir / "poster.pdf"
        self._write_pdf(image, w_mm, h_mm, pdf_path)

        compliance = check_poster_compliance(
            spec, image.width, image.height, w_mm, h_mm, pdf_path=pdf_path
        )
        (self._run_dir / "audit.json").write_text(
            json.dumps({"findings": audit_findings, "repair_rounds": rounds}, indent=2),
            encoding="utf-8",
        )
        output = GenerativePosterOutput(
            run_dir=str(self._run_dir),
            png_path=str(png_path),
            pdf_path=str(pdf_path),
            venue=venue_name,
            venue_spec_year=spec.year,
            size_mm=(w_mm, h_mm),
            figures_policy=figures,
            grounding=grounding,
            audit_findings=audit_findings,
            repair_rounds=rounds,
            figure_decisions=figure_decisions,
            compliance=compliance,
            cost_usd=getattr(self._cost, "total_cost", None),
        )
        (self._run_dir / "poster_output.json").write_text(
            output.model_dump_json(indent=2), encoding="utf-8"
        )
        self._emit(
            "poster_complete",
            run_dir=str(self._run_dir),
            compliance_passed=compliance.passed,
            findings=len(audit_findings),
        )
        return output

    def _composite_qr(self, image: Image.Image, url: str) -> Image.Image:
        """Paste a real scannable QR into the top-right corner."""
        import qrcode

        poster = image.convert("RGB")
        side = max(120, int(min(poster.width, poster.height) * 0.07))
        qr = qrcode.make(url).convert("RGB").resize((side, side))
        pad = side // 8
        framed = Image.new("RGB", (side + 2 * pad, side + 2 * pad), "white")
        framed.paste(qr, (pad, pad))
        margin = int(min(poster.width, poster.height) * 0.02)
        poster.paste(framed, (poster.width - framed.width - margin, margin))
        return poster

    async def _extract_figures(self, paper_pdf: Path) -> list:
        """Extract the paper's figures and select the poster-worthy ones."""
        from paperbanana.poster.agents.figure_detector import FigureDetectorAgent
        from paperbanana.poster.agents.paper_metadata import PaperMetadataAgent
        from paperbanana.poster.figure_embed import select_poster_figures
        from paperbanana.poster.ingest import ingest_paper

        kwargs = {"prompt_dir": self._prompt_dir}
        assets = await ingest_paper(
            Path(paper_pdf),
            FigureDetectorAgent(self._vlm, **kwargs),
            PaperMetadataAgent(self._vlm, **kwargs),
            self._run_dir / "paper_assets",
            extract_dpi=self.settings.poster_extract_dpi,
        )
        return select_poster_figures(assets.figures)

    async def _embed_figures(
        self,
        image,
        selected,
        policy,
        w_mm,
        w_px,
        h_px,
        grounding,
        slot_block,
        qr_note,
        venue_notes,
        orientation,
        spec,
    ):
        """Detect magenta slots and composite each figure (real/reauthored).

        Bounded regeneration if the design didn't leave the expected slot
        count; a persistent mismatch is a hard error pointing at
        --figures generated.
        """
        from paperbanana.poster.agents.faithfulness import FaithfulnessAgent
        from paperbanana.poster.agents.figure_curator import FigureCuratorAgent
        from paperbanana.poster.figure_embed import (
            SlotCountError,
            composite_into_slot,
            detect_slots,
            prepare_figure,
        )

        slots = detect_slots(image)
        for _ in range(2):  # the model sometimes draws the wrong slot count
            if len(slots) == len(selected):
                break
            self._emit("slot_retry", detected=len(slots), expected=len(selected))
            prompt = POSTER_PROMPT.format(
                orientation=orientation,
                width_mm=w_mm,
                height_mm=spec.dimensions.height_mm,
                venue=spec.display_name,
                qr_note=qr_note,
                venue_notes=venue_notes,
                slots=slot_block,
                grounding=grounding,
                repair="",
            )
            image = await self._image_gen.generate(
                prompt=prompt, width=w_px, height=h_px, quality="high"
            )
            slots = detect_slots(image)
        if len(slots) != len(selected):
            raise SlotCountError(
                f"design left {len(slots)} figure slots, expected {len(selected)}; "
                "re-run with --figures generated or fewer figures."
            )

        kwargs = {"prompt_dir": self._prompt_dir}
        curator = FigureCuratorAgent(self._vlm, **kwargs)
        faithfulness = FaithfulnessAgent(self._vlm, **kwargs)
        reauthor_template = (Path(self._prompt_dir) / "poster" / "reauthor_edit.txt").read_text(
            encoding="utf-8"
        )
        palette = {
            "primary": "#1A3A6B",
            "secondary": "#4A6FA5",
            "accent": "#E8A33D",
            "background": "#FFFFFF",
        }
        decisions = []
        for slot, fig in zip(slots, selected):
            slot_width_mm = slot.w / image.width * w_mm
            chosen, choice = await prepare_figure(
                fig,
                slot_width_mm,
                policy=policy,
                curator=curator,
                faithfulness=faithfulness,
                image_gen=self._image_gen,
                palette=palette,
                reauthor_template=reauthor_template,
                min_dpi=spec.text_rules.min_image_dpi,
                out_dir=self._run_dir / "paper_assets",
            )
            composite_into_slot(image, slot, chosen)
            decisions.append(choice.model_dump())
            self._emit("figure_embedded", figure=fig.id, source=choice.source)
        return image, decisions

    def _write_pdf(self, image: Image.Image, width_mm: float, height_mm: float, out: Path) -> None:
        """Single-page PDF at the exact physical poster size (vector page,
        embedded raster). No LibreOffice."""
        import io

        from reportlab.lib.units import mm
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfgen import canvas

        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        buf.seek(0)
        c = canvas.Canvas(str(out), pagesize=(width_mm * mm, height_mm * mm))
        c.drawImage(ImageReader(buf), 0, 0, width=width_mm * mm, height=height_mm * mm)
        c.showPage()
        c.save()
