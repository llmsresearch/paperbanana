"""PosterPipeline integration test: mocked providers, real deterministic core."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from paperbanana.core.config import Settings
from paperbanana.poster.convert import SofficeNotFoundError, find_soffice
from paperbanana.poster.pipeline import PosterPipeline

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"
SPEC_DIR = Path(__file__).resolve().parents[2] / "data" / "venue_specs"


def _has_soffice() -> bool:
    try:
        find_soffice()
        return True
    except SofficeNotFoundError:
        return False


needs_soffice = pytest.mark.skipif(not _has_soffice(), reason="LibreOffice not installed")


class _RoutedVLM:
    """Mock VLM that routes on prompt content instead of call order."""

    name = "mock"
    model_name = "mock-model"
    cost_tracker = None

    async def generate(self, prompt, images=None, **kwargs):
        if "bibliographic metadata" in prompt.lower() or '"affiliations"' in prompt:
            return json.dumps(
                {
                    "title": "A Synthetic Paper About Posters",
                    "authors": ["Ada Lovelace"],
                    "affiliations": ["Analytical Engines Lab"],
                    "abstract": "We automate poster generation.",
                }
            )
        if "Locate every FIGURE" in prompt:
            if "page 2" in prompt:
                return json.dumps(
                    [
                        {
                            "kind": "figure",
                            "figure_number": "Figure 1",
                            "caption": "Figure 1: The synthetic figure.",
                            "bbox": [160, 245, 820, 560],
                        }
                    ]
                )
            return "[]"
        if "storyboard" in prompt.lower():
            return json.dumps(
                {
                    "columns": 3,
                    "qr_url": None,
                    "new_figure_briefs": {},
                    "panels": [
                        {
                            "id": "motivation",
                            "role": "motivation",
                            "title": "Why",
                            "order": 1,
                            "weight": 1.0,
                            "text_blocks": ["- Posters take days to make"],
                            "figure_ids": [],
                        },
                        {
                            "id": "method",
                            "role": "method",
                            "title": "Approach",
                            "order": 2,
                            "weight": 1.4,
                            "text_blocks": ["- Two-phase agentic pipeline"],
                            "figure_ids": ["fig1"],
                        },
                        {
                            "id": "results",
                            "role": "results",
                            "title": "Results",
                            "order": 3,
                            "weight": 1.0,
                            "text_blocks": ["- Big wins on all metrics"],
                            "figure_ids": [],
                        },
                    ],
                }
            )
        if "visual identity" in prompt:
            return json.dumps(
                {
                    "palette": {
                        "primary": "#1A3A6B",
                        "secondary": "#4A6FA5",
                        "accent": "#E8A33D",
                        "background": "#FFFFFF",
                        "panel_bg": "#F5F7FA",
                        "text": "#1A1A1A",
                    },
                    "font_heading": "Helvetica",
                    "font_body": "Helvetica",
                    "type_scale_pt": {
                        "title": 72,
                        "authors": 36,
                        "affiliation": 26,
                        "heading": 40,
                        "body": 26,
                        "caption": 19,
                        "footnote": 15,
                    },
                }
            )
        if "print-production expert" in prompt:
            return json.dumps({"decision": "reuse", "reason": "crisp enough"})
        if "meticulous reviewer" in prompt:
            return json.dumps({"blocking": False, "summary": "Looks good.", "edit_ops": []})
        if "tightening text" in prompt:
            raise AssertionError("no panel should overflow in this fixture")
        raise AssertionError(f"unrouted mock prompt: {prompt[:120]}")


class _MockImageGen:
    name = "mock-imagen"
    model_name = "mock-image-model"

    async def generate(
        self,
        prompt,
        negative_prompt=None,
        width=1024,
        height=1024,
        seed=None,
        aspect_ratio=None,
        quality=None,
        images=None,
    ):
        return Image.new("RGB", (2000, 1200), "lightyellow")


@pytest.fixture
def paper_pdf(tmp_path: Path) -> Path:
    import fitz

    doc = fitz.open()
    page1 = doc.new_page(width=612, height=792)
    page1.insert_text((72, 100), "A Synthetic Paper About Posters", fontsize=16)
    page1.insert_text((72, 170), "Abstract", fontsize=12)
    page1.insert_text((72, 190), "We automate poster generation.", fontsize=10)
    page2 = doc.new_page(width=612, height=792)
    page2.insert_text((72, 90), "3 Method", fontsize=12)
    rect = fitz.Rect(100, 200, 500, 400)
    page2.draw_rect(rect, color=(0, 0, 1), width=2)
    page2.insert_text((100, 420), "Figure 1: The synthetic figure.", fontsize=9)
    path = tmp_path / "paper.pdf"
    doc.save(str(path))
    doc.close()
    return path


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        output_dir=str(tmp_path / "outputs"),
        prompt_dir=str(PROMPT_DIR),
        save_prompts=False,
        poster_refinement_iterations=1,
    )


@needs_soffice
async def test_pipeline_end_to_end(paper_pdf: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PAPERBANANA_VENUE_SPEC_DIR", "/nonexistent")
    monkeypatch.setenv("PAPERBANANA_LESSONS_DIR", str(tmp_path / "lessons"))
    monkeypatch.chdir(Path(__file__).resolve().parents[2])  # builtin specs resolve relatively
    pipeline = PosterPipeline(
        settings=_settings(tmp_path),
        vlm_client=_RoutedVLM(),
        image_gen_client=_MockImageGen(),
    )
    output = await pipeline.generate(paper_pdf, venue="neurips")

    assert Path(output.pptx_path).is_file()
    assert Path(output.pdf_path).is_file()
    assert Path(output.preview_path).is_file()
    assert output.preflight.passed, [c.id for c in output.preflight.failures]
    assert output.figure_decisions and output.figure_decisions[0].decision == "reuse"

    # PDF physical size must equal the NeurIPS generation default 48x36in.
    import fitz

    with fitz.open(output.pdf_path) as doc:
        page = doc[0]
        assert page.rect.width / 72 == pytest.approx(48.0, abs=0.06)
        assert page.rect.height / 72 == pytest.approx(36.0, abs=0.06)

    # IR + provenance persisted.
    ir = json.loads(Path(output.ir_path).read_text())
    assert ir["assets"]["fig1"]["provenance"]["decision"] == "reuse"
    run_dir = Path(output.run_dir)
    assert (run_dir / "storyboard.json").is_file()
    assert (run_dir / "preflight_report.md").is_file()


def test_pipeline_fails_fast_without_soffice(tmp_path: Path, monkeypatch):
    import paperbanana.poster.convert as convert_mod

    monkeypatch.setattr("shutil.which", lambda _: None)
    monkeypatch.setattr(convert_mod, "KNOWN_SOFFICE_LOCATIONS", ("/no/such/place",))
    with pytest.raises(SofficeNotFoundError):
        PosterPipeline(
            settings=_settings(tmp_path),
            vlm_client=_RoutedVLM(),
            image_gen_client=_MockImageGen(),
        )
