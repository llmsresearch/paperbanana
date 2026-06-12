"""Ingestion tests with a synthetic PDF and a scripted mock VLM."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paperbanana.poster.agents.figure_detector import DetectedRegion, FigureDetectorAgent
from paperbanana.poster.agents.paper_metadata import PaperMetadataAgent
from paperbanana.poster.ingest import IngestError, ingest_paper, render_page

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"


class _MockVLM:
    """Routed VLM: page detections are concurrent, so responses key on the
    prompt content (metadata vs. per-page detection), never on call order."""

    name = "mock"
    model_name = "mock-model"
    cost_tracker = None

    def __init__(self, metadata: str, page_detections: dict[int, str]):
        self._metadata = metadata
        self._page_detections = page_detections
        self.calls: list[dict] = []

    async def generate(self, prompt, images=None, **kwargs):
        self.calls.append({"prompt": prompt, "n_images": len(images or [])})
        if "bibliographic metadata" in prompt.lower():
            return self._metadata
        for page, response in self._page_detections.items():
            if f"page {page} of an academic paper" in prompt:
                return response
        raise AssertionError(f"unrouted mock prompt: {prompt[:120]}")


@pytest.fixture
def paper_pdf(tmp_path: Path) -> Path:
    """A 2-page PDF with searchable text including a 'Figure 1' caption."""
    import fitz

    doc = fitz.open()
    page1 = doc.new_page(width=612, height=792)
    page1.insert_text((72, 100), "A Synthetic Paper About Posters", fontsize=16)
    page1.insert_text((72, 130), "Ada Lovelace, Alan Turing", fontsize=10)
    page1.insert_text((72, 170), "Abstract", fontsize=12)
    page1.insert_text((72, 190), "We study automated poster generation.", fontsize=10)
    page2 = doc.new_page(width=612, height=792)
    page2.insert_text((72, 90), "3 Method", fontsize=12)
    page2.insert_text((72, 110), "Our method has two phases.", fontsize=10)
    rect = fitz.Rect(100, 200, 500, 400)
    page2.draw_rect(rect, color=(0, 0, 1), width=2)
    page2.insert_text((100, 420), "Figure 1: The synthetic figure.", fontsize=9)
    path = tmp_path / "paper.pdf"
    doc.save(str(path))
    doc.close()
    return path


def _metadata_response() -> str:
    return json.dumps(
        {
            "title": "A Synthetic Paper About Posters",
            "authors": ["Ada Lovelace", "Alan Turing"],
            "affiliations": ["Analytical Engines Lab"],
            "abstract": "We study automated poster generation.",
        }
    )


def _page2_detection() -> str:
    # Page is 612x792pt; figure rect (100,200)-(500,400) + caption at y=420.
    # Normalized to 0-1000: x 163-816, y 252-545 (incl. caption).
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


async def test_ingest_end_to_end(paper_pdf: Path, tmp_path: Path):
    vlm = _MockVLM(_metadata_response(), {1: "[]", 2: _page2_detection()})
    detector = FigureDetectorAgent(vlm, prompt_dir=str(PROMPT_DIR))
    metadata_agent = PaperMetadataAgent(vlm, prompt_dir=str(PROMPT_DIR))
    out_dir = tmp_path / "paper_assets"

    assets = await ingest_paper(paper_pdf, detector, metadata_agent, out_dir)

    assert assets.title == "A Synthetic Paper About Posters"
    assert assets.authors == ["Ada Lovelace", "Alan Turing"]
    assert assets.page_count == 2
    assert any("Method" in s.heading for s in assets.sections)

    assert len(assets.figures) == 1
    fig = assets.figures[0]
    assert fig.id == "fig1"
    assert fig.page == 2
    assert fig.caption_anchored, "caption 'Figure 1' is in the text layer inside the bbox"
    crop = Path(fig.image_path)
    assert crop.is_file()
    from PIL import Image

    with Image.open(crop) as img:
        assert img.width > 100 and img.height > 100

    assert (out_dir / "paper_assets.json").is_file()


async def test_ingest_unanchored_caption(paper_pdf: Path, tmp_path: Path):
    detection = json.dumps(
        [
            {
                "kind": "figure",
                "figure_number": "Figure 9",  # not in the text layer
                "caption": "Figure 9: Phantom.",
                "bbox": [160, 245, 820, 560],
            }
        ]
    )
    vlm = _MockVLM(_metadata_response(), {1: "[]", 2: detection})
    assets = await ingest_paper(
        paper_pdf,
        FigureDetectorAgent(vlm, prompt_dir=str(PROMPT_DIR)),
        PaperMetadataAgent(vlm, prompt_dir=str(PROMPT_DIR)),
        tmp_path / "out",
    )
    assert assets.figures[0].caption_anchored is False


async def test_ingest_dedupes_numbered_figures(paper_pdf: Path, tmp_path: Path):
    dupe = json.loads(_page2_detection())
    dupe.append(dict(dupe[0]))  # same Figure 1 twice
    vlm = _MockVLM(_metadata_response(), {1: "[]", 2: json.dumps(dupe)})
    assets = await ingest_paper(
        paper_pdf,
        FigureDetectorAgent(vlm, prompt_dir=str(PROMPT_DIR)),
        PaperMetadataAgent(vlm, prompt_dir=str(PROMPT_DIR)),
        tmp_path / "out",
    )
    assert len(assets.figures) == 1


async def test_ingest_bad_detection_json_raises(paper_pdf: Path, tmp_path: Path):
    vlm = _MockVLM(_metadata_response(), {1: "no json here", 2: "[]"})
    with pytest.raises(ValueError, match="no JSON array"):
        await ingest_paper(
            paper_pdf,
            FigureDetectorAgent(vlm, prompt_dir=str(PROMPT_DIR)),
            PaperMetadataAgent(vlm, prompt_dir=str(PROMPT_DIR)),
            tmp_path / "out",
        )


async def test_ingest_missing_pdf_raises(tmp_path: Path):
    vlm = _MockVLM("{}", {})
    with pytest.raises(IngestError, match="not found"):
        await ingest_paper(
            tmp_path / "missing.pdf",
            FigureDetectorAgent(vlm, prompt_dir=str(PROMPT_DIR)),
            PaperMetadataAgent(vlm, prompt_dir=str(PROMPT_DIR)),
            tmp_path / "out",
        )


def test_render_page(paper_pdf: Path):
    img = render_page(paper_pdf, 0, dpi=100)
    # 612x792pt at 100dpi -> 850x1100px
    assert img.size == (850, 1100)


def test_degenerate_bbox_raises():
    region = DetectedRegion(kind="figure", caption="x", bbox=[500, 500, 400, 600])
    with pytest.raises(ValueError, match="degenerate"):
        region.bbox_norm()
