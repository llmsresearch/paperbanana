"""Tests for orchestration planning helpers."""

from __future__ import annotations

from paperbanana.core.orchestrate import split_paper_sections


def test_split_paper_sections_ignores_pdf_running_headers_and_page_numbers():
    """Noisy PDF extraction should still keep real section boundaries."""
    noisy_text = "\n".join(
        [
            "PaperBanana 2026",
            "1",
            "A Very Good Paper Title",
            "",
            "Abstract",
            "We summarize the work here.",
            "",
            "PaperBanana 2026",
            "2",
            "1 Introduction",
            "This page includes motivation and setup.",
            "",
            "PaperBanana 2026",
            "3",
            "2 Method",
            "We describe the encoder, retriever, and critic.",
            "",
            "PaperBanana 2026",
            "4",
            "3 Experiments",
            "We compare against strong baselines.",
            "",
            "PaperBanana 2026",
            "5",
        ]
    )

    sections = split_paper_sections(noisy_text)

    headings = [section["heading"] for section in sections]
    assert headings == ["Abstract", "1 Introduction", "2 Method", "3 Experiments"]
    assert all("PaperBanana 2026" not in section["content"] for section in sections)
    assert all(section["content"] not in {"1", "2", "3", "4", "5"} for section in sections)
