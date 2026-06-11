"""Tests for the GitHub Action LaTeX section extractor."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[2] / "integrations" / "github-action" / "extract_section.py"
_spec = importlib.util.spec_from_file_location("extract_section", _SCRIPT)
extract_section = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(extract_section)


SAMPLE = r"""
\documentclass{article}
\begin{document}
\section{Introduction}
Intro text here.
\section{Methodology}
We propose a two-phase pipeline. % inline comment
The first phase retrieves examples (5\% of the corpus).
\subsection{Planning}
The planner produces a layout.
\section{Experiments}
Results here.
\end{document}
"""


def test_extracts_section_body():
    body = extract_section.extract(SAMPLE, "Methodology")
    assert "two-phase pipeline" in body
    assert "Intro text" not in body
    assert "Results here" not in body


def test_includes_subsections():
    body = extract_section.extract(SAMPLE, "Methodology")
    assert r"\subsection{Planning}" in body
    assert "planner produces a layout" in body


def test_match_is_case_insensitive_substring():
    body = extract_section.extract(SAMPLE, "method")
    assert "two-phase pipeline" in body


def test_strips_comments_but_keeps_escaped_percent():
    body = extract_section.extract(SAMPLE, "Methodology")
    assert "inline comment" not in body
    assert r"5\% of the corpus" in body


def test_starred_section_and_end_document_boundary():
    text = r"""
\section*{Method}
Body of the starred section.
\end{document}
trailing junk
"""
    body = extract_section.extract(text, "Method")
    assert "Body of the starred section." in body
    assert "trailing junk" not in body


def test_missing_section_lists_available():
    with pytest.raises(LookupError, match="Introduction"):
        extract_section.extract(SAMPLE, "Conclusion")


def test_no_sections_at_all():
    with pytest.raises(LookupError, match="no \\\\section commands"):
        extract_section.extract("just plain text", "Method")


def test_braces_in_section_title():
    text = r"""
\section{Method \& \textbf{Approach}}
Brace-heavy title body.
\section{Other}
Nope.
"""
    body = extract_section.extract(text, "approach")
    assert "Brace-heavy title body." in body
    assert "Nope" not in body
