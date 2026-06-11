#!/usr/bin/env python3
"""Extract a LaTeX section body for PaperBanana.

Used by the PaperBanana GitHub Action to pull the methodology section out of a
paper's .tex source before figure generation. Stdlib only — runs standalone on
the Actions runner before paperbanana itself is needed.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_SECTION_CMD = re.compile(r"\\section\*?\s*\{")
_END_DOCUMENT = re.compile(r"\\end\s*\{document\}")


def strip_comments(text: str) -> str:
    """Remove LaTeX comments (unescaped % to end of line), preserving \\%."""
    out_lines = []
    for line in text.splitlines():
        search_from = 0
        while True:
            idx = line.find("%", search_from)
            if idx == -1:
                out_lines.append(line)
                break
            backslashes = 0
            k = idx - 1
            while k >= 0 and line[k] == "\\":
                backslashes += 1
                k -= 1
            if backslashes % 2 == 0:
                out_lines.append(line[:idx])
                break
            search_from = idx + 1
    return "\n".join(out_lines)


def find_sections(text: str) -> list[dict]:
    """Locate every \\section{...} with brace-aware title parsing.

    Returns dicts with title, cmd_start (offset of the backslash) and
    body_start (offset just past the closing brace of the title).
    """
    sections = []
    for match in _SECTION_CMD.finditer(text):
        depth = 1
        i = match.end()
        while i < len(text) and depth > 0:
            char = text[i]
            if char == "\\":
                i += 2
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            i += 1
        title = text[match.end() : i - 1].strip()
        sections.append({"title": title, "cmd_start": match.start(), "body_start": i})
    return sections


def extract(text: str, section_name: str) -> str:
    """Return the body of the first section whose title contains section_name.

    Matching is case-insensitive. The body runs from the end of the section
    heading to the next \\section, \\end{document}, or end of file — whichever
    comes first — so subsections are included.
    """
    text = strip_comments(text)
    sections = find_sections(text)
    if not sections:
        raise LookupError("no \\section commands found in the file")

    needle = section_name.lower()
    chosen_idx = next(
        (i for i, s in enumerate(sections) if needle in s["title"].lower()),
        None,
    )
    if chosen_idx is None:
        available = ", ".join(repr(s["title"]) for s in sections)
        raise LookupError(f"no section matching {section_name!r}; available: {available}")

    chosen = sections[chosen_idx]
    if chosen_idx + 1 < len(sections):
        end = sections[chosen_idx + 1]["cmd_start"]
    else:
        end_doc = _END_DOCUMENT.search(text, chosen["body_start"])
        end = end_doc.start() if end_doc else len(text)

    body = text[chosen["body_start"] : end].strip()
    body = re.sub(r"\n{3,}", "\n\n", body)
    if not body:
        raise LookupError(f"section {chosen['title']!r} matched but its body is empty")
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", required=True, help="Path to the .tex source file")
    parser.add_argument(
        "--section",
        default="Method",
        help="Section title to extract (case-insensitive substring match)",
    )
    parser.add_argument("--out", required=True, help="Where to write the extracted text")
    args = parser.parse_args()

    tex_path = Path(args.tex)
    if not tex_path.is_file():
        print(f"error: tex file not found: {tex_path}", file=sys.stderr)
        return 1

    try:
        body = extract(tex_path.read_text(encoding="utf-8"), args.section)
    except LookupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf-8")
    print(f"extracted {len(body)} chars from {tex_path} -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
