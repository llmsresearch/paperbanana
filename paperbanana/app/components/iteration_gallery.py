"""Reusable iteration history gallery component."""

from __future__ import annotations

from paperbanana.core.types import GenerationOutput, IterationRecord


def build_gallery_items(result: GenerationOutput) -> list[tuple[str, str]]:
    """Convert GenerationOutput iterations into Gradio Gallery items.

    Returns list of (image_path, caption) tuples.
    """
    items = []
    for i, record in enumerate(result.iterations):
        caption = f"Iter {record.iteration}"
        if record.critique and record.critique.needs_revision:
            caption += f" | {record.critique.summary[:80]}"
        elif record.critique:
            caption += " | Publication-ready"
        items.append((record.image_path, caption))
    return items


def format_critique_detail(record: IterationRecord) -> str:
    """Format a single iteration's critique for display."""
    lines = [f"**Iteration {record.iteration}**"]
    if record.critique:
        if record.critique.needs_revision:
            lines.append("Status: Needs revision")
            for j, sug in enumerate(record.critique.critic_suggestions, 1):
                lines.append(f"  {j}. {sug}")
        else:
            lines.append("Status: Publication-ready")
    else:
        lines.append("Status: No critique available")
    return "\n".join(lines)


def format_all_critiques(result: GenerationOutput) -> str:
    """Format all iteration critiques for display in a textbox."""
    if not result.iterations:
        return "No iterations recorded."
    sections = [format_critique_detail(r) for r in result.iterations]
    return "\n\n---\n\n".join(sections)
