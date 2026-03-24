"""PaperBanana Studio — local browser UI for diagram, plot, and evaluation workflows."""

from __future__ import annotations

__all__ = ["launch_studio", "build_studio_app"]


def launch_studio(**kwargs):
    """Start the Gradio studio (requires ``pip install 'paperbanana[studio]'``)."""
    from paperbanana.studio.app import launch_studio as _launch

    return _launch(**kwargs)


def build_studio_app(**kwargs):
    """Build the Gradio Blocks app without launching (for tests and embedding)."""
    from paperbanana.studio.app import build_studio_app as _build

    return _build(**kwargs)
