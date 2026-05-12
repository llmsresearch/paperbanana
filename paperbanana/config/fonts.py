"""Font configuration and priority loading for diagram rendering."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["FontConfig", "get_default_font_config"]


@dataclass
class FontConfig:
    """Font configuration with priority-based font loading and fallbacks."""

    primary_fonts: list[str]
    fallback_fonts: list[str] | None = None

    def __post_init__(self) -> None:
        """Validate and set defaults."""
        if not self.primary_fonts:
            raise ValueError("primary_fonts must not be empty")
        if self.fallback_fonts is None:
            self.fallback_fonts = ["Helvetica", "Arial", "sans-serif"]

    def get_font_string(self, separator: str = ",") -> str:
        """Get a font string suitable for Graphviz.

        Combines primary and fallback fonts with the specified separator.

        Args:
            separator: Font separator for the output string (default: ",")

        Returns:
            A font string like "Tahoma,Helvetica,Arial"
        """
        all_fonts = self.primary_fonts + (self.fallback_fonts or [])
        return separator.join(all_fonts)

    def get_first_available_font(self) -> str:
        """Get the first font from the priority list.

        Returns:
            The first font in the primary fonts list.
        """
        return self.primary_fonts[0]


def get_default_font_config() -> FontConfig:
    """Get the default font configuration.

    Uses Tahoma as the primary font with fallbacks to Helvetica and Arial.

    Returns:
        FontConfig with Tahoma as primary font.
    """
    return FontConfig(
        primary_fonts=["Tahoma"],
        fallback_fonts=["Helvetica", "Arial", "sans-serif"],
    )


def get_legacy_font_config() -> FontConfig:
    """Get the legacy font configuration (Helvetica-based).

    Returns:
        FontConfig with Helvetica as primary font.
    """
    return FontConfig(
        primary_fonts=["Helvetica"],
        fallback_fonts=["Arial", "sans-serif"],
    )
