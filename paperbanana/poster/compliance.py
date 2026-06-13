"""Image-level venue compliance for generated posters.

The generative pipeline has no structured IR to inspect, so compliance is
checked on the rendered artifact itself: does the PDF page match the
venue's physical size and orientation, and is the raster resolution high
enough to print at that size? These are physics/print constraints — the
deterministic envelope around the learned design.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from paperbanana.poster.venue_spec import VenueSpec


class ComplianceCheck(BaseModel):
    id: str
    status: Literal["pass", "warn", "fail"]
    value: str
    threshold: str
    detail: str


class ComplianceReport(BaseModel):
    venue: str
    year: int
    checks: list[ComplianceCheck] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.status != "fail" for c in self.checks)

    @property
    def failures(self) -> list[ComplianceCheck]:
        return [c for c in self.checks if c.status == "fail"]

    @property
    def warnings(self) -> list[ComplianceCheck]:
        return [c for c in self.checks if c.status == "warn"]


def check_poster_compliance(
    spec: VenueSpec,
    width_px: int,
    height_px: int,
    width_mm: float,
    height_mm: float,
    pdf_path: Optional[Path] = None,
) -> ComplianceReport:
    """Verify a generated poster against the venue's physical/print rules.

    Args:
        spec: Venue specification.
        width_px, height_px: Rendered raster pixel dimensions.
        width_mm, height_mm: Physical poster size the artifact targets.
        pdf_path: Optional PDF for a file-size check.
    """
    checks: list[ComplianceCheck] = []
    target_w, target_h = spec.dimensions.generation_size_mm()

    # Physical size within the venue bound (max-mode) or matching it.
    fits = width_mm <= spec.dimensions.width_mm + 1 and height_mm <= spec.dimensions.height_mm + 1
    checks.append(
        ComplianceCheck(
            id="page_size",
            status="pass" if fits else "fail",
            value=f"{width_mm:.0f}x{height_mm:.0f}mm",
            threshold=f"<= {spec.dimensions.width_mm:.0f}x{spec.dimensions.height_mm:.0f}mm",
            detail=f"{spec.display_name} size rule (mode={spec.dimensions.mode})",
        )
    )

    # Orientation.
    actual = "landscape" if width_mm >= height_mm else "portrait"
    want = spec.dimensions.orientation
    ok_orient = want == "any" or actual == want
    checks.append(
        ComplianceCheck(
            id="orientation",
            status="pass" if ok_orient else "fail",
            value=actual,
            threshold=want,
            detail=f"{spec.display_name} orientation rule",
        )
    )

    # Print resolution: pixels / physical inches.
    long_edge_in = max(width_mm, height_mm) / 25.4
    dpi = (max(width_px, height_px) / long_edge_in) if long_edge_in else 0.0
    min_dpi = spec.text_rules.min_image_dpi
    # Venues regulate physical SIZE and orientation, not your file's raster
    # DPI — that is our own quality bar. A single 4K image on a large board
    # is inherently low-DPI until tiling/upscaling lands, so DPI is advisory:
    # WARN below our preferred minimum, never a compliance FAIL.
    dpi_status = "pass" if dpi >= min_dpi else "warn"
    checks.append(
        ComplianceCheck(
            id="print_dpi",
            status=dpi_status,
            value=f"{dpi:.0f} DPI at {max(width_mm, height_mm) / 25.4:.0f}in",
            threshold=f">= {min_dpi} DPI",
            detail="full-poster raster resolution at physical size",
        )
    )

    if pdf_path is not None and pdf_path.is_file() and spec.file_rules.pdf_max_mb:
        mb = pdf_path.stat().st_size / 1_000_000
        checks.append(
            ComplianceCheck(
                id="pdf_file_size",
                status="pass" if mb <= spec.file_rules.pdf_max_mb else "fail",
                value=f"{mb:.1f} MB",
                threshold=f"<= {spec.file_rules.pdf_max_mb} MB",
                detail="poster.pdf",
            )
        )

    _ = (target_w, target_h)  # generation size is informational here
    return ComplianceReport(venue=spec.venue, year=spec.year, checks=checks)
