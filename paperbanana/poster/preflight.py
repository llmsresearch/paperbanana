"""Deterministic print-fidelity and venue-compliance checks.

Preflight runs twice per iteration loop pass (feeding failures into the
critic) and once more on the final artifacts (the shipped report). Every
check is computable from the IR, the venue spec, and the rendered files
— no model involvement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from paperbanana.poster.renderer import (
    PANEL_PADDING_MM,
    measure_panel_required_height_mm,
    resolve_figure_placement,
)
from paperbanana.poster.types import (
    FigureElement,
    PosterIR,
    PreflightCheck,
    PreflightReport,
    TextElement,
    TextLevel,
)
from paperbanana.poster.venue_spec import RequiredElementRule, VenueSpec

#: Framework legibility floors (pt), independent of venue rules; grounded in
#: the standard "readable from 1.5-2m viewing distance" guidance for posters.
DEFAULT_LEGIBILITY_MIN_PT: dict[TextLevel, float] = {
    "title": 60,
    "authors": 32,
    "affiliation": 24,
    "heading": 32,
    "body": 24,
    "caption": 18,
    "footnote": 14,
    "banner": 40,
    "big_number": 64,
}

#: Dimension tolerance for 'exact' venue sizes, in mm.
EXACT_SIZE_TOLERANCE_MM = 2.0


def run_preflight(
    ir: PosterIR,
    spec: VenueSpec,
    pdf_path: Optional[Path] = None,
    png_path: Optional[Path] = None,
) -> PreflightReport:
    """Run all checks; file checks are skipped when paths are not given."""
    checks: list[PreflightCheck] = []
    checks.append(_check_page_size(ir, spec))
    checks.append(_check_orientation(ir, spec))
    checks.extend(_check_font_minima(ir, spec))
    checks.extend(_check_image_dpi(ir, spec))
    checks.extend(_check_required_elements(ir, spec))
    checks.extend(_check_text_overflow(ir))
    checks.extend(_check_contrast(ir))
    checks.extend(_check_caption_anchoring(ir))
    checks.append(_check_callout_count(ir))
    checks.extend(_check_faithfulness(ir))
    if pdf_path is not None:
        checks.append(_check_pdf_size(pdf_path, spec))
    if png_path is not None:
        checks.append(_check_png_pixels(png_path, spec))
    return PreflightReport(checks=checks)


def render_preflight_markdown(report: PreflightReport) -> str:
    """Human-readable preflight report."""
    icon = {"pass": "PASS", "fail": "FAIL", "warn": "WARN"}
    lines = [
        "# Poster Preflight Report",
        "",
        f"**Result: {'PASSED' if report.passed else 'FAILED'}** "
        f"({len(report.failures)} failures, {len(report.warnings)} warnings)",
        "",
        "| Check | Status | Value | Threshold | Detail |",
        "|---|---|---|---|---|",
    ]
    for c in report.checks:
        lines.append(f"| {c.id} | {icon[c.status]} | {c.value} | {c.threshold} | {c.detail} |")
    return "\n".join(lines) + "\n"


def _check_page_size(ir: PosterIR, spec: VenueSpec) -> PreflightCheck:
    dims = spec.dimensions
    value = f"{ir.size.width_mm:.0f}x{ir.size.height_mm:.0f}mm"
    if dims.mode == "exact":
        ok = (
            abs(ir.size.width_mm - dims.width_mm) <= EXACT_SIZE_TOLERANCE_MM
            and abs(ir.size.height_mm - dims.height_mm) <= EXACT_SIZE_TOLERANCE_MM
        )
        threshold = f"exactly {dims.width_mm:.0f}x{dims.height_mm:.0f}mm"
    else:
        ok = ir.size.width_mm <= dims.width_mm and ir.size.height_mm <= dims.height_mm
        threshold = f"within {dims.width_mm:.0f}x{dims.height_mm:.0f}mm"
    return PreflightCheck(
        id="page_size",
        status="pass" if ok else "fail",
        value=value,
        threshold=threshold,
        detail=f"{spec.display_name} ({dims.mode} rule)",
    )


def _check_orientation(ir: PosterIR, spec: VenueSpec) -> PreflightCheck:
    required = spec.dimensions.orientation
    ok = required == "any" or ir.orientation == required
    return PreflightCheck(
        id="orientation",
        status="pass" if ok else "fail",
        value=ir.orientation,
        threshold=required,
        detail=f"{spec.display_name} orientation rule",
    )


def _check_font_minima(ir: PosterIR, spec: VenueSpec) -> list[PreflightCheck]:
    checks = []
    levels_in_use = {
        el.level for panel in ir.panels for el in panel.elements if isinstance(el, TextElement)
    }
    levels_in_use.add("heading")  # panel titles render at heading level
    levels_in_use.add("caption")  # figure captions render at caption level
    for level in sorted(levels_in_use):
        actual = ir.style.type_scale_pt.get(level)
        if actual is None:
            checks.append(
                PreflightCheck(
                    id=f"font_minima.{level}",
                    status="fail",
                    value="(unset)",
                    threshold="defined in type scale",
                    detail=f"text level '{level}' is used but has no size in the type scale",
                )
            )
            continue
        venue_min = spec.text_rules.min_pt.get(level, 0)
        floor = max(venue_min, DEFAULT_LEGIBILITY_MIN_PT.get(level, 0))
        source = "venue rule" if venue_min >= floor and venue_min > 0 else "legibility floor"
        checks.append(
            PreflightCheck(
                id=f"font_minima.{level}",
                status="pass" if actual >= floor else "fail",
                value=f"{actual:.0f}pt",
                threshold=f">= {floor:.0f}pt",
                detail=source,
            )
        )
    return checks


def _check_image_dpi(ir: PosterIR, spec: VenueSpec) -> list[PreflightCheck]:
    checks = []
    min_dpi = spec.text_rules.min_image_dpi
    for panel in ir.panels:
        for el in panel.elements:
            if not isinstance(el, FigureElement):
                continue
            asset = ir.assets[el.asset_id]
            placement = resolve_figure_placement(
                panel, el, asset.width_px, asset.height_px, page_height_mm=ir.size.height_mm
            )
            dpi = asset.effective_dpi(placement.width_mm)
            checks.append(
                PreflightCheck(
                    id=f"image_dpi.{el.asset_id}",
                    status="pass" if dpi >= min_dpi else "fail",
                    value=f"{dpi:.0f} DPI at {placement.width_mm:.0f}mm",
                    threshold=f">= {min_dpi} DPI",
                    detail=(
                        f"{asset.width_px}px wide, decision={asset.provenance.decision}, "
                        f"panel '{panel.id}'"
                    ),
                )
            )
    return checks


def _check_required_elements(ir: PosterIR, spec: VenueSpec) -> list[PreflightCheck]:
    checks = []
    for rule in spec.required_elements:
        checker = _ELEMENT_RULE_CHECKERS[rule.rule]
        ok, value = checker(ir, rule)
        checks.append(
            PreflightCheck(
                id=f"required.{rule.id}",
                status="pass" if ok else rule.severity,
                value=value,
                threshold="present",
                detail=rule.description,
            )
        )
    return checks


def _rule_text_level_present(ir: PosterIR, rule: RequiredElementRule) -> tuple[bool, str]:
    level = rule.params.get("level")
    found = any(
        isinstance(el, TextElement) and el.level == level
        for panel in ir.panels
        for el in panel.elements
    )
    return found, f"text level '{level}' {'found' if found else 'missing'}"


def _rule_element_kind_present(ir: PosterIR, rule: RequiredElementRule) -> tuple[bool, str]:
    kind = rule.params.get("kind")
    found = any(el.kind == kind for panel in ir.panels for el in panel.elements)
    return found, f"element kind '{kind}' {'found' if found else 'missing'}"


def _rule_text_contains(ir: PosterIR, rule: RequiredElementRule) -> tuple[bool, str]:
    needle = rule.params.get("text", "")
    found = any(
        isinstance(el, TextElement) and needle.lower() in el.content.lower()
        for panel in ir.panels
        for el in panel.elements
    )
    return found, f"text '{needle}' {'found' if found else 'missing'}"


def _rule_panel_role_present(ir: PosterIR, rule: RequiredElementRule) -> tuple[bool, str]:
    role = rule.params.get("role")
    found = any(p.role == role for p in ir.panels)
    return found, f"panel role '{role}' {'found' if found else 'missing'}"


_ELEMENT_RULE_CHECKERS: dict[str, Callable[[PosterIR, RequiredElementRule], tuple[bool, str]]] = {
    "text_level_present": _rule_text_level_present,
    "element_kind_present": _rule_element_kind_present,
    "text_contains": _rule_text_contains,
    "panel_role_present": _rule_panel_role_present,
}


def _check_text_overflow(ir: PosterIR) -> list[PreflightCheck]:
    checks = []
    for panel in ir.panels_in_order():
        if panel.bbox is None:
            continue
        available = panel.bbox.h_mm - 2 * PANEL_PADDING_MM
        required = measure_panel_required_height_mm(panel, ir)
        checks.append(
            PreflightCheck(
                id=f"text_overflow.{panel.id}",
                status="pass" if required <= available else "fail",
                value=f"{required:.0f}mm required",
                threshold=f"<= {available:.0f}mm available",
                detail=f"panel '{panel.id}' at current type scale (incl. safety margin)",
            )
        )
    return checks


def _relative_luminance(hex_color: str) -> float:
    v = hex_color.lstrip("#")
    rgb = [int(v[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    r, g, b = linear
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(fg_hex: str, bg_hex: str) -> float:
    """WCAG contrast ratio between two hex colors."""
    l1 = _relative_luminance(fg_hex)
    l2 = _relative_luminance(bg_hex)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def _check_contrast(ir: PosterIR) -> list[PreflightCheck]:
    checks = []
    for panel in ir.panels:
        is_header = panel.role == "header"
        is_accent = panel.emphasis == "accent" or any(el.kind == "banner" for el in panel.elements)
        if is_header:
            fg, bg = ir.style.palette["background"], ir.style.palette["primary"]
        elif is_accent:
            fg, bg = ir.style.palette["background"], ir.style.palette["accent"]
        else:
            fg, bg = ir.style.palette["text"], ir.style.palette["panel_bg"]
        ratio = contrast_ratio(fg, bg)
        sizes = [
            ir.style.type_scale_pt[el.level] for el in panel.elements if isinstance(el, TextElement)
        ]
        if not sizes:
            continue
        # WCAG: 3.0 suffices for large text (>= 18pt bold / 24pt); poster text
        # levels are all "large", but body copy still gets the stricter bar.
        required = 3.0 if min(sizes) >= 24 else 4.5
        checks.append(
            PreflightCheck(
                id=f"contrast.{panel.id}",
                status="pass" if ratio >= required else "fail",
                value=f"{ratio:.1f}:1",
                threshold=f">= {required}:1",
                detail=f"{fg} on {bg} (WCAG)",
            )
        )
    return checks


def _check_callout_count(ir: PosterIR) -> PreflightCheck:
    from paperbanana.poster.types import MAX_BIG_NUMBERS

    count = sum(1 for panel in ir.panels for el in panel.elements if el.kind == "big_number")
    if count > MAX_BIG_NUMBERS:
        status = "fail"
    elif count == MAX_BIG_NUMBERS:
        status = "warn"
    else:
        status = "pass"
    return PreflightCheck(
        id="callout_count",
        status=status,
        value=str(count),
        threshold=f"<= {MAX_BIG_NUMBERS}",
        detail="big-number callouts lose impact beyond a couple per poster",
    )


def _check_caption_anchoring(ir: PosterIR) -> list[PreflightCheck]:
    checks = []
    for asset in ir.assets.values():
        if asset.provenance.origin == "paper" and not asset.provenance.caption_anchored:
            checks.append(
                PreflightCheck(
                    id=f"caption_anchor.{asset.id}",
                    status="warn",
                    value="not anchored",
                    threshold="caption matched to PDF text layer",
                    detail=(
                        f"figure '{asset.id}' detection could not be cross-checked against "
                        "the paper's caption text; verify the crop manually"
                    ),
                )
            )
    return checks


def _check_faithfulness(ir: PosterIR) -> list[PreflightCheck]:
    checks = []
    for asset in ir.assets.values():
        if asset.provenance.faithfulness == "failed":
            checks.append(
                PreflightCheck(
                    id=f"faithfulness.{asset.id}",
                    status="fail",
                    value="failed",
                    threshold="verified or not_required",
                    detail=f"re-authored figure '{asset.id}' failed the faithfulness gate",
                )
            )
        elif asset.provenance.decision == "reauthor":
            checks.append(
                PreflightCheck(
                    id=f"faithfulness.{asset.id}",
                    status="pass" if asset.provenance.faithfulness == "verified" else "fail",
                    value=asset.provenance.faithfulness,
                    threshold="verified",
                    detail=f"re-authored figure '{asset.id}' must pass the faithfulness gate",
                )
            )
    return checks


def _check_pdf_size(pdf_path: Path, spec: VenueSpec) -> PreflightCheck:
    size_mb = pdf_path.stat().st_size / (1024 * 1024)
    limit = spec.file_rules.pdf_max_mb
    ok = limit is None or size_mb <= limit
    return PreflightCheck(
        id="pdf_file_size",
        status="pass" if ok else "fail",
        value=f"{size_mb:.1f} MB",
        threshold=f"<= {limit} MB" if limit else "(no venue limit)",
        detail=str(pdf_path.name),
    )


def _check_png_pixels(png_path: Path, spec: VenueSpec) -> PreflightCheck:
    from PIL import Image

    with Image.open(png_path) as img:
        w, h = img.size
    limit = spec.file_rules.png_max_px
    ok = limit is None or max(w, h) <= limit
    return PreflightCheck(
        id="png_pixel_limit",
        status="pass" if ok else "fail",
        value=f"{w}x{h}px",
        threshold=f"max dim <= {limit}px" if limit else "(no venue limit)",
        detail=str(png_path.name),
    )
