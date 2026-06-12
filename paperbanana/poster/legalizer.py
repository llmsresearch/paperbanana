"""Proposal legalization: validate, repair mechanically, or send back.

The split honors the no-fallback rule:

- **Fatal** violations change design intent (missing content, ungrounded
  callouts, structural nonsense). They are returned to the proposer as
  feedback for a bounded number of re-proposals, then raise
  :class:`LayoutProposalError`. Never auto-fixed.
- **Repairable** violations are mechanical grid problems (span past the
  band edge, columns too narrow). Each repair is an explicit, logged
  :class:`RepairAction` that lands in the IR's ``LayoutProvenance`` —
  visible in every run artifact.
"""

from __future__ import annotations

import structlog

from paperbanana.poster.proposal import (
    LayoutProposal,
    ProposedBand,
    RepairAction,
    Violation,
)
from paperbanana.poster.types import (
    Band,
    BannerElement,
    BigNumberElement,
    LayoutProvenance,
    Panel,
    PosterIR,
    Storyboard,
)

logger = structlog.get_logger()

MAX_BANDS = 5
MIN_COLUMN_WIDTH_MM = 200.0
BANNER_PANEL_ID = "takeaway_banner"


class LayoutProposalError(ValueError):
    """No legal proposal could be obtained within the re-proposal budget."""

    def __init__(self, violations: list[Violation], rounds: int):
        self.violations = violations
        listing = "; ".join(f"{v.code}({v.target}): {v.detail}" for v in violations[:8])
        super().__init__(
            f"layout proposer produced no legal structure after {rounds} attempt(s); "
            f"remaining violations: {listing}"
        )


def validate_proposal(
    proposal: LayoutProposal,
    storyboard: Storyboard,
    figure_asset_ids: set[str],
    page_width_mm: float,
    margin_mm: float,
    gutter_mm: float,
) -> list[Violation]:
    """All violations in a proposal, fatal and repairable."""
    violations: list[Violation] = []
    bands_by_id: dict[str, ProposedBand] = {}
    for band in proposal.bands:
        if band.id in bands_by_id:
            violations.append(
                Violation(
                    code="rename_duplicate_band",
                    fatal=False,
                    target=band.id,
                    detail="duplicate band id",
                )
            )
        bands_by_id[band.id] = band

    headers = [b for b in proposal.bands if b.kind == "header"]
    if not headers:
        violations.append(
            Violation(
                code="add_header_band",
                fatal=False,
                target="header",
                detail="proposal lacks a header band; one will be inserted at the top",
            )
        )
    elif len(headers) > 1:
        violations.append(
            Violation(
                code="multiple_header_bands",
                fatal=True,
                target=", ".join(b.id for b in headers),
                detail="exactly one header band is allowed",
            )
        )
    banners = [b for b in proposal.bands if b.kind == "banner"]
    if len(banners) > 1:
        violations.append(
            Violation(
                code="multiple_banner_bands",
                fatal=True,
                target=", ".join(b.id for b in banners),
                detail="at most one banner band is allowed",
            )
        )
    if len(proposal.bands) > MAX_BANDS:
        violations.append(
            Violation(
                code="too_many_bands",
                fatal=True,
                target=str(len(proposal.bands)),
                detail=f"at most {MAX_BANDS} bands; simplify the structure",
            )
        )
    if proposal.use_banner and not any(b.kind == "banner" for b in proposal.bands):
        violations.append(
            Violation(
                code="add_banner_band",
                fatal=False,
                target="banner",
                detail="use_banner=true but no banner band declared; inserting one",
            )
        )
    if proposal.use_banner and not storyboard.takeaway:
        violations.append(
            Violation(
                code="banner_without_takeaway",
                fatal=True,
                target="use_banner",
                detail="use_banner=true but the storyboard has no takeaway sentence",
            )
        )
    if banners and not proposal.use_banner:
        violations.append(
            Violation(
                code="drop_empty_band",
                fatal=False,
                target=banners[0].id,
                detail="banner band declared but use_banner=false",
            )
        )

    sb_ids = {p.id for p in storyboard.panels}
    placed_ids: set[str] = set()
    content_w = page_width_mm - 2 * margin_mm
    for placement in proposal.placements:
        if placement.panel_id not in sb_ids:
            violations.append(
                Violation(
                    code="unknown_panel",
                    fatal=True,
                    target=placement.panel_id,
                    detail=f"not a storyboard panel (known: {sorted(sb_ids)})",
                )
            )
            continue
        if placement.panel_id in placed_ids:
            violations.append(
                Violation(
                    code="duplicate_placement",
                    fatal=True,
                    target=placement.panel_id,
                    detail="panel placed more than once",
                )
            )
        placed_ids.add(placement.panel_id)
        band = bands_by_id.get(placement.band_id)
        if band is None:
            violations.append(
                Violation(
                    code="unknown_band_ref",
                    fatal=True,
                    target=placement.panel_id,
                    detail=f"references unknown band '{placement.band_id}'",
                )
            )
            continue
        if band.kind != "body":
            violations.append(
                Violation(
                    code="promote_full_span",
                    fatal=False,
                    target=placement.panel_id,
                    detail=f"panel in {band.kind} band must span the full band",
                )
            )
        if placement.column + placement.col_span > band.columns:
            violations.append(
                Violation(
                    code="clamp_col_span",
                    fatal=False,
                    target=placement.panel_id,
                    detail=(
                        f"occupies columns [{placement.column}, "
                        f"{placement.column + placement.col_span}) of {band.columns}"
                    ),
                )
            )
    missing = sb_ids - placed_ids
    for panel_id in sorted(missing):
        violations.append(
            Violation(
                code="missing_panel",
                fatal=True,
                target=panel_id,
                detail="storyboard panel has no placement",
            )
        )
    for band in proposal.bands:
        if band.kind == "body":
            members = [p for p in proposal.placements if p.band_id == band.id]
            if not members:
                violations.append(
                    Violation(
                        code="drop_empty_band",
                        fatal=False,
                        target=band.id,
                        detail="body band with no placements",
                    )
                )
            col_w = (content_w - (band.columns - 1) * gutter_mm) / band.columns
            if col_w < MIN_COLUMN_WIDTH_MM:
                violations.append(
                    Violation(
                        code="reduce_band_columns",
                        fatal=False,
                        target=band.id,
                        detail=(
                            f"{band.columns} columns -> {col_w:.0f}mm wide; "
                            f"minimum is {MIN_COLUMN_WIDTH_MM:.0f}mm"
                        ),
                    )
                )

    stat_ids = {s.id for s in storyboard.key_stats}
    for callout in proposal.callouts:
        if callout.key_stat_id not in stat_ids:
            violations.append(
                Violation(
                    code="callout_unknown_stat",
                    fatal=True,
                    target=callout.key_stat_id,
                    detail=f"not a storyboard key_stat (known: {sorted(stat_ids)})",
                )
            )
        if callout.panel_id not in sb_ids:
            violations.append(
                Violation(
                    code="callout_unknown_panel",
                    fatal=True,
                    target=callout.panel_id,
                    detail="callout targets a panel that does not exist",
                )
            )
    if proposal.hero_figure_id and proposal.hero_figure_id not in figure_asset_ids:
        violations.append(
            Violation(
                code="hero_figure_unknown",
                fatal=True,
                target=proposal.hero_figure_id,
                detail=f"not a curated figure (known: {sorted(figure_asset_ids)})",
            )
        )
    return violations


def repair(
    proposal: LayoutProposal,
    violations: list[Violation],
    page_width_mm: float,
    margin_mm: float,
    gutter_mm: float,
) -> tuple[LayoutProposal, list[RepairAction]]:
    """Apply mechanical repairs for all non-fatal violations, logged."""
    data = proposal.model_dump()
    actions: list[RepairAction] = []
    content_w = page_width_mm - 2 * margin_mm

    for violation in violations:
        if violation.fatal:
            continue
        if violation.code == "add_banner_band":
            header_idx = next((i for i, b in enumerate(data["bands"]) if b["kind"] == "header"), -1)
            data["bands"].insert(header_idx + 1, {"id": "banner", "kind": "banner", "columns": 1})
            actions.append(
                RepairAction(
                    op="add_banner_band",
                    target="banner",
                    before="(absent)",
                    after="banner band after header",
                    reason=violation.detail,
                )
            )
        elif violation.code == "add_header_band":
            data["bands"].insert(0, {"id": "header", "kind": "header", "columns": 1})
            actions.append(
                RepairAction(
                    op="add_header_band",
                    target="header",
                    before="(absent)",
                    after="header band at top",
                    reason=violation.detail,
                )
            )
        elif violation.code == "rename_duplicate_band":
            seen: set[str] = set()
            for band in data["bands"]:
                base = band["id"]
                while band["id"] in seen:
                    band["id"] = f"{band['id']}-2"
                if band["id"] != base:
                    for placement in data["placements"]:
                        if placement["band_id"] == base:
                            placement["band_id"] = band["id"]
                    actions.append(
                        RepairAction(
                            op="rename_duplicate_band",
                            target=base,
                            before=base,
                            after=band["id"],
                            reason="duplicate band id",
                        )
                    )
                seen.add(band["id"])
        elif violation.code == "drop_empty_band":
            kept = [b for b in data["bands"] if b["id"] != violation.target]
            if len(kept) < len(data["bands"]):
                data["bands"] = kept
                actions.append(
                    RepairAction(
                        op="drop_empty_band",
                        target=violation.target,
                        before="declared",
                        after="removed",
                        reason=violation.detail,
                    )
                )
        elif violation.code == "reduce_band_columns":
            for band in data["bands"]:
                if band["id"] != violation.target:
                    continue
                before = band["columns"]
                max_cols = max(1, int((content_w + gutter_mm) // (MIN_COLUMN_WIDTH_MM + gutter_mm)))
                band["columns"] = min(before, max_cols)
                actions.append(
                    RepairAction(
                        op="reduce_band_columns",
                        target=band["id"],
                        before=str(before),
                        after=str(band["columns"]),
                        reason=violation.detail,
                    )
                )
        elif violation.code in ("clamp_col_span", "promote_full_span"):
            bands_by_id = {b["id"]: b for b in data["bands"]}
            for placement in data["placements"]:
                if placement["panel_id"] != violation.target:
                    continue
                band = bands_by_id.get(placement["band_id"])
                if band is None:
                    continue
                before = f"col={placement['column']} span={placement['col_span']}"
                if violation.code == "promote_full_span":
                    placement["column"] = 0
                    placement["col_span"] = band["columns"]
                else:
                    placement["column"] = min(placement["column"], band["columns"] - 1)
                    placement["col_span"] = max(
                        1, min(placement["col_span"], band["columns"] - placement["column"])
                    )
                actions.append(
                    RepairAction(
                        op=violation.code,
                        target=placement["panel_id"],
                        before=before,
                        after=f"col={placement['column']} span={placement['col_span']}",
                        reason=violation.detail,
                    )
                )
    repaired = LayoutProposal(**data)
    for action in actions:
        logger.info(
            "Layout proposal repaired",
            op=action.op,
            target=action.target,
            before=action.before,
            after=action.after,
        )
    return repaired, actions


def build_ir_from_proposal(
    draft_ir: PosterIR,
    proposal: LayoutProposal,
    storyboard: Storyboard,
    repairs: list[RepairAction],
    proposal_index: int = 0,
    exemplar_ids: list[str] | None = None,
    reproposal_rounds: int = 0,
) -> PosterIR:
    """Apply a legal proposal's structure to the content-complete draft IR."""
    data = draft_ir.model_dump()
    # Re-proposals rebuild from the current IR, which may already hold a
    # synthesized banner panel — strip it; it is re-added below if used.
    data["panels"] = [p for p in data["panels"] if p["id"] != BANNER_PANEL_ID]
    for panel_dump in data["panels"]:
        # Callouts are proposal-driven; strip stale ones before re-adding.
        panel_dump["elements"] = [
            el for el in panel_dump["elements"] if el.get("kind") != "big_number"
        ] or panel_dump["elements"]
    bands = [
        Band(id=b.id, kind=b.kind, order=i, columns=b.columns).model_dump()
        for i, b in enumerate(proposal.bands)
    ]
    header_band_id = next(b["id"] for b in bands if b["kind"] == "header")

    panels = []
    order = 0
    for panel_dump in sorted(data["panels"], key=lambda p: p["order"]):
        panel_dump["bbox"] = None
        if panel_dump["role"] == "header":
            panel_dump.update(band_id=header_band_id, column=0, col_span=1, order=order)
            panels.append(panel_dump)
            order += 1
    # Banner panel (synthesized) sits directly after the header in reading order.
    if proposal.use_banner:
        banner_band_id = next(b["id"] for b in bands if b["kind"] == "banner")
        panels.append(
            Panel(
                id=BANNER_PANEL_ID,
                role="takeaway",
                order=order,
                band_id=banner_band_id,
                column=0,
                col_span=1,
                elements=[BannerElement(content=storyboard.takeaway or "")],
            ).model_dump()
        )
        order += 1
    callouts_by_panel: dict[str, list[BigNumberElement]] = {}
    stats_by_id = {s.id: s for s in storyboard.key_stats}
    for callout in proposal.callouts:
        stat = stats_by_id[callout.key_stat_id]
        callouts_by_panel.setdefault(callout.panel_id, []).append(
            BigNumberElement(value=stat.value, label=stat.label)
        )
    for placement in proposal.placements:
        panel_dump = next((p for p in data["panels"] if p["id"] == placement.panel_id), None)
        if panel_dump is None:
            raise ValueError(
                f"proposal places panel '{placement.panel_id}' which is not in the "
                "draft IR — validate against the draft's panel set first"
            )
        panel_dump.update(
            band_id=placement.band_id,
            column=placement.column,
            col_span=placement.col_span,
            emphasis=placement.emphasis,
            order=order,
            bbox=None,
        )
        for big_number in callouts_by_panel.get(placement.panel_id, []):
            panel_dump["elements"].insert(0, big_number.model_dump())
        panels.append(panel_dump)
        order += 1

    data["panels"] = panels
    data["bands"] = bands
    data["layout_provenance"] = LayoutProvenance(
        proposal_index=proposal_index,
        exemplar_ids=exemplar_ids or [],
        repairs=[r.model_dump() for r in repairs],
        reproposal_rounds=reproposal_rounds,
    ).model_dump()
    return PosterIR(**data)
