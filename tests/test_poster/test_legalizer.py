"""Legalizer tests: violations, repairs, and IR construction from proposals."""

from __future__ import annotations

from paperbanana.poster.legalizer import (
    BANNER_PANEL_ID,
    build_ir_from_proposal,
    repair,
    validate_proposal,
)
from paperbanana.poster.proposal import (
    LayoutProposal,
    ProposedBand,
    ProposedCallout,
    ProposedPlacement,
)
from paperbanana.poster.types import KeyStat, PosterIR, Storyboard, StoryboardPanel

PAGE_W, MARGIN, GUTTER = 1219.2, 20.0, 10.0


def _storyboard(with_stats: bool = True) -> Storyboard:
    return Storyboard(
        panels=[
            StoryboardPanel(id="motivation", role="motivation", order=1, text_blocks=["- a"]),
            StoryboardPanel(id="method", role="method", order=2, text_blocks=["- b"]),
            StoryboardPanel(id="results", role="results", order=3, text_blocks=["- c"]),
        ],
        columns=3,
        takeaway="One key message about the work.",
        key_stats=(
            [KeyStat(id="stat1", value="3.4x", label="faster", source_panel="results")]
            if with_stats
            else []
        ),
    )


def _proposal(**overrides) -> LayoutProposal:
    base = dict(
        bands=[
            ProposedBand(id="header", kind="header"),
            ProposedBand(id="body", kind="body", columns=3),
        ],
        placements=[
            ProposedPlacement(panel_id="motivation", band_id="body", column=0),
            ProposedPlacement(panel_id="method", band_id="body", column=1),
            ProposedPlacement(panel_id="results", band_id="body", column=2),
        ],
        use_banner=False,
        callouts=[],
        rationale="test",
    )
    base.update(overrides)
    return LayoutProposal(**base)


def _validate(proposal, storyboard=None, figures=frozenset({"fig1"})):
    return validate_proposal(
        proposal, storyboard or _storyboard(), set(figures), PAGE_W, MARGIN, GUTTER
    )


def test_clean_proposal_has_no_violations():
    assert _validate(_proposal()) == []


def test_missing_panel_is_fatal():
    proposal = _proposal(
        placements=[
            ProposedPlacement(panel_id="motivation", band_id="body", column=0),
            ProposedPlacement(panel_id="method", band_id="body", column=1),
        ]
    )
    violations = _validate(proposal)
    assert any(v.code == "missing_panel" and v.fatal and v.target == "results" for v in violations)


def test_unknown_panel_and_band_are_fatal():
    proposal = _proposal(
        placements=[
            ProposedPlacement(panel_id="ghost", band_id="body", column=0),
            ProposedPlacement(panel_id="motivation", band_id="nowhere", column=0),
            ProposedPlacement(panel_id="method", band_id="body", column=1),
            ProposedPlacement(panel_id="results", band_id="body", column=2),
        ]
    )
    codes = {v.code for v in _validate(proposal)}
    assert {"unknown_panel", "unknown_band_ref"} <= codes


def test_duplicate_placement_is_fatal():
    proposal = _proposal(
        placements=[
            ProposedPlacement(panel_id="motivation", band_id="body", column=0),
            ProposedPlacement(panel_id="motivation", band_id="body", column=1),
            ProposedPlacement(panel_id="method", band_id="body", column=1),
            ProposedPlacement(panel_id="results", band_id="body", column=2),
        ]
    )
    assert any(v.code == "duplicate_placement" and v.fatal for v in _validate(proposal))


def test_callout_must_reference_known_stat():
    proposal = _proposal(callouts=[ProposedCallout(panel_id="results", key_stat_id="invented")])
    violations = _validate(proposal)
    assert any(v.code == "callout_unknown_stat" and v.fatal for v in violations)


def test_hero_must_be_curated_figure():
    proposal = _proposal(hero_figure_id="not_a_figure")
    assert any(v.code == "hero_figure_unknown" and v.fatal for v in _validate(proposal))


def test_banner_without_takeaway_is_fatal():
    storyboard = _storyboard()
    storyboard = storyboard.model_copy(update={"takeaway": None})
    proposal = _proposal(use_banner=True)
    violations = _validate(proposal, storyboard=storyboard)
    assert any(v.code == "banner_without_takeaway" and v.fatal for v in violations)


def test_missing_header_band_is_repaired():
    proposal = _proposal(bands=[ProposedBand(id="body", kind="body", columns=3)])
    violations = _validate(proposal)
    assert any(v.code == "add_header_band" and not v.fatal for v in violations)
    repaired, actions = repair(proposal, violations, PAGE_W, MARGIN, GUTTER)
    assert repaired.bands[0].kind == "header"
    assert any(a.op == "add_header_band" for a in actions)


def test_span_overflow_is_clamped_and_logged():
    proposal = _proposal(
        placements=[
            ProposedPlacement(panel_id="motivation", band_id="body", column=2, col_span=3),
            ProposedPlacement(panel_id="method", band_id="body", column=0),
            ProposedPlacement(panel_id="results", band_id="body", column=1),
        ]
    )
    violations = _validate(proposal)
    assert any(v.code == "clamp_col_span" and not v.fatal for v in violations)
    repaired, actions = repair(proposal, violations, PAGE_W, MARGIN, GUTTER)
    fixed = next(p for p in repaired.placements if p.panel_id == "motivation")
    assert fixed.column + fixed.col_span <= 3
    assert any(a.op == "clamp_col_span" for a in actions)


def _storyboard_with_figure() -> Storyboard:
    sb = _storyboard()
    sb.panels[1].figure_ids = ["fig1"]  # method panel carries a figure
    return sb


def test_text_panel_wide_span_is_clamped_but_figure_hero_is_not():
    proposal = _proposal(
        placements=[
            # text panel illegally spanning 2 columns -> clamp to 1
            ProposedPlacement(panel_id="motivation", band_id="body", column=0, col_span=2),
            # figure panel spanning 2 columns -> legal hero, untouched
            ProposedPlacement(panel_id="method", band_id="body", column=1, col_span=2),
            ProposedPlacement(panel_id="results", band_id="body", column=0),
        ],
        bands=[
            ProposedBand(id="header", kind="header"),
            ProposedBand(id="body", kind="body", columns=3),
        ],
    )
    sb = _storyboard_with_figure()
    violations = _validate(proposal, storyboard=sb)
    assert any(
        v.code == "clamp_text_span" and v.target == "motivation" and not v.fatal for v in violations
    )
    assert not any(v.target == "method" and v.code == "clamp_text_span" for v in violations)
    repaired, actions = repair(proposal, violations, PAGE_W, MARGIN, GUTTER)
    motivation = next(p for p in repaired.placements if p.panel_id == "motivation")
    method = next(p for p in repaired.placements if p.panel_id == "method")
    assert motivation.col_span == 1  # text clamped
    assert method.col_span == 2  # figure hero preserved
    assert any(a.op == "clamp_text_span" for a in actions)


def test_too_many_columns_reduced_for_min_width():
    proposal = _proposal(
        bands=[
            ProposedBand(id="header", kind="header"),
            ProposedBand(id="body", kind="body", columns=6),  # 1179mm -> ~188mm/col
        ]
    )
    violations = _validate(proposal)
    assert any(v.code == "reduce_band_columns" for v in violations)
    repaired, actions = repair(proposal, violations, PAGE_W, MARGIN, GUTTER)
    assert repaired.bands[1].columns <= 5
    assert any(a.op == "reduce_band_columns" for a in actions)


def test_banner_band_inserted_when_use_banner():
    proposal = _proposal(use_banner=True)  # no banner band declared
    violations = _validate(proposal)
    assert any(v.code == "add_banner_band" and not v.fatal for v in violations)
    repaired, actions = repair(proposal, violations, PAGE_W, MARGIN, GUTTER)
    kinds = [b.kind for b in repaired.bands]
    assert "banner" in kinds and kinds.index("banner") == kinds.index("header") + 1


def test_build_ir_applies_structure(poster_ir: PosterIR):
    storyboard = _storyboard()
    proposal = _proposal(
        use_banner=True,
        bands=[
            ProposedBand(id="header", kind="header"),
            ProposedBand(id="banner", kind="banner"),
            ProposedBand(id="body", kind="body", columns=3),
        ],
        placements=[
            ProposedPlacement(panel_id="motivation", band_id="body", column=0),
            ProposedPlacement(panel_id="method", band_id="body", column=1, col_span=2),
            ProposedPlacement(panel_id="results", band_id="body", column=0, emphasis="accent"),
        ],
        callouts=[ProposedCallout(panel_id="results", key_stat_id="stat1")],
    )
    # Draft IR with matching panel ids.
    data = poster_ir.model_dump()
    for panel, new_id in zip(data["panels"][1:], ("motivation", "method", "results")):
        panel["id"] = new_id
    draft = PosterIR(**data)

    ir = build_ir_from_proposal(draft, proposal, storyboard, repairs=[], reproposal_rounds=1)
    assert {b.kind for b in ir.bands} == {"header", "banner", "body"}
    banner_panel = next(p for p in ir.panels if p.id == BANNER_PANEL_ID)
    assert banner_panel.elements[0].kind == "banner"
    assert banner_panel.elements[0].content == storyboard.takeaway
    method = next(p for p in ir.panels if p.id == "method")
    assert method.col_span == 2
    results = next(p for p in ir.panels if p.id == "results")
    assert results.emphasis == "accent"
    assert results.elements[0].kind == "big_number"
    assert results.elements[0].value == "3.4x"
    assert ir.layout_provenance.reproposal_rounds == 1
    # Rebuilding from the produced IR must not duplicate banner/callouts.
    ir2 = build_ir_from_proposal(ir, proposal, storyboard, repairs=[])
    assert sum(1 for p in ir2.panels if p.id == BANNER_PANEL_ID) == 1
    results2 = next(p for p in ir2.panels if p.id == "results")
    assert sum(1 for el in results2.elements if el.kind == "big_number") == 1


def test_repaired_clean_proposal_roundtrip():
    violations = _validate(_proposal())
    repaired, actions = repair(_proposal(), violations, PAGE_W, MARGIN, GUTTER)
    assert actions == []
    assert repaired.model_dump() == _proposal().model_dump()
