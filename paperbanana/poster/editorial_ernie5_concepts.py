"""Three full, explicitly authored ERNIE 5.0 editorial poster concepts."""

from __future__ import annotations

import json
from pathlib import Path

from paperbanana.poster.editorial_ernie5 import (
    ARGUMENT,
    EVIDENCE,
    INK,
    MUTED_INK,
    PAPER,
    PLATE,
    RULE,
    SUPPORT,
    _style,
)
from paperbanana.poster.editorial_renderer import Box, EditorialSvg

WIDTH = 1600
HEIGHT = 1200


def render_all_concepts(repo_root: Path) -> list[Path]:
    plan = json.loads(
        (repo_root / "outputs/editorial_handoff/ernie5/editorial_plan.json").read_text(
            encoding="utf-8"
        )
    )
    sources = plan["source_of_truth"]["prepared_figures"]
    figures = {key: repo_root / value for key, value in sources.items()}
    output_dir = repo_root / "outputs/editorial_handoff/ernie5/concepts"
    outputs = []
    for concept_id, renderer in (
        ("concept_a_architecture_spread", _render_concept_a),
        ("concept_b_narrative_columns", _render_concept_b),
        ("concept_c_technical_plate", _render_concept_c),
    ):
        svg = EditorialSvg(WIDTH, HEIGHT, PAPER)
        renderer(svg, plan, figures)
        outputs.append(svg.write(output_dir / f"{concept_id}.svg"))
    return outputs


def _title(svg: EditorialSvg, claim: str, concept: str) -> None:
    svg.rect(Box(0, 0, WIDTH, 14), fill=ARGUMENT)
    svg.text("ERNIE 5.0", Box(62, 38, 420, 82), _style(66, display=True), element_id="title")
    svg.text(
        claim,
        Box(482, 38, 1010, 96),
        _style(27, ARGUMENT, 600, line_height=1.15),
        element_id="scientific-argument",
    )
    svg.text(
        f"ERNIE Team · Baidu                     {concept}",
        Box(64, 142, 1428, 28),
        _style(15, MUTED_INK, 500),
        element_id="identity",
    )
    svg.line([(64, 184), (1536, 184)], stroke=RULE, stroke_width=2)


def _render_concept_a(svg: EditorialSvg, plan: dict, figures: dict[str, Path]) -> None:
    copy = plan["three_minute_walkthrough"]
    _title(svg, plan["scientific_argument"], "CONCEPT A · ARCHITECTURE-LED SPREAD")

    svg.text("THE BREAK", Box(64, 220, 190, 28), _style(14, EVIDENCE, 700), element_id="a-break")
    svg.text(
        copy[0]["copy"],
        Box(64, 254, 262, 138),
        _style(18, line_height=1.25),
        element_id="a-problem",
    )
    svg.text(
        "THE UNIFICATION",
        Box(354, 220, 240, 28),
        _style(14, ARGUMENT, 700),
        element_id="a-unification",
    )
    svg.rect(Box(344, 256, 1090, 508), fill=PLATE, stroke=RULE, stroke_width=1.3)
    svg.image(figures["figure_1"], Box(360, 270, 1058, 478), element_id="a-figure-1")

    target_one = (512, 526)
    target_two = (840, 470)
    for number, target, origin in (("1", target_one, (326, 442)), ("2", target_two, (1448, 386))):
        svg.circle(origin, 17, fill=EVIDENCE)
        svg.text(
            number,
            Box(origin[0] - 6, origin[1] - 15, 18, 22),
            _style(17, PAPER, 700),
            element_id=f"a-marker-{number}",
        )
        elbow_x = 336 if number == "1" else 1438
        svg.line(
            [origin, (elbow_x, origin[1]), (elbow_x, target[1]), target],
            stroke=EVIDENCE,
            stroke_width=3,
        )
        svg.circle(target, 8, fill=EVIDENCE)

    svg.text(
        "Serialized modalities",
        Box(64, 432, 238, 30),
        _style(19, EVIDENCE, 700),
        element_id="a-note-1-title",
    )
    svg.text(
        copy[1]["copy"], Box(64, 470, 246, 156), _style(15, line_height=1.25), element_id="a-note-1"
    )
    svg.text(
        "One shared expert pool",
        Box(1445, 424, 120, 54),
        _style(18, EVIDENCE, 700, line_height=1.1),
        element_id="a-note-2-title",
    )
    svg.text(
        copy[2]["copy"],
        Box(1445, 494, 120, 210),
        _style(13, line_height=1.25),
        element_id="a-note-2",
    )

    svg.line([(64, 806), (1536, 806)], stroke=ARGUMENT, stroke_width=4)
    svg.text(
        "ROUTING BEHAVIOR",
        Box(64, 830, 270, 28),
        _style(14, SUPPORT, 700),
        element_id="a-routing-label",
    )
    svg.image(figures["figure_8"], Box(64, 870, 300, 252), element_id="a-figure-8")
    svg.text(
        copy[2]["copy"],
        Box(386, 870, 300, 122),
        _style(16, line_height=1.25),
        element_id="a-routing-copy",
    )

    svg.text(
        "ELASTIC DEPLOYMENT",
        Box(730, 830, 300, 28),
        _style(14, EVIDENCE, 700),
        element_id="a-elastic-label",
    )
    svg.image(figures["figure_4"], Box(730, 874, 402, 214), element_id="a-figure-4")
    _comparison_strip(svg, Box(1164, 842, 372, 252), plan, "a")
    svg.text(
        copy[5]["copy"],
        Box(386, 1030, 746, 70),
        _style(22, ARGUMENT, 700, line_height=1.15),
        element_id="a-takeaway",
    )
    _footer(
        svg, "A", "Architecture is the visual thesis; routing and elasticity unfold beneath it."
    )


def _render_concept_b(svg: EditorialSvg, plan: dict, figures: dict[str, Path]) -> None:
    copy = plan["three_minute_walkthrough"]
    _title(svg, plan["scientific_argument"], "CONCEPT B · NARRATIVE COLUMNS")
    svg.rect(Box(42, 218, 10, 864), fill=EVIDENCE)
    svg.text("01", Box(76, 220, 70, 54), _style(38, EVIDENCE, 700), element_id="b-step-1")
    svg.text(
        "WHY UNIFY?", Box(76, 280, 248, 38), _style(22, INK, 700), element_id="b-problem-title"
    )
    svg.text(
        copy[0]["copy"],
        Box(76, 334, 250, 170),
        _style(18, line_height=1.26),
        element_id="b-problem",
    )
    svg.line([(76, 526), (300, 526)], stroke=EVIDENCE, stroke_width=4)
    svg.text("INPUTS", Box(76, 560, 150, 28), _style(14, ARGUMENT, 700), element_id="b-input-label")
    svg.text(
        "TEXT\nIMAGE · VIDEO\nAUDIO",
        Box(76, 600, 238, 144),
        _style(25, ARGUMENT, 600, line_height=1.35),
        element_id="b-inputs",
    )
    svg.text(
        copy[5]["copy"],
        Box(76, 828, 252, 146),
        _style(20, SUPPORT, 700, line_height=1.22),
        element_id="b-takeaway",
    )

    svg.text(
        "02 · ONE SHARED BACKBONE",
        Box(370, 220, 470, 30),
        _style(15, ARGUMENT, 700),
        element_id="b-architecture-label",
    )
    svg.image(figures["figure_1"], Box(362, 278, 772, 444), element_id="b-figure-1")
    svg.line([(362, 746), (1134, 746)], stroke=ARGUMENT, stroke_width=3)
    svg.text(
        copy[1]["copy"],
        Box(370, 772, 360, 112),
        _style(17, line_height=1.25),
        element_id="b-unified-copy",
    )
    svg.text(
        copy[2]["copy"],
        Box(758, 772, 366, 138),
        _style(17, line_height=1.25),
        element_id="b-routing-copy",
    )
    _comparison_strip(svg, Box(370, 942, 754, 150), plan, "b")

    svg.text(
        "03 · EVIDENCE",
        Box(1182, 220, 300, 30),
        _style(15, EVIDENCE, 700),
        element_id="b-evidence-label",
    )
    svg.image(figures["figure_8"], Box(1182, 270, 332, 304), element_id="b-figure-8")
    svg.text(
        "Shared rules, task-shaped use",
        Box(1182, 590, 332, 34),
        _style(20, SUPPORT, 700),
        element_id="b-routing-title",
    )
    svg.text(
        copy[2]["copy"],
        Box(1182, 636, 332, 120),
        _style(15, line_height=1.25),
        element_id="b-routing-evidence",
    )
    svg.image(figures["figure_4"], Box(1182, 792, 332, 176), element_id="b-figure-4")
    svg.text(
        copy[3]["copy"],
        Box(1182, 980, 332, 116),
        _style(15, line_height=1.25),
        element_id="b-elastic-copy",
    )
    _footer(
        svg,
        "B",
        "Unequal columns turn problem, mechanism, and evidence into one left-to-right argument.",
    )


def _render_concept_c(svg: EditorialSvg, plan: dict, figures: dict[str, Path]) -> None:
    copy = plan["three_minute_walkthrough"]
    _title(svg, plan["scientific_argument"], "CONCEPT C · ANNOTATED TECHNICAL PLATE")
    svg.text(
        "READ THE PLATE",
        Box(64, 220, 220, 30),
        _style(14, EVIDENCE, 700),
        element_id="c-entry-label",
    )
    svg.text(
        copy[0]["copy"],
        Box(64, 260, 238, 150),
        _style(17, line_height=1.25),
        element_id="c-problem",
    )

    plate = Box(326, 218, 1018, 612)
    svg.rect(plate, fill=PLATE, stroke=RULE, stroke_width=1.4)
    svg.image(figures["figure_1"], Box(342, 234, 986, 580), element_id="c-figure-1")
    _plate_annotation(svg, "1", (306, 430), (486, 546), "Serialize every modality", "c-1")
    _plate_annotation(svg, "2", (1370, 328), (836, 478), "Route through one expert pool", "c-2")
    _plate_annotation(svg, "3", (1370, 640), (1112, 640), "Predict text, vision, and audio", "c-3")

    svg.rect(Box(54, 744, 410, 386), fill=PLATE)
    svg.text(
        "WHAT THE ROUTER LEARNS",
        Box(64, 752, 330, 28),
        _style(14, SUPPORT, 700),
        element_id="c-routing-label",
    )
    svg.image(figures["figure_8"], Box(64, 790, 390, 330), element_id="c-figure-8")
    svg.line([(454, 930), (482, 930), (482, 866), (532, 866)], stroke=SUPPORT, stroke_width=4)
    svg.text(
        "Shared rules, task-shaped specialization",
        Box(498, 828, 250, 62),
        _style(18, SUPPORT, 700, line_height=1.12),
        element_id="c-routing-title",
    )
    svg.text(
        copy[2]["copy"],
        Box(498, 900, 250, 146),
        _style(14, line_height=1.25),
        element_id="c-routing-copy",
    )

    svg.text(
        "HOW ONE RUN BECOMES MANY",
        Box(786, 850, 350, 28),
        _style(14, EVIDENCE, 700),
        element_id="c-elastic-label",
    )
    svg.image(figures["figure_4"], Box(780, 890, 354, 188), element_id="c-figure-4")
    svg.line([(1134, 986), (1152, 986)], stroke=EVIDENCE, stroke_width=4)
    _comparison_strip(svg, Box(1160, 846, 376, 246), plan, "c")
    svg.text(
        copy[5]["copy"],
        Box(498, 1054, 636, 70),
        _style(18, ARGUMENT, 700, line_height=1.18),
        element_id="c-takeaway",
    )
    _footer(
        svg,
        "C",
        "Annotations explain the plate; insets verify routing behavior and deployment elasticity.",
    )


def _plate_annotation(
    svg: EditorialSvg,
    number: str,
    origin: tuple[float, float],
    target: tuple[float, float],
    label: str,
    element_id: str,
) -> None:
    svg.circle(origin, 17, fill=EVIDENCE)
    svg.text(
        number,
        Box(origin[0] - 6, origin[1] - 15, 18, 22),
        _style(17, PAPER, 700),
        element_id=f"{element_id}-number",
    )
    label_x = 64 if origin[0] < WIDTH / 2 else 1388
    svg.text(
        label,
        Box(label_x, origin[1] - 28, 166, 64),
        _style(16, EVIDENCE, 700, line_height=1.1),
        element_id=f"{element_id}-label",
    )
    elbow = 316 if origin[0] < WIDTH / 2 else 1354
    svg.line(
        [origin, (elbow, origin[1]), (elbow, target[1]), target], stroke=EVIDENCE, stroke_width=3
    )
    svg.circle(target, 8, fill=EVIDENCE)


def _comparison_strip(svg: EditorialSvg, box: Box, plan: dict, prefix: str) -> None:
    evidence = next(item for item in plan["evidence_plan"] if item["figure"] == "table_12_reset")
    svg.text(
        "TABLE 12 · PERFORMANCE / EFFICIENCY",
        Box(box.x, box.y, box.width, 24),
        _style(12, EVIDENCE, 700),
        element_id=f"{prefix}-table-title",
    )
    row_height = 32
    start_y = box.y + 38
    for index, row in enumerate(evidence["rows"]):
        y = start_y + index * row_height
        color = ARGUMENT if index == 0 else (EVIDENCE if index == 1 else SUPPORT)
        svg.line([(box.x, y + 27), (box.x + box.width, y + 27)], stroke=RULE, stroke_width=1)
        svg.text(
            row["model"],
            Box(box.x, y, box.width * 0.72, 26),
            _style(12, INK, 600),
            element_id=f"{prefix}-table-model-{index}",
        )
        svg.text(
            row["average"],
            Box(box.x + box.width * 0.83, y, box.width * 0.17, 26),
            _style(16, color, 700),
            element_id=f"{prefix}-table-average-{index}",
        )
    fact_y = start_y + len(evidence["rows"]) * row_height + 12
    svg.text(
        ">15% faster decoding at 25% routing top-k",
        Box(box.x, fact_y, box.width, 26),
        _style(14, EVIDENCE, 700),
        element_id=f"{prefix}-speed-fact",
    )
    svg.text(
        "53.7% activated · 35.8% total parameters",
        Box(box.x, fact_y + 30, box.width, 26),
        _style(14, SUPPORT, 700),
        element_id=f"{prefix}-parameter-fact",
    )


def _footer(svg: EditorialSvg, concept: str, note: str) -> None:
    svg.line([(64, 1150), (1536, 1150)], stroke=RULE, stroke_width=2)
    svg.text(
        note,
        Box(64, 1164, 1260, 24),
        _style(12, MUTED_INK, 500),
        element_id=f"{concept.lower()}-footer",
    )
    svg.text(
        f"ERNIE 5.0 · {concept}",
        Box(1380, 1164, 156, 24),
        _style(12, ARGUMENT, 700),
        element_id=f"{concept.lower()}-index",
    )


if __name__ == "__main__":
    for output in render_all_concepts(Path.cwd()):
        print(output)
