"""Authored ERNIE 5.0 editorial poster proof."""

from __future__ import annotations

import json
from pathlib import Path

from paperbanana.poster.editorial_renderer import Box, EditorialSvg, TextStyle

CANVAS_WIDTH = 1600
CANVAS_HEIGHT = 1200

INK = "#17232c"
MUTED_INK = "#59666d"
PAPER = "#f7f6f1"
PLATE = "#fffefd"
ARGUMENT = "#147f91"
EVIDENCE = "#dc573f"
SUPPORT = "#527c63"
RULE = "#c8d1ce"

SANS_FONT = Path("/System/Library/Fonts/Avenir Next.ttc")
DISPLAY_FONT = Path("/System/Library/Fonts/NewYork.ttf")


def render_ernie5_proof(repo_root: Path) -> Path:
    """Render the first architecture-led technical plate proof as SVG."""
    plan_path = repo_root / "outputs/editorial_handoff/ernie5/editorial_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    copy = plan["proof_copy"]

    for font_path in (SANS_FONT, DISPLAY_FONT):
        if not font_path.is_file():
            raise FileNotFoundError(f"Required editorial font not found: {font_path}")

    figure_1 = repo_root / plan["source_of_truth"]["prepared_figures"]["figure_1"]
    figure_8 = repo_root / plan["source_of_truth"]["prepared_figures"]["figure_8"]
    output_path = repo_root / "outputs/editorial_handoff/ernie5/proof/technical_plate.svg"

    svg = EditorialSvg(CANVAS_WIDTH, CANVAS_HEIGHT, PAPER)
    _draw_title(svg, copy)
    _draw_entry(svg, copy)
    plate_box = _draw_architecture_plate(svg, figure_1, copy)
    _draw_annotations(svg, plate_box, copy, plan["annotation_targets"])
    _draw_routing_evidence(svg, figure_8, copy, plate_box)
    _draw_footer(svg, copy)
    return svg.write(output_path)


def _style(
    size: float,
    fill: str = INK,
    weight: int = 400,
    *,
    display: bool = False,
    line_height: float = 1.2,
) -> TextStyle:
    return TextStyle(
        family="New York" if display else "Avenir Next",
        font_path=DISPLAY_FONT if display else SANS_FONT,
        size=size,
        fill=fill,
        weight=weight,
        line_height=line_height,
    )


def _draw_title(svg: EditorialSvg, copy: dict[str, str]) -> None:
    svg.rect(Box(0, 0, 22, 214), fill=ARGUMENT)
    svg.text(copy["title"], Box(68, 42, 460, 92), _style(76, display=True), element_id="title")
    svg.text(
        copy["subtitle"],
        Box(540, 52, 920, 98),
        _style(31, ARGUMENT, 600, line_height=1.15),
        element_id="subtitle",
    )
    svg.text(
        copy["authors"],
        Box(70, 154, 450, 34),
        _style(18, MUTED_INK, 500),
        element_id="authors",
    )
    svg.line([(540, 174), (1528, 174)], stroke=RULE, stroke_width=2)


def _draw_entry(svg: EditorialSvg, copy: dict[str, str]) -> None:
    svg.text(
        copy["entry_label"],
        Box(68, 246, 262, 36),
        _style(14, EVIDENCE, 700),
        element_id="entry-label",
    )
    svg.text(
        copy["entry_copy"],
        Box(68, 288, 250, 210),
        _style(20, INK, 500, line_height=1.28),
        element_id="entry-copy",
    )
    svg.line([(68, 500), (274, 500)], stroke=EVIDENCE, stroke_width=5)


def _draw_architecture_plate(svg: EditorialSvg, figure_path: Path, copy: dict[str, str]) -> Box:
    svg.text(
        copy["plate_label"],
        Box(356, 238, 500, 30),
        _style(15, ARGUMENT, 700),
        element_id="plate-label",
    )
    plate_box = Box(350, 286, 1010, 580)
    svg.rect(plate_box, fill=PLATE, stroke=RULE, stroke_width=1.5)
    svg.image(figure_path, Box(366, 302, 978, 548), element_id="figure-1")
    return plate_box


def _draw_annotations(
    svg: EditorialSvg,
    plate_box: Box,
    copy: dict[str, str],
    targets: list[dict],
) -> None:
    first_target = _target_point(plate_box, targets[0]["target_norm"])
    second_target = _target_point(plate_box, targets[1]["target_norm"])

    svg.circle((78, 534), 18, fill=EVIDENCE)
    svg.text("1", Box(72, 519, 20, 24), _style(18, PAPER, 700), element_id="marker-1")
    svg.text(
        copy["annotation_1_title"],
        Box(112, 510, 214, 50),
        _style(20, EVIDENCE, 700),
        element_id="annotation-1-title",
    )
    svg.text(
        copy["annotation_1_body"],
        Box(112, 550, 210, 116),
        _style(16, INK, 500, line_height=1.28),
        element_id="annotation-1-body",
    )
    svg.line(
        [(322, 610), (342, 610), (342, first_target[1]), first_target],
        stroke=EVIDENCE,
        stroke_width=3,
    )
    svg.circle(first_target, 9, fill=EVIDENCE)

    svg.circle((1398, 336), 18, fill=EVIDENCE)
    svg.text("2", Box(1392, 321, 20, 24), _style(18, PAPER, 700), element_id="marker-2")
    svg.text(
        copy["annotation_2_title"],
        Box(1430, 312, 150, 56),
        _style(19, EVIDENCE, 700, line_height=1.08),
        element_id="annotation-2-title",
    )
    svg.text(
        copy["annotation_2_body"],
        Box(1398, 386, 172, 130),
        _style(15, INK, 500, line_height=1.28),
        element_id="annotation-2-body",
    )
    svg.line(
        [(1398, 366), (1376, 366), (1376, second_target[1]), second_target],
        stroke=EVIDENCE,
        stroke_width=3,
    )
    svg.circle(second_target, 9, fill=EVIDENCE)


def _draw_routing_evidence(
    svg: EditorialSvg,
    figure_path: Path,
    copy: dict[str, str],
    plate_box: Box,
) -> None:
    inset_box = Box(1198, 742, 340, 354)
    svg.line(
        [(plate_box.x + plate_box.width * 0.58, plate_box.y + plate_box.height), (1130, 922)],
        stroke=SUPPORT,
        stroke_width=4,
    )
    svg.text(
        copy["bridge_copy"],
        Box(686, 914, 416, 82),
        _style(22, SUPPORT, 600, line_height=1.2),
        element_id="routing-bridge",
    )
    svg.text(
        copy["routing_label"],
        Box(inset_box.x, 684, inset_box.width, 32),
        _style(14, SUPPORT, 700),
        element_id="routing-label",
    )
    svg.image(figure_path, inset_box, element_id="figure-8")
    svg.text(
        copy["routing_caption"],
        Box(688, 992, 468, 112),
        _style(15, MUTED_INK, 500, line_height=1.28),
        element_id="routing-caption",
    )


def _draw_footer(svg: EditorialSvg, copy: dict[str, str]) -> None:
    svg.line([(68, 1142), (1538, 1142)], stroke=RULE, stroke_width=2)
    svg.text(
        copy["proof_footer"],
        Box(68, 1155, 1120, 26),
        _style(13, MUTED_INK, 500),
        element_id="proof-footer",
    )
    svg.text(
        "FIG. 01 / FIG. 08",
        Box(1354, 1155, 184, 24),
        _style(13, ARGUMENT, 700),
        element_id="figure-index",
    )


def _target_point(plate_box: Box, target_norm: list[float]) -> tuple[float, float]:
    return (
        plate_box.x + plate_box.width * target_norm[0],
        plate_box.y + plate_box.height * target_norm[1],
    )


if __name__ == "__main__":
    print(render_ernie5_proof(Path.cwd()))
