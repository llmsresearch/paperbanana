"""Small SVG primitives for authored editorial poster compositions."""

from __future__ import annotations

import base64
import mimetypes
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from PIL import ImageFont

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


class TextOverflowError(ValueError):
    """Raised when measured text does not fit its authored allocation."""


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class TextStyle:
    family: str
    font_path: Path
    size: float
    fill: str
    weight: int = 400
    line_height: float = 1.2
    letter_spacing: float = 0.0


def _svg_tag(name: str) -> str:
    return f"{{{SVG_NS}}}{name}"


class EditorialSvg:
    """Structured SVG document with measured text and embedded raster evidence."""

    def __init__(self, width: int, height: int, background: str) -> None:
        self.width = width
        self.height = height
        self.root = ET.Element(
            _svg_tag("svg"),
            {
                "width": str(width),
                "height": str(height),
                "viewBox": f"0 0 {width} {height}",
                "role": "img",
            },
        )
        self.rect(Box(0, 0, width, height), fill=background)

    def group(self, *, element_id: str | None = None) -> ET.Element:
        attributes = {"id": element_id} if element_id else {}
        return ET.SubElement(self.root, _svg_tag("g"), attributes)

    def rect(
        self,
        box: Box,
        *,
        fill: str,
        stroke: str | None = None,
        stroke_width: float = 0,
        parent: ET.Element | None = None,
    ) -> ET.Element:
        attributes = {
            "x": f"{box.x:g}",
            "y": f"{box.y:g}",
            "width": f"{box.width:g}",
            "height": f"{box.height:g}",
            "fill": fill,
        }
        if stroke:
            attributes.update({"stroke": stroke, "stroke-width": f"{stroke_width:g}"})
        return ET.SubElement(
            parent if parent is not None else self.root, _svg_tag("rect"), attributes
        )

    def line(
        self,
        points: list[tuple[float, float]],
        *,
        stroke: str,
        stroke_width: float,
        parent: ET.Element | None = None,
    ) -> ET.Element:
        return ET.SubElement(
            parent if parent is not None else self.root,
            _svg_tag("polyline"),
            {
                "points": " ".join(f"{x:g},{y:g}" for x, y in points),
                "fill": "none",
                "stroke": stroke,
                "stroke-width": f"{stroke_width:g}",
                "stroke-linecap": "round",
                "stroke-linejoin": "round",
            },
        )

    def circle(
        self,
        center: tuple[float, float],
        radius: float,
        *,
        fill: str,
        parent: ET.Element | None = None,
    ) -> ET.Element:
        return ET.SubElement(
            parent if parent is not None else self.root,
            _svg_tag("circle"),
            {
                "cx": f"{center[0]:g}",
                "cy": f"{center[1]:g}",
                "r": f"{radius:g}",
                "fill": fill,
            },
        )

    def image(
        self,
        image_path: Path,
        box: Box,
        *,
        element_id: str,
        parent: ET.Element | None = None,
    ) -> ET.Element:
        if not image_path.is_file():
            raise FileNotFoundError(f"Editorial figure not found: {image_path}")
        mime_type = mimetypes.guess_type(image_path.name)[0]
        if mime_type not in {"image/png", "image/jpeg"}:
            raise ValueError(f"Unsupported editorial figure format: {image_path}")
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return ET.SubElement(
            parent if parent is not None else self.root,
            _svg_tag("image"),
            {
                "id": element_id,
                "x": f"{box.x:g}",
                "y": f"{box.y:g}",
                "width": f"{box.width:g}",
                "height": f"{box.height:g}",
                "preserveAspectRatio": "xMidYMid meet",
                f"{{{XLINK_NS}}}href": f"data:{mime_type};base64,{encoded}",
            },
        )

    def text(
        self,
        content: str,
        box: Box,
        style: TextStyle,
        *,
        element_id: str,
        parent: ET.Element | None = None,
    ) -> Box:
        lines = self._wrap(content, box.width, style)
        line_height = style.size * style.line_height
        rendered_height = len(lines) * line_height
        if rendered_height > box.height:
            raise TextOverflowError(
                f"Text '{element_id}' needs {rendered_height:.1f}px but has {box.height:.1f}px"
            )
        text = ET.SubElement(
            parent if parent is not None else self.root,
            _svg_tag("text"),
            {
                "id": element_id,
                "x": f"{box.x:g}",
                "y": f"{box.y + style.size:g}",
                "fill": style.fill,
                "font-family": style.family,
                "font-size": f"{style.size:g}",
                "font-weight": str(style.weight),
                "letter-spacing": f"{style.letter_spacing:g}",
            },
        )
        for index, line in enumerate(lines):
            tspan = ET.SubElement(
                text,
                _svg_tag("tspan"),
                {
                    "x": f"{box.x:g}",
                    "dy": "0" if index == 0 else f"{line_height:g}",
                },
            )
            tspan.text = line
        return Box(box.x, box.y, box.width, rendered_height)

    def write(self, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tree = ET.ElementTree(self.root)
        ET.indent(tree, space="  ")
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
        return output_path

    @staticmethod
    def _wrap(content: str, max_width: float, style: TextStyle) -> list[str]:
        font = ImageFont.truetype(str(style.font_path), round(style.size))
        lines: list[str] = []
        for paragraph in content.splitlines() or [""]:
            words = paragraph.split()
            if not words:
                lines.append("")
                continue
            current = words[0]
            if EditorialSvg._measure(current, font, style.letter_spacing) > max_width:
                raise TextOverflowError(
                    f"Word '{current}' exceeds its {max_width:.1f}px allocation"
                )
            for word in words[1:]:
                candidate = f"{current} {word}"
                if EditorialSvg._measure(candidate, font, style.letter_spacing) <= max_width:
                    current = candidate
                    continue
                lines.append(current)
                current = word
                if EditorialSvg._measure(current, font, style.letter_spacing) > max_width:
                    raise TextOverflowError(
                        f"Word '{current}' exceeds its {max_width:.1f}px allocation"
                    )
            lines.append(current)
        return lines

    @staticmethod
    def _measure(content: str, font: ImageFont.FreeTypeFont, letter_spacing: float) -> float:
        left, _, right, _ = font.getbbox(content)
        return right - left + max(0, len(content) - 1) * letter_spacing
