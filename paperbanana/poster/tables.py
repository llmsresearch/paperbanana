"""Table intelligence: parse table crops, re-set them cleanly, or rechart.

Tables are the judged-weakest poster element. Two upgrades over placing
the crop: ``reset_table`` re-renders the parsed table as clean vector-
crisp matplotlib at 300 DPI with the winning row bolded; ``rechart``
turns the key comparison into a chart. Both start from a VLM parse of
the crop into structured data, which doubles as the numeric ground
truth for the faithfulness gate.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal, Optional

import structlog
from PIL import Image
from pydantic import BaseModel, Field

from paperbanana.core.utils import extract_json

logger = structlog.get_logger()

ChartKind = Literal["bar", "grouped_bar", "line", "scatter"]

RENDER_DPI = 300


class TableData(BaseModel):
    """A parsed table: the numeric ground truth for re-setting/recharting."""

    columns: list[str] = Field(min_length=1)
    rows: list[list[str]] = Field(min_length=1)
    title: Optional[str] = None
    highlight_row: Optional[int] = Field(
        default=None, ge=0, description="Row index of 'our method' / the winner"
    )

    def numbers(self) -> list[str]:
        """Every numeric token in the table — the faithfulness checklist."""
        tokens = []
        for row in self.rows:
            for cell in row:
                tokens.extend(re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", str(cell)))
        return tokens


async def parse_table(crop: Image.Image, vlm, prompt_dir: Path, caption: str = "") -> TableData:
    """VLM parse of a table crop into structured rows/columns."""
    template = (Path(prompt_dir) / "poster" / "table_parser.txt").read_text(encoding="utf-8")
    prompt = template.format(caption=caption or "(none)")
    raw = await vlm.generate(prompt=prompt, images=[crop], response_format="json", temperature=0.1)
    data = extract_json(raw)
    if not isinstance(data, dict):
        raise ValueError(f"table parser returned no JSON object: {raw[:400]!r}")
    table = TableData(**data)
    widths = {len(r) for r in table.rows}
    if widths != {len(table.columns)}:
        raise ValueError(
            f"parsed table is ragged: {len(table.columns)} columns but row widths {sorted(widths)}"
        )
    return table


def render_table_matplotlib(
    table: TableData,
    palette: dict[str, str],
    out_path: Path,
    width_mm: float,
) -> Path:
    """Deterministic clean re-set of a table at print DPI.

    Vector-crisp text, alternating row tint, highlighted winner row —
    the typography is ours, the numbers are the parse's.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(table.rows)
    width_in = width_mm / 25.4
    height_in = max(1.2, 0.42 * (n_rows + 1) + (0.5 if table.title else 0.0))
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=RENDER_DPI)
    ax.axis("off")
    if table.title:
        ax.set_title(table.title, fontsize=13, color=palette["primary"], pad=10, weight="bold")

    mpl_table = ax.table(
        cellText=table.rows,
        colLabels=table.columns,
        cellLoc="center",
        loc="center",
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(11)
    mpl_table.scale(1, 1.6)
    for (row, _col), cell in mpl_table.get_celld().items():
        cell.set_edgecolor("#FFFFFF")
        if row == 0:
            cell.set_facecolor(palette["primary"])
            cell.set_text_props(color=palette["background"], weight="bold")
        elif table.highlight_row is not None and row == table.highlight_row + 1:
            cell.set_facecolor(palette["accent"])
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor(palette["panel_bg"])
        else:
            cell.set_facecolor(palette["background"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=palette["background"])
    plt.close(fig)
    logger.info("Table re-set", path=str(out_path), rows=n_rows)
    return out_path


def table_to_plot_payload(table: TableData, chart_kind: Optional[str]) -> dict[str, Any]:
    """Raw-data payload for the plot pipeline's code generation."""
    return {
        "columns": table.columns,
        "rows": table.rows,
        "title": table.title,
        "requested_chart_kind": chart_kind or "bar",
        "highlight_row": table.highlight_row,
    }


def verify_chart_numbers(table: TableData, chart_numbers: list[str]) -> list[str]:
    """Numeric tokens from the table that are missing in the chart's data.

    Used to ground the faithfulness check: the chart must carry the
    table's numbers (subset relation — a chart may legitimately show a
    column subset, so caller decides the threshold).
    """
    chart_set = set(chart_numbers)
    return [n for n in table.numbers() if n not in chart_set]
