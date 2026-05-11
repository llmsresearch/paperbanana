"""Render Rajah 3.1 (Proses Kajian) in UMS thesis style.

Output: 1280x720 PNG, white background, black 0.75 pt borders,
sharp rectangles for process nodes, pill shapes for Mula/Tamat,
sans-serif (Tahoma preferred; falls back to Liberation Sans / Arial).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# ── UMS style constants ─────────────────────────────────────────────
FONT_CANDIDATES = ["Tahoma", "Liberation Sans", "Arial", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = FONT_CANDIDATES

LINE_WIDTH = 0.9   # ≈ 0.75 pt at render
BORDER = "black"
FILL = "white"
FONT_SIZE = 9
PHASE_FONT_SIZE = 10
TERMINATOR_FONT_SIZE = 10

# ── Diagram content (from §3.5 Proses Kajian) ───────────────────────
PHASES = [
    {
        "title": "FASA 1\nPembentukan Asas Kajian",
        "steps": [
            "Memahami konteks kajian:\nsekolah rendah luar bandar Sabah",
            "Mendefinisikan masalah kajian",
            "Sorotan literatur dan\nmengenal pasti jurang",
            "Membangunkan model kajian",
            "Membentuk persoalan,\nobjektif dan hipotesis",
        ],
    },
    {
        "title": "FASA 2\nPembangunan Metodologi & Instrumen",
        "steps": [
            "Reka bentuk kuantitatif:\ntinjauan keratan lintang",
            "Operasionalisasi konstruk\ndan pengukuran",
            "Pembangunan soal selidik",
            "Penentuan unit analisis dan\nreka bentuk persampelan",
            "Pra-ujian instrumen:\nkesahan kandungan dan muka",
            "Kajian rintis:\nkebolehpercayaan",
            "Pengumpulan data",
        ],
    },
    {
        "title": "FASA 3\nAnalisis Data dan Pelaporan",
        "steps": [
            "Saringan data menggunakan\nSPSS versi 29",
            "Pengujian hipotesis melalui\nPLS-SEM dalam SmartPLS",
            "Tafsiran dapatan",
            "Perbincangan, rumusan\ndan pendokumentasian",
        ],
    },
]

# ── Canvas: 1280 × 720 → 12.80 × 7.20 in @ 100 dpi ──────────────────
WIDTH_IN, HEIGHT_IN = 12.80, 7.20
fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN), dpi=100, facecolor="white")
# Coordinate system: x 0–160, y 0–90 (matches 16:9 aspect → unit ratio).
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 160)
ax.set_ylim(0, 90)
ax.set_aspect("equal")
ax.axis("off")

# Column geometry (x is unit-coords). Use 3 columns; equal pitch.
COL_X = [30, 80, 130]
COL_W = 38
BOX_H = 7.0
GAP = 1.2
PHASE_HEADER_H = 7.5

TOP = 81.0
BOT = 7.0
TERMINATOR_TOP_Y = 86.0
TERMINATOR_BOT_Y = 4.0


def draw_process_box(ax, cx, cy, text):
    rect = Rectangle(
        (cx - COL_W / 2, cy - BOX_H / 2),
        COL_W,
        BOX_H,
        linewidth=LINE_WIDTH,
        edgecolor=BORDER,
        facecolor=FILL,
    )
    ax.add_patch(rect)
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=FONT_SIZE,
        color="black",
    )


def draw_terminator(ax, cx, cy, text):
    """Pill shape for Mula / Tamat — standard ANSI flowchart terminator."""
    w, h = 16.0, 5.0
    pill = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle="round,pad=0,rounding_size=2.4",
        linewidth=LINE_WIDTH,
        edgecolor=BORDER,
        facecolor=FILL,
    )
    ax.add_patch(pill)
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=TERMINATOR_FONT_SIZE,
        color="black",
        fontweight="bold",
    )


def draw_phase_header(ax, cx, top_y, title):
    rect = Rectangle(
        (cx - COL_W / 2, top_y - PHASE_HEADER_H),
        COL_W,
        PHASE_HEADER_H,
        linewidth=LINE_WIDTH,
        edgecolor=BORDER,
        facecolor=FILL,
        linestyle="--",
    )
    ax.add_patch(rect)
    ax.text(
        cx,
        top_y - PHASE_HEADER_H / 2,
        title,
        ha="center",
        va="center",
        fontsize=PHASE_FONT_SIZE,
        color="black",
        fontweight="bold",
    )


def draw_arrow(ax, x1, y1, x2, y2, head=True):
    style = "-|>" if head else "-"
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        mutation_scale=12,
        linewidth=LINE_WIDTH,
        color=BORDER,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arrow)


def draw_line(ax, x1, y1, x2, y2):
    ax.plot([x1, x2], [y1, y2], color=BORDER, linewidth=LINE_WIDTH)


# ── Draw each column ────────────────────────────────────────────────
column_box_centres: list[list[float]] = []
for col_idx, phase in enumerate(PHASES):
    cx = COL_X[col_idx]
    draw_phase_header(ax, cx, TOP, phase["title"])

    n = len(phase["steps"])
    span_top = TOP - PHASE_HEADER_H - 1.5
    span_bot = BOT + 6
    if n == 1:
        ys = [(span_top + span_bot) / 2]
    else:
        step = (span_top - span_bot) / (n - 1)
        ys = [span_top - BOX_H / 2 - i * step for i in range(n)]
        # Recompute step relative to centres so first centre is span_top - h/2
        span = (span_top - BOX_H / 2) - (span_bot + BOX_H / 2)
        step = span / (n - 1)
        ys = [span_top - BOX_H / 2 - i * step for i in range(n)]

    centres = []
    for y, text in zip(ys, phase["steps"]):
        draw_process_box(ax, cx, y, text)
        centres.append(y)
    column_box_centres.append(centres)

    # Arrows between adjacent boxes in the same column
    for y_from, y_to in zip(centres[:-1], centres[1:]):
        draw_arrow(ax, cx, y_from - BOX_H / 2, cx, y_to + BOX_H / 2)

# ── Mula → top of FASA 1 ────────────────────────────────────────────
draw_terminator(ax, COL_X[0], TERMINATOR_TOP_Y, "Mula")
draw_arrow(
    ax,
    COL_X[0],
    TERMINATOR_TOP_Y - 2.5,
    COL_X[0],
    TOP,
)

# ── Tamat ← bottom of FASA 3 ────────────────────────────────────────
draw_terminator(ax, COL_X[2], TERMINATOR_BOT_Y, "Tamat")
last_y3 = column_box_centres[2][-1]
draw_arrow(
    ax,
    COL_X[2],
    last_y3 - BOX_H / 2,
    COL_X[2],
    TERMINATOR_BOT_Y + 2.5,
)

# ── Inter-column connectors (Z-shape) ───────────────────────────────
# col1 last box → col2 first box
last_y1 = column_box_centres[0][-1]
first_y2 = column_box_centres[1][0]
mid_x_12 = (COL_X[0] + COL_X[1]) / 2
# right out of col1 last box → horizontal → up → into col2 first box from left
draw_line(ax, COL_X[0] + COL_W / 2, last_y1, mid_x_12, last_y1)
draw_line(ax, mid_x_12, last_y1, mid_x_12, first_y2)
draw_arrow(ax, mid_x_12, first_y2, COL_X[1] - COL_W / 2, first_y2)

# col2 last box → col3 first box
last_y2 = column_box_centres[1][-1]
first_y3 = column_box_centres[2][0]
mid_x_23 = (COL_X[1] + COL_X[2]) / 2
draw_line(ax, COL_X[1] + COL_W / 2, last_y2, mid_x_23, last_y2)
draw_line(ax, mid_x_23, last_y2, mid_x_23, first_y3)
draw_arrow(ax, mid_x_23, first_y3, COL_X[2] - COL_W / 2, first_y3)

# ── Save ────────────────────────────────────────────────────────────
out_path = "/tmp/figbuild/rajah_3_1_proses_kajian.png"
fig.savefig(out_path, dpi=100, facecolor="white")
print(f"Saved: {out_path}")
