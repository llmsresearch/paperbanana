"""Generate Rajah 3.1: Proses Kajian in UMS thesis style.

Style requirements (Gaya UMS 2018):
- Font: Tahoma 11 (Liberation Sans used as render-time substitute when
  Tahoma is unavailable; open the SVG in Word/PowerPoint to switch fonts).
- Black text on white background; minimal grayscale for phase headers.
- Caption ``Rajah 3.1: Proses Kajian`` placed at bottom-left of the figure.
- A4 portrait friendly aspect.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Prefer Tahoma if available; fall back to a close sans-serif.
_PREFERRED = ["Tahoma", "Liberation Sans", "DejaVu Sans"]
_available = {f.name for f in fm.fontManager.ttflist}
FONT = next((f for f in _PREFERRED if f in _available), "sans-serif")
plt.rcParams["font.family"] = FONT

PHASES = [
    {
        "title": "FASA 1\nPEMBENTUKAN ASAS KAJIAN",
        "steps": [
            "Memahami konteks kajian sekolah\nrendah luar bandar Sabah",
            "Mendefinisikan masalah kajian",
            "Sorotan literatur dan teori",
            "Mengenal pasti jurang kajian",
            "Membangunkan model kajian",
            "Membentuk persoalan, objektif\ndan hipotesis kajian",
        ],
    },
    {
        "title": "FASA 2\nPEMBANGUNAN METODOLOGI\nDAN INSTRUMEN",
        "steps": [
            "Reka bentuk kuantitatif:\ntinjauan keratan lintang",
            "Operasionalisasi konstruk\ndan pengukuran",
            "Pembangunan soal selidik",
            "Penentuan unit analisis dan\nreka bentuk persampelan",
            "Pra-ujian instrumen: kesahan\nkandungan dan kebolehpercayaan",
            "Kajian rintis",
            "Pengumpulan data",
        ],
    },
    {
        "title": "FASA 3\nANALISIS DATA DAN PELAPORAN",
        "steps": [
            "Saringan data menggunakan\nSPSS versi 29",
            "Pengujian hipotesis melalui\nPLS-SEM dalam SmartPLS",
            "Tafsiran dapatan",
            "Perbincangan, rumusan\ndan pendokumentasian",
        ],
    },
]

# Layout constants (data units).
COL_W = 5.8
COL_GAP = 0.9
BOX_W = 4.8
BOX_H = 0.95
V_GAP = 0.45
HEADER_H = 1.55
TOP_PAD = 0.6
TERM_H = 0.75  # Mula / Tamat ovals

n_cols = len(PHASES)
fig_w_in = 13.5
# Determine tallest column to size the figure.
max_steps = max(len(p["steps"]) for p in PHASES)
col_inner_h = TOP_PAD + TERM_H + V_GAP + HEADER_H + V_GAP + (max_steps * BOX_H) + (max_steps - 1) * V_GAP + V_GAP + TERM_H + 0.4
fig_h_in = col_inner_h * 0.55

fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
ax.set_xlim(0, n_cols * COL_W + (n_cols - 1) * COL_GAP + 0.5)
ax.set_ylim(0, col_inner_h)
ax.invert_yaxis()
ax.set_aspect("equal")
ax.axis("off")

EDGE = "#000000"
FILL = "#FFFFFF"
HEADER_FILL = "#E8E8E8"


def rect(x, y, w, h, *, fill=FILL, lw=1.1, radius=0.08):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=lw,
        edgecolor=EDGE,
        facecolor=fill,
    )
    ax.add_patch(patch)


def oval(cx, cy, w, h, label):
    e = mpatches.Ellipse((cx, cy), w, h, linewidth=1.2, edgecolor=EDGE, facecolor=FILL)
    ax.add_patch(e)
    ax.text(cx, cy, label, ha="center", va="center", fontsize=11, fontweight="bold")


def arrow(x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.1,
            color=EDGE,
        )
    )


def col_x(idx: int) -> float:
    return 0.25 + idx * (COL_W + COL_GAP)


# Track terminal y-positions so we can draw inter-phase arrows.
phase_top_y = []
phase_bottom_y = []
phase_cx = []

for idx, phase in enumerate(PHASES):
    x0 = col_x(idx)
    cx = x0 + COL_W / 2
    phase_cx.append(cx)
    y = TOP_PAD

    # Header
    rect(x0, y, COL_W, HEADER_H, fill=HEADER_FILL, lw=1.2, radius=0.05)
    ax.text(
        cx,
        y + HEADER_H / 2,
        phase["title"],
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    y += HEADER_H + V_GAP

    # Mula oval only at the top of Phase 1
    if idx == 0:
        oval(cx, y + TERM_H / 2, 2.0, TERM_H, "Mula")
        phase_top_y.append(y + TERM_H)
        prev_bottom = y + TERM_H
        y += TERM_H + V_GAP
    else:
        phase_top_y.append(y)
        prev_bottom = y

    # Steps
    for step in phase["steps"]:
        box_x = x0 + (COL_W - BOX_W) / 2
        rect(box_x, y, BOX_W, BOX_H)
        ax.text(cx, y + BOX_H / 2, step, ha="center", va="center", fontsize=10.5)
        # Arrow from previous element
        arrow(cx, prev_bottom, cx, y - 0.02)
        prev_bottom = y + BOX_H
        y += BOX_H + V_GAP

    # Tamat oval only at the bottom of Phase 3
    if idx == n_cols - 1:
        oval(cx, y + TERM_H / 2, 2.0, TERM_H, "Tamat")
        arrow(cx, prev_bottom, cx, y - 0.02)
        prev_bottom = y + TERM_H

    phase_bottom_y.append(prev_bottom)

# Horizontal connectors between phases: route through the gap between columns.
# From the right edge of the last box of phase i, go right into the gap,
# then up to the level of phase i+1's header, then left into the left edge
# of the first box of phase i+1.
for i in range(n_cols - 1):
    # Right edge of last box of phase i (use BOX_W centered on column).
    x_box_right_i = phase_cx[i] + BOX_W / 2
    # Left edge of first box of phase i+1.
    x_box_left_next = phase_cx[i + 1] - BOX_W / 2
    # Mid-gap x (between phase columns)
    x_mid = (col_x(i) + COL_W + col_x(i + 1)) / 2

    y_out = phase_bottom_y[i] - BOX_H / 2  # mid of last box of phase i
    y_in = phase_top_y[i + 1] + BOX_H / 2  # mid of first box of phase i+1
    # If phase i+1 starts with header (not terminator), phase_top_y is just
    # below the header, so y_in lands on the first step's mid: that is fine.

    # Segment 1: from right edge of last box of phase i to mid-gap
    ax.plot(
        [x_box_right_i, x_mid],
        [y_out, y_out],
        color=EDGE,
        linewidth=1.1,
    )
    # Segment 2: vertical in the gap, from y_out up/down to y_in
    ax.plot(
        [x_mid, x_mid],
        [y_out, y_in],
        color=EDGE,
        linewidth=1.1,
    )
    # Segment 3: from mid-gap to left edge of first box of phase i+1 (with arrow)
    ax.add_patch(
        FancyArrowPatch(
            (x_mid, y_in),
            (x_box_left_next - 0.02, y_in),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.1,
            color=EDGE,
        )
    )

# Caption (bottom-left, per UMS style for figures)
fig.text(
    0.06,
    0.03,
    "Rajah 3.1: Proses Kajian",
    ha="left",
    va="bottom",
    fontsize=11,
    fontweight="bold",
)

plt.subplots_adjust(left=0.02, right=0.99, top=0.99, bottom=0.07)

out_png = "/home/user/paperbanana/docs/thesis/rajah_3_1_proses_kajian.png"
out_svg = "/home/user/paperbanana/docs/thesis/rajah_3_1_proses_kajian.svg"
fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_png}")
print(f"Saved: {out_svg}")
print(f"Font used: {FONT}")
