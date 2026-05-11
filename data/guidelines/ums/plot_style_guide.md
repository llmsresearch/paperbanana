# UMS 2018 Statistical Plot Aesthetics Guide

Adapted from *Garis Panduan Penyerahan dan Penulisan Tesis/Disertasi
Gaya UMS* (Pusat Pengajian Pascasiswazah, Universiti Malaysia Sabah,
2018). This guide adapts the UMS thesis style for statistical plots
(carta, graf) embedded in the *Dapatan Kajian* / *Results* chapter
of a UMS thesis or disertasi.

---

## 1. The "UMS Look"

A UMS thesis plot reads like a journal report from a Malaysian
government statistics publication: monochrome-first, sans-serif,
densely labelled, and intended to be photocopied without loss of
information. Plots must remain legible after **black-and-white
printing on 80 gsm A4 paper** because three hard-cover copies are
submitted (Perpustakaan, Fakulti, Penyelia).

- **Vibe:** Formal, archival, government-report.
- **Backgrounds:** Strict white only. No seaborn grey, no tinted
  axes panels.
- **Accessibility:** Distinguish series by **line style** and
  **marker shape** first, colour second. The plot must still be
  decodable in grayscale.
- **Language:** Axis labels, legend, and caption in the same
  language as the thesis body (Bahasa Melayu or Bahasa Inggeris).
  Foreign-language terms are italicised.

---

## 2. Detailed Style Options

### A. Color Palettes

*Design Philosophy: Default is grayscale. Colour is only used when
it carries information, and even then it must survive grayscale
printing.*

**Categorical Data**

- **Default (preferred):** Grayscale ramp — black, dark grey
  (`#555555`), medium grey (`#999999`), light grey (`#CCCCCC`).
  Series are further separated by **hatching** (solid, diagonal
  stripes, dots, cross-hatch) for bars and by **marker shape**
  (circle, square, triangle, diamond) for lines/scatter.
- **Colour fallback (≤ 4 categories):** Dark blue (`#1F3864`),
  dark red (`#8B0000`), dark green (`#1E5631`), ochre (`#B8860B`).
  Avoid neon/saturated primaries.
- **Forbidden:** Jet/rainbow palette, neon green/pink, gradient bar
  fills, semi-transparent pastel fills as the only differentiator.

**Sequential / Heatmaps**

- Preferred: a **single-hue** ramp from white to dark grey
  (or white → dark blue if colour is essential).
- Acceptable: viridis (perceptually uniform, prints adequately in
  grayscale).
- Forbidden: jet, rainbow.

**Diverging**

- Black-to-white-to-black is not allowed (loses sign on grayscale).
- Use dark red ↔ white ↔ dark blue; ensure midpoint is true white.

### B. Axes, Grids, Spines

- **Spines:** All four sides drawn ("boxed"), 0.75 pt solid black.
- **Ticks:** Facing inward, 0.75 pt black, length ~3 pt. Minor
  ticks may be omitted.
- **Gridlines:** Thin dashed light grey (`#CCCCCC`), 0.5 pt,
  rendered *behind* the data. Horizontal gridlines only for bar
  charts; both axes for scatter/line.
- **Origin:** Always show numeric axis ranges; never crop the
  y-axis baseline of a bar chart without an explicit break symbol.
- **Units:** Always state units in the axis label, e.g.
  "Suhu (°C)", "Pendapatan (RM/bulan)".

### C. Layout & Typography

- **Font family:** Tahoma for axis labels, tick labels, legend,
  and in-plot annotations. This matches §6.2 b of the UMS
  guideline.
- **Sizes (after scaling to print size):**
  - Axis title: 11 pt, regular.
  - Tick labels: 9 pt.
  - Legend text: 9 pt.
  - In-plot annotations: 9 pt; minimum 8 pt after scaling.
- **Variables / mathematical symbols:** Italic. A serif (Cambria
  Math) is acceptable for true LaTeX-style maths but not for
  ordinary labels.
- **Rotation:** Rotate x-axis tick labels 45° only when horizontal
  labels overlap; otherwise keep horizontal.
- **Legend:** Inside the axes box (preferred top-right) or
  horizontally above the plot. No legend frame shadows. If the
  number of series ≤ 6, prefer **direct labels** on the data
  rather than a legend.

### D. Page Geometry

- Plot must fit within the UMS text area: left margin 38 mm,
  right/top/bottom 28 mm on A4 (210 × 297 mm). Useful drawing
  width ≈ **144 mm**.
- A single full-width plot: ~144 × 90 mm.
- Two side-by-side plots: ~70 × 70 mm each, with a 4 mm gutter.
- Caption (`Jadual X.Y:` or `Rajah X.Y:`) must not be counted
  inside the drawing area.

### E. Caption Placement

UMS has **different caption positions for tables vs. figures**
(§6.7.3):

- **Tables (`Jadual`):** Title **above** the table, **centred**,
  bolded. Source line below the table, **left-aligned**.
  - Line 1: **`Jadual X.Y : <Tajuk Jadual>`** (Tahoma 11 pt bold)
  - Table body
  - `Sumber: <citation>` (Tahoma 11 pt regular, left)
- **Figures / Plots (`Rajah`):** Title **below** the plot, **left
  aligned**. Source line directly under the title.
  - Plot body
  - **`Rajah X.Y: <Tajuk Rajah>`** (Tahoma 11 pt; "Rajah X.Y" bold)
  - `Sumber: <citation>` (Tahoma 11 pt regular, left)

`X` is the chapter number, `Y` the running figure/table number
within that chapter; restart `Y` at 1 in each new chapter.

---

## 3. Type-Specific Guidelines

### Bar Charts & Histograms

- Black bar outlines (0.75 pt), white or grey fills with **hatch
  patterns** (`////`, `\\\\`, `....`, `xxxx`) to distinguish
  groups.
- Bars grouped tightly within a category; whitespace between
  categories (gap ≈ 1 bar-width).
- Error bars: solid black, flat (T-shaped) caps, 0.75 pt.

### Line Charts

- Marker on every data point (circle, square, triangle, diamond) —
  required for grayscale legibility.
- Solid line for primary series, dashed for baseline / secondary.
- Confidence intervals: light grey (`#DDDDDD`) fill or thin dashed
  outline. Never neon pastel bands.

### Scatter Plots

- Encode categories by **marker shape**, not just colour.
- Markers fully opaque; black outline preferred.
- For 3D scatter, include drop-lines to the x–y floor; keep all
  three pane backgrounds white.

### Heatmaps

- Cells strictly square aspect ratio.
- Annotate each cell with the numeric value (Tahoma 9 pt, black on
  light cells, white on dark cells).
- Thin white cell borders (0.5 pt) only; never thick coloured
  grids.

### Box / Violin Plots

- Box outlined in black, fill white or hatched. Median line solid
  black, 1.0 pt.
- Outliers as small black circles (`o`), 4 pt.

### Pie / Donut Charts

- Use sparingly. Donut preferred over pie. Maximum 5 slices; merge
  the rest into "Lain-lain" / "Others".
- Thick white slice borders (1.0 pt). Direct label each slice with
  category name and percentage; no legend.

### Radar / Spider Charts

- Polygon outline solid black; fill at α = 0.15 if colour used,
  α = 0 (no fill) preferred.

### Dot / Lollipop Plots

- Thin connecting line to axis, filled black circle marker.

---

## 4. Common Pitfalls (Avoid These)

- **Excel default 3D bars, shadows, gradient fills.** Forbidden.
- **Rainbow / Jet colormap.** Perceptually misleading and prints
  poorly in grayscale.
- **Seaborn grey background panel.** UMS requires white.
- **Caption above a figure / below a table.** Reversed convention —
  UMS uses the opposite.
- **Centre-aligned figure caption.** UMS aligns figure captions to
  the left.
- **Tick labels in Times New Roman / Calibri / Arial.** Use
  Tahoma.
- **Y-axis truncation without a break symbol.** Misleading.
- **Tiny tick labels (< 8 pt at print size).**
- **Legend without distinguishing marker shape or hatch** — fails
  in grayscale photocopies.
- **Decorative emoji, flags, or coloured backgrounds.**

---

## 5. Reference Anchors

- Body font and size: §6.2 b, c (Tahoma 11 pt, black).
- Margins: §6.6.3 (38 mm left, 28 mm elsewhere).
- Table caption: §6.7.3 (title above, centred, bold; source below,
  left).
- Figure caption: §6.7.3 (title and source below, left-aligned).
- Numbering: §6.7.2 (Bab → 1.1 → 1.1.1).
- Foreign-language italicisation: §6.1.
- Paper: §6.13 (A4, 80 gsm simili, white).
- Single-sided printing: §6.14.1.
