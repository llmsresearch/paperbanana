# NeurIPS 2025 Statistical Plot Aesthetics Guide

## 1. The "NeurIPS Look": A High-Level Overview

The prevailing aesthetic for 2025 is defined by **precision, accessibility,
and high contrast**. The "default" academic look has shifted away from
bare-bones styling toward a more graphic, publication-ready presentation.

* **Vibe:** Professional, clean, and information-dense.
* **Backgrounds:** There is a heavy bias toward **stark white backgrounds**
  for maximum contrast in print and PDF reading, though the "Seaborn-style"
  light grey background remains an accepted variant.
* **Accessibility:** A strong emphasis on distinguishing data not just by
  color, but by texture (patterns) and shape (markers) to support black-and-white
  printing and colorblind readers.

---

## 2. Detailed Style Options

### **Color Palettes**

* **Categorical Data:**
  * **Soft Pastels:** Matte, low-saturation colors (salmon, sky blue,
    mint, lavender) are frequently used to prevent visual fatigue.
  * **Muted Earth Tones:** "Academic" palettes using olive, beige, slate
    grey, and navy.
  * **High-Contrast Primaries:** Used sparingly when categories must be
    distinct (e.g., deep orange vs. vivid purple).
  * **Accessibility Mode:** A growing trend involves combining color
    with **geometric patterns** (hatches, dots, stripes) to differentiate
    categories.
* **Sequential & Heatmaps:**
  * **Perceptually Uniform:** "Viridis" (blue-to-yellow) and "Magma/
    Plasma" (purple-to-orange) are the standard.
  * **Diverging:** "Coolwarm" (blue-to-red) is used for positive/negative
    value splits.
  * **Avoid:** The traditional "Jet/Rainbow" scale is almost entirely absent.

### **Axes & Grids**

* **Grid Style:**
  * **Visibility:** Grid lines are almost rarely solid. Common choices
    include **fine dashed ('--')** or **dotted (':')** lines in light gray.
  * **Placement:** Grids are consistently rendered *behind* data
    elements (low Z-order).
* **Spines (Borders):**
  * **The "Boxed" Look:** A full enclosure (black spines on all 4 sides)
    is very common.
  * **The "Open" Look:** Removing the top and right spines for a
    minimalist appearance.
* **Ticks:**
  * **Style:** Ticks are generally subtle, facing inward, or removed
    entirely in favor of grid alignment.

### **Layout & Typography**

* **Typography:**
  * **Font Family:** Exclusively **Sans-Serif** (resembling Helvetica,
    Arial, or DejaVu Sans). Serif fonts are rarely used for labels.
  * **Label Rotation:** X-axis labels are rotated **45 degrees** only
    when necessary to prevent overlap; otherwise, horizontal orientation
    is preferred.
* **Legends:**
  * **Internal Placement:** Floating the legend *inside* the plot area (
    top-left or top-right) to maximize the "data-ink ratio."
  * **Top Horizontal:** Placing the legend in a single row above the
    plot title.
* **Annotations:**
  * **Direct Labeling:** Instead of forcing readers to reference a
    legend, text is often placed directly next to lines or on top of bars.

---

## 3. Type-Specific Guidelines

### **Bar Charts & Histograms**

* **Borders:** Two distinct styles are accepted:
  * **High-Definition:** Using **black outlines** around colored bars
    for a "comic-book" or high-contrast look.
  * **Borderless:** Solid color fills with no outline (often used with
    light grey backgrounds).
* **Grouping:** Bars are grouped tightly, with significant whitespace
  between categorical groups.
* **Error Bars:** Consistently styled with **black, flat caps**.

### **Line Charts**

* **Markers:** A critical observation: Lines almost always include **geometric
  markers** (circles, squares, diamonds) at data points, rather than just
  being smooth strokes.
* **Line Styles:** Use **dashed lines** ('--') for theoretical limits,
  baselines, or secondary data, and **solid lines** for primary experimental
  data.
* **Uncertainty:** Represented by semi-transparent **shaded bands** (
  confidence intervals) rather than simple vertical error bars.

### **Tree & Pie/Donut Charts**

* **Separators:** Thick **white borders** are standard to separate slices
  or treemap blocks.
* **Structure:** Thick **Donut charts** are preferred over traditional Pie
  charts.
* **Emphasis:** "Exploding" (detaching) a specific slice is a common
  technique to highlight a key statistic.

### **Scatter Plots**

* **Shape Coding:** Use different marker shapes (e.g., circles vs.
  triangles) to encode a categorical dimension alongside color.
* **Fills:** Markers are typically solid and fully opaque.
* **3D Plots:** Depth is emphasized by drawing "walls" with grids or using
  drop-lines to the "floor" of the plot.

### **Heatmaps**

* **Aspect Ratio:** Cells are almost strictly **square**.
* **Annotation:** Writing the exact value (in white or black text) **inside
  the cell** is highly preferred over relying solely on a color bar.
* **Borders:** Cells are often borderless (smooth gradient look) or
  separated by very thin white lines.

### **Radar Charts**

* **Fills:** The polygon area uses **translucent fills** (alpha ~0.2) to
  show grid lines underneath.
* **Perimeter:** The outer boundary is marked by a solid, darker line.

### **Miscellaneous**

* **Dot Plots:** Used as a modern alternative to bar charts; often styled
  as "lollipops" (dots connected to the axis by a thin line).

---

## 4. Common Pitfalls (What to Avoid)

* **The "Excel Default" Look:** Avoid heavy 3D effects on bars, shadow
  drops, or serif fonts (Times New Roman) on axes.
* **The "Rainbow" Map:** Avoid the Jet/Rainbow colormap; it is considered
  outdated and perceptually misleading.
* **Ambiguous Lines:** A line chart *without* markers can look ambiguous
  if data points are sparse; always add markers.
* **Over-reliance on Color:** Failing to use patterns or shapes to
  distinguish groups makes the plot inaccessible to colorblind readers.
* **Cluttered Grids:** Avoid solid black grid lines; they compete with the
  data. Always use light grey/dashed grids.
