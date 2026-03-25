# ICML Statistical Plot Aesthetics Guide

## 1. The "ICML Look"

ICML plots prioritize **precision and compactness**. The two-column
format means plots must remain legible at smaller sizes. Clean lines,
clear legends, and high data-ink ratio are essential. Avoid decoration
that does not encode information.

---

## 2. Detailed Style Options

### **Color Palettes**

* **Categorical:** Muted but distinguishable colors. A common palette
  uses navy, teal, coral, and slate. Avoid neon or fully saturated hues.
* **Sequential:** Viridis or Plasma for heatmaps. Perceptually uniform
  colormaps are expected.
* **Diverging:** Coolwarm (blue-to-red) for positive/negative splits.
* **Accessibility:** Combine color with marker shapes and line styles
  to support grayscale printing.

### **Axes & Grids**

* **Grid lines:** Light grey dashed lines behind data. Never solid black.
* **Spines:** Either all four sides (boxed) or remove top and right
  (open). Be consistent across figures in the same paper.
* **Tick labels:** Sans-serif, minimum 7 pt at final print size.
  Avoid rotated labels when horizontal fits.

### **Layout & Typography**

* **Font:** Sans-serif throughout (Helvetica, Arial, DejaVu Sans).
* **Legends:** Inside the plot area when space permits; otherwise
  placed horizontally above or below the plot.
* **Annotations:** Direct labeling on lines or bars is preferred over
  legend-only identification.
* **Subplot spacing:** Tight but with clear separation. Use shared
  axes where appropriate to save space.

---

## 3. Type-Specific Guidelines

### **Line Charts**
* Always include markers at data points (circles, squares, triangles).
* Solid lines for primary results, dashed for baselines.
* Shaded bands for confidence intervals (alpha 0.2-0.3).

### **Bar Charts**
* Thin black outlines or borderless fills. Group bars tightly.
* Error bars with flat caps in black.
* Hatching patterns as a secondary differentiator for accessibility.

### **Heatmaps**
* Square cells with numeric annotations inside.
* Viridis or Plasma colormap. Include a labeled colorbar.

### **Scatter Plots**
* Marker shape encodes one dimension, color encodes another.
* Solid opaque markers. Add trend lines where meaningful.

---

## 4. Common Pitfalls

* **Unreadable text** when scaled to column width. Always verify at
  final print size.
* **Too many overlapping lines** without distinguishing markers or styles.
* **Missing units** on axes.
* **Rainbow colormaps** (Jet) — use perceptually uniform alternatives.
* **Legends that obscure data** — reposition or use direct labeling.
