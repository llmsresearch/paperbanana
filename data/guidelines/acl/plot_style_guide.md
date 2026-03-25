# ACL Statistical Plot Aesthetics Guide

## 1. The "ACL Look"

ACL plots emphasize **readability and comparison**. Papers frequently
include performance tables alongside plots, so figures must justify
their visual form by revealing patterns that tables cannot. Bar charts
comparing models across tasks and line charts showing training dynamics
are the most common plot types.

---

## 2. Detailed Style Options

### **Color Palettes**

* **Categorical:** Muted, well-separated colors. Common choices: navy,
  teal, salmon, olive. One color per model or method, used consistently
  across all figures in the paper.
* **Sequential:** Viridis or Blues for heatmaps (e.g., attention maps,
  confusion matrices).
* **Emphasis:** Bold color (red, gold) reserved for the proposed method;
  baselines in neutral tones (grey, light blue).

### **Axes & Grids**

* **Grid lines:** Light grey dashed lines, behind data elements.
* **Spines:** Boxed (all four sides) is most common in ACL papers.
* **Axis labels:** Clear, with units. Font size must survive two-column
  scaling (minimum 8 pt at print).

### **Layout & Typography**

* **Font:** Sans-serif throughout. Match the paper's body font when
  possible.
* **Legends:** Inside the plot area (top-left or top-right) or as a
  shared horizontal legend above grouped subplots.
* **Subplots:** Common for multi-task or multi-dataset comparisons.
  Use shared y-axes and consistent x-axis ordering.

---

## 3. Type-Specific Guidelines

### **Bar Charts (Most Common)**
* Group by task/dataset on x-axis, models as grouped bars.
* Include numeric values above or inside bars for precise comparison.
* Error bars for multiple runs, with flat caps.

### **Line Charts**
* Training curves, learning rate schedules, or scaling experiments.
* Markers at evaluation points. Shaded confidence bands.
* Log scale on x-axis for large-scale experiments.

### **Heatmaps**
* Attention visualizations: token-by-token grids with Viridis colormap.
* Confusion matrices: annotated cells with counts or percentages.
* Square aspect ratio for symmetric matrices.

### **Box / Violin Plots**
* Distribution comparisons across models or datasets.
* Outlier markers as small dots. Median line clearly visible.

---

## 4. Common Pitfalls

* **Bar charts without numeric labels** — when differences are small,
  readers need exact values.
* **Inconsistent model ordering** across subplots.
* **Attention heatmaps without axis labels** — always label source
  and target tokens.
* **Overcrowded legends** — simplify or use direct annotation.
* **Missing significance indicators** — use asterisks or brackets
  for statistical significance where applicable.
