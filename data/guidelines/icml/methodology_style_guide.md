# ICML Method Diagram Aesthetics Guide

## 1. The "ICML Look"

ICML diagrams favor **clarity and density**. The style leans toward
compact, information-rich layouts that fit the two-column format well.
Compared to NeurIPS, ICML papers tend to use slightly more saturated
colors and tighter spacing, reflecting the venue's emphasis on rigorous
technical communication over visual flair.

---

## 2. Detailed Style Options

### **A. Color Palettes**

*Design Philosophy: Functional color coding with moderate saturation.
Each color should map to a distinct semantic role.*

**Background Fills**

* **Primary approach:** White or very light grey backgrounds with
  colored borders to delineate stages.
* **Alternative:** Light blue-grey (#EDF2F7) or pale warm grey (#F7F7F5)
  for grouping containers.
* **Avoid:** Strongly tinted pastels that reduce readability in the
  narrow two-column layout.

**Functional Element Colors**

* **Core modules:** Medium-saturation blues (#4A90D9) and greens (#5BA585).
* **Trainable vs. frozen:** Warm tones (orange, coral) for trainable;
  cool tones (steel blue, grey) for frozen or pre-trained components.
* **Loss / objective:** Red or deep magenta, used sparingly for emphasis.
* **Data flow highlights:** Teal or gold for key paths.

### **B. Shapes & Containers**

* **Process blocks:** Rounded rectangles with moderate corner radius (4-8 px).
* **Mathematical operations:** Small circles or diamonds placed inline.
* **Data tensors:** Flat grids or stacked rectangles; avoid heavy 3D
  unless dimensionality is the point.
* **Grouping:** Thin solid or dashed borders around logical stages.
  Avoid heavy drop shadows.

### **C. Lines & Arrows**

* **Standard flow:** Solid dark grey or black arrows, 1-2 px weight.
* **Gradient / auxiliary flow:** Dashed lines in a lighter color.
* **Connector style:** Orthogonal elbows for architecture diagrams;
  gentle curves for high-level system flows.
* **Arrowheads:** Simple filled triangles; avoid oversized or ornamental heads.

### **D. Typography & Icons**

* **Labels:** Sans-serif (Helvetica, Arial). Bold for module names,
  regular weight for annotations.
* **Math variables:** Serif italic (LaTeX style), consistent with the
  paper body.
* **Icons:** Minimal use. Gears for processing, snowflakes for frozen
  parameters. Keep icons small and monochrome.

### **E. Layout & Composition**

* **Flow:** Left-to-right is standard; top-to-bottom for hierarchical
  architectures.
* **Compactness:** ICML's two-column format demands tighter layouts.
  Eliminate unnecessary whitespace but keep logical grouping clear.
* **Alignment:** Strict grid alignment. Misaligned elements stand out
  more in narrow columns.
* **Figure width:** Design for single-column (3.25 in) or full-width
  (6.75 in) with appropriate detail level for each.

---

## 3. Common Pitfalls

* **Overly wide diagrams** that lose detail when scaled to column width.
* **Too many colors** without a legend or consistent mapping.
* **Tiny text** that becomes unreadable at print scale (minimum 7 pt).
* **Inconsistent line styles** between data flow and gradient flow.
* **Decorative elements** (shadows, gradients, 3D effects) that add
  clutter without information.

---

## 4. Domain-Specific Styles

**Optimization / Theory Papers:**
* Minimalist diagrams with graph nodes and mathematical annotations.
* Grayscale with one accent color for key results.

**Representation Learning:**
* Embedding space visualizations with color-coded clusters.
* Use arrows to show transformations between spaces.

**Reinforcement Learning:**
* Agent-environment loop diagrams with clear state/action/reward labels.
* Dashed lines for policy updates, solid for environment transitions.
