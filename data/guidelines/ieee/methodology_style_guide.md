# IEEE Method Diagram Aesthetics Guide

## 1. The "IEEE Look"

IEEE diagrams follow a **formal, engineering-oriented** aesthetic.
The style reflects the venue's roots in electrical engineering and
systems design: clean block diagrams, precise labeling, and a
preference for structured, grid-aligned layouts. IEEE papers tend
toward more conservative visuals compared to ML-focused venues.

---

## 2. Detailed Style Options

### **A. Color Palettes**

*Design Philosophy: Restrained use of color. Many IEEE papers still
target black-and-white or grayscale printing, so color must be
supplementary, not the sole differentiator.*

**Background Fills**

* **Primary approach:** White backgrounds with black or dark grey
  borders for all containers.
* **Shading:** Light grey fills (#F0F0F0, #E8E8E8) to distinguish
  subsystem blocks. Avoid colored backgrounds unless functionally
  necessary.
* **Colored accents (optional):** Muted blue (#6699CC) for highlights,
  muted red (#CC6666) for critical paths or errors.

**Functional Element Colors**

* **Standard blocks:** White fill with black borders (the classic
  engineering block diagram).
* **Signal paths:** Black for primary, grey for secondary.
* **If using color:** Stick to 3-4 colors maximum. Blue for processing,
  green for input/output, red for error/feedback, grey for storage.

### **B. Shapes & Containers**

* **Process blocks:** Sharp-cornered rectangles are standard in IEEE.
  Rounded corners are acceptable but less traditional.
* **Decision points:** Diamonds for conditional logic (flowchart style).
* **Summing junctions:** Small circles with + or - signs (control
  systems convention).
* **Multiplexers / switches:** Trapezoids or labeled selector blocks.
* **Databases / storage:** Cylinders, following standard conventions.
* **Grouping:** Dashed rectangles for subsystems, with a label in the
  top-left corner.

### **C. Lines & Arrows**

* **Signal flow:** Solid black arrows with simple arrowheads.
* **Feedback loops:** Clearly marked with direction arrows and gain
  labels where applicable.
* **Bus lines:** Thicker lines with a slash and bus width annotation
  (e.g., /8 for 8-bit bus).
* **Connector style:** Orthogonal routing (right angles) is strongly
  preferred. Avoid curved connectors in block diagrams.

### **D. Typography & Icons**

* **Labels:** Sans-serif (Arial, Helvetica) or the paper's body font.
  All caps for major subsystem names is common.
* **Signal labels:** Placed directly on or adjacent to arrows.
* **Math:** LaTeX-style variables in serif italic, consistent with
  equation numbering in the paper.
* **Icons:** Minimal. Standard engineering symbols (antenna, amplifier
  triangle, ADC/DAC blocks) where applicable.

### **E. Layout & Composition**

* **Flow:** Left-to-right for signal processing chains; top-to-bottom
  for layered architectures.
* **Grid alignment:** Strict. IEEE reviewers notice misalignment.
* **Symmetry:** Balance the diagram. Symmetric architectures should
  look symmetric.
* **Figure captions:** IEEE style uses "Fig. 1." format. Ensure the
  diagram is self-explanatory with its caption.
* **Single vs. double column:** Design for the target width. Most IEEE
  transactions use two-column layout.

---

## 3. Common Pitfalls

* **Over-reliance on color** — the diagram must work in grayscale.
  Use patterns, line styles, or labels as primary differentiators.
* **Rounded, pastel aesthetics** borrowed from ML venues — IEEE
  reviewers may perceive this as informal.
* **Missing signal labels** on connecting arrows.
* **Inconsistent block sizing** — related blocks should share dimensions.
* **Decorative gradients or shadows** — keep fills flat and borders crisp.

---

## 4. Domain-Specific Styles

**Signal Processing / Communications:**
* Classic block diagrams with summing junctions and transfer functions.
* Frequency domain representations alongside time domain.

**Computer Architecture / Hardware:**
* Register-transfer level diagrams with bus annotations.
* Pipeline stage diagrams with clock boundary markers.

**Robotics / Control Systems:**
* Feedback loop diagrams with plant, controller, and sensor blocks.
* State machine diagrams with labeled transitions.

**Deep Learning (IEEE format):**
* Network architecture as a linear chain of labeled blocks.
* Keep the ML diagram style but with sharper corners, less color,
  and more formal labeling than NeurIPS or ICML conventions.
