# NeurIPS 2025 Method Diagram Aesthetics Guide

## 1. The "NeurIPS Look"

The prevailing aesthetic for 2025 is **"Soft Tech & Scientific Pastels."**
Gone are the days of harsh primary colors and sharp black boxes. The modern
NeurIPS diagram feels approachable yet precise. It utilizes high-value (
light) backgrounds to organize complexity, reserving saturation for the
most critical active elements. The vibe balances **clean modularity** (
clear separation of parts) with **narrative flow** (clear left-to-right
progression).

---

## 2. Detailed Style Options

### **A. Color Palettes**

*Design Philosophy: Use color to group logic, not just to decorate. Avoid
fully saturated backgrounds.*

**Background Fills (The "Zone" Strategy)**

*Used to encapsulate stages (e.g., "Pre-training phase") or environments.*

* **Most papers use:** Very light, desaturated pastels (Opacity ~10-15%).
* **Aesthetically pleasing options include:**
  * **Cream / Beige** (e.g., '#F5F5DC') - *Warm, academic feel.*
  * **Pale Blue / Ice** (e.g., '#E6F3FF') - *Clean, technical feel.*
  * **Mint / Sage** (e.g., '#E0F2F1') - *Soft, organic feel.*
  * **Pale Lavender** (e.g., '#F3E5F5') - *Distinctive, modern feel.*
* **Alternative (~20%):** White backgrounds with colored *dashed borders*
  for a high-contrast, minimalist look (common in theoretical papers).

**Functional Element Colors**

* **For "Active" Modules (Encoders, MLP, Attention):** Medium saturation
  is preferred.
  * *Common pairings:* Blue/Orange, Green/Purple, or Teal/Pink.
  * *Observation:* Colors are often used to distinguish **status**
    rather than component type:
    * **Trainable Elements:** Often Warm tones (Red, Orange, Deep Pink).
    * **Frozen/Static Elements:** Often Cool tones (Grey, Ice Blue, Cyan).
* **For Highlights/Results:** High saturation (Primary Red, Bright Gold)
  is strictly reserved for "Error/Loss," "Ground Truth," or the final
  output.

### **B. Shapes & Containers**

*Design Philosophy: "Softened Geometry." Sharp corners are for data; rounded
corners are for processes.*

**Core Components**

* **Process Nodes (The Standard):** Rounded Rectangles (Corner radius 5-10
  px). This is the dominant shape (~80%) for generic layers or steps.
* **Tensors & Data:**
  * **3D Stacks/Cuboids:** Used to imply depth/volume (e.g., B x H x W).
  * **Flat Squares/Grids:** Used for matrices, tokens, or attention maps.
* **Cylinders:** Exclusively reserved for Databases, Buffers, or Memory.

**Grouping & Hierarchy**

* **The "Macro-Micro" Pattern:** A solid, light-colored container
  represents the global view, with a specific module (e.g., "Attention
  Block") connected via lines to a "zoomed-in" detailed breakout box.
* **Borders:**
  * **Solid:** For physical components.
  * **Dashed:** Highly prevalent for indicating "Logical Stages,"
    "Optional Paths," or "Scopes."

### **C. Lines & Arrows**

*Design Philosophy: Line style dictates flow type.*

**Connector Styles**

* **Orthogonal / Elbow (Right Angles):** Most papers use this for **Network
  Architectures** (implies precision, matrices, and tensors).
* **Curved / Bezier:** Common choices include this for **System Logic,
  Feedback Loops, or High-Level Data Flow** (implies narrative and
  connection).

**Line Semantics**

* **Solid Black/Grey:** Standard data flow (Forward pass).
* **Dashed Lines:** Universally recognized as "Auxiliary Flow."
  * *Used for:* Gradient updates, Skip connections, or Loss calculations.
* **Integrated Math:** Standard operators (plus for Add, times for
  Concat/Multiply) are frequently placed *directly* on the line or
  intersection.

### **D. Typography & Icons**

*Design Philosophy: Strict separation between "Labeling" and "Math."*

**Typography**

* **Labels (Module Names):** **Sans-Serif** (Arial, Roboto, Helvetica).
  * *Style:* Bold for headers, Regular for details.
* **Variables (Math):** **Serif** (Times New Roman, LaTeX default).
  * *Rule:* If it is a variable in your equation (e.g., x, theta,
    L), it **must** be Serif and Italicized in the diagram.

**Iconography Options**

* **For Model State:**
  * *Trainable:* Fire, Lightning.
  * *Frozen:* Snowflake, Padlock, Stop Sign (Greyed out).
* **For Operations:**
  * *Inspection:* Magnifying Glass.
  * *Processing/Computation:* Gear, Monitor.
* **For Content:**
  * *Text/Prompt:* Document, Chat Bubble.
  * *Image:* Actual thumbnail of an image (not just a square).

### **E. Layout & Composition**

* **Flow direction:** Left-to-right for sequential pipelines; top-to-bottom
  for hierarchical architectures. Be consistent within a diagram.
* **Alignment:** All elements should snap to an implicit grid. No floating
  or randomly placed components.
* **Spacing:** Consistent gaps between elements. Components within the same
  group should be closer together than components in different groups.
* **Balance:** Distribute visual weight evenly. Avoid heavy clusters on one
  side with empty space on the other.
* **Whitespace:** Use whitespace intentionally to separate phases, stages,
  or conceptual groups. Whitespace is a design element, not wasted space.

---

## 3. Common Pitfalls (How to Look "Amateur")

* **The "PowerPoint Default" Look:** Using standard Blue/Orange presets
  with heavy black outlines.
* **Font Mixing:** Using Times New Roman for "Encoder" labels (makes the
  paper look dated to the 1990s).
* **Inconsistent Dimension:** Mixing flat 2D boxes and 3D isometric cubes
  without a clear reason (e.g., 2D for logic, 3D for tensors is fine;
  random mixing is not).
* **Primary Backgrounds:** Using saturated Yellow or Blue backgrounds for
  grouping (distracts from the content).
* **Ambiguous Arrows:** Using the same line style for "Data Flow" and
  "Gradient Flow."

---

## 4. Domain-Specific Styles

**If you are writing an AGENT / LLM Paper:**
* **Vibe:** Illustrative, Narrative, "Friendly.", Cartoony.
* **Key Elements:** Use "User Interface" aesthetics. Chat bubbles for
  prompts, document icons for retrieval.
* **Characters:** It is common to use cute 2D vector robots, human avatars,
  or emojis to humanize the agent's reasoning steps.

**If you are writing a COMPUTER VISION / 3D Paper:**
* **Vibe:** Spatial, Dense, Geometric.
* **Key Elements:** Frustums (camera cones), Ray lines, and Point Clouds.
* **Color:** Often uses RGB color coding to denote axes or channel
  correspondence. Use heatmaps (Rainbow/Viridis) to show activation.

**If you are writing a THEORETICAL / OPTIMIZATION Paper:**
* **Vibe:** Minimalist, Abstract, "Textbook."
* **Key Elements:** Focus on graph nodes (circles) and manifolds (planes/
  surfaces).
* **Color:** Restrained. Mostly Grayscale/Black/White with one highlight
  color (e.g., Gold or Blue). Avoid "cartoony" elements.

**If you are writing a GENERATIVE / LEARNING Paper:**
* **Vibe:** Dynamic, Process-oriented.
* **Key Elements:** Use noise/denoising visual metaphors, latent space
  representations, and distribution visualizations.
* **Color:** Gradual color transitions to indicate progressive refinement
  or generation stages.
