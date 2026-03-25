# ACL Method Diagram Aesthetics Guide

## 1. The "ACL Look"

ACL diagrams reflect the NLP community's preference for **narrative
clarity**. Diagrams often tell a story: input text flows through
processing stages to produce output. The style favors horizontal
pipelines with clearly labeled stages, making heavy use of text
examples and linguistic annotations within the diagram itself.

---

## 2. Detailed Style Options

### **A. Color Palettes**

*Design Philosophy: Soft, readable colors that do not compete with
the text examples embedded in diagrams.*

**Background Fills**

* **Primary approach:** White backgrounds with light-colored containers
  (pale blue, pale green, light yellow) to group pipeline stages.
* **Text highlighting:** Use colored backgrounds behind example tokens
  or spans to show entity types, attention, or alignments.
* **Avoid:** Dark or saturated backgrounds that reduce readability of
  embedded text samples.

**Functional Element Colors**

* **Encoder blocks:** Blues and teals.
* **Decoder / generation:** Greens and warm yellows.
* **Attention mechanisms:** Orange or coral highlights.
* **Loss / training signals:** Red, used sparingly.
* **Text spans / entities:** Distinct pastel highlights per category
  (e.g., light blue for Person, light pink for Location).

### **B. Shapes & Containers**

* **Process blocks:** Rounded rectangles, often wider than tall to
  accommodate text labels.
* **Text examples:** Rectangles with visible text content inside,
  sometimes with token-level colored backgrounds.
* **Embeddings:** Narrow vertical stacks or horizontal bars.
* **Grouping:** Light-colored rectangular containers with headers.
  Dashed borders for optional or conditional paths.

### **C. Lines & Arrows**

* **Standard flow:** Left-to-right solid arrows for the main pipeline.
* **Attention / alignment:** Curved or diagonal lines connecting source
  and target tokens, often with varying opacity to show weight.
* **Skip connections:** Dashed curved arrows.
* **Cross-attention:** Bidirectional arrows or parallel lines between
  encoder and decoder stacks.

### **D. Typography & Icons**

* **Labels:** Sans-serif, clean and readable. Bold for component names.
* **Example text:** Monospace or serif font to distinguish from labels,
  often quoted or in a distinct box.
* **Math:** LaTeX-style serif italic for formulas and variables.
* **Icons:** Speech bubbles for dialogue, document icons for retrieval,
  magnifying glass for search. Keep them simple line-art style.

### **E. Layout & Composition**

* **Flow:** Predominantly left-to-right, reflecting the sequential
  nature of text processing.
* **Vertical stacking:** Encoder-decoder architectures often use
  top-to-bottom layout with cross-connections.
* **Example integration:** Show actual text snippets flowing through
  the pipeline, not just abstract tensor shapes.
* **Annotation:** Label every non-obvious component. NLP audiences
  expect explicit stage names.

---

## 3. Common Pitfalls

* **Missing text examples** — NLP diagrams without concrete examples
  feel abstract and harder to follow.
* **Overcrowded token-level diagrams** that become unreadable.
* **Inconsistent entity coloring** across sub-figures.
* **Unlabeled arrows** between encoder and decoder.
* **Tiny font in embedded text** — must remain readable at print size.

---

## 4. Domain-Specific Styles

**Information Extraction / NER:**
* Token sequences with colored span highlights.
* Tables showing predicted vs. gold labels.

**Machine Translation / Seq2Seq:**
* Side-by-side source and target with alignment lines.
* Attention heatmaps as inset panels.

**Dialogue / Conversational AI:**
* Chat-bubble style layouts with agent and user turns.
* Pipeline from input utterance through intent/slot extraction to response.

**Retrieval-Augmented Generation:**
* Document retrieval shown as a search step with ranked results.
* Retrieved passages flowing into the generation module.
