# UMS 2018 Methodology Diagram / Figure Aesthetics Guide

Adapted from *Garis Panduan Penyerahan dan Penulisan Tesis/Disertasi
Gaya UMS* (Pusat Pengajian Pascasiswazah, Universiti Malaysia Sabah,
2018). This guide adapts the UMS thesis style for methodology figures
(rajah) and conceptual diagrams embedded in a UMS thesis or
disertasi.

---

## 1. The "UMS Look"

UMS thesis figures favour **clarity, formality, and reproducibility**
over decorative visual language. A methodology diagram in a UMS
thesis should read like a printed textbook figure: monochrome or
near-monochrome, sans-serif, with strict alignment, and a caption
placed *below* the figure (left-aligned) rather than above. Diagrams
must remain legible after black-and-white photocopying because the
final submission is a hard-cover bound printout on 80 gsm A4 paper.

- **Vibe:** Formal, academic, government-document feel.
- **Backgrounds:** Plain white only. No coloured page tint, no
  gradients, no drop shadows.
- **Reproducibility:** Diagram must remain readable in grayscale and
  at 1:1 print size on A4.
- **Language:** Labels in Bahasa Melayu *or* Bahasa Inggeris,
  matching the thesis body. Foreign-language terms (e.g. Latin,
  English inside a Malay thesis) are *italicised*.

---

## 2. Detailed Style Options

### A. Color Palettes

*Design Philosophy: Use colour sparingly and only when it carries
information. The figure must still be interpretable when printed in
black & white.*

**Background Fills**

- Use plain white (`#FFFFFF`) only. Never tint the figure background.
- Grouping zones, if needed, use **light grey** (`#F2F2F2`) or
  thin **dashed black** borders — never pastel fills.

**Line and Text Colour**

- Default line and text colour is **black** (`#000000`).
- Greys (`#666666`, `#999999`) are allowed for auxiliary elements
  (gridlines, secondary arrows, de-emphasised nodes).
- If colour is used to distinguish categories, restrict to a palette
  of **at most 4 colours** drawn from: dark blue (`#1F3864`), dark
  red (`#8B0000`), dark green (`#1E5631`), ochre (`#B8860B`).
  Saturated/neon colours are forbidden.

**Highlights**

- A single accent colour may be used for the *one* concept the figure
  is trying to emphasise. Everything else stays black/grey.

### B. Shapes & Containers

*Design Philosophy: Plain geometric shapes with thin black outlines.
The diagram should look like it was drawn in Microsoft Word's
SmartArt rather than in a glossy slide deck.*

- **Process nodes:** Rectangles (not rounded), 0.75 pt black border,
  white fill.
- **Decision nodes:** Diamonds.
- **Data / I/O:** Parallelograms.
- **Stored data / database:** Cylinders.
- **Grouping container:** Thin dashed black rectangle with a small
  bold label in the top-left corner, set in Tahoma 9 pt.
- **Avoid:** Drop shadows, bevels, 3D extrusions, glow effects,
  rounded "soft tech" containers.

### C. Lines & Arrows

- **Default arrow:** Solid black line, 0.75 pt, with a small filled
  triangular arrowhead. Strictly orthogonal (right-angle) routing
  when connecting process boxes.
- **Feedback / iteration:** Same solid arrow but with a small label
  on the line (e.g. "ulangan", "iterate").
- **Auxiliary flow** (e.g. data exchange, optional path): dashed
  black line, same weight.
- Operators (`+`, `×`, `Σ`) placed on the line are allowed, set in
  the same font family as labels.
- Do **not** use curved Bezier arrows for the main pipeline — reserve
  curves only for clearly non-linear feedback loops.

### D. Typography & Icons

- **Font family:** Tahoma for all labels, captions, and embedded
  text. This matches the UMS body text rule (§6.2 b of the
  guideline).
- **Label size:** 11 pt for primary labels, 9 pt for sub-labels and
  in-figure annotations. Never smaller than 8 pt after final
  scaling.
- **Weight:** Regular for ordinary labels; **bold** only for
  container/zone headers and for the figure caption stem
  ("Rajah 4.2:" / "Figure 4.2:").
- **Variables and mathematical symbols:** Italic, may switch to a
  serif (e.g. Cambria Math, Times) only for true LaTeX-style maths.
- **Capitalisation:** Title Case for headers (Huruf besar pada
  setiap awal perkataan, kecuali kata sendi nama dan kata hubung).
  Sentence case for descriptive sub-labels.
- **Icons:** Optional and minimal. If used, line-art (outline-only)
  icons in black. No emoji, no flat colour illustrations, no
  cartoon characters.

### E. Layout & Composition

- **Page geometry:** Figure must fit within the UMS text area:
  left margin 38 mm, right/top/bottom margin 28 mm on A4 (210 ×
  297 mm). Useful drawing width is approximately **144 mm**.
- **Flow direction:** Left-to-right for sequential pipelines;
  top-to-bottom for hierarchical/breakdown diagrams. Pick one
  primary axis and stay consistent within the figure.
- **Alignment:** All nodes snap to a shared grid; arrowheads align
  with node centres.
- **Whitespace:** At least 6 mm clear space between adjacent nodes;
  at least 10 mm clear space between the figure body and the
  caption.
- **Aspect ratio:** Landscape figures are acceptable; if rotated
  90°, the top of the figure must face the binding (left) edge.

### F. Caption ("Tajuk Rajah")

UMS places the figure title and source **below** the figure, aligned
to the **left** (§6.7.3, format Rajah).

- Line 1 (caption): `Rajah X.Y: <Tajuk Rajah>`
  - `Rajah X.Y` is bold; the descriptive title is regular weight.
  - `X` is the chapter number, `Y` the running figure number within
    that chapter. Restart `Y` at 1 in every new chapter.
- Line 2 (source, optional): `Sumber: <citation in APA style>`
  - One blank line (1.5 line-spacing) between the figure and the
    caption; single line-spacing between caption and source.
- The same scheme applies to photos (Foto X.Y) and musical
  notation (Notasi X.Y).

---

## 3. Common Pitfalls (Avoid These)

- **Coloured pastel backgrounds** ("Soft Tech" NeurIPS look).
  Forbidden — UMS requires plain white.
- **Mixed fonts:** Times New Roman, Calibri, Arial, Comic Sans
  inside the figure. Use Tahoma only (Cambria Math acceptable for
  maths).
- **Caption above the figure:** This is the table convention, not
  the figure convention. Figure captions go *below*.
- **Centre-aligned figure caption:** UMS aligns figure captions to
  the left.
- **Drop shadows, glows, gradients, 3D bevels.**
- **Saturated rainbow palettes** or palettes that lose meaning in
  grayscale.
- **Abbreviations inside the figure title** (§6.3 a — singkatan
  tidak dibenarkan dalam tajuk).
- **Decorative emoji or cartoon icons.**
- **Text smaller than 8 pt** at final print size.

---

## 4. Domain-Specific Adjustments

UMS does not prescribe per-discipline visual styles, but typical
choices observed in UMS theses are:

- **Sains Sosial / Pendidikan / Perniagaan:** Conceptual framework
  ("kerangka konseptual") diagrams with labelled boxes and arrows.
  Strict orthogonal layout, no icons.
- **Sains & Teknologi / Kejuruteraan:** Block diagrams, flowcharts
  (ANSI shapes), schematics. Greyscale acceptable.
- **Sains Hayat / Pertanian:** Workflow / experimental design
  figures with sample sizes annotated. Mild use of one accent
  colour to indicate the treatment group is acceptable.
- **Seni / Muzik:** Notasi muzik captioned as "Notasi X.Y:"
  following §6.9.

---

## 5. Reference Anchors

- Body font and size: §6.2 b, c (Tahoma 11 pt, black).
- Margins: §6.6.3 (28/28/38/28 mm).
- Numbering: §6.7.2 (1.1, 1.1.1; max three numeric levels).
- Figure caption placement: §6.7.3 (Format Rajah).
- Title constraints: §6.3 a (no abbreviations; ≤ 20 words; uppercase
  for the thesis title).
- Foreign-language italicisation: §6.1.
