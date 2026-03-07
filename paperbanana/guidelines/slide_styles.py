"""Slide style presets borrowed from baoyu-slide-deck, elite-powerpoint-designer, and scientific-slides."""

from __future__ import annotations

SLIDE_STYLE_PRESETS: dict[str, dict[str, str]] = {
    # ── Baoyu Slide Deck presets (16) ────────────────────────────────
    "blueprint": {
        "source": "baoyu-slide-deck",
        "feel": "Engineering precision, analytical clarity",
        "auto_select": "architecture, system, data, analysis, technical",
        "prompt": """\
## Style: Blueprint

Design Aesthetic: Clean, structured visual metaphors using blueprints, diagrams, and schematics. Precise, analytical and aesthetically refined. Information presented in grid-based layouts with engineering precision.

Background: Blueprint Off-White (#FAF8F5) with subtle grid overlay, light engineering paper feel.

Typography: Clean sans-serif headlines (Neue Haas Grotesk style), bold weight for titles. Elegant serif for body explanations. Technical, authoritative presence.

Color Palette:
- Background: Blueprint Paper #FAF8F5
- Grid Lines: Light Gray #E5E5E5
- Primary Text: Deep Slate #334155
- Primary Accent: Engineering Blue #2563EB
- Secondary Accent: Navy Blue #1E3A5F
- Tertiary: Light Blue #BFDBFE
- Warning: Amber #F59E0B

Visual Elements: Precise lines with consistent stroke weights. Technical schematics and clean vector graphics. Thin line work in technical drawing style. Connection lines use straight lines or 90-degree angles only. Dimension lines and measurement indicators.

Rules: Maintain consistent line weights. Use grid alignment for all elements. Keep color palette restrained. Create clear visual hierarchy through scale. Use geometric precision for all shapes. No hand-drawn or organic shapes. No decorative flourishes.""",
    },
    "chalkboard": {
        "source": "baoyu-slide-deck",
        "feel": "Classroom warmth, educational",
        "auto_select": "classroom, teaching, school, chalkboard",
        "prompt": """\
## Style: Chalkboard

Design Aesthetic: Warm classroom feel with organic hand-drawn elements on a dark green chalkboard surface. Educational, approachable, and nostalgic.

Background: Dark green chalkboard (#2D4A3E) with subtle chalk dust texture and slight wear patterns.

Typography: Handwritten chalk-style fonts for headlines. Slightly irregular letter spacing for authenticity. White or cream chalk color for text.

Color Palette:
- Background: Chalkboard Green #2D4A3E
- Primary Text: Chalk White #F5F0E8
- Accent 1: Yellow Chalk #F4D35E
- Accent 2: Blue Chalk #7EC8E3
- Accent 3: Pink Chalk #E8A0BF
- Accent 4: Orange Chalk #F4845F

Visual Elements: Hand-drawn borders, arrows, and underlines. Chalk-style diagrams and illustrations. Organic shapes with slightly rough edges. Dashed and dotted chalk lines. Star and asterisk decorations for emphasis.""",
    },
    "corporate": {
        "source": "baoyu-slide-deck",
        "feel": "Business credibility, institutional trust",
        "auto_select": "investor, quarterly, business, corporate",
        "prompt": """\
## Style: Corporate

Design Aesthetic: Clean, professional business presentation with geometric layouts. Trustworthy, data-driven, enterprise-ready. Think Microsoft/IBM product presentations.

Background: Clean white (#FFFFFF) or very light warm gray (#F3F2F1).

Typography: Modern geometric sans-serif (Segoe UI style). Semibold for titles, regular for body. Professional, highly readable at all sizes.

Color Palette:
- Background: Warm Gray #F3F2F1
- Primary: Navy #003366
- Secondary: Steel Blue #0078D4
- Text Primary: Dark Charcoal #323130
- Text Secondary: Medium Gray #605E5C
- Accent: Gold #D4AF37

Visual Elements: Clean geometric shapes. Grid-based balanced layouts. Professional data visualizations. Subtle shadows for depth. Consistent spacing and alignment. No decorative elements.""",
    },
    "minimal": {
        "source": "baoyu-slide-deck",
        "feel": "Maximum sophistication, executive focus",
        "auto_select": "executive, minimal, clean, simple",
        "prompt": """\
## Style: Minimal

Design Aesthetic: Ultra-clean with maximum whitespace. Only essential elements. Sophisticated restraint. Executive-level clarity.

Background: Pure white (#FFFFFF) with no texture or decoration.

Typography: Light-weight geometric sans-serif. Very large titles with thin strokes. Minimal body text. Extreme hierarchy through size contrast.

Color Palette:
- Background: White #FFFFFF
- Primary Text: Near Black #1A1A1A
- Secondary Text: Medium Gray #6B7280
- Single Accent: One carefully chosen color for emphasis
- Dividers: Light Gray #E5E7EB

Visual Elements: Maximum whitespace (60%+ of slide). Single focal point per slide. Thin hairline dividers if needed. No icons, no borders, no shadows. Content speaks through typography and space.""",
    },
    "sketch-notes": {
        "source": "baoyu-slide-deck",
        "feel": "Friendly learning, approachable education",
        "auto_select": "tutorial, learn, education, guide, beginner",
        "prompt": """\
## Style: Sketch Notes

Design Aesthetic: Hand-drawn sketchnote style with warm, approachable feel. Educational and friendly, like illustrated lecture notes. Organic and playful.

Background: Warm off-white paper (#FFF8F0) with subtle paper texture.

Typography: Handwritten-style fonts with varied sizes. Bold marker-style for headlines. Thin pen-style for annotations. Mixed case for friendliness.

Color Palette:
- Background: Warm Paper #FFF8F0
- Primary Ink: Dark Brown #3D2B1F
- Accent 1: Coral #FF6B6B
- Accent 2: Teal #4ECDC4
- Accent 3: Mustard #FFE66D
- Accent 4: Sky Blue #95E1D3
- Highlight: Light Yellow #FFEAA7

Visual Elements: Hand-drawn boxes, arrows, and banners. Doodle-style icons and illustrations. Thought bubbles and speech bubbles. Underlines and circles for emphasis. Small decorative elements (stars, dots, swirls).""",
    },
    "watercolor": {
        "source": "baoyu-slide-deck",
        "feel": "Artistic, natural, lifestyle",
        "auto_select": "lifestyle, wellness, travel, artistic",
        "prompt": """\
## Style: Watercolor

Design Aesthetic: Soft watercolor washes with organic, flowing shapes. Artistic and delicate. Natural, calming, and visually rich. Minimal text with large visual areas.

Background: Soft cream (#FAF7F2) with subtle watercolor wash edges.

Typography: Elegant humanist sans-serif or light serif. Thin strokes for an airy feel. Minimal text — visuals dominate.

Color Palette:
- Background: Cream #FAF7F2
- Wash 1: Soft Blue #B8D4E3
- Wash 2: Blush Pink #F2C4C4
- Wash 3: Sage Green #B5C7A3
- Wash 4: Lavender #C5B4E3
- Text: Warm Dark Gray #4A4A4A

Visual Elements: Watercolor wash backgrounds and borders. Soft gradient color transitions. Organic, flowing shapes. Delicate line illustrations. Floral or natural motifs.""",
    },
    "dark-atmospheric": {
        "source": "baoyu-slide-deck",
        "feel": "Cinematic, entertainment",
        "auto_select": "entertainment, music, gaming, atmospheric",
        "prompt": """\
## Style: Dark Atmospheric

Design Aesthetic: Dark cinematic mood with dramatic lighting. Editorial magazine quality on dark backgrounds. Moody, sophisticated, immersive.

Background: Near black (#1A1A2E) or deep dark blue (#16213E) with subtle gradient.

Typography: Bold editorial serif or dramatic sans-serif headlines. High contrast white or light text on dark. Elegant, magazine-style layout.

Color Palette:
- Background: Dark Navy #1A1A2E
- Text: Bright White #FFFFFF
- Accent 1: Electric Blue #4CC9F0
- Accent 2: Magenta #F72585
- Accent 3: Amber #F59E0B
- Subtle: Dark Gray #374151

Visual Elements: Dramatic gradient overlays. Glowing accent elements. Cinematic lighting effects. Bold typography as visual element. Moody atmospheric backgrounds.""",
    },
    "notion": {
        "source": "baoyu-slide-deck",
        "feel": "SaaS professional, data-forward",
        "auto_select": "saas, product, dashboard, metrics",
        "prompt": """\
## Style: Notion

Design Aesthetic: Clean SaaS product style inspired by Notion, Linear, and modern productivity tools. Neutral, functional, information-dense but organized.

Background: Pure white (#FFFFFF) or very light gray (#FAFAFA).

Typography: Clean geometric sans-serif (Inter style). Regular weight for most text. Medium weight for emphasis. Monospace for data and code.

Color Palette:
- Background: White #FFFFFF
- Text: Near Black #191919
- Secondary Text: Gray #787774
- Border: Light Gray #E5E5E3
- Accent: Notion Blue #2F80ED
- Tags: Soft pastels (pink #FFE2DD, green #DBEDDB, blue #D3E5EF)

Visual Elements: Card-style containers with subtle borders. Clean data tables. Toggle/accordion indicators. Tag-style labels with pastel backgrounds. Checkbox-style list items. Dense but well-organized layouts.""",
    },
    "bold-editorial": {
        "source": "baoyu-slide-deck",
        "feel": "Magazine impact, keynote drama",
        "auto_select": "launch, marketing, keynote, magazine",
        "prompt": """\
## Style: Bold Editorial

Design Aesthetic: High-impact magazine editorial with vibrant colors and dramatic typography. Bold, confident, attention-grabbing. Product launches and keynotes.

Background: White (#FFFFFF) or bold solid color blocks.

Typography: Extra-bold condensed sans-serif for headlines. Dramatic size contrast between title and body. All-caps for section headers. Editorial serif for body text.

Color Palette:
- Background: White #FFFFFF or bold color blocks
- Primary: Vibrant Red #E63946
- Secondary: Deep Blue #1D3557
- Accent: Electric Yellow #FFD700
- Text: Near Black #0A0A0A
- Contrast: Pure White #FFFFFF

Visual Elements: Large bold typography as visual element. Color-block backgrounds. Full-bleed images with text overlays. Dramatic asymmetric layouts. Pull quotes and oversized numbers.""",
    },
    "editorial-infographic": {
        "source": "baoyu-slide-deck",
        "feel": "Publication quality, informative",
        "auto_select": "explainer, journalism, science communication",
        "prompt": """\
## Style: Editorial Infographic

Design Aesthetic: Data-rich editorial infographic style. Cool tones, dense but clear information display. Publication-quality data journalism.

Background: White (#FFFFFF) or very light cool gray (#F8FAFC).

Typography: Clean editorial sans-serif. Clear hierarchy from headline to data labels. Condensed fonts for dense data.

Color Palette:
- Background: Cool White #F8FAFC
- Primary Text: Dark Slate #0F172A
- Secondary Text: Slate #475569
- Data 1: Teal #0D9488
- Data 2: Indigo #4F46E5
- Data 3: Amber #D97706
- Data 4: Rose #E11D48

Visual Elements: Clean data visualizations. Icon-driven statistics. Structured grid layouts. Numbered sequences. Callout boxes with data highlights. Small multiples for comparisons.""",
    },
    "fantasy-animation": {
        "source": "baoyu-slide-deck",
        "feel": "Magical, storytelling",
        "auto_select": "story, fantasy, animation, magical",
        "prompt": """\
## Style: Fantasy Animation

Design Aesthetic: Vibrant animated fantasy style. Magical, whimsical, and colorful. Organic hand-drawn feel with rich illustration. Storytelling through visuals.

Background: Gradient skies or magical landscapes with soft colors.

Typography: Whimsical handwritten or rounded display fonts. Playful and expressive. Varied sizes for storytelling rhythm.

Color Palette:
- Background: Soft gradient (sky blue to lavender)
- Primary: Deep Purple #6C3483
- Accent 1: Golden #F1C40F
- Accent 2: Emerald #27AE60
- Accent 3: Coral #E74C3C
- Magic: Sparkle White #FFFFFF with glow

Visual Elements: Illustrated characters or icons. Magical sparkle effects. Organic flowing shapes. Storybook-style borders. Rich, layered backgrounds.""",
    },
    "intuition-machine": {
        "source": "baoyu-slide-deck",
        "feel": "Technical briefing, bilingual documentation",
        "auto_select": "briefing, academic, research, bilingual",
        "prompt": """\
## Style: Intuition Machine

Design Aesthetic: Technical briefing style with cool analytical precision. Dense information presented clearly. Academic and research-oriented. Bilingual-friendly layouts.

Background: White (#FFFFFF) or very light blue-gray (#F0F4F8).

Typography: Technical sans-serif (monospace for data, sans-serif for text). Clear hierarchy. Dense but readable. CJK-compatible font choices.

Color Palette:
- Background: Light Blue-Gray #F0F4F8
- Primary Text: Dark Slate #1E293B
- Secondary Text: Slate #64748B
- Accent 1: Blue #3B82F6
- Accent 2: Teal #0D9488
- Accent 3: Violet #8B5CF6
- Border: Gray #CBD5E1

Visual Elements: Structured data tables. Technical diagrams. Numbered reference systems. Dense callout boxes. Multi-column layouts. Citation-style references.""",
    },
    "pixel-art": {
        "source": "baoyu-slide-deck",
        "feel": "Retro gaming, developer culture",
        "auto_select": "gaming, retro, pixel, developer",
        "prompt": """\
## Style: Pixel Art

Design Aesthetic: 8-bit retro pixel art style. Chunky pixels, vibrant colors, nostalgic gaming aesthetic. Fun and developer-friendly.

Background: Dark pixel background (#1A1A2E) or retro game screen green (#0F380F).

Typography: Pixel/bitmap-style fonts. Monospace for all text. Chunky, clearly readable characters.

Color Palette:
- Background: Dark #1A1A2E
- Text: Bright Green #00FF41
- Accent 1: Pixel Red #FF004D
- Accent 2: Pixel Blue #29ADFF
- Accent 3: Pixel Yellow #FFEC27
- Accent 4: Pixel Pink #FF77A8

Visual Elements: Pixel art icons and illustrations. 8-bit style borders and frames. Retro game UI elements. Scanline or CRT effects. Blocky geometric shapes.""",
    },
    "scientific": {
        "source": "baoyu-slide-deck",
        "feel": "Academic precision, research quality",
        "auto_select": "biology, chemistry, medical, scientific",
        "prompt": """\
## Style: Scientific

Design Aesthetic: Academic scientific illustration for biological pathways, chemical processes, and technical systems. Clean, precise diagrams with proper labeling. Textbook quality illustrations and academic journal figures.

Background: Off-White (#FAFAFA) or Light Blue-Gray (#F0F4F8). No texture or very subtle paper grain.

Typography: Clean serif font (Times New Roman style) for formal academic headlines. Sans-serif for diagram labels and annotations. Clear, readable at small sizes. Consistent sizing hierarchy.

Color Palette:
- Background: Off-White #FAFAFA
- Primary Text: Dark Slate #1E293B
- Label Text: Medium Gray #475569
- Pathway 1: Teal #0D9488
- Pathway 2: Blue #3B82F6
- Pathway 3: Purple #8B5CF6
- Membrane: Amber #F59E0B
- Alert: Red #EF4444
- Positive: Green #22C55E

Visual Elements: Precise, consistent line weights. Labeled modular components with distinct colors. Flow arrows for movement. Chemical formulas and molecular notation. Cross-section and pathway diagrams. Numbered step sequences. Process summary boxes.""",
    },
    "vector-illustration": {
        "source": "baoyu-slide-deck",
        "feel": "Flat design, friendly creative",
        "auto_select": "creative, children, kids, cute",
        "prompt": """\
## Style: Vector Illustration

Design Aesthetic: Modern flat vector illustration style. Vibrant, friendly, and creative. Clean geometric shapes with bold colors. Suitable for children's content and creative presentations.

Background: White (#FFFFFF) or soft pastel (#F5F0FF).

Typography: Rounded humanist sans-serif. Friendly and approachable. Bold for headlines, regular for body.

Color Palette:
- Background: Soft Lavender #F5F0FF
- Primary: Bright Blue #4361EE
- Accent 1: Coral #FF6B6B
- Accent 2: Mint #00C9A7
- Accent 3: Sunny Yellow #FFD93D
- Accent 4: Soft Purple #C77DFF
- Text: Dark Charcoal #2B2D42

Visual Elements: Flat vector illustrations with no gradients. Bold geometric shapes. Rounded corners on all elements. Character illustrations with simple features. Playful icons and decorative elements.""",
    },
    "vintage": {
        "source": "baoyu-slide-deck",
        "feel": "Historical, heritage storytelling",
        "auto_select": "history, heritage, vintage, expedition",
        "prompt": """\
## Style: Vintage

Design Aesthetic: Warm vintage aesthetic with aged paper textures. Historical, heritage storytelling feel. Editorial typography with classic proportions. Think vintage travel posters and old book illustrations.

Background: Aged paper (#F4ECD8) with subtle texture, slight yellowing, and worn edges.

Typography: Classic editorial serif fonts. Display serif for headlines. Elegant letterforms with traditional proportions. Warm dark brown text.

Color Palette:
- Background: Aged Paper #F4ECD8
- Primary Text: Dark Brown #3C2415
- Accent 1: Burgundy #800020
- Accent 2: Forest Green #2D5016
- Accent 3: Navy #1B3A5C
- Accent 4: Mustard #C9A227
- Faded: Sepia #704214

Visual Elements: Ornamental borders and frames. Vintage illustration style. Decorative dividers and flourishes. Woodcut or engraving-inspired graphics. Stamp and seal elements. Aged paper textures.""",
    },

    # ── Elite Powerpoint Designer presets (3 new) ────────────────────
    "tech-keynote": {
        "source": "elite-powerpoint-designer",
        "feel": "Apple/Tesla premium minimalism",
        "auto_select": "product launch, demo, tech reveal, premium",
        "prompt": """\
## Style: Tech Keynote

Design Aesthetic: Premium tech product launch style inspired by Apple keynotes and Tesla reveals. Ultra-minimalist with dramatic single focal points. Every pixel is intentional. Maximum impact through restraint.

Background: Pure Black (#000000) or Pure White (#FFFFFF). No in-between. No textures.

Typography: SF Pro Display style — ultra-clean sans-serif. Title: 72-96pt bold. One word or phrase per line. Extreme negative space around text. Text IS the design.

Color Palette:
- Background: Black #000000 or White #FFFFFF
- Text: White #FFFFFF (on black) or Black #000000 (on white)
- Single Accent: Apple Blue #0071E3
- Secondary Text: Medium Gray #8E8E93
- No other colors unless showing a product

Visual Elements: Single product/hero image per slide. Full-bleed photography. Extreme whitespace (70%+ empty). No borders, no lines, no boxes. Gradient only for product backgrounds. Dramatic scale — one giant number or word. Cinematic transitions between concepts.""",
    },
    "creative-bold": {
        "source": "elite-powerpoint-designer",
        "feel": "Google/Airbnb energetic innovation",
        "auto_select": "innovation, design showcase, startup culture, creative agency",
        "prompt": """\
## Style: Creative Bold

Design Aesthetic: Energetic, design-forward style inspired by Google I/O and Airbnb presentations. Bold color combinations, dynamic layouts, and playful asymmetry. Projects confidence and innovation.

Background: White (#FFFFFF) with bold color accent blocks, or full-color backgrounds.

Typography: Product Sans or Montserrat style — geometric sans-serif. Title: 64-84pt bold. Playful size variations. Mixed alignment for dynamism.

Color Palette:
- Primary: Google Red #EA4335
- Secondary: Google Blue #4285F4
- Accent 1: Google Yellow #FBBC05
- Accent 2: Google Green #34A853
- Text: Dark Gray #202124
- Secondary Text: Medium Gray #5F6368

Visual Elements: Dynamic asymmetric layouts. Bold color-block backgrounds. Playful icon illustrations. Vibrant gradients (not subtle). Oversized numbers and statistics. Card-style content blocks with rounded corners. Energetic, movement-suggesting compositions.""",
    },
    "financial-elite": {
        "source": "elite-powerpoint-designer",
        "feel": "Goldman Sachs/McKinsey sophistication",
        "auto_select": "finance, investment, consulting, M&A, private equity",
        "prompt": """\
## Style: Financial Elite

Design Aesthetic: Ultra-sophisticated financial presentation style. Goldman Sachs annual report meets McKinsey deck. Understated elegance with serif typography and gold accents. Authority through restraint.

Background: Pure White (#FFFFFF). Clean, no texture.

Typography: Garamond or Georgia — elegant serif fonts. Title: 60pt semibold. Body: 26pt regular. Traditional hierarchy. Centered layout preferred. Letter-spacing slightly tracked for elegance.

Color Palette:
- Background: White #FFFFFF
- Primary: Charcoal #2C3E50
- Accent: Gold #D4AF37
- Secondary: Slate Gray #7F8C8D
- Text: Dark Navy #2C3E50
- Data Positive: Dark Green #1B5E20
- Data Negative: Dark Red #B71C1C

Visual Elements: Clean data tables with thin hairlines. Understated charts (no 3D, no gradients). Gold accent lines as dividers. Traditional centered compositions. Metric callouts with serif numerals. No icons, no illustrations — data speaks. Minimal everything — sophistication through absence.""",
    },

    # ── Scientific Slides topic presets (4 new) ──────────────────────
    "biotech": {
        "source": "scientific-slides",
        "feel": "Life sciences, biotechnology, genomics",
        "auto_select": "biotech, genomics, drug discovery, pharmaceutical, life sciences",
        "prompt": """\
## Style: Biotech

Design Aesthetic: Modern life sciences presentation style. Clean and professional with teal-coral color scheme reflecting biological systems. Suitable for biotech conferences, genomics talks, and pharmaceutical presentations.

Background: White (#FFFFFF) or very light teal tint (#F0FDFA).

Typography: Clean sans-serif (Arial or Calibri). Title: 40-54pt bold. Body: 24-28pt regular. Clear figure labels at 18-24pt.

Color Palette:
- Background: White #FFFFFF
- Primary: Teal #0A9396
- Accent: Coral #EE6C4D
- Secondary: Cream #F4F1DE
- Text: Charcoal #2C2C2C
- Data 1: Teal #0A9396
- Data 2: Coral #EE6C4D
- Data 3: Navy #264653
- Highlight: Burgundy #780000

Visual Elements: Clean molecular and pathway diagrams. Color-blind safe data visualizations (blue/orange preferred). High-resolution microscopy or structural images. Generous whitespace (40-50%). Direct labeling on plots rather than legends. Professional figure annotations.""",
    },
    "neuroscience": {
        "source": "scientific-slides",
        "feel": "Brain research, cognitive science, neural systems",
        "auto_select": "neuroscience, brain, cognitive, neural, fMRI, EEG",
        "prompt": """\
## Style: Neuroscience

Design Aesthetic: Deep, dramatic neuroscience presentation style. Purple-magenta color scheme evoking neural activity and brain imaging. Modern and visually striking while maintaining scientific rigor.

Background: White (#FFFFFF) or very dark purple (#1A0A2E) for impact slides.

Typography: Modern sans-serif (Helvetica or Futura). Title: 40-54pt bold. Body: 24-28pt regular. Clean, highly readable.

Color Palette:
- Background: White #FFFFFF or Dark Purple #1A0A2E
- Primary: Deep Purple #722880
- Accent: Magenta #D72D51
- Secondary: Electric Blue #3B82F6
- Text: Dark Charcoal #1A1A1A (light bg) or White #FFFFFF (dark bg)
- Neural 1: Purple #722880
- Neural 2: Magenta #D72D51
- Neural 3: Cyan #06B6D4
- Activation: Hot Orange #F97316

Visual Elements: Brain imaging visualizations (fMRI, EEG topoplots). Neural network diagrams. Activation heatmaps. Dark background slides for brain images. High-contrast data visualizations. Progressive build for complex neural pathways.""",
    },
    "ml-ai": {
        "source": "scientific-slides",
        "feel": "Machine learning, artificial intelligence, deep learning",
        "auto_select": "machine learning, AI, deep learning, neural network, transformer, LLM",
        "prompt": """\
## Style: ML/AI

Design Aesthetic: Modern machine learning and AI presentation style. Bold, technical, and forward-looking. Red-orange energy palette conveying computational power and innovation. Clean architecture diagrams.

Background: White (#FFFFFF) or Dark Gray (#1F2937) for technical architecture slides.

Typography: Modern geometric sans-serif (Inter or Roboto). Monospace (JetBrains Mono style) for model names, hyperparameters, and code. Title: 40-54pt bold. Body: 24-28pt.

Color Palette:
- Background: White #FFFFFF or Dark #1F2937
- Primary: Bold Red #E74C3C
- Accent: Orange #F39C12
- Secondary: Dark Gray #2C2C2C
- Code: Monospace on light gray #F3F4F6
- Data 1: Red #E74C3C
- Data 2: Blue #3498DB
- Data 3: Green #2ECC71
- Loss curve: Orange gradient

Visual Elements: Neural network architecture diagrams. Training loss/accuracy curves. Attention heatmaps. Model comparison tables. Code snippets with syntax highlighting. Performance benchmark charts. Confusion matrices. GPU/compute icons.""",
    },
    "environmental": {
        "source": "scientific-slides",
        "feel": "Environmental science, ecology, sustainability",
        "auto_select": "environment, ecology, climate, sustainability, conservation, earth science",
        "prompt": """\
## Style: Environmental

Design Aesthetic: Natural, earthy environmental science presentation style. Sage-terracotta palette evoking landscapes and natural systems. Warm but scientific. Suitable for ecology, climate science, and sustainability talks.

Background: Cream (#FAF7F2) or very light sage (#F0F7F0).

Typography: Humanist sans-serif (Gill Sans or Avenir style). Warm, approachable but professional. Title: 40-54pt bold. Body: 24-28pt regular.

Color Palette:
- Background: Cream #FAF7F2
- Primary: Sage Green #87A96B
- Accent: Terracotta #E07A5F
- Secondary: Earth Brown #8B6F47
- Text: Dark Charcoal #2C2C2C
- Data 1: Forest Green #2D5016
- Data 2: Terracotta #E07A5F
- Data 3: Sky Blue #4AA3DF
- Water: Ocean Blue #1B6B93

Visual Elements: Map visualizations and spatial data. Climate/weather diagrams. Ecosystem flow charts. Satellite imagery. Natural color gradients. Species illustrations. Time-series environmental data. Geographic information overlays.""",
    },
}


# ── Public API ──────────────────────────────────────────────────────


def list_styles() -> list[str]:
    """Return sorted list of all available style preset names."""
    return sorted(SLIDE_STYLE_PRESETS.keys())


def get_style_prompt(style_name: str) -> str:
    """Return the image-generation prompt instructions for a named style.

    Raises:
        KeyError: If the style name is not found.
    """
    key = style_name.lower().replace(" ", "-")
    if key not in SLIDE_STYLE_PRESETS:
        available = ", ".join(list_styles())
        raise KeyError(
            f"Unknown slide style '{style_name}'. "
            f"Available styles: {available}"
        )
    return SLIDE_STYLE_PRESETS[key]["prompt"]


def get_style_info(style_name: str) -> dict[str, str]:
    """Return full metadata for a named style."""
    key = style_name.lower().replace(" ", "-")
    if key not in SLIDE_STYLE_PRESETS:
        raise KeyError(f"Unknown slide style '{style_name}'.")
    return SLIDE_STYLE_PRESETS[key]


def match_style(content_signals: str) -> str | None:
    """Auto-detect best style preset from content signal keywords.

    Returns the style name if a match is found, None otherwise.
    """
    signals_lower = content_signals.lower()
    best_match = None
    best_score = 0

    for name, preset in SLIDE_STYLE_PRESETS.items():
        keywords = [k.strip() for k in preset["auto_select"].split(",")]
        score = sum(1 for kw in keywords if kw in signals_lower)
        if score > best_score:
            best_score = score
            best_match = name

    return best_match if best_score > 0 else None
