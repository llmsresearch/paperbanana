#!/bin/zsh
# Demo posters: 5 top 2026 LLM technical papers, rendered with Google
# Nano Banana Pro (Gemini-3-Pro-Image) via Atlas Cloud at 4k. Local only.
cd "/Users/i0539825/Desktop/Research Paper/PaperBanana" || exit 1
LOG=research/poster_run_nbp.log
: > "$LOG"

# slug:venue:arxiv_id
PAPERS=(
  "kimi_k2_5:neurips:2602.02276"
  "ernie5:iclr:2602.04705"
  "nemotron3_super:neurips:2604.12374"
  "minimax_m2:icml:2605.26494"
  "step3_5_flash:iclr:2602.10604"
)

for pair in $PAPERS; do
  slug="${pair%%:*}"; rest="${pair#*:}"; venue="${rest%%:*}"; id="${rest##*:}"
  echo "================ $slug ($venue, $id) $(date +%H:%M:%S) ================" | tee -a "$LOG"
  paperbanana poster \
    --paper "research/source_papers/${slug}.pdf" \
    --venue "$venue" \
    --figures auto \
    --qr-url "https://arxiv.org/abs/${id}" \
    --budget 3 \
    --vlm-provider atlas \
    --image-provider atlas_imagen \
    --image-model "google/nano-banana-pro/text-to-image" \
    --output-dir "examples/posters/${slug}" \
    --verbose >> "$LOG" 2>&1
  echo "---- exit=$? for $slug $(date +%H:%M:%S) ----" | tee -a "$LOG"
done
echo "ALL DONE $(date +%H:%M:%S)" | tee -a "$LOG"
