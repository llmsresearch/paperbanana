#!/bin/zsh
# Regenerate showcase posters for 5 fresh 2026 LLM papers (local only).
cd "/Users/i0539825/Desktop/Research Paper/PaperBanana" || exit 1
LOG=research/poster_run.log
: > "$LOG"

# slug:venue:arxiv_id
PAPERS=(
  "kimi_k2_5:neurips:2602.02276"
  "minimax_m2:icml:2605.26494"
  "nemotron3_super:neurips:2604.12374"
  "vibethinker_3b:iclr:2606.16140"
  "zaya1_8b:icml:2605.05365"
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
    --image-provider openai_imagen \
    --output-dir "examples/posters/${slug}" \
    --verbose >> "$LOG" 2>&1
  echo "---- exit=$? for $slug $(date +%H:%M:%S) ----" | tee -a "$LOG"
done
echo "ALL DONE $(date +%H:%M:%S)" | tee -a "$LOG"
