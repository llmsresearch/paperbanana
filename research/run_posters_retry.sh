#!/bin/zsh
# Re-run the two papers that hit transient failures (VLM JSON / Atlas timeout).
cd "/Users/i0539825/Desktop/Research Paper/PaperBanana" || exit 1
LOG=research/poster_run_retry.log
: > "$LOG"

PAPERS=(
  "minimax_m2:icml:2605.26494"
  "step3_5_flash:iclr:2602.10604"
)

for pair in $PAPERS; do
  slug="${pair%%:*}"; rest="${pair#*:}"; venue="${rest%%:*}"; id="${rest##*:}"
  rm -rf "examples/posters/${slug}"
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
