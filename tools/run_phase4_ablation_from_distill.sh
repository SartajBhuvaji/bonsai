#!/usr/bin/env bash
# Resume Phase 4 after both 6000-sample teacher caches (Steps 3b–3c only).
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /root/bonsai
source /root/venv/bin/activate
set -a
# shellcheck disable=SC1091
source /root/bonsai/.env
set +a
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"
HF_ARGS=(--hf-token "$HF_TOKEN")
LOG=/root/phase4_pipeline.log
exec > >(tee -a "$LOG") 2>&1

# shellcheck disable=SC1091
source /root/phase4_state.env
: "${CANDIDATE:?missing CANDIDATE in phase4_state.env}"
: "${BASELINE_CACHE:?missing BASELINE_CACHE}"
: "${LORA_CACHE:?missing LORA_CACHE}"

echo "=== Phase 4 resume from distill $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

echo "--- Step 3b: full distill vs baseline (500 steps) ---"
python tools/train_distilled_candidate.py \
  --candidate "$CANDIDATE" \
  --teacher-cache "$BASELINE_CACHE" \
  --run-name distill-aggressive-vs-baseline \
  --push-to-hub sartajbhuvaji/bonsai-distilled-aggressive-vs-baseline \
  "${HF_ARGS[@]}"
BASELINE_DISTILL_RUN=$(ls -dt runs/*-distill-aggressive-vs-baseline | head -1)
echo "BASELINE_DISTILL_RUN=${BASELINE_DISTILL_RUN}" | tee -a /root/phase4_state.env

echo "--- Step 3c: full distill vs lora-corrected (500 steps) ---"
python tools/train_distilled_candidate.py \
  --candidate "$CANDIDATE" \
  --teacher-cache "$LORA_CACHE" \
  --run-name distill-aggressive-vs-corrected \
  --push-to-hub sartajbhuvaji/bonsai-distilled-aggressive-vs-corrected \
  "${HF_ARGS[@]}"
CORRECTED_DISTILL_RUN=$(ls -dt runs/*-distill-aggressive-vs-corrected | head -1)
echo "CORRECTED_DISTILL_RUN=${CORRECTED_DISTILL_RUN}" | tee -a /root/phase4_state.env

echo "--- Final metrics ---"
for run in "$BASELINE_DISTILL_RUN" "$CORRECTED_DISTILL_RUN"; do
  echo "== $run =="
  tail -3 "$run/metrics.jsonl"
done

echo "=== Phase 4 ablation pipeline done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
