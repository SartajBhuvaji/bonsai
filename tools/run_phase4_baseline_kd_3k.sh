#!/usr/bin/env bash
# Phase 4 incremental step 1: baseline teacher only, 3000-step KD + n=40 HumanEval-TS.
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

LOG="${LOG:-/root/phase4_baseline_kd_3k.log}"
exec > >(tee -a "$LOG") 2>&1

echo "=== Phase 4 baseline KD 3k + HumanEval n=40 start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

STATE=/root/phase4_state.env
if [[ -f "$STATE" ]]; then
  # shellcheck disable=SC1090
  source "$STATE"
fi

CANDIDATE="${CANDIDATE:-runs/20260927T161454Z-e94ad0-build-candidate-aggressive-depth-experts/checkpoints/candidate}"
BASELINE_CACHE="${BASELINE_CACHE:-runs/20260927T163225Z-91fcb3-cache-teacher-logits-baseline/checkpoints/baseline-topk-logits.pt}"
BASELINE_CACHE_HF="${BASELINE_CACHE_HF:-sartajbhuvaji/bonsai-teacher-logits-baseline}"
DISTILL_HUB="${DISTILL_HUB:-sartajbhuvaji/bonsai-distilled-aggressive-vs-baseline-3k}"
STEPS="${STEPS:-3000}"

echo "CANDIDATE=${CANDIDATE}"
echo "BASELINE_CACHE=${BASELINE_CACHE}"
echo "STEPS=${STEPS}"

if [[ ! -f "$BASELINE_CACHE" ]]; then
  echo "--- baseline teacher cache missing; recomputing 6000 samples ---"
  python tools/cache_teacher_logits.py --teacher baseline --num-samples 6000 \
    --push-to-hub "$BASELINE_CACHE_HF" "${HF_ARGS[@]}"
  BASELINE_CACHE_RUN=$(ls -dt runs/*-cache-teacher-logits-baseline | head -1)
  BASELINE_CACHE="${BASELINE_CACHE_RUN}/checkpoints/baseline-topk-logits.pt"
  echo "BASELINE_CACHE=${BASELINE_CACHE}" | tee -a "$STATE"
else
  echo "--- reusing baseline teacher cache: ${BASELINE_CACHE} ---"
fi

echo "--- distill vs baseline teacher (${STEPS} steps) ---"
python tools/train_distilled_candidate.py \
  --candidate "$CANDIDATE" \
  --teacher-cache "$BASELINE_CACHE" \
  --run-name "distill-aggressive-vs-baseline-${STEPS}" \
  --steps "$STEPS" \
  --push-to-hub "$DISTILL_HUB" \
  "${HF_ARGS[@]}"

DISTILL_RUN=$(ls -dt runs/*distill-aggressive-vs-baseline-"${STEPS}" | head -1)
echo "DISTILL_RUN_3K=${DISTILL_RUN}" | tee -a "$STATE"
echo "tail metrics: ${DISTILL_RUN}/metrics.jsonl"

echo "--- HumanEval-TS n=40 on ${DISTILL_HUB} ---"
./tools/run_humaneval_ts_sample.sh "$DISTILL_HUB" "distill-vs-baseline-humaneval-ts-n40-3k" 40

HE_RUN=$(ls -dt runs/*distill-vs-baseline-humaneval-ts-n40-3k | head -1)
python - <<PY
import json
from pathlib import Path
m = json.loads((Path("$HE_RUN") / "humaneval-ts.json").read_text())
print("humaneval-ts pass@1:", m["multiple-ts"]["pass@1"])
print("run_dir:", "$HE_RUN")
PY

echo "=== done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
