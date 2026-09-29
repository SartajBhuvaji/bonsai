#!/usr/bin/env bash
# Phase 4 schedule fix: baseline teacher, 3000-step KD with warmup_steps=450 (15% of 3000).
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

LOG="${LOG:-/root/phase4_baseline_kd_3k_warmup450.log}"
exec > >(tee -a "$LOG") 2>&1

echo "=== Phase 4 baseline KD 3k warmup450 + HumanEval n=40 start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

CANDIDATE="${CANDIDATE:-sartajbhuvaji/bonsai-candidate-aggressive-depth-experts}"
TEACHER_CACHE="${TEACHER_CACHE:-sartajbhuvaji/bonsai-teacher-logits-baseline}"
DISTILL_HUB="${DISTILL_HUB:-sartajbhuvaji/bonsai-distilled-aggressive-vs-baseline-3k-warmup450}"
STEPS="${STEPS:-3000}"
WARMUP="${WARMUP:-450}"
RUN_SLUG="${RUN_SLUG:-distill-aggressive-vs-baseline-3000-warmup450}"
HE_SLUG="${HE_SLUG:-distill-vs-baseline-humaneval-ts-n40-3k-warmup450}"

echo "CANDIDATE=${CANDIDATE}"
echo "TEACHER_CACHE=${TEACHER_CACHE}"
echo "STEPS=${STEPS} WARMUP=${WARMUP} peak_lr=2e-4 (default)"
echo "DISTILL_HUB=${DISTILL_HUB}"

echo "--- distill vs baseline teacher (${STEPS} steps, warmup ${WARMUP}) ---"
python tools/train_distilled_candidate.py \
  --candidate "$CANDIDATE" \
  --teacher-cache "$TEACHER_CACHE" \
  --run-name "$RUN_SLUG" \
  --steps "$STEPS" \
  --warmup-steps "$WARMUP" \
  --push-to-hub "$DISTILL_HUB" \
  "${HF_ARGS[@]}"

DISTILL_RUN=$(ls -dt runs/*"${RUN_SLUG}"* | head -1)
echo "DISTILL_RUN=${DISTILL_RUN}"
echo "tail metrics: ${DISTILL_RUN}/metrics.jsonl"

echo "--- HumanEval-TS n=40 on ${DISTILL_HUB} ---"
./tools/run_humaneval_ts_sample.sh "$DISTILL_HUB" "$HE_SLUG" 40

HE_RUN=$(ls -dt runs/*"${HE_SLUG}"* | head -1)
python - <<PY
import json
from pathlib import Path
m = json.loads((Path("$HE_RUN") / "humaneval-ts.json").read_text())
print("humaneval-ts pass@1:", m["multiple-ts"]["pass@1"])
print("run_dir:", "$HE_RUN")
PY

echo "=== done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
