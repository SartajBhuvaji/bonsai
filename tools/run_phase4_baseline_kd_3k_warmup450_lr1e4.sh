#!/usr/bin/env bash
# Baseline teacher 3k KD: warmup450, peak_lr=1e-4; HumanEval final + best-val checkpoints.
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

LOG="${LOG:-/root/phase4_baseline_kd_3k_warmup450_lr1e4.log}"
exec > >(tee -a "$LOG") 2>&1

echo "=== Phase 4 baseline KD 3k warmup450 lr1e-4 start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

CANDIDATE="${CANDIDATE:-sartajbhuvaji/bonsai-candidate-aggressive-depth-experts}"
TEACHER_CACHE="${TEACHER_CACHE:-sartajbhuvaji/bonsai-teacher-logits-baseline}"
DISTILL_HUB="${DISTILL_HUB:-sartajbhuvaji/bonsai-distilled-aggressive-vs-baseline-3k-warmup450-lr1e4}"
DISTILL_HUB_BEST="${DISTILL_HUB_BEST:-${DISTILL_HUB}-best}"
STEPS="${STEPS:-3000}"
WARMUP="${WARMUP:-450}"
PEAK_LR="${PEAK_LR:-1e-4}"
RUN_SLUG="${RUN_SLUG:-distill-aggressive-vs-baseline-3000-warmup450-lr1e4}"

echo "CANDIDATE=${CANDIDATE}"
echo "TEACHER_CACHE=${TEACHER_CACHE}"
echo "STEPS=${STEPS} WARMUP=${WARMUP} peak_lr=${PEAK_LR}"
echo "DISTILL_HUB=${DISTILL_HUB}"
echo "DISTILL_HUB_BEST=${DISTILL_HUB_BEST}"

echo "--- distill vs baseline teacher ---"
python tools/train_distilled_candidate.py \
  --candidate "$CANDIDATE" \
  --teacher-cache "$TEACHER_CACHE" \
  --run-name "$RUN_SLUG" \
  --steps "$STEPS" \
  --warmup-steps "$WARMUP" \
  --peak-lr "$PEAK_LR" \
  --push-to-hub "$DISTILL_HUB" \
  --push-to-hub-best "$DISTILL_HUB_BEST" \
  "${HF_ARGS[@]}"

DISTILL_RUN=$(ls -dt runs/*"${RUN_SLUG}"* | head -1)
echo "DISTILL_RUN=${DISTILL_RUN}"
cat "${DISTILL_RUN}/best_checkpoint.json" 2>/dev/null || true

echo "--- HumanEval-TS n=40 FINAL on ${DISTILL_HUB} ---"
./tools/run_humaneval_ts_sample.sh "$DISTILL_HUB" "distill-vs-baseline-humaneval-ts-n40-3k-warmup450-lr1e4-final" 40
HE_FINAL=$(ls -dt runs/*humaneval*n40*lr1e4-final* | head -1)

echo "--- HumanEval-TS n=40 BEST-VAL on ${DISTILL_HUB_BEST} ---"
./tools/run_humaneval_ts_sample.sh "$DISTILL_HUB_BEST" "distill-vs-baseline-humaneval-ts-n40-3k-warmup450-lr1e4-best" 40
HE_BEST=$(ls -td runs/*humaneval*n40*lr1e4-best* | head -1)

python - <<PY
import json
from pathlib import Path
for label, run, hub in [
    ("final", "$HE_FINAL", "$DISTILL_HUB"),
    ("best", "$HE_BEST", "$DISTILL_HUB_BEST"),
]:
    p = Path(run) / "humaneval-ts.json"
    m = json.loads(p.read_text())
    print(f"humaneval-ts pass@1 ({label}, {hub}):", m["multiple-ts"]["pass@1"])
    print(f"  run_dir: {run}")
PY

echo "=== done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
