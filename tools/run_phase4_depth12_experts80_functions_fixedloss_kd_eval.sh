#!/usr/bin/env bash
# Fixed KD loss (PR #71): retrain on existing function-mode teacher cache + HumanEval n=40.
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /root/bonsai
source /root/venv/bin/activate
set -a
# shellcheck disable=SC1091
source /root/bonsai/.env
set +a
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"

LOG="${LOG:-/root/phase4_depth12_experts80_functions_fixedloss_pipeline.log}"
exec > >(tee -a "$LOG") 2>&1

CANDIDATE_HUB="${CANDIDATE_HUB:-sartajbhuvaji/bonsai-candidate-depth12-experts80}"
CACHE_HUB="${CACHE_HUB:-sartajbhuvaji/bonsai-teacher-logits-baseline-functions}"
DISTILL_HUB="${DISTILL_HUB:-sartajbhuvaji/bonsai-distilled-depth12-experts80-functions-vs-baseline-3k-warmup450-lr1e4-fixedloss}"
DISTILL_HUB_BEST="${DISTILL_HUB_BEST:-${DISTILL_HUB}-best}"
RUN_SLUG="${RUN_SLUG:-distill-depth12-experts80-functions-vs-baseline-3000-warmup450-lr1e4-fixedloss}"

echo "=== fixed-loss KD + eval start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "CACHE_HUB=${CACHE_HUB} (reuse; no re-cache)"
echo "DISTILL_HUB=${DISTILL_HUB}"

KD_WALL_START=$(date +%s)
python tools/train_distilled_candidate.py \
  --candidate "$CANDIDATE_HUB" \
  --teacher-cache "$CACHE_HUB" \
  --run-name "$RUN_SLUG" \
  --steps 3000 \
  --warmup-steps 450 \
  --peak-lr 1e-4 \
  --push-to-hub "$DISTILL_HUB" \
  --push-to-hub-best "$DISTILL_HUB_BEST" \
  --hub-public \
  --hf-token "$HF_TOKEN"
KD_WALL_END=$(date +%s)
echo "KD_WALL_CLOCK_SEC=$((KD_WALL_END - KD_WALL_START))"

DISTILL_RUN=$(ls -dt runs/*"${RUN_SLUG}"* | head -1)
echo "DISTILL_RUN=${DISTILL_RUN}"
BEST_JSON="${DISTILL_RUN}/best_checkpoint.json"
cat "$BEST_JSON"

BEST_STEP=$(python3 -c "import json; print(json.load(open('$BEST_JSON'))['best_step'])")
FINAL_STEP=2999
echo "BEST_STEP=${BEST_STEP} FINAL_STEP=${FINAL_STEP}"

if [ "$BEST_STEP" = "$FINAL_STEP" ]; then
  echo "--- HumanEval n=40 on ${DISTILL_HUB} (best == final @ step ${BEST_STEP}) ---"
  ./tools/run_humaneval_ts_sample.sh \
    "$DISTILL_HUB" \
    distill-depth12-experts80-functions-fixedloss-humaneval-ts-n40 \
    40
  HE_FINAL=$(ls -dt runs/*fixedloss-humaneval-ts-n40* | head -1)
  HE_BEST=""
else
  echo "--- HumanEval n=40 FINAL on ${DISTILL_HUB} ---"
  ./tools/run_humaneval_ts_sample.sh \
    "$DISTILL_HUB" \
    distill-depth12-experts80-functions-fixedloss-humaneval-ts-n40-final \
    40
  HE_FINAL=$(ls -dt runs/*fixedloss-humaneval-ts-n40-final* | head -1)
  echo "--- HumanEval n=40 BEST on ${DISTILL_HUB_BEST} ---"
  ./tools/run_humaneval_ts_sample.sh \
    "$DISTILL_HUB_BEST" \
    distill-depth12-experts80-functions-fixedloss-humaneval-ts-n40-best \
    40
  HE_BEST=$(ls -dt runs/*fixedloss-humaneval-ts-n40-best* | head -1)
fi

python3 - <<PY
import json
from pathlib import Path
for label, run, hub in [
    ("final", "$HE_FINAL", "$DISTILL_HUB"),
] + ([("best", "$HE_BEST", "$DISTILL_HUB_BEST")] if "$HE_BEST" else []):
    p = Path(run) / "humaneval-ts.json"
    m = json.loads(p.read_text())
    print(f"humaneval-ts pass@1 ({label}, {hub}):", m["multiple-ts"]["pass@1"])
    print(f"  run_dir: {run}")
PY

echo "=== done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
