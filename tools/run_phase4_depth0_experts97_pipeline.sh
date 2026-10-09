#!/usr/bin/env bash
# Experts-only isolation (depth_n=0, expert_n=97): build candidate, baseline-teacher KD
# (warmup450 + peak_lr=1e-4 + 3000 steps), HumanEval-TS n=40 on final + best-val.
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /root/bonsai
source /root/venv/bin/activate
set -a
# shellcheck disable=SC1091
source /root/bonsai/.env
set +a
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"
HUB_PUBLIC="${HUB_PUBLIC:-1}"
HF_ARGS=(--hf-token "$HF_TOKEN")
if [ "$HUB_PUBLIC" = "1" ]; then
  HF_ARGS+=(--hub-public)
fi

LOG="${LOG:-/root/phase4_depth0_experts97_pipeline.log}"
exec > >(tee -a "$LOG") 2>&1

CANDIDATE_HUB="${CANDIDATE_HUB:-sartajbhuvaji/bonsai-candidate-depth0-experts97}"
TEACHER_CACHE="${TEACHER_CACHE:-sartajbhuvaji/bonsai-teacher-logits-baseline}"
DISTILL_HUB="${DISTILL_HUB:-sartajbhuvaji/bonsai-distilled-depth0-experts97-vs-baseline-3k-warmup450-lr1e4}"
DISTILL_HUB_BEST="${DISTILL_HUB_BEST:-${DISTILL_HUB}-best}"
STEPS="${STEPS:-3000}"
WARMUP="${WARMUP:-450}"
PEAK_LR="${PEAK_LR:-1e-4}"
RUN_SLUG="${RUN_SLUG:-distill-depth0-experts97-vs-baseline-3000-warmup450-lr1e4}"

echo "=== Phase 4 depth0/experts97 isolation pipeline start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "CANDIDATE_HUB=${CANDIDATE_HUB}"
echo "TEACHER_CACHE=${TEACHER_CACHE}"
echo "DISTILL_HUB=${DISTILL_HUB}"
echo "DISTILL_HUB_BEST=${DISTILL_HUB_BEST}"

echo "HUB_PUBLIC=${HUB_PUBLIC} (1 = public Hub uploads)"

echo "--- build candidate (depth_n=0, expert_n=97) ---"
python tools/build_candidate.py \
  --depth-n 0 \
  --expert-n 97 \
  --push-to-hub "$CANDIDATE_HUB" \
  "${HF_ARGS[@]}"

echo "--- distill vs baseline teacher (${STEPS} steps, warmup ${WARMUP}, peak_lr ${PEAK_LR}) ---"
python tools/train_distilled_candidate.py \
  --candidate "$CANDIDATE_HUB" \
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
./tools/run_humaneval_ts_sample.sh "$DISTILL_HUB" "distill-depth0-experts97-humaneval-ts-n40-lr1e4-final" 40
HE_FINAL=$(ls -dt runs/*depth0-experts97*lr1e4-final* | head -1)

echo "--- HumanEval-TS n=40 BEST-VAL on ${DISTILL_HUB_BEST} ---"
./tools/run_humaneval_ts_sample.sh "$DISTILL_HUB_BEST" "distill-depth0-experts97-humaneval-ts-n40-lr1e4-best" 40
HE_BEST=$(ls -dt runs/*depth0-experts97*lr1e4-best* | head -1)

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
