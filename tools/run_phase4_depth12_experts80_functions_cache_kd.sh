#!/usr/bin/env bash
# Subtasks 1–2 only: function-mode teacher cache (12k) + full-param depth12/experts80 KD.
# No HumanEval — subtasks 3–4 on a fresh GPU tomorrow.
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /root/bonsai
source /root/venv/bin/activate
set -a
# shellcheck disable=SC1091
source /root/bonsai/.env
set +a
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"

LOG="${LOG:-/root/phase4_depth12_experts80_functions_cache_kd.log}"
exec > >(tee -a "$LOG") 2>&1

CACHE_HUB="${CACHE_HUB:-sartajbhuvaji/bonsai-teacher-logits-baseline-functions}"
SAMPLES_HUB="${SAMPLES_HUB:-sartajbhuvaji/bonsai-kd-function-samples-baseline-12k}"
CANDIDATE_HUB="${CANDIDATE_HUB:-sartajbhuvaji/bonsai-candidate-depth12-experts80}"
DISTILL_HUB="${DISTILL_HUB:-sartajbhuvaji/bonsai-distilled-depth12-experts80-functions-vs-baseline-3k-warmup450-lr1e4}"
DISTILL_HUB_BEST="${DISTILL_HUB_BEST:-${DISTILL_HUB}-best}"
RUN_SLUG="${RUN_SLUG:-distill-depth12-experts80-functions-vs-baseline-3000-warmup450-lr1e4}"
STEPS="${STEPS:-3000}"
WARMUP="${WARMUP:-450}"
PEAK_LR="${PEAK_LR:-1e-4}"
NUM_SAMPLES="${NUM_SAMPLES:-12000}"

HF_ARGS=(--hf-token "$HF_TOKEN")
TRAIN_HF_ARGS=("${HF_ARGS[@]}" --hub-public)

echo "=== Phase 4 depth12/experts80 function-mode cache + KD start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "CACHE_HUB=${CACHE_HUB} (new repo; do not overwrite raw-file baseline cache)"
echo "SAMPLES_HUB=${SAMPLES_HUB} (function text dataset, separate from logits .pt)"
echo "DISTILL_HUB=${DISTILL_HUB}"
echo "DISTILL_HUB_BEST=${DISTILL_HUB_BEST}"

echo "--- Subtask 1: cache teacher logits (function mode, n=${NUM_SAMPLES}) ---"
CACHE_WALL_START=$(date +%s)
python tools/cache_teacher_logits.py \
  --teacher baseline \
  --sample-mode function \
  --num-samples "$NUM_SAMPLES" \
  --push-to-hub "$CACHE_HUB" \
  "${HF_ARGS[@]}"
CACHE_WALL_END=$(date +%s)
CACHE_WALL_SEC=$((CACHE_WALL_END - CACHE_WALL_START))
echo "CACHE_WALL_CLOCK_SEC=${CACHE_WALL_SEC} ($(printf '%dh %dm %ds' $((CACHE_WALL_SEC/3600)) $((CACHE_WALL_SEC%3600/60)) $((CACHE_WALL_SEC%60))))"

CACHE_RUN=$(ls -dt runs/*cache-teacher-logits-baseline* 2>/dev/null | head -1 || true)
echo "CACHE_RUN=${CACHE_RUN}"
if [ -n "${CACHE_RUN:-}" ] && [ -f "${CACHE_RUN}/config.json" ]; then
  cat "${CACHE_RUN}/config.json"
fi

python - <<'PY' "$HF_TOKEN" "$CACHE_HUB"
import sys
from huggingface_hub import HfApi
token, repo = sys.argv[1], sys.argv[2]
api = HfApi(token=token)
api.update_repo_settings(repo, private=False, token=token, repo_type="dataset")
print(f"dataset {repo} set public")
PY

CACHE_PT=$(find "${CACHE_RUN}/checkpoints" -name '*-topk-logits.pt' | head -1)
echo "--- Subtask 1b: push function sample texts to ${SAMPLES_HUB} ---"
SAMPLES_WALL_START=$(date +%s)
python tools/push_function_samples_to_hub.py \
  --from-cache-pt "$CACHE_PT" \
  --push-to-hub "$SAMPLES_HUB" \
  --hub-public \
  --write-jsonl "${CACHE_RUN}/checkpoints/function-samples-hub.jsonl" \
  "${HF_ARGS[@]}"
SAMPLES_WALL_END=$(date +%s)
echo "SAMPLES_WALL_CLOCK_SEC=$((SAMPLES_WALL_END - SAMPLES_WALL_START))"

echo "--- Subtask 2: full-parameter KD vs function cache (${STEPS} steps, warmup ${WARMUP}, peak_lr ${PEAK_LR}) ---"
KD_WALL_START=$(date +%s)
python tools/train_distilled_candidate.py \
  --candidate "$CANDIDATE_HUB" \
  --teacher-cache "$CACHE_HUB" \
  --run-name "$RUN_SLUG" \
  --steps "$STEPS" \
  --warmup-steps "$WARMUP" \
  --peak-lr "$PEAK_LR" \
  --push-to-hub "$DISTILL_HUB" \
  --push-to-hub-best "$DISTILL_HUB_BEST" \
  "${TRAIN_HF_ARGS[@]}"
KD_WALL_END=$(date +%s)
KD_WALL_SEC=$((KD_WALL_END - KD_WALL_START))
echo "KD_WALL_CLOCK_SEC=${KD_WALL_SEC}"

DISTILL_RUN=$(ls -dt runs/*"${RUN_SLUG}"* | head -1)
echo "DISTILL_RUN=${DISTILL_RUN}"
cat "${DISTILL_RUN}/best_checkpoint.json" 2>/dev/null || true

echo "=== done $(date -u +%Y-%m-%dT%H:%M:%SZ) (no eval; subtasks 3–4 tomorrow) ==="
