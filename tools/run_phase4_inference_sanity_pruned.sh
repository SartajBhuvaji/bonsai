#!/usr/bin/env bash
# Step 1: load each pruned checkpoint in vLLM and SGLang (single prompt).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "${VENV:-/root/venv}/bin/activate"
GPU_MEM="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
LOG="$ROOT/docs/debug/phase4_inference_sanity_pruned.log"
: >"$LOG"

MODELS=(
  "depth0_experts64|sartajbhuvaji/bonsai-distilled-depth0-experts64-500-warmup75-lr1e4-fixedloss-best"
  "depth16_experts0|sartajbhuvaji/bonsai-distilled-depth16-experts0-500-warmup75-lr1e4-fixedloss-best"
  "depth0_experts8|sartajbhuvaji/bonsai-distilled-depth0-experts8-500-warmup75-lr1e4-fixedloss-best"
)

export VLLM_MAX_MODEL_LEN="$MAX_MODEL_LEN"

echo "========== vLLM dependency pins ==========" | tee -a "$LOG"
bash tools/setup_inference_engine_deps.sh vllm 2>&1 | tee -a "$LOG"

for entry in "${MODELS[@]}"; do
  slug="${entry%%|*}"
  model="${entry#*|}"
  echo "========== ${slug} / vllm ==========" | tee -a "$LOG"
  python tools/phase4_inference_sanity_check.py \
    --model "$model" \
    --engine vllm \
    --gpu-memory-utilization "$GPU_MEM" \
    2>&1 | tee -a "$LOG"
done

echo "========== SGLang dependency pins ==========" | tee -a "$LOG"
bash tools/setup_inference_engine_deps.sh sglang 2>&1 | tee -a "$LOG"

for entry in "${MODELS[@]}"; do
  slug="${entry%%|*}"
  model="${entry#*|}"
  echo "========== ${slug} / sglang ==========" | tee -a "$LOG"
  python tools/phase4_inference_sanity_check.py \
    --model "$model" \
    --engine sglang \
    --gpu-memory-utilization "$GPU_MEM" \
    2>&1 | tee -a "$LOG"
done

echo "All pruned sanity checks passed." | tee -a "$LOG"
