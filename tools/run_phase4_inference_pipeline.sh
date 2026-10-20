#!/usr/bin/env bash
# Step 1 sanity (pruned only), then full sweep if sanity passes.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
VENV="${VENV:-/root/venv}"
source "${VENV}/bin/activate"

LOG="$ROOT/docs/debug/phase4_inference_pipeline.log"
: >"$LOG"

echo "[$(date -Iseconds)] vLLM deps" | tee -a "$LOG"
bash tools/setup_inference_engine_deps.sh vllm 2>&1 | tee -a "$LOG"

if [[ "${SKIP_SANITY:-0}" != "1" ]]; then
  echo "[$(date -Iseconds)] sanity (pruned)" | tee -a "$LOG"
  bash tools/run_phase4_inference_sanity_pruned.sh 2>&1 | tee -a "$LOG"
else
  echo "[$(date -Iseconds)] sanity skipped (SKIP_SANITY=1)" | tee -a "$LOG"
fi

if [[ "${VLLM_ONLY:-0}" == "1" ]]; then
  echo "[$(date -Iseconds)] vLLM sweep only (VLLM_ONLY=1)" | tee -a "$LOG"
  VLLM_ONLY=1 bash tools/run_phase4_inference_benchmark_sweep.sh 2>&1 | tee -a "$LOG"
elif [[ "${SGLANG_ONLY:-0}" == "1" ]]; then
  echo "[$(date -Iseconds)] SGLang sweep only (SGLANG_ONLY=1)" | tee -a "$LOG"
  SGLANG_ONLY=1 bash tools/run_phase4_inference_benchmark_sweep.sh 2>&1 | tee -a "$LOG"
else
  echo "[$(date -Iseconds)] full sweep" | tee -a "$LOG"
  bash tools/run_phase4_inference_benchmark_sweep.sh 2>&1 | tee -a "$LOG"
fi

echo "[$(date -Iseconds)] pipeline done" | tee -a "$LOG"
