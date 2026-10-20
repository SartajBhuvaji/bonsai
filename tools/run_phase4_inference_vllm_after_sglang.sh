#!/usr/bin/env bash
# Wait for the active SGLang sweep, then re-run vLLM benchmarks (fixed client URL).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "${VENV:-/root/venv}/bin/activate"
LOG="$ROOT/docs/debug/phase4_inference_vllm_rerun.log"

MARKER="$ROOT/docs/debug/phase4_inference_sglang_sweep.done"
echo "[$(date -Iseconds)] waiting for SGLang sweep marker: $MARKER" | tee "$LOG"
while [[ ! -f "$MARKER" ]]; do
  if ! pgrep -f "[b]ash tools/run_phase4_inference_benchmark_sweep.sh" >/dev/null; then
    if ls "$ROOT/docs/debug/phase4_inference_benchmark_"*_sglang_c32.json >/dev/null 2>&1; then
      touch "$MARKER"
      break
    fi
  fi
  sleep 30
done
echo "[$(date -Iseconds)] starting VLLM_ONLY rerun" | tee -a "$LOG"
SKIP_SANITY=1 VLLM_ONLY=1 bash tools/run_phase4_inference_pipeline.sh 2>&1 | tee -a "$LOG"
echo "[$(date -Iseconds)] VLLM_ONLY rerun done" | tee -a "$LOG"
