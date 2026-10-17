#!/usr/bin/env bash
# Full HumanEval-TS (no --limit) for four Phase 4 checkpoints. Run on GPU box as root.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "${VENV:-/root/venv}/bin/activate"
set -a
source .env
set +a
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

if ! command -v tsc >/dev/null 2>&1; then
  echo "tsc required for multiple-ts" >&2
  exit 1
fi

run_one() {
  local prefix="$1"
  local model="$2"
  local slug="$3"
  local log="/root/${prefix}_humaneval_ts_full_pipeline.log"
  : >"$log"
  echo "========== ${prefix} ==========" | tee -a "$log"
  echo "model=${model}" | tee -a "$log"
  local run_dir
  run_dir="$(python -c "from bonsai.tracking import new_run; print(new_run('${slug}'))")"
  echo "RUN_DIR=${run_dir}" | tee -a "$log"
  stdbuf -o0 -e0 accelerate launch tools/bigcode_eval_main.py \
    --model "$model" \
    --precision bf16 \
    --use_auth_token \
    --tasks multiple-ts \
    --max_length_generation 512 \
    --batch_size 8 \
    --allow_code_execution \
    --save_generations \
    --metric_output_path "${run_dir}/humaneval-ts.json" \
    --save_generations_path "${run_dir}/humaneval-ts-generations.json" \
    2>&1 | tee -a "$log"
  python -c "
import json
from pathlib import Path
p = Path('${run_dir}/humaneval-ts.json')
d = json.loads(p.read_text())
assert d['config']['limit'] is None, d['config']['limit']
print('pass@1=', d['multiple-ts']['pass@1'])
" | tee -a "$log"
  echo "METRICS_SRC=${run_dir}/humaneval-ts.json" | tee -a "$log"
  echo "GENS_SRC=${run_dir}/humaneval-ts-generations_multiple-ts.json" | tee -a "$log"
  echo "========== ${prefix} done ==========" | tee -a "$log"
}

run_one \
  phase4_aggressive_depth_experts_fixedloss_short \
  sartajbhuvaji/bonsai-distilled-aggressive-depth-experts-500-warmup75-lr1e4-fixedloss \
  distill-aggressive-depth-experts-fixedloss-short-humaneval-ts-full

run_one \
  phase4_depth0_experts97_fixedloss_short \
  sartajbhuvaji/bonsai-distilled-depth0-experts97-500-warmup75-lr1e4-fixedloss \
  distill-depth0-experts97-fixedloss-short-humaneval-ts-full

run_one \
  phase4_depth12_experts80_rawfile_fixedloss \
  sartajbhuvaji/bonsai-distilled-depth12-experts80-rawfile-vs-baseline-3k-warmup450-lr1e4-fixedloss-best \
  distill-depth12-experts80-rawfile-fixedloss-humaneval-ts-full-best

run_one \
  phase4_depth22_experts92_fixedloss_short \
  sartajbhuvaji/bonsai-distilled-depth22-experts92-500-warmup75-lr1e4-fixedloss \
  distill-depth22-experts92-fixedloss-short-humaneval-ts-full

echo "=== all four full-suite evals done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
