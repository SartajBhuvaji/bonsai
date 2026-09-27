#!/usr/bin/env bash
# HumanEval-TS pass@1 on the first N problems (default 40), project-standard gen settings.
# Usage: HF_TOKEN=... ./tools/run_humaneval_ts_sample.sh <hf-model-id> <run-name-slug> [limit]

set -euo pipefail

MODEL="${1:?model repo id required}"
RUN_SLUG="${2:?run name slug required}"
LIMIT="${3:-40}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN must be set (private distilled checkpoints)." >&2
  exit 1
fi

export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

source "${VENV:-/root/venv}/bin/activate"

if ! command -v tsc >/dev/null 2>&1; then
  echo "tsc not found — install TypeScript before HumanEval-TS (npm install -g typescript)." >&2
  exit 1
fi

RUN_DIR="$(python -c "from bonsai.tracking import new_run; print(new_run('${RUN_SLUG}'))")"
echo "model=${MODEL}"
echo "run_dir=${RUN_DIR}"
echo "limit=${LIMIT}"

stdbuf -o0 -e0 accelerate launch tools/bigcode_eval_main.py \
  --model "$MODEL" \
  --precision bf16 \
  --use_auth_token \
  --tasks multiple-ts \
  --limit "$LIMIT" \
  --limit_start 0 \
  --max_length_generation 512 \
  --batch_size 8 \
  --allow_code_execution \
  --save_generations \
  --metric_output_path "${RUN_DIR}/humaneval-ts.json" \
  --save_generations_path "${RUN_DIR}/humaneval-ts-generations.json"

python -c "
import json
from pathlib import Path
p = Path('${RUN_DIR}/humaneval-ts.json')
data = json.loads(p.read_text())
pass1 = data['multiple-ts']['pass@1']
print(f'pass@1={pass1}')
"
