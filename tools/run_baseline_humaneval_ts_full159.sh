#!/usr/bin/env bash
# Re-run unpruned baseline HumanEval-TS with the same harness as Phase 4 full-suite
# runs (159-problem multiple-ts, batch_size=8, no --limit). Run on GPU as root.
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /root/bonsai
source /root/venv/bin/activate
set -a
# shellcheck disable=SC1091
source /root/bonsai/.env
set +a
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"

if ! command -v tsc >/dev/null 2>&1; then
  echo "tsc required for multiple-ts" >&2
  exit 1
fi

ARCHIVE_PREFIX="${ARCHIVE_PREFIX:-/root/phase4_baseline_humaneval_ts_full159}"
LOG="${ARCHIVE_PREFIX}_pipeline.log"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Base}"

: >"$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "=== baseline HumanEval-TS full159 start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "model=${MODEL}"

run_dir="$(python -c "from bonsai.tracking import new_run; print(new_run('baseline-humaneval-ts-full159'))")"
echo "RUN_DIR=${run_dir}"

stdbuf -o0 -e0 accelerate launch tools/bigcode_eval_main.py \
  --model "$MODEL" \
  --precision bf16 \
  --use_auth_token \
  --tasks multiple-ts \
  --max_length_generation 512 \
  --batch_size 8 \
  --allow_code_execution \
  --save_generations \
  --metric_output_path "${run_dir}/humaneval-ts.json" \
  --save_generations_path "${run_dir}/humaneval-ts-generations.json"

python3 - <<PY
import json
from pathlib import Path

run_dir = Path("${run_dir}")
metrics_path = run_dir / "humaneval-ts.json"
gens_path = run_dir / "humaneval-ts-generations_multiple-ts.json"
d = json.loads(metrics_path.read_text())
assert d["config"]["limit"] is None, d["config"]["limit"]
gens = json.loads(gens_path.read_text())
n_probs = len(gens)
pass1 = d["multiple-ts"]["pass@1"]
prefix = Path("${ARCHIVE_PREFIX}")
prefix.parent.mkdir(parents=True, exist_ok=True)
Path(f"{prefix}_humaneval_ts_full.json").write_text(metrics_path.read_text())
Path(f"{prefix}_humaneval_generations_full.json").write_text(gens_path.read_text())
lines = []
for i in range(min(3, n_probs)):
    t = gens[i][0]
    lines += ["=" * 72, f"problem {i} (len={len(t)})", t]
Path(f"{prefix}_humaneval_generations_full_problems_0-2.txt").write_text(
    "\n".join(lines) + "\n"
)
print("n_problems", n_probs)
print("pass@1", pass1)
print("k_correct_implied", round(pass1 * n_probs))
PY

echo "=== baseline HumanEval-TS full159 done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
