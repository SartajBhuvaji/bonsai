#!/usr/bin/env bash
# Phase 4 ablation tiebreaker: n=40 HumanEval-TS on both distilled students.

set -euo pipefail

LOG="${LOG:-/root/phase4_humaneval_n40.log}"
exec > >(tee -a "$LOG") 2>&1

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "${VENV:-/root/venv}/bin/activate"

if ! command -v tsc >/dev/null 2>&1; then
  echo "tsc not found — HumanEval-TS execution requires TypeScript (npm install -g typescript)." >&2
  exit 1
fi

echo "=== Phase 4 distilled HumanEval-TS n=40 === $(date -u +%Y-%m-%dT%H:%M:%SZ)"

BASELINE="sartajbhuvaji/bonsai-distilled-aggressive-vs-baseline"
CORRECTED="sartajbhuvaji/bonsai-distilled-aggressive-vs-corrected"
BASELINE_RUN="${BASELINE_RUN:-$(ls -td runs/*distill-vs-baseline-humaneval-ts-n40 2>/dev/null | head -1 || true)}"

score_saved_generations() {
  local run_dir="$1"
  local gens="${run_dir}/humaneval-ts-generations_multiple-ts.json"
  if [[ ! -f "$gens" ]]; then
    echo "missing generations: $gens" >&2
    return 1
  fi
  echo "scoring only (load_generations_path) run_dir=${run_dir}"
  stdbuf -o0 -e0 accelerate launch tools/bigcode_eval_main.py \
    --tasks multiple-ts \
    --limit 40 \
    --limit_start 0 \
    --allow_code_execution \
    --load_generations_path "$gens" \
    --metric_output_path "${run_dir}/humaneval-ts.json"
}

echo "--- vs baseline teacher distill ---"
if [[ -n "$BASELINE_RUN" && -f "${BASELINE_RUN}/humaneval-ts-generations_multiple-ts.json" \
  && ! -f "${BASELINE_RUN}/humaneval-ts.json" ]]; then
  score_saved_generations "$BASELINE_RUN"
elif [[ -n "$BASELINE_RUN" && -f "${BASELINE_RUN}/humaneval-ts.json" ]]; then
  echo "baseline metrics already at ${BASELINE_RUN}/humaneval-ts.json — skip"
else
  ./tools/run_humaneval_ts_sample.sh "$BASELINE" "distill-vs-baseline-humaneval-ts-n40" 40
  BASELINE_RUN="$(ls -td runs/*distill-vs-baseline-humaneval-ts-n40 | head -1)"
fi

echo "--- vs LoRA teacher distill ---"
CORRECTED_RUN="$(ls -td runs/*distill-vs-corrected-humaneval-ts-n40 2>/dev/null | head -1 || true)"
if [[ -n "$CORRECTED_RUN" && -f "${CORRECTED_RUN}/humaneval-ts.json" ]]; then
  echo "corrected metrics already at ${CORRECTED_RUN}/humaneval-ts.json — skip"
else
  ./tools/run_humaneval_ts_sample.sh "$CORRECTED" "distill-vs-corrected-humaneval-ts-n40" 40
  CORRECTED_RUN="$(ls -td runs/*distill-vs-corrected-humaneval-ts-n40 | head -1)"
fi

echo "=== Summary ==="
python - <<PY
import json
from pathlib import Path

def load(run_glob):
    run = sorted(Path("runs").glob(run_glob), reverse=True)[0]
    data = json.loads((run / "humaneval-ts.json").read_text())
    return run.name, data["multiple-ts"]["pass@1"]

b_run, b_p1 = load("*distill-vs-baseline-humaneval-ts-n40")
c_run, c_p1 = load("*distill-vs-corrected-humaneval-ts-n40")
print(f"baseline_teacher_distill: {b_run} pass@1={b_p1}")
print(f"corrected_teacher_distill: {c_run} pass@1={c_p1}")
PY

echo "=== done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
