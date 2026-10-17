#!/usr/bin/env bash
# Experts-only axis fill (depth_n=0): build → fixed-loss short KD → full HumanEval-TS.
# Order: e112 → e64 → e32. B300 /root/bonsai.
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /root/bonsai
source /root/venv/bin/activate
set -a
# shellcheck disable=SC1091
source /root/bonsai/.env
set +a
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"

MASTER_LOG="${MASTER_LOG:-/root/phase4_pure_axis_fill_pipeline.log}"
: >"$MASTER_LOG"
exec > >(tee -a "$MASTER_LOG") 2>&1

CACHE_HUB="${CACHE_HUB:-sartajbhuvaji/bonsai-teacher-logits-baseline}"
STEPS="${STEPS:-500}"
WARMUP="${WARMUP:-75}"
PEAK_LR="${PEAK_LR:-1e-4}"

humaneval_full() {
  local model="$1"
  local slug="$2"
  local log="$3"
  echo "--- full HumanEval-TS (no --limit) model=${model} ---" | tee -a "$log"
  local run_dir
  run_dir="$(python -c "from bonsai.tracking import new_run; print(new_run('${slug}'))")"
  echo "HE_RUN_DIR=${run_dir}" | tee -a "$log"
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
assert d['config']['limit'] is None
print('pass@1=', d['multiple-ts']['pass@1'])
" | tee -a "$log"
  echo "HE_METRICS=${run_dir}/humaneval-ts.json" | tee -a "$log"
  echo "HE_GENS=${run_dir}/humaneval-ts-generations_multiple-ts.json" | tee -a "$log"
}

run_candidate() {
  local LABEL="$1"
  local DEPTH_N="$2"
  local EXPERT_N="$3"
  local ARCHIVE_PREFIX="$4"
  local CANDIDATE_HUB="$5"
  local DISTILL_HUB="$6"
  local RUN_SLUG="$7"
  local HE_SLUG="$8"

  local LOG="${ARCHIVE_PREFIX}_pipeline.log"

  if [ -f "${ARCHIVE_PREFIX}_run_summary.json" ]; then
    echo "SKIP ${LABEL} (run_summary exists)" | tee -a "$MASTER_LOG"
    return 0
  fi

  if [ -f "${ARCHIVE_PREFIX}_build_summary.json" ]; then
    echo "Resume ${LABEL}: build already archived, starting at KD" | tee -a "$LOG" | tee -a "$MASTER_LOG"
  else
    : >"$LOG"
    echo ""
    echo "========== ${LABEL} BUILD start $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
    local BUILD_WALL_START BUILD_WALL_END BUILD_WALL
    BUILD_WALL_START=$(date +%s)
    python tools/build_candidate.py \
      --depth-n "$DEPTH_N" \
      --expert-n "$EXPERT_N" \
      --push-to-hub "$CANDIDATE_HUB" \
      --hub-public \
      --hf-token "$HF_TOKEN"
    BUILD_WALL_END=$(date +%s)
    BUILD_WALL=$((BUILD_WALL_END - BUILD_WALL_START))
    echo "BUILD_WALL_CLOCK_SEC=${BUILD_WALL}" | tee -a "$LOG"

    local BUILD_RUN
    BUILD_RUN=$(ls -dt runs/*"build-candidate-depth${DEPTH_N}-experts${EXPERT_N}"* | head -1)
    echo "BUILD_RUN=${BUILD_RUN}" | tee -a "$LOG"
    cp "${BUILD_RUN}/config.json" "${ARCHIVE_PREFIX}_build_config.json"
    python3 - <<PY | tee -a "$LOG"
import json
from pathlib import Path
build = Path("$BUILD_RUN")
rows = [json.loads(l) for l in (build / "metrics.jsonl").read_text().splitlines() if l.strip()]
zs = rows[-1].get("zero_shot_loss") if rows else None
out = {
    "build_run": str(build),
    "depth_n": int("$DEPTH_N"),
    "expert_n": int("$EXPERT_N"),
    "candidate_hub": "$CANDIDATE_HUB",
    "build_wall_clock_sec": int("$BUILD_WALL"),
    "zero_shot_loss": zs,
    "metrics": rows,
}
Path("${ARCHIVE_PREFIX}_build_summary.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
PY
  fi

  echo "========== ${LABEL} KD start ==========" | tee -a "$LOG"
  local DISTILL_HUB_BEST="${DISTILL_HUB}-best"
  local KD_WALL_START KD_WALL_END KD_WALL
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
    --hub-public \
    --hf-token "$HF_TOKEN" 2>&1 | tee -a "$LOG"
  KD_WALL_END=$(date +%s)
  KD_WALL=$((KD_WALL_END - KD_WALL_START))
  echo "KD_WALL_CLOCK_SEC=${KD_WALL}" | tee -a "$LOG"

  local DISTILL_RUN BEST_JSON
  DISTILL_RUN=$(ls -dt runs/*"${RUN_SLUG}"* | head -1)
  BEST_JSON="${DISTILL_RUN}/best_checkpoint.json"
  cp "${DISTILL_RUN}/metrics.jsonl" "${ARCHIVE_PREFIX}_kd_metrics.jsonl"
  cp "$BEST_JSON" "${ARCHIVE_PREFIX}_best_checkpoint.json"

  local BEST_STEP FINAL_STEP EVAL_MODEL
  read -r BEST_STEP FINAL_STEP <<<"$(python3 - <<PY
import json
from pathlib import Path
best = json.loads(Path("$BEST_JSON").read_text())
vals = []
last = -1
for line in Path("$DISTILL_RUN/metrics.jsonl").read_text().splitlines():
    row = json.loads(line)
    if "step" in row:
        last = row["step"]
    if "val_loss" in row:
        vals.append((row["step"], row["val_loss"]))
final = vals[-1][0] if vals else last
print(best["best_step"], final)
PY
)"
  if [ "$BEST_STEP" = "$FINAL_STEP" ]; then
    EVAL_MODEL="$DISTILL_HUB"
    HE_SLUG="${HE_SLUG}-final"
  else
    EVAL_MODEL="$DISTILL_HUB_BEST"
    HE_SLUG="${HE_SLUG}-best"
  fi
  echo "EVAL_MODEL=${EVAL_MODEL} BEST_STEP=${BEST_STEP} FINAL_STEP=${FINAL_STEP}" | tee -a "$LOG"

  humaneval_full "$EVAL_MODEL" "$HE_SLUG" "$LOG"

  python3 - <<PY | tee -a "$LOG"
import json
from pathlib import Path

def pass_at_1(metrics_path: str) -> float:
    return json.loads(Path(metrics_path).read_text())["multiple-ts"]["pass@1"]

he_run = None
for line in Path("$LOG").read_text().splitlines():
    if line.startswith("HE_METRICS="):
        he_run = line.split("=", 1)[1]
        break
assert he_run
he_pass = pass_at_1(he_run)
best = json.loads(Path("$BEST_JSON").read_text())
build = json.loads(Path("${ARCHIVE_PREFIX}_build_summary.json").read_text())
vals = []
for line in Path("$DISTILL_RUN/metrics.jsonl").read_text().splitlines():
    row = json.loads(line)
    if "val_loss" in row:
        vals.append((row["step"], row["val_loss"]))
final_step, final_val = vals[-1]

def human(sec: int) -> str:
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m}m {s}s"

summary = {
    "label": "$LABEL",
    "depth_n": int("$DEPTH_N"),
    "expert_n": int("$EXPERT_N"),
    "build_wall_clock_sec": build.get("build_wall_clock_sec"),
    "zero_shot_loss": build.get("zero_shot_loss"),
    "build_run": build.get("build_run"),
    "kd_wall_clock_sec": int("$KD_WALL"),
    "kd_wall_clock_human": human(int("$KD_WALL")),
    "steps": int("$STEPS"),
    "warmup_steps": int("$WARMUP"),
    "peak_lr": float("$PEAK_LR"),
    "teacher_cache_hub": "$CACHE_HUB",
    "candidate_hub": "$CANDIDATE_HUB",
    "distill_hub": "$DISTILL_HUB",
    "eval_hub": "$EVAL_MODEL",
    "val_best_step": best["best_step"],
    "val_best": best["best_val_loss"],
    "val_final_step": final_step,
    "val_final": final_val,
    "humaneval_full_pass_at_1": he_pass,
    "humaneval_full_metrics": he_run,
    "distill_run": "$DISTILL_RUN",
}
Path("${ARCHIVE_PREFIX}_run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
Path("${ARCHIVE_PREFIX}_humaneval_ts_full.json").write_text(Path(he_run).read_text())
gens = Path(he_run).parent / "humaneval-ts-generations_multiple-ts.json"
if gens.exists():
    Path("${ARCHIVE_PREFIX}_humaneval_generations_full.json").write_text(gens.read_text())
print(json.dumps(summary, indent=2))
print("humaneval_full_pass@1:", he_pass)
PY

  cp "$LOG" "${ARCHIVE_PREFIX}_humaneval_ts_full_pipeline.log"
  echo "========== ${LABEL} done $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
}

echo "=== experts-only fill pipeline start $(date -u +%Y-%m-%dT%H:%M:%SZ) GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="

run_candidate "depth0/experts112" 0 112 \
  /root/phase4_pure_axis_depth0_experts112_fixedloss_short \
  sartajbhuvaji/bonsai-candidate-depth0-experts112 \
  sartajbhuvaji/bonsai-distilled-depth0-experts112-500-warmup75-lr1e4-fixedloss \
  distill-depth0-experts112-vs-baseline-500-warmup75-lr1e4-fixedloss \
  distill-depth0-experts112-fixedloss-short-humaneval-ts-full

run_candidate "depth0/experts64" 0 64 \
  /root/phase4_pure_axis_depth0_experts64_fixedloss_short \
  sartajbhuvaji/bonsai-candidate-depth0-experts64 \
  sartajbhuvaji/bonsai-distilled-depth0-experts64-500-warmup75-lr1e4-fixedloss \
  distill-depth0-experts64-vs-baseline-500-warmup75-lr1e4-fixedloss \
  distill-depth0-experts64-fixedloss-short-humaneval-ts-full

run_candidate "depth0/experts32" 0 32 \
  /root/phase4_pure_axis_depth0_experts32_fixedloss_short \
  sartajbhuvaji/bonsai-candidate-depth0-experts32 \
  sartajbhuvaji/bonsai-distilled-depth0-experts32-500-warmup75-lr1e4-fixedloss \
  distill-depth0-experts32-vs-baseline-500-warmup75-lr1e4-fixedloss \
  distill-depth0-experts32-fixedloss-short-humaneval-ts-full

echo "=== experts-only fill all done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
