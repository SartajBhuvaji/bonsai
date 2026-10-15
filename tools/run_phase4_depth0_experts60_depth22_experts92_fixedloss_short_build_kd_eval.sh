#!/usr/bin/env bash
# Build depth0/experts60 + depth22/experts92, fixed-loss KD (500/warmup75), HumanEval n=40.
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /root/bonsai
source /root/venv/bin/activate
set -a
# shellcheck disable=SC1091
source /root/bonsai/.env
set +a
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"

LOG="${LOG:-/root/phase4_depth0_experts60_depth22_experts92_fixedloss_short_pipeline.log}"
exec > >(tee -a "$LOG") 2>&1

CACHE_HUB="${CACHE_HUB:-sartajbhuvaji/bonsai-teacher-logits-baseline}"
STEPS="${STEPS:-500}"
WARMUP="${WARMUP:-75}"
PEAK_LR="${PEAK_LR:-1e-4}"

build_arm() {
  local LABEL="$1"
  local ARCHIVE_PREFIX="$2"
  local DEPTH_N="$3"
  local EXPERT_N="$4"
  local CANDIDATE_HUB="$5"

  echo ""
  echo "========== BUILD ${LABEL} start $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
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
  echo "BUILD_WALL_CLOCK_SEC=${BUILD_WALL}"

  local BUILD_RUN
  BUILD_RUN=$(ls -dt runs/*"build-candidate-depth${DEPTH_N}-experts${EXPERT_N}"* | head -1)
  echo "BUILD_RUN=${BUILD_RUN}"
  cp "${BUILD_RUN}/config.json" "${ARCHIVE_PREFIX}_build_config.json"
  python3 - <<PY
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
  echo "========== BUILD ${LABEL} done =========="
}

run_arm() {
  local LABEL="$1"
  local ARCHIVE_PREFIX="$2"
  local CANDIDATE_HUB="$3"
  local DISTILL_HUB="$4"
  local RUN_SLUG="$5"
  local HE_SLUG_PREFIX="$6"

  echo ""
  echo "========== KD ${LABEL} start $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
  echo "CANDIDATE=${CANDIDATE_HUB} CACHE=${CACHE_HUB} (reuse)"
  echo "DISTILL_HUB=${DISTILL_HUB}"
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
    --hf-token "$HF_TOKEN"
  KD_WALL_END=$(date +%s)
  KD_WALL=$((KD_WALL_END - KD_WALL_START))
  echo "KD_WALL_CLOCK_SEC=${KD_WALL}"

  local DISTILL_RUN BEST_JSON
  DISTILL_RUN=$(ls -dt runs/*"${RUN_SLUG}"* | head -1)
  echo "DISTILL_RUN=${DISTILL_RUN}"
  BEST_JSON="${DISTILL_RUN}/best_checkpoint.json"
  cat "$BEST_JSON"

  cp "${DISTILL_RUN}/metrics.jsonl" "${ARCHIVE_PREFIX}_kd_metrics.jsonl"
  cp "$BEST_JSON" "${ARCHIVE_PREFIX}_best_checkpoint.json"

  local BEST_STEP FINAL_STEP
  read -r BEST_STEP FINAL_STEP <<<"$(python3 - <<PY
import json
from pathlib import Path
run = Path("$DISTILL_RUN")
vals = []
last_step = -1
for line in (run / "metrics.jsonl").read_text().splitlines():
    row = json.loads(line)
    if "step" in row:
        last_step = row["step"]
    if "val_loss" in row:
        vals.append((row["step"], row["val_loss"]))
trajectory = "\n".join(f"{s}\t{v:.6f}" for s, v in vals)
Path("${ARCHIVE_PREFIX}_kd_val_trajectory.txt").write_text(trajectory + "\n")
best = json.loads(Path("$BEST_JSON").read_text())
final_step = vals[-1][0] if vals else last_step
print(best["best_step"], final_step)
print(f"val_best_step={best['best_step']} val_best={best['best_val_loss']}", flush=True)
if vals:
    print(f"val_final_step={vals[-1][0]} val_final={vals[-1][1]}", flush=True)
PY
)"
  echo "BEST_STEP=${BEST_STEP} FINAL_STEP=${FINAL_STEP}"

  local HE_FINAL HE_BEST
  if [ "$BEST_STEP" = "$FINAL_STEP" ]; then
    echo "--- HumanEval n=40 on ${DISTILL_HUB} (best == final @ step ${BEST_STEP}) ---"
    ./tools/run_humaneval_ts_sample.sh \
      "$DISTILL_HUB" \
      "${HE_SLUG_PREFIX}-humaneval-ts-n40" \
      40
    HE_FINAL=$(ls -dt runs/*"${HE_SLUG_PREFIX}"-humaneval-ts-n40* | head -1)
    HE_BEST=""
  else
    echo "--- HumanEval n=40 BEST on ${DISTILL_HUB_BEST} ---"
    ./tools/run_humaneval_ts_sample.sh \
      "$DISTILL_HUB_BEST" \
      "${HE_SLUG_PREFIX}-humaneval-ts-n40-best" \
      40
    HE_BEST=$(ls -dt runs/*"${HE_SLUG_PREFIX}"-humaneval-ts-n40-best* | head -1)
    echo "--- HumanEval n=40 FINAL on ${DISTILL_HUB} ---"
    ./tools/run_humaneval_ts_sample.sh \
      "$DISTILL_HUB" \
      "${HE_SLUG_PREFIX}-humaneval-ts-n40-final" \
      40
    HE_FINAL=$(ls -dt runs/*"${HE_SLUG_PREFIX}"-humaneval-ts-n40-final* | head -1)
  fi

  python3 - <<PY
import json
from pathlib import Path

def pass_at_1(run_dir: str) -> float:
    p = Path(run_dir) / "humaneval-ts.json"
    return json.loads(p.read_text())["multiple-ts"]["pass@1"]

he_final = pass_at_1("$HE_FINAL")
he_best = pass_at_1("$HE_BEST") if "$HE_BEST" else he_final
best = json.loads(Path("$BEST_JSON").read_text())
vals = []
for line in Path("$DISTILL_RUN/metrics.jsonl").read_text().splitlines():
    row = json.loads(line)
    if "val_loss" in row:
        vals.append((row["step"], row["val_loss"]))
final_step, final_val = vals[-1]
kd_wall = int("$KD_WALL")
build_summary = json.loads(Path("${ARCHIVE_PREFIX}_build_summary.json").read_text())

def human(sec: int) -> str:
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m}m {s}s"

summary = {
    "build_wall_clock_sec": build_summary.get("build_wall_clock_sec"),
    "build_wall_clock_human": human(int(build_summary.get("build_wall_clock_sec", 0))),
    "zero_shot_loss": build_summary.get("zero_shot_loss"),
    "build_run": build_summary.get("build_run"),
    "kd_wall_clock_sec": kd_wall,
    "kd_wall_clock_human": human(kd_wall),
    "steps": int("$STEPS"),
    "warmup_steps": int("$WARMUP"),
    "peak_lr": float("$PEAK_LR"),
    "teacher_cache_hub": "$CACHE_HUB",
    "candidate_hub": "$CANDIDATE_HUB",
    "val_best_step": best["best_step"],
    "val_best": best["best_val_loss"],
    "val_final_step": final_step,
    "val_final": final_val,
    "humaneval_final_pass_at_1": he_final,
    "humaneval_best_pass_at_1": he_best,
    "distill_run": "$DISTILL_RUN",
    "humaneval_final_run": "$HE_FINAL",
    "humaneval_best_run": "$HE_BEST" or "$HE_FINAL",
}
out = Path("${ARCHIVE_PREFIX}_run_summary.json")
out.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
for label, run in ([("best", "$HE_BEST")] if "$HE_BEST" else []) + [("final", "$HE_FINAL")]:
    hub = "$DISTILL_HUB_BEST" if label == "best" else "$DISTILL_HUB"
    print(f"humaneval-ts pass@1 ({label}, {hub}):", pass_at_1(run))
    print(f"  run_dir: {run}")
PY

  cp "$LOG" "${ARCHIVE_PREFIX}_pipeline.log"
  echo "========== KD ${LABEL} done $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
}

echo "=== build + fixed-loss short (depth0/e60, depth22/e92) start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

build_arm \
  "depth0/experts60" \
  "/root/phase4_depth0_experts60_fixedloss_short" \
  0 60 \
  "sartajbhuvaji/bonsai-candidate-depth0-experts60"

run_arm \
  "depth0/experts60" \
  "/root/phase4_depth0_experts60_fixedloss_short" \
  "sartajbhuvaji/bonsai-candidate-depth0-experts60" \
  "sartajbhuvaji/bonsai-distilled-depth0-experts60-500-warmup75-lr1e4-fixedloss" \
  "distill-depth0-experts60-vs-baseline-500-warmup75-lr1e4-fixedloss" \
  "distill-depth0-experts60-fixedloss-short"

build_arm \
  "depth22/experts92" \
  "/root/phase4_depth22_experts92_fixedloss_short" \
  22 92 \
  "sartajbhuvaji/bonsai-candidate-depth22-experts92"

run_arm \
  "depth22/experts92" \
  "/root/phase4_depth22_experts92_fixedloss_short" \
  "sartajbhuvaji/bonsai-candidate-depth22-experts92" \
  "sartajbhuvaji/bonsai-distilled-depth22-experts92-500-warmup75-lr1e4-fixedloss" \
  "distill-depth22-experts92-vs-baseline-500-warmup75-lr1e4-fixedloss" \
  "distill-depth22-experts92-fixedloss-short"

echo "=== all done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
