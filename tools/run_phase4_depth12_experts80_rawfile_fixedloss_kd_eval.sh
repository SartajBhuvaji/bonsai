#!/usr/bin/env bash
# Fixed KD loss (PR #71): retrain on ORIGINAL raw-file teacher cache + HumanEval n=40.
# Isolates data shape vs loss fix (function-mode cache not used).
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /root/bonsai
source /root/venv/bin/activate
set -a
# shellcheck disable=SC1091
source /root/bonsai/.env
set +a
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"

LOG="${LOG:-/root/phase4_depth12_experts80_rawfile_fixedloss_pipeline.log}"
exec > >(tee -a "$LOG") 2>&1

CANDIDATE_HUB="${CANDIDATE_HUB:-sartajbhuvaji/bonsai-candidate-depth12-experts80}"
CACHE_HUB="${CACHE_HUB:-sartajbhuvaji/bonsai-teacher-logits-baseline}"
DISTILL_HUB="${DISTILL_HUB:-sartajbhuvaji/bonsai-distilled-depth12-experts80-rawfile-vs-baseline-3k-warmup450-lr1e4-fixedloss}"
DISTILL_HUB_BEST="${DISTILL_HUB_BEST:-${DISTILL_HUB}-best}"
RUN_SLUG="${RUN_SLUG:-distill-depth12-experts80-rawfile-vs-baseline-3000-warmup450-lr1e4-fixedloss}"
ARCHIVE_PREFIX="${ARCHIVE_PREFIX:-/root/phase4_depth12_experts80_rawfile_fixedloss}"

echo "=== raw-file fixed-loss KD + eval start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "CACHE_HUB=${CACHE_HUB} (reuse; no re-cache)"
echo "DISTILL_HUB=${DISTILL_HUB}"

KD_WALL_START=$(date +%s)
python tools/train_distilled_candidate.py \
  --candidate "$CANDIDATE_HUB" \
  --teacher-cache "$CACHE_HUB" \
  --run-name "$RUN_SLUG" \
  --steps 3000 \
  --warmup-steps 450 \
  --peak-lr 1e-4 \
  --push-to-hub "$DISTILL_HUB" \
  --push-to-hub-best "$DISTILL_HUB_BEST" \
  --hub-public \
  --hf-token "$HF_TOKEN"
KD_WALL_END=$(date +%s)
KD_WALL=$((KD_WALL_END - KD_WALL_START))
echo "KD_WALL_CLOCK_SEC=${KD_WALL}"

DISTILL_RUN=$(ls -dt runs/*"${RUN_SLUG}"* | head -1)
echo "DISTILL_RUN=${DISTILL_RUN}"
BEST_JSON="${DISTILL_RUN}/best_checkpoint.json"
cat "$BEST_JSON"

cp "${DISTILL_RUN}/metrics.jsonl" "${ARCHIVE_PREFIX}_kd_metrics.jsonl"
cp "$BEST_JSON" "${ARCHIVE_PREFIX}_best_checkpoint.json"
python3 - <<PY
import json
from pathlib import Path
run = Path("$DISTILL_RUN")
vals = []
for line in (run / "metrics.jsonl").read_text().splitlines():
    row = json.loads(line)
    if "val_loss" in row:
        vals.append((row["step"], row["val_loss"]))
trajectory = "\n".join(f"{s}\t{v:.6f}" for s, v in vals)
Path("${ARCHIVE_PREFIX}_kd_val_trajectory.txt").write_text(trajectory + "\n")
best = json.loads(Path("$BEST_JSON").read_text())
final_step, final_val = vals[-1]
print(f"val_best_step={best['best_step']} val_best={best['best_val_loss']}")
print(f"val_final_step={final_step} val_final={final_val}")
PY

BEST_STEP=$(python3 -c "import json; print(json.load(open('$BEST_JSON'))['best_step'])")
FINAL_STEP=2999
echo "BEST_STEP=${BEST_STEP} FINAL_STEP=${FINAL_STEP}"

if [ "$BEST_STEP" = "$FINAL_STEP" ]; then
  echo "--- HumanEval n=40 on ${DISTILL_HUB} (best == final @ step ${BEST_STEP}) ---"
  ./tools/run_humaneval_ts_sample.sh \
    "$DISTILL_HUB" \
    distill-depth12-experts80-rawfile-fixedloss-humaneval-ts-n40 \
    40
  HE_FINAL=$(ls -dt runs/*rawfile-fixedloss-humaneval-ts-n40* | head -1)
  HE_BEST=""
else
  echo "--- HumanEval n=40 FINAL on ${DISTILL_HUB} ---"
  ./tools/run_humaneval_ts_sample.sh \
    "$DISTILL_HUB" \
    distill-depth12-experts80-rawfile-fixedloss-humaneval-ts-n40-final \
    40
  HE_FINAL=$(ls -dt runs/*rawfile-fixedloss-humaneval-ts-n40-final* | head -1)
  echo "--- HumanEval n=40 BEST on ${DISTILL_HUB_BEST} ---"
  ./tools/run_humaneval_ts_sample.sh \
    "$DISTILL_HUB_BEST" \
    distill-depth12-experts80-rawfile-fixedloss-humaneval-ts-n40-best \
    40
  HE_BEST=$(ls -dt runs/*rawfile-fixedloss-humaneval-ts-n40-best* | head -1)
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
wall = int("$KD_WALL")
h, r = divmod(wall, 3600)
m, s = divmod(r, 60)
summary = {
    "kd_wall_clock_sec": wall,
    "kd_wall_clock_human": f"{h}h {m}m {s}s",
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
for label, run in [("final", "$HE_FINAL")] + ([("best", "$HE_BEST")] if "$HE_BEST" else []):
    hub = "$DISTILL_HUB" if label == "final" else "$DISTILL_HUB_BEST"
    print(f"humaneval-ts pass@1 ({label}, {hub}):", pass_at_1(run))
    print(f"  run_dir: {run}")
PY

cp "$LOG" "${ARCHIVE_PREFIX}_pipeline.log"
echo "=== done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "Archive prefix on box: ${ARCHIVE_PREFIX}_*"
