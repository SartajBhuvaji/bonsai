#!/usr/bin/env bash
# Resume after KD finished but Hub push / eval failed (e.g. private storage quota).
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /root/bonsai
source /root/venv/bin/activate
set -a
# shellcheck disable=SC1091
source /root/bonsai/.env
set +a
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"

DISTILL_HUB="${DISTILL_HUB:-sartajbhuvaji/bonsai-distilled-depth0-experts97-vs-baseline-3k-warmup450-lr1e4}"
DISTILL_HUB_BEST="${DISTILL_HUB_BEST:-${DISTILL_HUB}-best}"
RUN_DIR="${RUN_DIR:-/root/bonsai/runs/20261008T220546Z-b50ff3-distill-depth0-experts97-vs-baseline-3000-warmup450-lr1e4}"
LOG="${LOG:-/root/phase4_depth0_experts97_resume.log}"
exec > >(tee -a "$LOG") 2>&1

echo "=== resume public Hub + HumanEval $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

python - <<'PY'
import os
from huggingface_hub import HfApi

token = os.environ["HF_TOKEN"]
api = HfApi(token=token)
for repo in [
    "sartajbhuvaji/bonsai-candidate-depth0-experts97",
    "sartajbhuvaji/bonsai-distilled-depth0-experts97-vs-baseline-3k-warmup450-lr1e4",
]:
    api.update_repo_settings(repo, private=False, token=token)
    print(f"public: {repo}")
PY

echo "--- push best-val (public) ---"
python tools/push_local_to_hub.py \
  "${RUN_DIR}/checkpoints/distilled-best" \
  "$DISTILL_HUB_BEST" \
  --hf-token "$HF_TOKEN" \
  --hub-public

echo "--- HumanEval-TS n=40 FINAL ---"
./tools/run_humaneval_ts_sample.sh "$DISTILL_HUB" "distill-depth0-experts97-humaneval-ts-n40-lr1e4-final" 40
HE_FINAL=$(ls -dt runs/*depth0-experts97*lr1e4-final* | head -1)

echo "--- HumanEval-TS n=40 BEST-VAL ---"
./tools/run_humaneval_ts_sample.sh "$DISTILL_HUB_BEST" "distill-depth0-experts97-humaneval-ts-n40-lr1e4-best" 40
HE_BEST=$(ls -dt runs/*depth0-experts97*lr1e4-best* | head -1)

python - <<PY
import json
from pathlib import Path
for label, run, hub in [
    ("final", "$HE_FINAL", "$DISTILL_HUB"),
    ("best", "$HE_BEST", "$DISTILL_HUB_BEST"),
]:
    p = Path(run) / "humaneval-ts.json"
    m = json.loads(p.read_text())
    print(f"humaneval-ts pass@1 ({label}, {hub}):", m["multiple-ts"]["pass@1"])
PY

echo "=== done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
