#!/usr/bin/env bash
# Resume subtask 2 only (cache + samples already on Hub).
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /root/bonsai
source /root/venv/bin/activate
set -a
# shellcheck disable=SC1091
source /root/bonsai/.env
set +a
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"

LOG="${LOG:-/root/phase4_depth12_experts80_functions_kd_only.log}"
exec > >(tee -a "$LOG") 2>&1

python - <<'PY'
import os
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
api = HfApi(token=token)
api.update_repo_settings(
    "sartajbhuvaji/bonsai-teacher-logits-baseline-functions",
    private=False,
    token=token,
    repo_type="dataset",
)
print("logits cache repo set public")
PY

CACHE_PT="/root/bonsai/runs/20261010T021652Z-91a1f6-cache-teacher-logits-baseline/checkpoints/Qwen--Qwen3-30B-A3B-Base-topk-logits.pt"
if [ -f "$CACHE_PT" ]; then
  python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
src = Path("/root/bonsai/runs/20261010T021652Z-91a1f6-cache-teacher-logits-baseline/checkpoints/Qwen--Qwen3-30B-A3B-Base-topk-logits.pt")
api = HfApi(token=token)
api.upload_file(
    path_or_fileobj=str(src),
    path_in_repo="baseline-topk-logits.pt",
    repo_id="sartajbhuvaji/bonsai-teacher-logits-baseline-functions",
    repo_type="dataset",
)
print("uploaded baseline-topk-logits.pt alias on Hub")
PY
fi

echo "=== KD subtask 2 resume $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
KD_WALL_START=$(date +%s)
python tools/train_distilled_candidate.py \
  --candidate sartajbhuvaji/bonsai-candidate-depth12-experts80 \
  --teacher-cache sartajbhuvaji/bonsai-teacher-logits-baseline-functions \
  --run-name distill-depth12-experts80-functions-vs-baseline-3000-warmup450-lr1e4 \
  --steps 3000 \
  --warmup-steps 450 \
  --peak-lr 1e-4 \
  --push-to-hub sartajbhuvaji/bonsai-distilled-depth12-experts80-functions-vs-baseline-3k-warmup450-lr1e4 \
  --push-to-hub-best sartajbhuvaji/bonsai-distilled-depth12-experts80-functions-vs-baseline-3k-warmup450-lr1e4-best \
  --hf-token "$HF_TOKEN" \
  --hub-public
echo "KD_WALL_CLOCK_SEC=$(( $(date +%s) - KD_WALL_START ))"
echo "=== KD done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
