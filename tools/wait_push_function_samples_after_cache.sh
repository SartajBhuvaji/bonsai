#!/usr/bin/env bash
# One-off helper when the main pipeline started before subtask 1b was added.
set -euo pipefail
cd /root/bonsai
source /root/venv/bin/activate
set -a
# shellcheck disable=SC1091
source /root/bonsai/.env
set +a

LOG="${LOG:-/root/phase4_depth12_experts80_functions_samples_push.log}"
exec > >(tee -a "$LOG") 2>&1

PIPELINE_LOG="${PIPELINE_LOG:-/root/phase4_depth12_experts80_functions_cache_kd.log}"
SAMPLES_HUB="${SAMPLES_HUB:-sartajbhuvaji/bonsai-kd-function-samples-baseline-12k}"

echo "waiting for teacher cache to finish in ${PIPELINE_LOG} ..."
until grep -q 'saved 12000 cached samples' "$PIPELINE_LOG" 2>/dev/null; do
  sleep 30
done

CACHE_PT=$(find /root/bonsai/runs -path '*cache-teacher-logits-baseline*' -name '*-topk-logits.pt' | head -1)
echo "CACHE_PT=${CACHE_PT}"
python tools/push_function_samples_to_hub.py \
  --from-cache-pt "$CACHE_PT" \
  --push-to-hub "$SAMPLES_HUB" \
  --hub-public \
  --write-jsonl "$(dirname "$CACHE_PT")/function-samples-hub.jsonl" \
  --hf-token "$HF_TOKEN"
echo "=== samples push done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
