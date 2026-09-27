#!/usr/bin/env bash
# Run an incremental Phase 4 script, then keep tmux open for log tail / restarts.
set -uo pipefail
export PYTHONUNBUFFERED=1
SCRIPT="${PHASE4_SCRIPT:-/root/bonsai/tools/run_phase4_baseline_kd_3k.sh}"
"$SCRIPT"
code=$?
echo ""
echo "Pipeline exited with code ${code} at $(date -u +%Y-%m-%dT%H:%M:%SZ)."
echo "Log: ${LOG:-/root/phase4_baseline_kd_3k.log}"
echo "Metrics: tail -f runs/<distill-run>/metrics.jsonl"
echo "tmux session kept open — inspect log, re-run steps, or exit."
exec bash -l
