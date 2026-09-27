#!/usr/bin/env bash
# Run the Phase 4 pipeline, then keep the tmux pane open for inspection/restarts.
set -uo pipefail
export PYTHONUNBUFFERED=1
SCRIPT="${PHASE4_SCRIPT:-/root/bonsai/tools/run_phase4_ablation_pipeline.sh}"
"$SCRIPT"
code=$?
echo ""
echo "Pipeline exited with code ${code} at $(date -u +%Y-%m-%dT%H:%M:%SZ)."
echo "tmux session kept open — inspect /root/phase4_pipeline.log, edit and re-run steps, or exit."
exec bash -l
