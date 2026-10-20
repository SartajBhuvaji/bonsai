#!/usr/bin/env bash
# Phase 4.3 inference-engine benchmark sweep (vLLM + SGLang). Run on GPU box as root.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
VENV="${VENV:-/root/venv}"
source "${VENV}/bin/activate"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
set -a
# HF_TOKEN optional for public models; sourced when present.
if [[ -f .env ]]; then source .env; fi
set +a
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

GPU_MEM="${GPU_MEMORY_UTILIZATION:-0.9}"
CONCURRENCIES="${CONCURRENCIES:-1,4,16,32}"
NUM_PROMPTS="${NUM_PROMPTS:-159}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-512}"
DEBUG_DIR="${DEBUG_DIR:-$ROOT/docs/debug}"
VLLM_PORT="${VLLM_PORT:-8000}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
MANIFEST="$DEBUG_DIR/phase4_inference_benchmark_manifest.json"
VLLM_ONLY="${VLLM_ONLY:-0}"
SGLANG_ONLY="${SGLANG_ONLY:-0}"
if [[ "$VLLM_ONLY" == "1" && -f "$MANIFEST" ]]; then
  python -c "
import json
from pathlib import Path
p = Path('${MANIFEST}')
rows = json.loads(p.read_text())
rows = [r for r in rows if r.get('engine') != 'vllm']
p.write_text(json.dumps(rows, indent=2) + '\n')
print(f'manifest: kept {len(rows)} non-vLLM rows')
"
elif [[ "$SGLANG_ONLY" == "1" && -f "$MANIFEST" ]]; then
  python -c "
import json
from pathlib import Path
p = Path('${MANIFEST}')
rows = json.loads(p.read_text())
rows = [r for r in rows if r.get('engine') != 'sglang']
p.write_text(json.dumps(rows, indent=2) + '\n')
print(f'manifest: kept {len(rows)} non-SGLang rows')
"
elif [[ "$VLLM_ONLY" != "1" && "$SGLANG_ONLY" != "1" ]]; then
  echo "[]" >"$MANIFEST"
fi

append_manifest() {
  python -c '
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
rows = json.loads(p.read_text())
rows.append(json.loads(sys.argv[2]))
p.write_text(json.dumps(rows, indent=2) + "\n")
' "$MANIFEST" "$1"
}

VLLM_VER="$(python -c 'import vllm; print(vllm.__version__)')"
SGLANG_VER="$(python -c 'import sglang; print(sglang.__version__)')"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

declare -A MODELS=(
  [baseline]="Qwen/Qwen3-30B-A3B-Base"
  [depth0_experts64]="sartajbhuvaji/bonsai-distilled-depth0-experts64-500-warmup75-lr1e4-fixedloss-best"
  [depth16_experts0]="sartajbhuvaji/bonsai-distilled-depth16-experts0-500-warmup75-lr1e4-fixedloss-best"
  [depth0_experts8]="sartajbhuvaji/bonsai-distilled-depth0-experts8-500-warmup75-lr1e4-fixedloss-best"
)

VLLM_DATASET="$DEBUG_DIR/phase4_inference_benchmark_humaneval_ts_vllm.jsonl"
SGLANG_DATASET="$DEBUG_DIR/phase4_inference_benchmark_humaneval_ts_sglang.jsonl"
PROMPT_STATS="$DEBUG_DIR/phase4_inference_benchmark_humaneval_ts_prompt_stats.json"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$MASTER_LOG"; }

stop_server() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
  sleep 3
}

wait_http() {
  local url="$1"
  local tries="${2:-120}"
  for _ in $(seq 1 "$tries"); do
    if curl -sf "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  return 1
}

run_vllm_bench() {
  local slug="$1"
  local model="$2"
  local conc="$3"
  local prefix="$DEBUG_DIR/phase4_inference_benchmark_${slug}_vllm_c${conc}"
  local result_json="${prefix}.json"
  local log_file="${prefix}.log"

  log "vLLM bench ${slug} concurrency=${conc}"
  vllm bench serve \
    --backend openai \
    --base-url "http://127.0.0.1:${VLLM_PORT}" \
    --model "$model" \
    --dataset-name custom \
    --dataset-path "$VLLM_DATASET" \
    --custom-output-len "$OUTPUT_TOKENS" \
    --skip-chat-template \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$conc" \
    --request-rate inf \
    --temperature 0 \
    --save-result \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,99 \
    --result-filename "$(basename "$result_json")" \
    --result-dir "$DEBUG_DIR" \
    --label "phase4_inference_benchmark_${slug}_vllm_c${conc}" \
    2>&1 | tee "$log_file"

  append_manifest "$(
    python -c "import json; print(json.dumps({
      'candidate': '${slug}',
      'model_id': '${model}',
      'engine': 'vllm',
      'concurrency': int('${conc}'),
      'gpu_memory_utilization': float('${GPU_MEM}'),
      'num_prompts': int('${NUM_PROMPTS}'),
      'output_tokens': int('${OUTPUT_TOKENS}'),
      'result_json': '${result_json}',
      'vllm_version': '${VLLM_VER}',
      'sglang_version': '${SGLANG_VER}',
      'gpu': '''${GPU_NAME}''',
      'prompt_stats_path': '${PROMPT_STATS}',
    }))"
  )"
}

run_sglang_bench() {
  local slug="$1"
  local model="$2"
  local conc="$3"
  local prefix="$DEBUG_DIR/phase4_inference_benchmark_${slug}_sglang_c${conc}"
  local result_json="${prefix}.json"
  local log_file="${prefix}.log"

  log "SGLang bench ${slug} concurrency=${conc}"
  python -m sglang.benchmark.serving \
    --backend sglang \
    --host 127.0.0.1 \
    --port "$SGLANG_PORT" \
    --model "$model" \
    --dataset-name custom \
    --dataset-path "$SGLANG_DATASET" \
    --sharegpt-output-len "$OUTPUT_TOKENS" \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$conc" \
    --request-rate inf \
    --output-file "$result_json" \
    2>&1 | tee "$log_file"

  append_manifest "$(
    python -c "import json; print(json.dumps({
      'candidate': '${slug}',
      'model_id': '${model}',
      'engine': 'sglang',
      'concurrency': int('${conc}'),
      'gpu_memory_utilization': float('${GPU_MEM}'),
      'num_prompts': int('${NUM_PROMPTS}'),
      'output_tokens': int('${OUTPUT_TOKENS}'),
      'result_json': '${result_json}',
      'vllm_version': '${VLLM_VER}',
      'sglang_version': '${SGLANG_VER}',
      'gpu': '''${GPU_NAME}''',
      'prompt_stats_path': '${PROMPT_STATS}',
    }))"
  )"
}

MASTER_LOG="$DEBUG_DIR/phase4_inference_benchmark_sweep_master.log"
if [[ "$VLLM_ONLY" == "1" || "$SGLANG_ONLY" == "1" ]]; then
  echo "[$(date -Iseconds)] partial sweep (VLLM_ONLY=${VLLM_ONLY} SGLANG_ONLY=${SGLANG_ONLY})" >>"$MASTER_LOG"
else
  : >"$MASTER_LOG"
fi

log "Exporting HumanEval-TS prompts"
python tools/export_humaneval_ts_inference_prompts.py --out-dir "$DEBUG_DIR"

IFS=',' read -r -a CONC_ARRAY <<<"$CONCURRENCIES"

if [[ "$SGLANG_ONLY" != "1" ]]; then
log "Installing vLLM-pinned kernel deps"
bash tools/setup_inference_engine_deps.sh vllm

for slug in baseline depth0_experts64 depth16_experts0 depth0_experts8; do
  model="${MODELS[$slug]}"
  log "========== vLLM server: ${slug} =========="
  stop_server
  vllm serve "$model" \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM" \
    --port "$VLLM_PORT" \
    >"$DEBUG_DIR/phase4_inference_benchmark_${slug}_vllm_server.log" 2>&1 &
  SERVER_PID=$!
  wait_http "http://127.0.0.1:${VLLM_PORT}/health" 180 || {
    log "vLLM server failed to start for ${slug}"
    tail -50 "$DEBUG_DIR/phase4_inference_benchmark_${slug}_vllm_server.log" || true
    exit 1
  }
  for conc in "${CONC_ARRAY[@]}"; do
    run_vllm_bench "$slug" "$model" "$conc"
  done
  stop_server
done
fi

if [[ "$VLLM_ONLY" != "1" ]]; then
log "Installing SGLang-pinned kernel deps"
bash tools/setup_inference_engine_deps.sh sglang

for slug in baseline depth0_experts64 depth16_experts0 depth0_experts8; do
  model="${MODELS[$slug]}"
  log "========== SGLang server: ${slug} =========="
  stop_server
  python -m sglang.launch_server \
    --model-path "$model" \
    --trust-remote-code \
    --context-length "$MAX_MODEL_LEN" \
    --mem-fraction-static "$GPU_MEM" \
    --port "$SGLANG_PORT" \
    >"$DEBUG_DIR/phase4_inference_benchmark_${slug}_sglang_server.log" 2>&1 &
  SERVER_PID=$!
  wait_http "http://127.0.0.1:${SGLANG_PORT}/health" 240 || {
    log "SGLang server failed to start for ${slug}"
    tail -50 "$DEBUG_DIR/phase4_inference_benchmark_${slug}_sglang_server.log" || true
    exit 1
  }
  for conc in "${CONC_ARRAY[@]}"; do
    run_sglang_bench "$slug" "$model" "$conc"
  done
  stop_server
done
touch "$DEBUG_DIR/phase4_inference_sglang_sweep.done"
fi

python tools/consolidate_phase4_inference_benchmark.py \
  --manifest "$MANIFEST" \
  --out "$DEBUG_DIR/phase4_inference_benchmark_summary.json"

log "Done. Summary: $DEBUG_DIR/phase4_inference_benchmark_summary.json"
