#!/usr/bin/env bash
# Phase 4 MISSION BRIEF (2026-09-27): build candidate, cache logits, distill x2.
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /root/bonsai
source /root/venv/bin/activate
set -a
# shellcheck disable=SC1091
source /root/bonsai/.env
set +a
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"
HF_ARGS=(--hf-token "$HF_TOKEN")
LOG=/root/phase4_pipeline.log
exec > >(tee -a "$LOG") 2>&1

echo "=== Phase 4 ablation pipeline start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

echo "--- Step 1: build pruned candidate ---"
python tools/build_candidate.py \
  --push-to-hub sartajbhuvaji/bonsai-candidate-aggressive-depth-experts \
  "${HF_ARGS[@]}"
CANDIDATE_RUN=$(ls -dt runs/*-build-candidate-aggressive-depth-experts | head -1)
CANDIDATE="${CANDIDATE_RUN}/checkpoints/candidate"
echo "CANDIDATE=${CANDIDATE}" | tee /root/phase4_state.env

echo "--- Step 2a: time baseline logit cache (50 samples) ---"
TIMING_START=$(date +%s)
python tools/cache_teacher_logits.py --teacher baseline --num-samples 50 "${HF_ARGS[@]}"
TIMING_END=$(date +%s)
ELAPSED=$((TIMING_END - TIMING_START))
echo "50-sample baseline cache took ${ELAPSED}s (~$((ELAPSED / 50))s/sample)"
EST_FULL=$((ELAPSED * 6000 / 50))
echo "Extrapolated 6000 samples: ~${EST_FULL}s (~$((EST_FULL / 3600))h)"
SMOKE_CACHE_RUN=$(ls -dt runs/*-cache-teacher-logits-baseline | head -1)
SMOKE_CACHE="${SMOKE_CACHE_RUN}/checkpoints/baseline-topk-logits.pt"

echo "--- Step 3a: smoke distill vs baseline (10 steps, 50-sample cache) ---"
export SMOKE_CANDIDATE="$CANDIDATE" SMOKE_CACHE="$SMOKE_CACHE"
python - <<'PY'
import os
from pathlib import Path
from typing import cast

import bitsandbytes as bnb
import torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

from bonsai.training import TrainConfig, train
from bonsai.training.distillation import make_kd_loss_fn

cache_path = Path(os.environ["SMOKE_CACHE"])
candidate = os.environ["SMOKE_CANDIDATE"]
model = cast(
    Qwen3MoeForCausalLM,
    AutoModelForCausalLM.from_pretrained(candidate, dtype=torch.bfloat16, device_map="auto"),
)
import sys

sys.path.insert(0, "/root/bonsai/tools")
import train_distilled_candidate as tdc

device = next(model.parameters()).device
cache, all_ids = tdc.load_teacher_cache(cache_path, device=device)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
split_at = max(1, int(len(all_ids) * 0.95))
cfg = TrainConfig(peak_lr=2e-4, min_lr=2e-5, warmup_steps=2, total_steps=10, batch_size=4)
history = train(
    model,
    all_ids[:split_at],
    all_ids[split_at:],
    cfg,
    loss_fn=make_kd_loss_fn(cache),
    optimizer=bnb.optim.AdamW8bit(model.parameters(), lr=cfg.peak_lr),
)
print("smoke OK", history[-1])
PY

echo "--- Step 2b: full baseline logit cache (6000) ---"
python tools/cache_teacher_logits.py --teacher baseline --num-samples 6000 "${HF_ARGS[@]}"
BASELINE_CACHE_RUN=$(ls -dt runs/*-cache-teacher-logits-baseline | head -1)
BASELINE_CACHE="${BASELINE_CACHE_RUN}/checkpoints/baseline-topk-logits.pt"
echo "BASELINE_CACHE=${BASELINE_CACHE}" | tee -a /root/phase4_state.env

echo "--- Step 2c: full lora-corrected logit cache (6000) ---"
python tools/cache_teacher_logits.py --teacher lora-corrected --num-samples 6000 "${HF_ARGS[@]}"
LORA_CACHE_RUN=$(ls -dt runs/*-cache-teacher-logits-lora-corrected | head -1)
LORA_CACHE="${LORA_CACHE_RUN}/checkpoints/lora-corrected-topk-logits.pt"
echo "LORA_CACHE=${LORA_CACHE}" | tee -a /root/phase4_state.env

echo "--- Step 3b: full distill vs baseline (500 steps) ---"
python tools/train_distilled_candidate.py \
  --candidate "$CANDIDATE" \
  --teacher-cache "$BASELINE_CACHE" \
  --run-name distill-aggressive-vs-baseline \
  --push-to-hub sartajbhuvaji/bonsai-distilled-aggressive-vs-baseline \
  "${HF_ARGS[@]}"
BASELINE_DISTILL_RUN=$(ls -dt runs/*-distill-aggressive-vs-baseline | head -1)
echo "BASELINE_DISTILL_RUN=${BASELINE_DISTILL_RUN}" | tee -a /root/phase4_state.env

echo "--- Step 3c: full distill vs lora-corrected (500 steps) ---"
python tools/train_distilled_candidate.py \
  --candidate "$CANDIDATE" \
  --teacher-cache "$LORA_CACHE" \
  --run-name distill-aggressive-vs-corrected \
  --push-to-hub sartajbhuvaji/bonsai-distilled-aggressive-vs-corrected \
  "${HF_ARGS[@]}"
CORRECTED_DISTILL_RUN=$(ls -dt runs/*-distill-aggressive-vs-corrected | head -1)
echo "CORRECTED_DISTILL_RUN=${CORRECTED_DISTILL_RUN}" | tee -a /root/phase4_state.env

echo "--- Final metrics ---"
for run in "$BASELINE_DISTILL_RUN" "$CORRECTED_DISTILL_RUN"; do
  echo "== $run =="
  tail -3 "$run/metrics.jsonl"
done

echo "=== Phase 4 ablation pipeline done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
