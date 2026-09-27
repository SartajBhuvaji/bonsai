---
library_name: transformers
license: apache-2.0
base_model: sartajbhuvaji/bonsai-candidate-aggressive-depth-experts
tags:
- knowledge-distillation
- moe
- code
- typescript
- qwen3
language:
- code
datasets:
- bigcode/the-stack-dedup
---

# Bonsai distilled — aggressive candidate vs baseline teacher

Knowledge-distilled **student** checkpoint from Phase 4 of [bonsai](https://github.com/SartajBhuvaji/bonsai): the pruned **aggressive depth+experts** MoE (~8.5B parameters) trained to match **`Qwen/Qwen3-30B-A3B-Base`** on a fixed top-k logit cache (baseline teacher arm of the teacher-correction ablation).

**Private repo** — request access to load weights for reproduction or eval.

## Model summary

| Field | Value |
|-------|--------|
| **Architecture** | Qwen3 MoE (pruned layers + pruned experts), same tokenizer as base Qwen3-30B-A3B |
| **Init checkpoint** | [`sartajbhuvaji/bonsai-candidate-aggressive-depth-experts`](https://huggingface.co/sartajbhuvaji/bonsai-candidate-aggressive-depth-experts) |
| **Teacher (distillation target)** | [`Qwen/Qwen3-30B-A3B-Base`](https://huggingface.co/Qwen/Qwen3-30B-A3B-Base) |
| **Distillation loss** | Forward KL on teacher **top-20** logits per token (truncated support, not full-vocab CE) |
| **Training run id** | `20260927T171637Z-3ddb90-distill-aggressive-vs-baseline` |
| **Final KD loss** | train **0.557**, val **0.535** (500 steps, last logged step 499) |

Compare with the paired arm distilled against the LoRA teacher-corrected checkpoint: `sartajbhuvaji/bonsai-distilled-aggressive-vs-corrected` (same student init and matched 6000-sample cache).

## Training data & cache

- **Corpus:** `bigcode/the-stack-dedup`, subdirectory `data/typescript`
- **Pool:** 6000 sequences, **seed 11** (same ids used for baseline and LoRA teacher caches)
- **Teacher cache:** 6000-sample top-20 logit file from `tools/cache_teacher_logits.py --teacher baseline`  
  (local run `runs/20260927T163225Z-91fcb3-cache-teacher-logits-baseline/`)
- **Train/val split:** 95% / 5% of cached sequences (~5700 / ~300)

## Training procedure

| Hyperparameter | Value |
|----------------|--------|
| Optimizer | AdamW 8-bit (`bitsandbytes`) |
| Steps | 500 |
| Batch size | 4 |
| Peak LR | 2e-4 |
| Min LR | 2e-5 |
| Warmup | 75 steps |
| Precision | bf16, gradient checkpointing enabled |
| Hardware (recorded run) | DigitalOcean H100, 2026-09-27 |

Entry point: `tools/train_distilled_candidate.py` → `bonsai.training.kd_run.distill_student`.

## Quick start

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "sartajbhuvaji/bonsai-distilled-aggressive-vs-baseline"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=True,
)
```

Use the same generation settings as other Qwen3 checkpoints in this project (`configs/qwen3-30b-a3b.toml`).

## Evaluation status

KD val loss is reported above; **HumanEval-TS / MBPP-TS** sample evals for this checkpoint are planned separately. Do not treat KD loss alone as a downstream coding benchmark.

Paper figures and machine-readable metrics: `docs/latex/figures/phase4_kd/` in the bonsai repo (`results.json`, refresh via `tools/refresh_phase4_kd_artifacts.py`).

## Related artifacts

| Artifact | Role |
|----------|------|
| `sartajbhuvaji/bonsai-importance-scores` | Pruning scores used to build the student init |
| `sartajbhuvaji/bonsai-candidate-aggressive-depth-experts` | Pre-distillation pruned weights |
| `sartajbhuvaji/bonsai-qwen3-30b-a3b-teacher-corrected-lora` | Alternate teacher (other ablation arm) |

## License

Derived from Qwen3-30B-A3B-Base and project pruning/KD code; follow the [Qwen model license](https://huggingface.co/Qwen/Qwen3-30B-A3B-Base) and The Stack terms for training data.

## Contact

Sartaj Bhuvaji — [sbhuvaji@seattleu.edu](mailto:sbhuvaji@seattleu.edu)
