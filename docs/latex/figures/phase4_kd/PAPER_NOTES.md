# Phase 4 KD figures — notes for paper integration

Auto-updated by `tools/refresh_phase4_kd_artifacts.py` (2026-09-27T18:39:50Z).

## Experimental setup (both arms)

- **Student:** same pruned aggressive depth+experts candidate (~8.5B params).
- **Teachers:** cached top-20 logits on **identical** 6000 TypeScript samples (seed 11).
  - Baseline: `Qwen/Qwen3-30B-A3B-Base`
  - Corrected: `sartajbhuvaji/bonsai-qwen3-30b-a3b-teacher-corrected-lora`
- **Loss:** forward KL on teacher top-k support (not full-vocab CE).
- **Training:** AdamW 8-bit, 500 steps, batch 4, peak_lr=2e-4, warmup 75 (see `config.json` per run).

## Results snapshot

```json
{
  "vs_baseline": {
    "run_id": "20260927T171637Z-3ddb90-distill-aggressive-vs-baseline",
    "steps_logged": 500,
    "last_step": 499,
    "complete": true,
    "final_train_loss": 0.5572705641388891,
    "final_val_loss": 0.5350065913299721,
    "peak_lr": 0.0002,
    "total_steps": 500,
    "batch_size": 4,
    "warmup_steps": 75
  },
  "vs_corrected": {
    "run_id": "20260927T175820Z-2e6846-distill-aggressive-vs-corrected",
    "steps_logged": 500,
    "last_step": 499,
    "complete": true,
    "final_train_loss": 0.5810125991702081,
    "final_val_loss": 0.533891138657927,
    "peak_lr": 0.0002,
    "total_steps": 500,
    "batch_size": 4,
    "warmup_steps": 75
  }
}
```

## HumanEval-TS n=40 (2026-09-27)

| Distilled student (teacher) | pass@1 |
|-----------------------------|--------|
| vs baseline Qwen3-30B | **0.0** (0/40) |
| vs LoRA-corrected teacher | **0.0** (0/40) |

KD val tied (~0.535 vs ~0.534); downstream sample also tied. Neither arm beats
the other on this slice — both are far below unpruned base (~0.67).

## Interpretation

- Lower **val KD** on the matched holdout means the student matched that teacher's distribution better.
- KD tie + HumanEval tie: **no evidence** that teacher correction changes the distilled student on either proxy at this training budget.

## Figures in this directory

| File | Use |
|------|-----|
| `distill_teacher_ablation_comparison.pdf` | Main training-curve panel (both teachers) |
| `distill_vs_baseline_kd_loss.pdf` | Baseline arm only (legacy single-panel) |
| `distill_final_val_kd_bar.pdf` | Bar chart once **both** runs hit 500 steps |
| `results.json` | Machine-readable final metrics for agents/LaTeX tables |

## Paper agent TODO

- [ ] Re-run refresh script after 3c completes; include bar chart with both teachers.
- [ ] Optional: small HumanEval-TS / MBPP-TS sample on both distilled checkpoints.
- [ ] Caption: top-k truncated KD; same student init; matched caches.
