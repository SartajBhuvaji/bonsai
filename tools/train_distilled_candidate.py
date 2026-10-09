"""Train a student checkpoint against one cached teacher's logits.

Phase 4 ablation wrapper around `bonsai.training.kd_run.distill_student`.
For a general teacher/student CLI without pruning, use `tools/run_kd_distillation.py`.
"""

from __future__ import annotations

import argparse

from bonsai.training import TrainConfig
from bonsai.training.kd_run import (
    DEFAULT_LORA_TARGET_MODULES,
    distill_student,
    resolve_teacher_cache_path,
)

DEFAULT_CONFIG = TrainConfig(
    peak_lr=2e-4,
    min_lr=2e-5,
    warmup_steps=75,
    total_steps=500,
    batch_size=4,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate", required=True, help="path or HF repo id of the student checkpoint"
    )
    parser.add_argument(
        "--teacher-cache",
        required=True,
        help="local path to teacher top-k logits .pt, or HF dataset repo id (org/name)",
    )
    parser.add_argument("--run-name", required=True, help="e.g. distill-aggressive-vs-baseline")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--push-to-hub", default=None)
    parser.add_argument(
        "--push-to-hub-best",
        default=None,
        help="HF repo for lowest-val checkpoint (default: <push-to-hub>-best)",
    )
    parser.add_argument("--hf-token", default=None)
    parser.add_argument(
        "--hub-public",
        action="store_true",
        help="push distilled checkpoints as public HF repos (default: private)",
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_CONFIG.total_steps)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size)
    parser.add_argument("--peak-lr", type=float, default=DEFAULT_CONFIG.peak_lr)
    parser.add_argument("--min-lr", type=float, default=DEFAULT_CONFIG.min_lr)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_CONFIG.warmup_steps)
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help=(
            "freeze the student and train a LoRA adapter instead of every weight "
            "(default: full-parameter). See plan.md 2026-10-09 for why this exists: "
            "full-parameter KD was found to overwrite real pre-pruning capability."
        ),
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default=",".join(DEFAULT_LORA_TARGET_MODULES),
        help="comma-separated module names, e.g. q_proj,k_proj,v_proj,o_proj",
    )
    args = parser.parse_args()

    config = TrainConfig(
        peak_lr=args.peak_lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        total_steps=args.steps,
        batch_size=args.batch_size,
    )
    teacher_cache = resolve_teacher_cache_path(args.teacher_cache, hf_token=args.hf_token)
    distill_student(
        args.candidate,
        teacher_cache,
        run_name=args.run_name,
        config=config,
        val_fraction=args.val_fraction,
        push_to_hub=args.push_to_hub,
        push_to_hub_best=args.push_to_hub_best,
        hf_token=args.hf_token,
        hub_private=not args.hub_public,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules.split(","),
    )


if __name__ == "__main__":
    main()
