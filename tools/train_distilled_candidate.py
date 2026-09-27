"""Train a student checkpoint against one cached teacher's logits.

Phase 4 ablation wrapper around `bonsai.training.kd_run.distill_student`.
For a general teacher/student CLI without pruning, use `tools/run_kd_distillation.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bonsai.training import TrainConfig
from bonsai.training.kd_run import distill_student

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
        "--teacher-cache", required=True, help="path to a teacher top-k logits .pt file"
    )
    parser.add_argument("--run-name", required=True, help="e.g. distill-aggressive-vs-baseline")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--push-to-hub", default=None)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--steps", type=int, default=DEFAULT_CONFIG.total_steps)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size)
    parser.add_argument("--peak-lr", type=float, default=DEFAULT_CONFIG.peak_lr)
    parser.add_argument("--min-lr", type=float, default=DEFAULT_CONFIG.min_lr)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_CONFIG.warmup_steps)
    args = parser.parse_args()

    config = TrainConfig(
        peak_lr=args.peak_lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        total_steps=args.steps,
        batch_size=args.batch_size,
    )
    distill_student(
        args.candidate,
        Path(args.teacher_cache),
        run_name=args.run_name,
        config=config,
        val_fraction=args.val_fraction,
        push_to_hub=args.push_to_hub,
        hf_token=args.hf_token,
    )


if __name__ == "__main__":
    main()
