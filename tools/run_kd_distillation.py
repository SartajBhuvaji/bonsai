#!/usr/bin/env python3
"""End-to-end top-k KD: cache a teacher's logits, then train any student model.

No pruning step — pass any Hugging Face repo id (or local path) for teacher and
student. The teacher is loaded only for the caching pass; training reads the
cache so a large teacher and a trainable student fit on one GPU separately.

Examples:

  # Full run (cache 1000 samples, then 200 training steps)
  python tools/run_kd_distillation.py \\
    --teacher Qwen/Qwen3-30B-A3B-Base \\
    --student Qwen/Qwen3-30B-A3B-Base \\
    --num-samples 1000 --steps 200 --batch-size 2

  # Reuse an existing cache file
  python tools/run_kd_distillation.py \\
    --teacher Qwen/Qwen3-30B-A3B-Base \\
    --student ./my-student-checkpoint \\
    --teacher-cache runs/.../teacher-topk-logits.pt \\
    --train-only

  # Cache only (share the .pt across multiple student runs)
  python tools/run_kd_distillation.py \\
    --teacher sartajbhuvaji/bonsai-qwen3-30b-a3b-teacher-corrected-lora \\
    --num-samples 6000 --cache-only

Needs: GPU, `pip install bitsandbytes`, HF token for gated datasets/models.

Flags (see also `--help`):

  Models & mode:
    --teacher          Hugging Face repo id or local path (teacher)
    --student          Hugging Face repo id or local path (student to train)
    --teacher-cache    Reuse an existing top-k logits .pt; skip teacher forward
    --cache-only       Only run the caching pass
    --train-only       Only train; requires --teacher-cache and --student

  Data (caching pass):
    --num-samples      How many sequences to draw for KD (default 6000)
    --seed             Shuffle seed for the sample draw (default 11)
    --top-k            Teacher top-k support per token (default 20)
    --dataset-id       Hub dataset repo (default bigcode/the-stack-dedup)
    --data-dir         Subdir within that dataset (default data/typescript)

  Training (student pass):
    --steps            Optimizer steps (default 500)
    --batch-size       Samples per step (default 4)
    --peak-lr          Peak learning rate (default 2e-4)
    --min-lr           Cosine floor learning rate (default 2e-5)
    --warmup-steps     Linear warmup length (default 75)
    --val-fraction     Holdout fraction of cached samples (default 0.05)
    --val-every        Run val KD every N steps (default 10)

  Output:
    --run-name         Prefix for runs/<timestamp>-<suffix>-<name>/ (auto if omitted)
    --push-to-hub      Push distilled checkpoint to this private HF repo id
    --hf-token         Hugging Face token (or set HF_TOKEN / HUGGING_FACE_HUB_TOKEN)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bonsai.training import TrainConfig
from bonsai.training.kd_run import cache_teacher_logit_file, distill_student, repo_slug

_FLAG_REFERENCE = """
Flag reference
--------------

Models & mode:
  --teacher, --student, --teacher-cache, --cache-only, --train-only

Data:
  --num-samples, --seed, --top-k, --dataset-id, --data-dir

Training:
  --steps, --batch-size, --peak-lr, --min-lr, --warmup-steps,
  --val-fraction, --val-every, --weight-decay

Output:
  --run-name, --push-to-hub, --hf-token
"""

DEFAULT_DATASET = "bigcode/the-stack-dedup"
DEFAULT_DATA_DIR = "data/typescript"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_FLAG_REFERENCE,
    )
    parser.add_argument(
        "--teacher",
        help="Hugging Face repo id or local path for the teacher model (required except --train-only)",
    )
    parser.add_argument(
        "--student",
        help="Hugging Face repo id or local path for the student to train",
    )
    parser.add_argument(
        "--teacher-cache",
        type=Path,
        help="Existing .pt from a prior cache run; skips teacher forward if set",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Only build --teacher-cache; do not load the student",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only train; requires --teacher-cache (and --student)",
    )
    parser.add_argument("--run-name", help="Run directory name prefix (auto-generated if omitted)")

    parser.add_argument(
        "--num-samples", type=int, default=6000, help="Training/calibration pool size"
    )
    parser.add_argument("--seed", type=int, default=11, help="Shuffle seed for sample draw")
    parser.add_argument("--top-k", type=int, default=20, help="Teacher top-k support per token")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)

    parser.add_argument("--steps", type=int, default=500, help="Optimizer steps")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--peak-lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=75)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument(
        "--val-every", type=int, default=10, help="Validation KD loss every N steps"
    )

    parser.add_argument("--push-to-hub", help="Private HF repo id for the distilled student")
    parser.add_argument("--hf-token", default=None)

    args = parser.parse_args()
    if not args.train_only and not args.teacher:
        parser.error("--teacher is required unless --train-only")
    if args.cache_only and args.train_only:
        parser.error("use at most one of --cache-only and --train-only")
    if args.train_only:
        if not args.teacher_cache or not args.student:
            parser.error("--train-only requires --teacher-cache and --student")
    elif args.cache_only:
        if args.student:
            parser.error("--cache-only does not take --student")
    else:
        if not args.student:
            parser.error("--student is required unless --cache-only")

    cache_path = args.teacher_cache
    if not args.train_only:
        assert args.teacher is not None
        _, cache_path = cache_teacher_logit_file(
            args.teacher,
            num_samples=args.num_samples,
            seed=args.seed,
            top_k=args.top_k,
            dataset_id=args.dataset_id,
            data_dir=args.data_dir,
            hf_token=args.hf_token,
            run_name=args.run_name and f"{args.run_name}-cache",
        )

    if args.cache_only:
        print(f"cache ready: {cache_path}")
        return

    assert cache_path is not None
    assert args.student is not None
    train_config = TrainConfig(
        peak_lr=args.peak_lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        total_steps=args.steps,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
    )
    default_run = f"kd-{repo_slug(args.student)}-from-{repo_slug(args.teacher)}"
    distill_student(
        args.student,
        cache_path,
        run_name=args.run_name or default_run,
        config=train_config,
        val_fraction=args.val_fraction,
        val_every=args.val_every,
        push_to_hub=args.push_to_hub,
        hf_token=args.hf_token,
        teacher_repo_id=args.teacher,
    )


if __name__ == "__main__":
    main()
