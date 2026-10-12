"""Cache one teacher's top-k logits over a fixed sample set -- inference-only.

Phase 4 preset teachers (`--teacher baseline|lora-corrected`) or any HF repo via
`--teacher-model`. For cache+train in one command see `tools/run_kd_distillation.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bonsai.training.kd_run import cache_teacher_logit_file, push_teacher_cache_to_hub

PRESET_TEACHERS = {
    "baseline": "Qwen/Qwen3-30B-A3B-Base",
    "lora-corrected": "sartajbhuvaji/bonsai-qwen3-30b-a3b-teacher-corrected-lora",
}

DATASET_ID = "bigcode/the-stack-dedup"
DATA_DIR = "data/typescript"
SEED = 11
TOP_K = 20


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--teacher",
        choices=list(PRESET_TEACHERS),
        help="project preset alias",
    )
    group.add_argument(
        "--teacher-model",
        help="any Hugging Face repo id or local path (overrides --teacher presets)",
    )
    parser.add_argument("--num-samples", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument(
        "--sample-mode",
        choices=["file", "function"],
        default="file",
        help=(
            "'file' (default): raw whole-file continuations, truncated to max_length. "
            "'function': one complete, bounded TypeScript function per sample "
            "(bonsai.data.functions) -- matches HumanEval-TS's shape instead of "
            "teaching arbitrary continuation."
        ),
    )
    parser.add_argument("--hf-token", default=None)
    parser.add_argument(
        "--push-to-hub",
        default=None,
        help="HF dataset repo id (uploads the .pt as baseline-topk-logits.pt)",
    )
    parser.add_argument(
        "--hub-public",
        action="store_true",
        help="Create/push the dataset repo as public (default: private).",
    )
    args = parser.parse_args()

    repo_id = args.teacher_model or PRESET_TEACHERS[args.teacher]
    run_name = None
    if args.teacher and not args.teacher_model:
        run_name = f"cache-teacher-logits-{args.teacher}"

    _run_dir, cache_path = cache_teacher_logit_file(
        repo_id,
        num_samples=args.num_samples,
        seed=args.seed,
        top_k=args.top_k,
        dataset_id=args.dataset_id,
        data_dir=args.data_dir,
        hf_token=args.hf_token,
        run_name=run_name,
        sample_mode=args.sample_mode,
    )
    if args.push_to_hub:
        push_teacher_cache_to_hub(
            Path(cache_path),
            args.push_to_hub,
            hf_token=args.hf_token,
            hub_public=args.hub_public,
        )


if __name__ == "__main__":
    main()
