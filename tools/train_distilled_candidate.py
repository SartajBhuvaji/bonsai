"""Train the pruned candidate against one cached teacher's logits.

Phase 4.1's actual ablation (plan.md, "Approaches..." handoff, 2026-09-27):
run this twice, once per `--teacher-cache` (baseline and LoRA-corrected),
identical hyperparameters and identical starting candidate both times
(same `tools/build_candidate.py` checkpoint), then compare. That
comparison -- not either run alone -- is what answers whether teacher
correction actually helps or hurts a real distilled student, the question
the whole teacher-correction investigation (secs. teacher-correction /
teacher-correction-lora in the paper) exists to resolve.

Uses `bonsai.training.train`/`train_step` unchanged, per their own
docstring's design intent, with `bonsai.training.distillation.make_kd_loss_fn`
as the `loss_fn` -- the teacher itself is never loaded here, only its
cached top-k logits (`tools/cache_teacher_logits.py`'s output), which is
what lets the pruned candidate's full training footprint fit a single
80GB card without a ~61GB teacher also resident.

8-bit AdamW (bitsandbytes), same reason as the full-parameter teacher
correction run (3.1): plain AdamW's optimizer state roughly doubles a
parameter's own footprint on top of the parameter and its gradient.

Needs a real GPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import bitsandbytes as bnb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

from bonsai.tracking import new_run, save_config_snapshot
from bonsai.training import TrainConfig, train
from bonsai.training.distillation import make_kd_loss_fn

# Matches 3.1's real correction run's shape (linear warmup, cosine decay) --
# not its literal numbers, which were tuned for full-parameter fine-tuning
# of the *teacher*, a different model size and a different loss. This
# candidate is far smaller (~8.4B vs ~30.5B), so its own peak LR is a
# starting point to sanity-check via the same kind of short smoke test
# every other real run in this project did first, not assumed correct.
DEFAULT_CONFIG = TrainConfig(
    peak_lr=2e-4,
    min_lr=2e-5,
    warmup_steps=75,
    total_steps=500,
    batch_size=4,
)


def load_teacher_cache(path: Path) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Load `tools/cache_teacher_logits.py`'s output, rebuilding the
    `id(input_ids) -> (topk_ids, topk_logprobs)` mapping `make_kd_loss_fn`
    expects, paired with the exact `input_ids` list `train()` should be
    called with (same tensor objects, so the `id()`-keyed lookup matches)."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    input_ids_list = []
    cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for sample in data["samples"]:
        input_ids = sample["input_ids"].unsqueeze(0)  # [1, seq_len], matches train()'s convention
        input_ids_list.append(input_ids)
        cache[id(input_ids)] = (sample["topk_ids"], sample["topk_logprobs"].float())
    return cache, input_ids_list  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate", required=True, help="path or HF repo id of the pruned candidate"
    )
    parser.add_argument(
        "--teacher-cache", required=True, help="path to a tools/cache_teacher_logits.py .pt output"
    )
    parser.add_argument("--run-name", required=True, help="e.g. distill-aggressive-vs-baseline")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--push-to-hub", default=None)
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    run_dir = new_run(args.run_name)
    save_config_snapshot(
        run_dir,
        {"candidate": args.candidate, "teacher_cache": args.teacher_cache, **vars(DEFAULT_CONFIG)},
    )

    print(f"Loading candidate {args.candidate} ...")
    model = cast(
        Qwen3MoeForCausalLM,
        AutoModelForCausalLM.from_pretrained(
            args.candidate, dtype=torch.bfloat16, device_map="auto"
        ),
    )
    model.gradient_checkpointing_enable()  # type: ignore[no-untyped-call]
    # Cheap insurance, not a confirmed requirement this time: 3.3 found this
    # actually necessary for a *frozen* base model with LoRA adapters (grad
    # checkpointing's recomputed segments had no requires_grad=True input to
    # propagate through). Every parameter here is trainable instead, so
    # embeddings already carry requires_grad -- but the call is a no-op when
    # unneeded, and cheaper to keep than to re-verify which case this is.
    model.enable_input_require_grads()
    tokenizer = AutoTokenizer.from_pretrained(args.candidate)

    cache, all_input_ids = load_teacher_cache(Path(args.teacher_cache))
    split_at = max(1, int(len(all_input_ids) * (1 - args.val_fraction)))
    train_input_ids, val_input_ids = all_input_ids[:split_at], all_input_ids[split_at:]
    print(f"{len(train_input_ids)} train samples, {len(val_input_ids)} val samples")

    loss_fn = make_kd_loss_fn(cache)
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=DEFAULT_CONFIG.peak_lr)

    history = train(
        model,
        train_input_ids,
        val_input_ids,
        DEFAULT_CONFIG,
        loss_fn=loss_fn,
        optimizer=optimizer,
        run_dir=run_dir,
    )
    print(f"final train_loss={history[-1]['train_loss']:.4f}")
    if "val_loss" in history[-1]:
        print(f"final val_loss={history[-1]['val_loss']:.4f}")

    checkpoint_dir = run_dir / "checkpoints" / "distilled"
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    print(f"saved to {checkpoint_dir}")

    if args.push_to_hub:
        # Private per plan.md 4.0's decided checkpoint-persistence policy.
        model.push_to_hub(args.push_to_hub, token=args.hf_token, private=True)
        tokenizer.push_to_hub(args.push_to_hub, token=args.hf_token, private=True)
        print(f"pushed to https://huggingface.co/{args.push_to_hub} (private)")


if __name__ == "__main__":
    main()
