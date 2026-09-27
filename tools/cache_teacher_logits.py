"""Cache one teacher's top-k logits over a fixed sample set -- once, inference-only.

Phase 4.1 (plan.md, decided 2026-09-21): forward-KL distillation needs the
teacher's output distribution for every training token, but holding a
~61GB teacher resident *during* a pruned candidate's training would push
total memory past what a single 80GB card has room for once the
candidate's own weights/gradients/optimizer state are also resident.
Caching the teacher's logits once, up front, removes the teacher from
memory entirely by the time student training starts
(`bonsai.training.distillation.make_kd_loss_fn` only ever reads this
cache back).

Run this once per teacher -- baseline (uncorrected) and LoRA-corrected --
against the *same* sample set (same seed), so both caches line up
sample-for-sample for a fair, matched-data ablation between them
(plan.md's "Approaches..." handoff, 2026-09-27: the actual open question
is whether the corrected teacher helps or hurts a real distilled student,
which this fair pairing is what makes answerable).

Needs a real GPU: the teacher (~61GB bf16) has to be loaded and run
forward, even though nothing about this pass needs gradients.
"""

from __future__ import annotations

import argparse
from typing import cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

from bonsai.config import DataConfig
from bonsai.data import list_shards, load_samples, make_split, tokenize_samples
from bonsai.tracking import log_metric, new_run, save_config_snapshot

DATASET_ID = "bigcode/the-stack-dedup"
DATA_DIR = "data/typescript"
TOP_K = 20
# Disjoint from every prior draw this project made against the retraining pool
# (3.1's 6,000-sample correction run, 3.4's re-filtered correction run) --
# a fresh seed, not a guarantee checked against those runs' own consumed
# hexshas (never persisted), same caveat plan.md 4.1 already logs.
SEED = 11


def cache_teacher_logits(
    model: Qwen3MoeForCausalLM,
    samples: list[torch.Tensor],
    *,
    top_k: int = TOP_K,
) -> list[dict[str, torch.Tensor]]:
    """Run `model` forward over each sample, no grad, returning per-sample
    `{"input_ids": ..., "topk_ids": ..., "topk_logprobs": ...}`.

    `topk_logprobs` is renormalized over just the top-`k` support (matches
    `bonsai.training.distillation.make_kd_loss_fn`'s expected input format
    exactly -- it does not renormalize a second time).
    """
    model.eval()  # type: ignore[no-untyped-call]
    cached = []
    with torch.no_grad():
        for input_ids in samples:
            logits = model(input_ids=input_ids).logits[0]  # [seq_len, vocab]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            topk_logprobs, topk_ids = torch.topk(log_probs, top_k, dim=-1)
            topk_logprobs = topk_logprobs - torch.logsumexp(topk_logprobs, dim=-1, keepdim=True)
            cached.append(
                {
                    "input_ids": input_ids[0].cpu(),
                    "topk_ids": topk_ids.cpu(),
                    "topk_logprobs": topk_logprobs.to(torch.float16).cpu(),
                }
            )
    return cached


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher",
        required=True,
        choices=["baseline", "lora-corrected"],
        help="which teacher to cache logits for",
    )
    parser.add_argument("--num-samples", type=int, default=6000)
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    repo_id = {
        "baseline": "Qwen/Qwen3-30B-A3B-Base",
        "lora-corrected": "sartajbhuvaji/bonsai-qwen3-30b-a3b-teacher-corrected-lora",
    }[args.teacher]

    run_dir = new_run(f"cache-teacher-logits-{args.teacher}")
    save_config_snapshot(
        run_dir,
        {"teacher": args.teacher, "repo_id": repo_id, "seed": SEED, "top_k": TOP_K},
    )

    print(f"Loading {repo_id} ...")
    model = cast(
        Qwen3MoeForCausalLM,
        AutoModelForCausalLM.from_pretrained(repo_id, dtype=torch.bfloat16, device_map="auto"),
    )
    tokenizer = AutoTokenizer.from_pretrained(repo_id)

    shards = list_shards(DATASET_ID, DATA_DIR, token=args.hf_token)
    base_data = DataConfig(dataset_id=DATASET_ID, data_dir=DATA_DIR)
    split = make_split(base_data, shards)
    samples = load_samples(split, "retraining", n=args.num_samples, seed=SEED, token=args.hf_token)
    tokenized = tokenize_samples(samples, tokenizer)
    input_ids_batches = [torch.tensor([ids]) for ids in tokenized.input_ids]
    print(f"{len(input_ids_batches)} samples, mean length {tokenized.mean_length:.0f} tokens")

    cached = cache_teacher_logits(model, input_ids_batches, top_k=TOP_K)
    log_metric(run_dir, num_cached=len(cached))

    out_path = run_dir / "checkpoints" / f"{args.teacher}-topk-logits.pt"
    torch.save({"seed": SEED, "top_k": TOP_K, "samples": cached}, out_path)
    print(f"saved {len(cached)} cached samples to {out_path}")


if __name__ == "__main__":
    main()
