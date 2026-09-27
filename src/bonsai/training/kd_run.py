"""Top-k logit caching and student KD training (no pruning required).

Used by `tools/run_kd_distillation.py` and the Phase 4 tool scripts.
Teacher and student are Hugging Face repo ids or local checkpoint paths.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

import bitsandbytes as bnb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

from bonsai.config import DataConfig
from bonsai.data import list_shards, load_samples, make_split, tokenize_samples
from bonsai.tracking import log_metric, new_run, save_config_snapshot
from bonsai.training import TrainConfig, train
from bonsai.training.distillation import make_kd_loss_fn


def _log(msg: str) -> None:
    print(msg, flush=True)


def repo_slug(repo_id: str) -> str:
    return repo_id.replace("/", "--")


def load_teacher_cache(
    path: Path, *, device: torch.device
) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor]], list[torch.Tensor]]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    input_ids_list: list[torch.Tensor] = []
    cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for sample in data["samples"]:
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        input_ids_list.append(input_ids)
        cache[id(input_ids)] = (
            sample["topk_ids"].to(device),
            sample["topk_logprobs"].float().to(device),
        )
    return cache, input_ids_list


def cache_teacher_logit_file(
    teacher_repo_id: str,
    *,
    num_samples: int,
    seed: int,
    top_k: int,
    dataset_id: str,
    data_dir: str,
    hf_token: str | None,
    run_name: str | None = None,
) -> tuple[Path, Path]:
    """Inference-only teacher forward; returns `(run_dir, cache_path)`."""
    run_dir = new_run(run_name or f"cache-teacher-logits-{repo_slug(teacher_repo_id)}")
    save_config_snapshot(
        run_dir,
        {
            "teacher_repo_id": teacher_repo_id,
            "num_samples": num_samples,
            "seed": seed,
            "top_k": top_k,
            "dataset_id": dataset_id,
            "data_dir": data_dir,
        },
    )

    _log(f"Listing shards for {dataset_id} / {data_dir} ...")
    shards = list_shards(dataset_id, data_dir, token=hf_token)
    _log(f"  {len(shards)} shard(s) listed")
    split = make_split(DataConfig(dataset_id=dataset_id, data_dir=data_dir), shards)
    _log(f"Streaming {num_samples} retraining samples (seed={seed}) ...")
    raw = load_samples(split, "retraining", n=num_samples, seed=seed, token=hf_token)
    _log(f"  loaded {len(raw)} raw sample(s)")

    _log(f"Loading tokenizer for {teacher_repo_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(teacher_repo_id, token=hf_token)
    _log("Tokenizing ...")
    tokenized = tokenize_samples(raw, tokenizer)
    _log(
        f"  {len(tokenized.input_ids)} sample(s), mean length {tokenized.mean_length:.0f} tokens"
    )

    _log(f"Loading teacher {teacher_repo_id} ...")
    model = cast(
        Qwen3MoeForCausalLM,
        AutoModelForCausalLM.from_pretrained(
            teacher_repo_id, dtype=torch.bfloat16, device_map="auto", token=hf_token
        ),
    )
    device = next(model.parameters()).device
    batches = [torch.tensor([ids], device=device) for ids in tokenized.input_ids]
    _log(f"Running teacher forward (top_k={top_k}) on {device} ...")

    model.eval()  # type: ignore[no-untyped-call]
    cached: list[dict[str, torch.Tensor]] = []
    n = len(batches)
    t0 = time.monotonic()
    log_every = min(50, max(1, num_samples // 20))
    with torch.no_grad():
        for i, input_ids in enumerate(batches):
            logits = model(input_ids=input_ids).logits[0]
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
            done = i + 1
            if done == 1 or done == n or (log_every > 0 and done % log_every == 0):
                elapsed = time.monotonic() - t0
                rate = done / elapsed if elapsed > 0 else 0.0
                remaining = (n - done) / rate if rate > 0 else 0.0
                _log(
                    f"  forward progress: {done}/{n} samples "
                    f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)"
                )

    log_metric(run_dir, num_cached=len(cached))
    out_path = run_dir / "checkpoints" / f"{repo_slug(teacher_repo_id)}-topk-logits.pt"
    _log(f"Writing cache to {out_path} ...")
    torch.save(
        {
            "teacher_repo_id": teacher_repo_id,
            "seed": seed,
            "top_k": top_k,
            "dataset_id": dataset_id,
            "data_dir": data_dir,
            "samples": cached,
        },
        out_path,
    )
    _log(f"saved {len(cached)} cached samples to {out_path}")
    del model
    torch.cuda.empty_cache()
    return run_dir, out_path


def distill_student(
    student_repo_id: str,
    teacher_cache: Path,
    *,
    run_name: str,
    config: TrainConfig,
    val_fraction: float = 0.05,
    val_every: int = 10,
    push_to_hub: str | None = None,
    hf_token: str | None = None,
    teacher_repo_id: str | None = None,
) -> Path:
    """Train `student_repo_id` against a cached teacher; returns `run_dir`."""
    from dataclasses import asdict

    run_dir = new_run(run_name)
    save_config_snapshot(
        run_dir,
        {
            "student": student_repo_id,
            "teacher_cache": str(teacher_cache),
            "teacher_repo_id": teacher_repo_id,
            "val_fraction": val_fraction,
            "val_every": val_every,
            **asdict(config),
        },
    )

    _log(f"Loading student {student_repo_id} ...")
    model = cast(
        Qwen3MoeForCausalLM,
        AutoModelForCausalLM.from_pretrained(
            student_repo_id, dtype=torch.bfloat16, device_map="auto", token=hf_token
        ),
    )
    model.gradient_checkpointing_enable()  # type: ignore[no-untyped-call]
    model.enable_input_require_grads()
    tokenizer = AutoTokenizer.from_pretrained(student_repo_id, token=hf_token)

    device = next(model.parameters()).device
    cache, all_input_ids = load_teacher_cache(teacher_cache, device=device)
    split_at = max(1, int(len(all_input_ids) * (1 - val_fraction)))
    train_ids, val_ids = all_input_ids[:split_at], all_input_ids[split_at:]
    _log(
        f"{len(train_ids)} train samples, {len(val_ids)} val samples "
        f"(metrics: {run_dir}/metrics.jsonl)"
    )

    loss_fn = make_kd_loss_fn(cache)
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=config.peak_lr)
    history = train(
        model,
        train_ids,
        val_ids,
        config,
        loss_fn=loss_fn,
        optimizer=optimizer,
        val_every=val_every,
        run_dir=run_dir,
    )
    _log(f"final train_loss={history[-1]['train_loss']:.4f}")
    if "val_loss" in history[-1]:
        _log(f"final val_loss={history[-1]['val_loss']:.4f}")

    checkpoint_dir = run_dir / "checkpoints" / "distilled"
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    _log(f"saved to {checkpoint_dir}")

    if push_to_hub:
        model.push_to_hub(push_to_hub, token=hf_token, private=True)
        tokenizer.push_to_hub(push_to_hub, token=hf_token, private=True)
        _log(f"pushed to https://huggingface.co/{push_to_hub} (private)")

    return run_dir


__all__ = [
    "cache_teacher_logit_file",
    "distill_student",
    "load_teacher_cache",
    "repo_slug",
]
