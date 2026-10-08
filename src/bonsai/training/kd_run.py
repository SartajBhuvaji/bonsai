"""Top-k logit caching and student KD training (no pruning required).

Used by `tools/run_kd_distillation.py` and the Phase 4 tool scripts.
Teacher and student are Hugging Face repo ids or local checkpoint paths.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import cast

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


DEFAULT_TEACHER_CACHE_FILENAME = "baseline-topk-logits.pt"


def resolve_teacher_cache_path(spec: str, *, hf_token: str | None) -> Path:
    """Local `.pt` path, or download from a private HF dataset repo id (`org/name`)."""
    path = Path(spec)
    if path.is_file():
        return path
    if path.exists() and not path.is_dir():
        raise FileNotFoundError(path)
    if path.exists() and path.is_dir():
        candidate = path / DEFAULT_TEACHER_CACHE_FILENAME
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"no {DEFAULT_TEACHER_CACHE_FILENAME} under {path}")

    if "/" in spec and not spec.startswith("runs/"):
        from huggingface_hub import hf_hub_download

        _log(f"teacher cache not local; downloading from dataset {spec} ...")
        return Path(
            hf_hub_download(
                spec,
                DEFAULT_TEACHER_CACHE_FILENAME,
                repo_type="dataset",
                token=hf_token,
            )
        )

    raise FileNotFoundError(f"teacher cache not found: {spec}")


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
    _log(f"  {len(tokenized.input_ids)} sample(s), mean length {tokenized.mean_length:.0f} tokens")

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


def push_teacher_cache_to_hub(
    cache_path: Path,
    repo_id: str,
    *,
    hf_token: str | None,
) -> None:
    """Upload a cached top-k logits file to a private HF dataset repo."""
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(cache_path),
        path_in_repo=cache_path.name,
        repo_id=repo_id,
        repo_type="dataset",
    )
    _log(f"pushed cache to https://huggingface.co/datasets/{repo_id}/{cache_path.name} (private)")


def distill_student(
    student_repo_id: str,
    teacher_cache: Path,
    *,
    run_name: str,
    config: TrainConfig,
    val_fraction: float = 0.05,
    val_every: int = 10,
    push_to_hub: str | None = None,
    push_to_hub_best: str | None = None,
    hf_token: str | None = None,
    teacher_repo_id: str | None = None,
) -> Path:
    """Train `student_repo_id` against a cached teacher; returns `run_dir`."""
    from dataclasses import asdict

    if push_to_hub_best is None and push_to_hub is not None:
        push_to_hub_best = f"{push_to_hub}-best"

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
    # transformers>=5.19 added a real type stub for gradient_checkpointing_enable
    # (no longer needs an ignore); enable_input_require_grads is still untyped.
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # type: ignore[no-untyped-call]
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
    best_val = float("inf")
    best_step = -1
    best_dir = run_dir / "checkpoints" / "distilled-best"

    def on_val(step: int, val_loss: float) -> None:
        nonlocal best_val, best_step
        if val_loss >= best_val:
            return
        best_val = val_loss
        best_step = step
        best_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(best_dir)
        tokenizer.save_pretrained(best_dir)
        _log(f"new best val_loss={val_loss:.4f} at step {step} -> {best_dir}")

    history = train(
        model,
        train_ids,
        val_ids,
        config,
        loss_fn=loss_fn,
        optimizer=optimizer,
        val_every=val_every,
        run_dir=run_dir,
        on_val=on_val,
    )
    _log(f"final train_loss={history[-1]['train_loss']:.4f}")
    if "val_loss" in history[-1]:
        _log(f"final val_loss={history[-1]['val_loss']:.4f}")
    if best_step >= 0:
        _log(f"best val_loss={best_val:.4f} at step {best_step}")
        (run_dir / "best_checkpoint.json").write_text(
            json.dumps(
                {"best_step": best_step, "best_val_loss": best_val, "path": str(best_dir)},
                indent=2,
            )
            + "\n"
        )

    checkpoint_dir = run_dir / "checkpoints" / "distilled"
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    _log(f"saved final weights to {checkpoint_dir}")

    if push_to_hub:
        # transformers.PreTrainedModel.push_to_hub is `@wraps(PushToHubMixin.push_to_hub)`
        # over a `(self, *args, **kwargs)` redefinition (modeling_utils.py) -- typeshed's
        # `functools.wraps` stub carries over the *original*, class-accessed function's own
        # `self` parameter, so mypy double-counts it and checks this call's first argument
        # against `PreTrainedModel` instead of `str`. Confirmed by reading transformers'
        # source directly, not guessed; a real call to a real, correctly-typed method.
        model.push_to_hub(push_to_hub, token=hf_token, private=True)  # type: ignore[arg-type]
        tokenizer.push_to_hub(push_to_hub, token=hf_token, private=True)
        _log(f"pushed final to https://huggingface.co/{push_to_hub} (private)")

    if push_to_hub_best and best_dir.is_dir() and any(best_dir.iterdir()):
        best_model = AutoModelForCausalLM.from_pretrained(
            best_dir, dtype=torch.bfloat16, device_map="cpu", token=hf_token
        )
        best_tok = AutoTokenizer.from_pretrained(best_dir, token=hf_token)
        best_model.push_to_hub(push_to_hub_best, token=hf_token, private=True)  # type: ignore[arg-type]
        best_tok.push_to_hub(push_to_hub_best, token=hf_token, private=True)
        _log(f"pushed best-val to https://huggingface.co/{push_to_hub_best} (private)")
        del best_model

    return run_dir


__all__ = [
    "cache_teacher_logit_file",
    "distill_student",
    "load_teacher_cache",
    "push_teacher_cache_to_hub",
    "repo_slug",
    "resolve_teacher_cache_path",
]
