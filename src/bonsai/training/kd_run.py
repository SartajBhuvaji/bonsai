"""Top-k logit caching and student KD training (no pruning required).

Used by `tools/run_kd_distillation.py` and the Phase 4 tool scripts.
Teacher and student are Hugging Face repo ids or local checkpoint paths.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

import bitsandbytes as bnb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

from bonsai.config import DataConfig
from bonsai.data import list_shards, load_samples, make_split, tokenize_samples
from bonsai.data.functions import load_function_samples
from bonsai.tracking import log_metric, new_run, save_config_snapshot
from bonsai.training import TrainConfig, train
from bonsai.training.distillation import make_kd_loss_fn

DEFAULT_LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


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
    sample_mode: Literal["file", "function"] = "file",
) -> tuple[Path, Path]:
    """Inference-only teacher forward; returns `(run_dir, cache_path)`.

    `sample_mode="function"` draws `bonsai.data.functions.load_function_samples`
    (one complete, bounded TypeScript function per sample) instead of raw
    whole-file continuations -- see that module's docstring for why: KD
    against raw-file continuations teaches a student to continue arbitrary
    code, never to finish one function and stop, which is the exact shape
    HumanEval-TS evaluates.
    """
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
            "sample_mode": sample_mode,
        },
    )

    _log(f"Listing shards for {dataset_id} / {data_dir} ...")
    shards = list_shards(dataset_id, data_dir, token=hf_token)
    _log(f"  {len(shards)} shard(s) listed")
    split = make_split(DataConfig(dataset_id=dataset_id, data_dir=data_dir), shards)
    _log(f"Streaming {num_samples} retraining samples (seed={seed}, mode={sample_mode}) ...")
    if sample_mode == "function":
        raw = load_function_samples(split, "retraining", n=num_samples, seed=seed, token=hf_token)
    else:
        raw = load_samples(split, "retraining", n=num_samples, seed=seed, token=hf_token)
    _log(f"  loaded {len(raw)} raw sample(s)")
    if sample_mode == "function":
        samples_path = run_dir / "checkpoints" / "function-samples.jsonl"
        samples_path.parent.mkdir(parents=True, exist_ok=True)
        with samples_path.open("w", encoding="utf-8") as f:
            for i, row in enumerate(raw):
                f.write(
                    json.dumps({"index": i, "content": row["content"]}, ensure_ascii=False) + "\n"
                )
        _log(f"  wrote {len(raw)} function sample(s) to {samples_path}")

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
    hub_private: bool = True,
    use_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Sequence[str] = DEFAULT_LORA_TARGET_MODULES,
) -> Path:
    """Train `student_repo_id` against a cached teacher; returns `run_dir`.

    `use_lora=True` freezes the student and trains a small attention-only
    LoRA adapter instead of every weight -- the same recipe plan.md 3.4
    already validated for *teacher* correction, applied here to the
    *student's* distillation step instead. Motivated by a real, measured
    finding (plan.md, 2026-10-09): a pruned-but-undistilled candidate
    scores a real 0.475 pass@1 on HumanEval-TS zero-shot, but the exact
    same candidate after 3000 steps of full-parameter KD scores 0.0 with
    total repetition collapse -- full-parameter distillation on a narrow,
    un-bounded-completion training pool (raw file continuations, never an
    isolated spec-to-function example) overwrites a real capability the
    pruned model already had. Constraining the update via LoRA is the
    direct test of whether that overwriting, not the distillation signal
    itself, is the cause.

    Both the final and best-val checkpoints are saved as real, standalone
    merged models regardless of `use_lora` -- `push_to_hub`/
    `push_to_hub_best` always receive a plain model that loads with
    `AutoModelForCausalLM.from_pretrained` alone, no PEFT-specific
    handling needed downstream (matching the released teacher-correction
    checkpoint's own convention). The *intermediate* best-val checkpoint
    saved mid-training is adapter-only when `use_lora` (cheap, no merge
    mid-training -- merging destructively collapses the adapter into the
    base, which would corrupt the model still being trained); it's merged
    into a full model only once, after training ends.
    """
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
            "use_lora": use_lora,
            "lora_r": lora_r if use_lora else None,
            "lora_alpha": lora_alpha if use_lora else None,
            "lora_dropout": lora_dropout if use_lora else None,
            "lora_target_modules": list(lora_target_modules) if use_lora else None,
            **asdict(config),
        },
    )

    _log(f"Loading student {student_repo_id} ...")
    base_model = cast(
        Qwen3MoeForCausalLM,
        AutoModelForCausalLM.from_pretrained(
            student_repo_id, dtype=torch.bfloat16, device_map="auto", token=hf_token
        ),
    )
    model: PreTrainedModel = base_model
    if use_lora:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(lora_target_modules),
            bias="none",
        )
        model = cast(PreTrainedModel, get_peft_model(base_model, lora_config))
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        _log(f"LoRA: {trainable:,} / {total:,} trainable ({trainable / total:.3%})")

    # transformers>=5.19 added a real type stub for gradient_checkpointing_enable
    # (no longer needs an ignore); enable_input_require_grads is still untyped.
    # Order matters for the LoRA case specifically: this has to come *after*
    # get_peft_model, not before -- plan.md 3.3's own real bug (frozen base +
    # gradient checkpointing silently zeroes adapter gradients without this
    # being called on the already-wrapped model).
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
    optimizer = bnb.optim.AdamW8bit(
        (p for p in model.parameters() if p.requires_grad), lr=config.peak_lr
    )
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
        # Adapter-only when use_lora (PeftModel.save_pretrained's own default) --
        # cheap, and merging mid-training would destructively collapse the
        # adapter into the base model still being trained. Merged into a real,
        # standalone model only once, after training ends, below.
        model.save_pretrained(best_dir)
        if not use_lora:
            tokenizer.save_pretrained(best_dir)
        _log(f"new best val_loss={val_loss:.4f} at step {step} -> {best_dir}")

    history = train(
        # train()/train_step() only ever call .train()/.eval()/__call__ on this --
        # a LoRA-wrapped model satisfies that structurally even though it isn't
        # nominally a Qwen3MoeForCausalLM.
        cast(Qwen3MoeForCausalLM, model),
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

    if use_lora:
        # CPU, not GPU: confirmed by a real OOM during teacher correction
        # (plan.md 3.3) that merge_and_unload's tensor-chunking needs more
        # headroom than a mostly-full GPU has left.
        _log("merging final LoRA adapter into base weights (CPU) ...")
        del optimizer
        torch.cuda.empty_cache()
        # PreTrainedModel.to is `@wraps(torch.nn.Module.to)` -- same mypy/typeshed
        # self-double-counting false positive as push_to_hub above, confirmed by
        # reading transformers' source directly.
        final_model: PreTrainedModel = model.to("cpu").merge_and_unload()  # type: ignore[arg-type]
    else:
        final_model = model

    checkpoint_dir = run_dir / "checkpoints" / "distilled"
    final_model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    _log(f"saved final weights to {checkpoint_dir}")

    if push_to_hub:
        # transformers.PreTrainedModel.push_to_hub is `@wraps(PushToHubMixin.push_to_hub)`
        # over a `(self, *args, **kwargs)` redefinition (modeling_utils.py) -- typeshed's
        # `functools.wraps` stub carries over the *original*, class-accessed function's own
        # `self` parameter, so mypy double-counts it and checks this call's first argument
        # against `PreTrainedModel` instead of `str`. Confirmed by reading transformers'
        # source directly, not guessed; a real call to a real, correctly-typed method.
        final_model.push_to_hub(push_to_hub, token=hf_token, private=hub_private)  # type: ignore[arg-type]
        tokenizer.push_to_hub(push_to_hub, token=hf_token, private=hub_private)
        vis = "private" if hub_private else "public"
        _log(f"pushed final to https://huggingface.co/{push_to_hub} ({vis})")

    if push_to_hub_best and best_dir.is_dir() and any(best_dir.iterdir()):
        if use_lora:
            from peft import PeftModel

            _log("merging best-val LoRA adapter into a fresh base copy (CPU) ...")
            fresh_base = AutoModelForCausalLM.from_pretrained(
                student_repo_id, dtype=torch.bfloat16, device_map="cpu", token=hf_token
            )
            best_peft = PeftModel.from_pretrained(fresh_base, best_dir)
            best_model: PreTrainedModel = best_peft.merge_and_unload()
        else:
            best_model = AutoModelForCausalLM.from_pretrained(
                best_dir, dtype=torch.bfloat16, device_map="cpu", token=hf_token
            )
        best_tok = AutoTokenizer.from_pretrained(student_repo_id, token=hf_token)
        best_model.push_to_hub(push_to_hub_best, token=hf_token, private=hub_private)  # type: ignore[arg-type]
        best_tok.push_to_hub(push_to_hub_best, token=hf_token, private=hub_private)
        vis = "private" if hub_private else "public"
        _log(f"pushed best-val to https://huggingface.co/{push_to_hub_best} ({vis})")
        del best_model

    return run_dir


__all__ = [
    "DEFAULT_LORA_TARGET_MODULES",
    "cache_teacher_logit_file",
    "distill_student",
    "load_teacher_cache",
    "push_teacher_cache_to_hub",
    "repo_slug",
    "resolve_teacher_cache_path",
]
