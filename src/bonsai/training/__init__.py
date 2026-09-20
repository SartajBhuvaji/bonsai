"""Fine-tuning loop: teacher correction (plain LM loss) now, distillation later.

Phase 3.1 (teacher correction) needs a real training loop -- everything
built before this (`bonsai.importance`, `bonsai.pruning`) only ever does
forward passes. Phase 4.1 (distillation-based retraining) needs the same
loop with a different loss (forward-KL against a teacher's logits, per
both Minitron papers) -- `loss_fn` is the pluggable seam for that, same
spirit as `bonsai.importance.aggregate`'s pluggable aggregation function,
so this loop doesn't need rewriting once 4.1 starts, only a different
`loss_fn` passed in.

The LR schedule (linear warmup, then cosine decay to a floor) matches the
*shape* of 2408.11796's Table 4 recipe (peak/min LR, warmup steps, cosine
decay) -- not their literal numbers, which were tuned for 94-380B tokens
across 32 DGX H100 nodes, wildly disproportionate to what this project
can run. The actual peak/min LR, warmup, and token budget for this
project are an empirical question (2.4/3.1's still-open token-budget
question), answered by sweeping this config, not assumed from the paper.

Batching is gradient accumulation over single-sample forward/backward
passes (`train_step`), not a padded batched forward: this project's
samples vary widely in length, and every other forward-pass call site in
`bonsai` already processes one sample at a time
(`bonsai.importance.depth.mean_loss`, the pruning sweeps). Each sample's
per-token-mean loss is weighted equally in the accumulated gradient
regardless of its own length -- a standard, deliberate simplification,
not an oversight -- so `batch_size` here means accumulation count, not a
padded tensor's leading dimension.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

LossFn = Callable[[Qwen3MoeForCausalLM, torch.Tensor], torch.Tensor]


def _lm_loss(model: Qwen3MoeForCausalLM, input_ids: torch.Tensor) -> torch.Tensor:
    """Default `loss_fn`: plain next-token cross-entropy against the model's own labels."""
    return model(input_ids=input_ids, labels=input_ids).loss  # type: ignore[no-any-return]


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """One training run's hyperparameters -- the unit a sweep varies.

    `total_steps` is optimizer steps, not samples or tokens; the actual
    token count a run consumes is `total_steps * batch_size *` (each
    sample's own length), which varies run to run since sequences aren't
    padded to a fixed length.
    """

    peak_lr: float
    min_lr: float
    warmup_steps: int
    total_steps: int
    batch_size: int = 1
    weight_decay: float = 0.0


def lr_schedule(step: int, config: TrainConfig) -> float:
    """Linear warmup to `peak_lr`, then cosine decay to `min_lr` by `total_steps`.

    Matches the shape of 2408.11796's Table 4 recipe (peak/min LR, warmup
    steps, cosine decay), not its literal numbers -- see module docstring.
    """
    if config.warmup_steps > 0 and step < config.warmup_steps:
        return config.peak_lr * (step + 1) / config.warmup_steps
    span = max(1, config.total_steps - config.warmup_steps)
    progress = min(1.0, (step - config.warmup_steps) / span)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_lr + (config.peak_lr - config.min_lr) * cosine


def train_step(
    model: Qwen3MoeForCausalLM,
    optimizer: torch.optim.Optimizer,
    input_ids_batch: Sequence[torch.Tensor],
    *,
    loss_fn: LossFn = _lm_loss,
) -> float:
    """One optimizer step, gradient-accumulated over `input_ids_batch`.

    One forward/backward pass per sample (see module docstring for why),
    each contributing `loss / len(input_ids_batch)` to the accumulated
    gradient before a single `optimizer.step()`. Returns the batch's mean
    loss (unscaled, for logging -- the scaling above is only applied to
    the backward pass, not this return value).
    """
    if not input_ids_batch:
        raise ValueError("input_ids_batch must not be empty")
    optimizer.zero_grad()
    total = 0.0
    for input_ids in input_ids_batch:
        loss = loss_fn(model, input_ids)
        (loss / len(input_ids_batch)).backward()  # type: ignore[no-untyped-call]
        total += float(loss.detach())
    optimizer.step()
    return total / len(input_ids_batch)


def train(
    model: Qwen3MoeForCausalLM,
    train_input_ids: Sequence[torch.Tensor],
    val_input_ids: Sequence[torch.Tensor],
    config: TrainConfig,
    *,
    loss_fn: LossFn = _lm_loss,
    optimizer: torch.optim.Optimizer | None = None,
    val_every: int = 10,
    run_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Run `config.total_steps` optimizer steps, periodically checking validation loss.

    `train_input_ids` is consumed `batch_size` samples at a time, cycling
    (via modulo) if `total_steps * batch_size` exceeds its length --
    correct for a fixed pre-tokenized pool used across a short
    hyperparameter sweep, not a claim that a full production run
    shouldn't stream fresh, non-repeating data instead.

    `optimizer` defaults to plain `torch.optim.AdamW`, constructed here
    from `config.peak_lr`/`config.weight_decay` -- pass a pre-built
    optimizer instead (e.g. `bitsandbytes.optim.AdamW8bit`) when full-
    parameter fine-tuning on a large model needs it: plain AdamW's states
    are ~2x a parameter's own footprint on top of the parameter and its
    gradient (~4x total per parameter, in the same dtype as the model),
    confirmed to OOM a naive `device_map="auto"`-sharded 30B-parameter
    model across 8x40GB GPUs even with gradient checkpointing enabled --
    an 8-bit optimizer roughly halves that. This function still owns the
    LR schedule regardless of which optimizer is passed (`lr_schedule`
    writes into `optimizer.param_groups` every step either way).

    Validation loss (mean `loss_fn` over all of `val_input_ids`, no
    gradient) is computed every `val_every` steps and on the final step.
    Returns the full per-step history (also logged live via
    `bonsai.tracking.log_metric` if `run_dir` is given, so a long run's
    progress survives a crash instead of living only in the return value).
    """
    if not train_input_ids or not val_input_ids:
        raise ValueError("train_input_ids and val_input_ids must both be non-empty")

    from bonsai.tracking import log_metric

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.peak_lr, weight_decay=config.weight_decay
        )
    model.train()
    history: list[dict[str, Any]] = []
    n = len(train_input_ids)

    for step in range(config.total_steps):
        lr = lr_schedule(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr

        batch = [
            train_input_ids[(step * config.batch_size + i) % n] for i in range(config.batch_size)
        ]
        train_loss = train_step(model, optimizer, batch, loss_fn=loss_fn)

        record: dict[str, Any] = {"step": step, "lr": lr, "train_loss": train_loss}
        if (step + 1) % val_every == 0 or step == config.total_steps - 1:
            model.eval()  # type: ignore[no-untyped-call]
            with torch.no_grad():
                val_loss = sum(float(loss_fn(model, ids)) for ids in val_input_ids) / len(
                    val_input_ids
                )
            model.train()
            record["val_loss"] = val_loss

        history.append(record)
        if run_dir is not None:
            log_metric(run_dir, **record)

    model.eval()  # type: ignore[no-untyped-call]
    return history


__all__ = ["LossFn", "TrainConfig", "lr_schedule", "train", "train_step"]
