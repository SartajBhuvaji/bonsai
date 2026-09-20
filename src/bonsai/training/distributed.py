"""FSDP-based multi-GPU training: keeps every GPU busy every step.

`bonsai.training`'s `train`/`train_step` shard the model via
`device_map="auto"`, which places different *layers* on different GPUs --
correct, and fine for a short exploratory sweep, but confirmed on a real
8x A100 run to leave only one GPU active at a time (forward and backward
each walk the same layer-by-layer chain in sequence, so every other GPU
sits near-idle waiting its turn). FSDP (Fully Sharded Data Parallel)
shards *parameters* instead: every GPU holds a slice of every layer, so
every GPU computes on every step, all at once.

This needs a genuinely different execution model -- one Python process
per GPU, launched via `accelerate launch`, with gradient/parameter
synchronization handled by `torch.distributed` collectives -- not a
single process importing `bonsai.training.train()` directly. Hence a
separate module rather than folding this into that function's existing
single-process loop, per the project's own preference to keep the two
launch paths distinct rather than one function silently branching its
entire execution model on an argument.

`TrainConfig` and `lr_schedule` are reused as-is from `bonsai.training`
(pure, no process-topology assumptions); only model construction and the
backward/optimizer-step mechanics differ here -- `accelerator.backward()`
in place of `loss.backward()`, gradient accumulation via `accelerate`'s
own `Accelerator(gradient_accumulation_steps=...)` instead of this
project's manual accumulation loop (accelerate's version correctly skips
gradient sync until the last micro-step of each window; replicating that
by hand across processes is exactly the kind of thing this library
already exists to get right once).

The FSDP wiring itself (this module) is validated by a real multi-GPU
run, not by CPU unit tests -- unlike every other real-forward-pass module
in this project, genuinely exercising `torch.distributed` collectives
needs more than one process actually talking to more than one device.
`TrainConfig`/`lr_schedule`'s own correctness is already covered by
`tests/test_training.py`; what's specific to this module (the FSDP plugin
construction, the accelerate-managed training loop) is exercised for real
in `scratch_teacher_correction_fsdp.py`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

from bonsai.training import LossFn, TrainConfig, _lm_loss, lr_schedule


def make_accelerator(*, gradient_accumulation_steps: int = 1) -> Accelerator:
    """An `Accelerator` configured for full-shard FSDP over `Qwen3MoeDecoderLayer` boundaries.

    Wrapping at the decoder-layer boundary (`transformer_cls_names_to_wrap`),
    not FSDP's default flat-parameter wrap, is what actually lets every GPU
    compute on every layer together -- `accelerate`'s own recommended
    auto-wrap unit for transformer models, MoE included (each
    `Qwen3MoeDecoderLayer` already bundles its attention and expert blocks
    into one wrappable unit). `activation_checkpointing=True` here replaces
    `bonsai.training`'s manual `model.gradient_checkpointing_enable()` call
    -- FSDP's own plugin option is the version meant to compose correctly
    with sharding, not a separately-applied wrapper.

    Deliberately *not* using `cpu_ram_efficient_loading`/`sync_module_states`
    (the "only rank 0 loads the full model, then broadcasts each wrapped
    unit to every other rank" path): confirmed on a real 8x A100 run that
    this broadcast step itself OOMs during FSDP's init (every rank hit
    ~38.8GB right as `_sync_module_params_and_buffers` ran, not during
    training) -- a real, reproduced failure, not a theoretical concern.
    The caller instead has every rank load its own full copy into CPU RAM
    (`AutoModelForCausalLM.from_pretrained(..., dtype=torch.bfloat16)`,
    no `device_map`) before `accelerator.prepare` shards it onto GPU; this
    instance's 1.7TB of system RAM comfortably holds 8 copies of a ~61GB
    model, and the simpler path avoids the broadcast step that failed.
    """
    fsdp_plugin = FullyShardedDataParallelPlugin(
        # `reshard_after_forward` is the current name for the old
        # `sharding_strategy` param -- FSDP1 (the default here; no
        # `fsdp_version=2`) still wants the string/enum form, not the
        # boolean FSDP2 accepts.
        reshard_after_forward="FULL_SHARD",
        auto_wrap_policy="transformer_based_wrap",
        transformer_cls_names_to_wrap=["Qwen3MoeDecoderLayer"],
        activation_checkpointing=True,
        state_dict_type="FULL_STATE_DICT",
        # Required whenever the optimizer needs real per-parameter shapes
        # (8-bit AdamW's block-wise quantization does): without this, FSDP
        # hands the optimizer one large flattened parameter per wrapped
        # unit instead of the original per-tensor parameters, which is a
        # well-documented source of exactly this kind of OOM -- confirmed
        # by reproducing it on a real 8x A100 run before adding this.
        use_orig_params=True,
    )
    return Accelerator(
        fsdp_plugin=fsdp_plugin, gradient_accumulation_steps=gradient_accumulation_steps
    )


def train_step(
    accelerator: Accelerator,
    model: Qwen3MoeForCausalLM,
    optimizer: torch.optim.Optimizer,
    input_ids_batch: Sequence[torch.Tensor],
    *,
    loss_fn: LossFn = _lm_loss,
) -> float:
    """One optimizer step, gradient-accumulated via `accelerator.accumulate`.

    Same per-sample-forward-pass shape as `bonsai.training.train_step`
    (sequences aren't padded), but the accumulation and gradient sync are
    `accelerate`'s own, not a manual loop -- `accelerator.accumulate(model)`
    skips gradient synchronization across processes until the batch's last
    micro-step, which a hand-rolled loop across multiple processes would
    otherwise get wrong (or need to reimplement `torch.distributed`'s own
    no_sync context manager to get right).
    """
    if not input_ids_batch:
        raise ValueError("input_ids_batch must not be empty")
    total = 0.0
    for input_ids in input_ids_batch:
        with accelerator.accumulate(model):
            loss = loss_fn(model, input_ids)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            total += float(loss.detach())
    return total / len(input_ids_batch)


def train(
    accelerator: Accelerator,
    model: Qwen3MoeForCausalLM,
    optimizer: torch.optim.Optimizer,
    train_input_ids: Sequence[torch.Tensor],
    val_input_ids: Sequence[torch.Tensor],
    config: TrainConfig,
    *,
    loss_fn: LossFn = _lm_loss,
    val_every: int = 10,
    run_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Run `config.total_steps` optimizer steps under FSDP, periodically checking validation loss.

    `model`/`optimizer` must already be passed through
    `accelerator.prepare(model, optimizer)` by the caller -- this function
    doesn't call `prepare` itself, since the caller also needs the
    prepared objects for anything else it does with them (e.g. deciding
    when to save a checkpoint).

    Logging (`bonsai.tracking.log_metric`) and the returned history are
    only meaningful on the main process -- every process runs the same
    training steps (that's the point of data-parallel-style gradient
    sync), but only `accelerator.is_main_process` writes to `run_dir` or
    returns a non-empty history, so callers don't need every rank to
    separately guard against duplicate writes.
    """
    if not train_input_ids or not val_input_ids:
        raise ValueError("train_input_ids and val_input_ids must both be non-empty")

    log_metric = None
    if run_dir is not None and accelerator.is_main_process:
        from bonsai.tracking import log_metric as _log_metric

        log_metric = _log_metric

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
        train_loss = train_step(accelerator, model, optimizer, batch, loss_fn=loss_fn)

        record: dict[str, Any] = {"step": step, "lr": lr, "train_loss": train_loss}
        if (step + 1) % val_every == 0 or step == config.total_steps - 1:
            model.eval()  # type: ignore[no-untyped-call]
            with torch.no_grad():
                val_loss = sum(float(loss_fn(model, ids)) for ids in val_input_ids) / len(
                    val_input_ids
                )
            model.train()
            record["val_loss"] = val_loss

        if accelerator.is_main_process:
            history.append(record)
            if log_metric is not None and run_dir is not None:
                log_metric(run_dir, **record)

    model.eval()  # type: ignore[no-untyped-call]
    return history


__all__ = ["make_accelerator", "train", "train_step"]
