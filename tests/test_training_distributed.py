"""Real-forward-and-backward-pass tests for `bonsai.training.distributed`.

Same tiny-model pattern as `test_training.py`, run through `accelerate`'s
real `Accelerator.prepare`/`accumulate`/`backward` API rather than mocking
it -- but single-process (this machine has one local GPU, not eight), so
this proves the control flow and API usage are correct, not that FSDP's
actual multi-GPU parameter sharding and gradient-sync collectives are.
That needs more than one process genuinely talking to more than one
device, which only a real multi-GPU `accelerate launch` run can exercise
-- see the module docstring in `bonsai/training/distributed.py` for what
that real validation looks like.
"""

import math
from copy import deepcopy
from pathlib import Path
from typing import cast

import torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

from bonsai.tracking import load_metrics, new_run
from bonsai.training import TrainConfig
from bonsai.training.distributed import make_accelerator, train, train_step

TINY_CONFIG = Qwen3MoeConfig(
    vocab_size=64,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    moe_intermediate_size=8,
    num_experts=4,
    num_experts_per_tok=2,
)


def tiny_model() -> Qwen3MoeForCausalLM:
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(deepcopy(TINY_CONFIG))  # type: ignore[no-untyped-call]
    return cast(Qwen3MoeForCausalLM, model)


def tiny_batch(n: int, device: torch.device, *, seed: int = 0) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [
        torch.randint(0, TINY_CONFIG.vocab_size, (1, 8), generator=gen).to(device) for _ in range(n)
    ]


def test_make_accelerator_builds_without_error() -> None:
    accelerator = make_accelerator(gradient_accumulation_steps=4)

    assert accelerator.gradient_accumulation_steps == 4


def test_train_step_updates_model_weights_via_accelerator() -> None:
    accelerator = make_accelerator()
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    model, optimizer = accelerator.prepare(model, optimizer)
    original = next(model.parameters()).data.clone()
    batch = tiny_batch(3, accelerator.device)

    train_step(accelerator, model, optimizer, batch)

    assert not torch.allclose(original, next(model.parameters()).data)


def test_train_step_rejects_empty_batch() -> None:
    accelerator = make_accelerator()
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    model, optimizer = accelerator.prepare(model, optimizer)

    try:
        train_step(accelerator, model, optimizer, [])
    except ValueError:
        return
    raise AssertionError("expected ValueError for an empty batch")


def test_train_runs_the_requested_number_of_steps() -> None:
    accelerator = make_accelerator()
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model, optimizer = accelerator.prepare(model, optimizer)
    config = TrainConfig(peak_lr=1e-3, min_lr=1e-4, warmup_steps=2, total_steps=5, batch_size=2)

    history = train(
        accelerator,
        model,
        optimizer,
        tiny_batch(4, accelerator.device),
        tiny_batch(2, accelerator.device),
        config,
        val_every=2,
    )

    assert len(history) == 5
    assert [r["step"] for r in history] == [0, 1, 2, 3, 4]


def test_train_reports_val_loss_on_schedule_and_final_step() -> None:
    accelerator = make_accelerator()
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model, optimizer = accelerator.prepare(model, optimizer)
    config = TrainConfig(peak_lr=1e-3, min_lr=1e-4, warmup_steps=1, total_steps=5, batch_size=1)

    history = train(
        accelerator,
        model,
        optimizer,
        tiny_batch(4, accelerator.device),
        tiny_batch(2, accelerator.device),
        config,
        val_every=2,
    )

    assert "val_loss" in history[1]
    assert "val_loss" in history[3]
    assert "val_loss" in history[4]
    assert "val_loss" not in history[0]
    assert "val_loss" not in history[2]


def test_train_logs_metrics_to_run_dir_on_main_process(tmp_path: Path) -> None:
    accelerator = make_accelerator()
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model, optimizer = accelerator.prepare(model, optimizer)
    config = TrainConfig(peak_lr=1e-3, min_lr=1e-4, warmup_steps=1, total_steps=3, batch_size=1)
    run_dir = new_run("test-distributed-training", root=tmp_path)

    history = train(
        accelerator,
        model,
        optimizer,
        tiny_batch(3, accelerator.device),
        tiny_batch(2, accelerator.device),
        config,
        val_every=1,
        run_dir=run_dir,
    )

    assert accelerator.is_main_process  # single-process test run always is
    logged = load_metrics(run_dir)
    assert len(logged) == len(history) == 3
    assert [m["step"] for m in logged] == [0, 1, 2]


def test_train_rejects_empty_train_or_val_data() -> None:
    accelerator = make_accelerator()
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model, optimizer = accelerator.prepare(model, optimizer)
    config = TrainConfig(peak_lr=1e-3, min_lr=1e-4, warmup_steps=1, total_steps=3)

    try:
        train(accelerator, model, optimizer, [], tiny_batch(2, accelerator.device), config)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty train_input_ids")

    try:
        train(accelerator, model, optimizer, tiny_batch(2, accelerator.device), [], config)
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty val_input_ids")


def test_train_step_reduces_loss_on_a_repeated_batch() -> None:
    accelerator = make_accelerator()
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-2)
    model, optimizer = accelerator.prepare(model, optimizer)
    batch = tiny_batch(2, accelerator.device)

    first_loss = train_step(accelerator, model, optimizer, batch)
    for _ in range(20):
        last_loss = train_step(accelerator, model, optimizer, batch)

    assert math.isfinite(last_loss)
    assert last_loss < first_loss
