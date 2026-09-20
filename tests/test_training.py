"""Real-forward-and-backward-pass tests for `bonsai.training`.

Same tiny-model pattern as the pruning test files: a real, tiny,
randomly-initialized Qwen3-MoE model on CPU, not a mocked optimizer step.
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
from bonsai.training import TrainConfig, lr_schedule, train, train_step

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


def tiny_batch(n: int, *, seed: int = 0) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randint(0, TINY_CONFIG.vocab_size, (1, 8), generator=gen) for _ in range(n)]


def test_lr_schedule_ramps_up_linearly_during_warmup() -> None:
    config = TrainConfig(peak_lr=1e-3, min_lr=1e-5, warmup_steps=10, total_steps=100)

    assert lr_schedule(0, config) == config.peak_lr * (1 / 10)
    assert lr_schedule(4, config) == config.peak_lr * (5 / 10)
    assert lr_schedule(9, config) == config.peak_lr * (10 / 10)


def test_lr_schedule_hits_peak_right_after_warmup() -> None:
    config = TrainConfig(peak_lr=1e-3, min_lr=1e-5, warmup_steps=10, total_steps=100)

    assert math.isclose(lr_schedule(10, config), config.peak_lr, rel_tol=1e-6)


def test_lr_schedule_decays_to_min_lr_by_final_step() -> None:
    config = TrainConfig(peak_lr=1e-3, min_lr=1e-5, warmup_steps=10, total_steps=100)

    assert math.isclose(lr_schedule(100, config), config.min_lr, rel_tol=1e-6)


def test_lr_schedule_zero_warmup_starts_at_peak() -> None:
    config = TrainConfig(peak_lr=1e-3, min_lr=1e-5, warmup_steps=0, total_steps=100)

    assert lr_schedule(0, config) == config.peak_lr


def test_train_step_updates_model_weights() -> None:
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    original = next(model.parameters()).data.clone()
    batch = tiny_batch(3)

    train_step(model, optimizer, batch)

    assert not torch.allclose(original, next(model.parameters()).data)


def test_train_step_returns_mean_loss_over_batch() -> None:
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    batch = tiny_batch(3)

    loss = train_step(model, optimizer, batch)

    assert math.isfinite(loss)
    assert loss > 0


def test_train_step_rejects_empty_batch() -> None:
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    try:
        train_step(model, optimizer, [])
    except ValueError:
        return
    raise AssertionError("expected ValueError for an empty batch")


def test_train_step_reduces_loss_on_a_repeated_batch() -> None:
    # Not a claim about real training dynamics -- just confirms gradients
    # actually flow and the optimizer actually reduces loss on a fixed
    # batch, the basic sanity check every real training loop needs.
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-2)
    batch = tiny_batch(2)

    first_loss = train_step(model, optimizer, batch)
    for _ in range(20):
        last_loss = train_step(model, optimizer, batch)

    assert last_loss < first_loss


def test_train_runs_the_requested_number_of_steps() -> None:
    model = tiny_model()
    config = TrainConfig(peak_lr=1e-3, min_lr=1e-4, warmup_steps=2, total_steps=5, batch_size=2)

    history = train(model, tiny_batch(4), tiny_batch(2), config, val_every=2)

    assert len(history) == 5
    assert [r["step"] for r in history] == [0, 1, 2, 3, 4]


def test_train_reports_val_loss_on_schedule_and_final_step() -> None:
    model = tiny_model()
    config = TrainConfig(peak_lr=1e-3, min_lr=1e-4, warmup_steps=1, total_steps=5, batch_size=1)

    history = train(model, tiny_batch(4), tiny_batch(2), config, val_every=2)

    assert "val_loss" in history[1]  # step index 1 -> step number 2, val_every=2
    assert "val_loss" in history[3]  # step index 3 -> step number 4
    assert "val_loss" in history[4]  # final step, always evaluated
    assert "val_loss" not in history[0]
    assert "val_loss" not in history[2]


def test_train_logs_metrics_to_run_dir(tmp_path: Path) -> None:
    model = tiny_model()
    config = TrainConfig(peak_lr=1e-3, min_lr=1e-4, warmup_steps=1, total_steps=3, batch_size=1)
    run_dir = new_run("test-training", root=tmp_path)

    history = train(model, tiny_batch(3), tiny_batch(2), config, val_every=1, run_dir=run_dir)

    logged = load_metrics(run_dir)
    assert len(logged) == len(history) == 3
    assert [m["step"] for m in logged] == [0, 1, 2]


def test_train_rejects_empty_train_or_val_data() -> None:
    model = tiny_model()
    config = TrainConfig(peak_lr=1e-3, min_lr=1e-4, warmup_steps=1, total_steps=3)

    try:
        train(model, [], tiny_batch(2), config)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty train_input_ids")

    try:
        train(model, tiny_batch(2), [], config)
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty val_input_ids")


def test_train_cycles_through_train_data_when_exhausted() -> None:
    # total_steps * batch_size (5*2=10) exceeds len(train_input_ids) (3),
    # so the loop must wrap around via modulo rather than raising.
    model = tiny_model()
    config = TrainConfig(peak_lr=1e-3, min_lr=1e-4, warmup_steps=1, total_steps=5, batch_size=2)

    history = train(model, tiny_batch(3), tiny_batch(2), config, val_every=5)

    assert len(history) == 5
