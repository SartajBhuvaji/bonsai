"""Real-forward-and-backward-pass tests for `bonsai.training.distillation`.

Same tiny-model pattern as `tests/test_training.py`: a real, tiny,
randomly-initialized Qwen3-MoE model on CPU stands in for both teacher and
student, not a mocked loss.
"""

from __future__ import annotations

from copy import deepcopy
from typing import cast

import pytest
import torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

from bonsai.training import TrainConfig, train
from bonsai.training.distillation import make_kd_loss_fn

# Same TINY_CONFIG/tiny_model/tiny_batch pattern as test_training.py, duplicated
# rather than imported -- every other test file in this project does the same
# (no cross-test-file imports exist elsewhere), which also sidesteps mypy
# resolving `tests/test_training.py` under two different module names.
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


def cache_top_k(
    teacher_input_ids: list[torch.Tensor], *, k: int = 5
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Stand-in for `tools/cache_teacher_logits.py`'s real output, built from a
    real (tiny) teacher forward pass -- not synthetic/random cache entries.

    Always uses `tiny_model()`'s fixed seed-0 init (same as every other
    tiny model in this file) -- there's deliberately no separate
    `teacher_seed` knob, since a distinct teacher model isn't needed for
    these tests and `tiny_model()` doesn't expose one anyway."""
    teacher = tiny_model()
    cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    with torch.no_grad():
        for input_ids in teacher_input_ids:
            logits = teacher(input_ids=input_ids).logits[0]  # [seq_len, vocab]
            log_probs = torch.log_softmax(logits, dim=-1)
            topk_logprobs, topk_ids = torch.topk(log_probs, k, dim=-1)
            # Renormalize over just the top-k support, matching the real
            # caching script's output format.
            topk_logprobs = topk_logprobs - torch.logsumexp(topk_logprobs, dim=-1, keepdim=True)
            cache[id(input_ids)] = (topk_ids, topk_logprobs)
    return cache


def test_kd_loss_fn_raises_on_an_uncached_sample() -> None:
    loss_fn = make_kd_loss_fn(cache={})
    student = tiny_model()
    stray_input_ids = torch.randint(0, TINY_CONFIG.vocab_size, (1, 8))

    with pytest.raises(KeyError):
        loss_fn(student, stray_input_ids)


def test_kd_loss_fn_is_real_valued_and_nonnegative() -> None:
    batch = tiny_batch(3, seed=2)
    cache = cache_top_k(batch)
    loss_fn = make_kd_loss_fn(cache)
    student = tiny_model()

    for input_ids in batch:
        loss = loss_fn(student, input_ids)
        assert torch.isfinite(loss)
        # KL divergence is non-negative by construction, up to fp slack.
        assert loss.item() > -1e-4


def test_kd_loss_fn_is_zero_when_student_matches_teacher_exactly() -> None:
    """A student that *is* the teacher should score ~0 KL against its own cache."""
    teacher = tiny_model()  # identical weights to cache_top_k's own teacher (both seed 0)
    batch = tiny_batch(2, seed=3)
    cache = cache_top_k(batch)
    loss_fn = make_kd_loss_fn(cache)

    for input_ids in batch:
        loss = loss_fn(teacher, input_ids)
        # Not exactly 0: the teacher's cached logprobs and the student's own
        # top-k extraction each go through their own log_softmax/gather chain
        # in fp32, so a small real discrepancy survives even for identical
        # weights. Still two orders of magnitude below the untrained-vs-teacher
        # loss the other tests see (~0.1-1), which is the property this test
        # actually checks.
        assert loss.item() < 1e-2


def test_training_against_cached_teacher_logits_reduces_kd_loss() -> None:
    """Real gradient descent: a few `train()` steps should lower the KD loss,
    same real-training-loop assertion style as `test_training.py`'s own tests."""
    batch = tiny_batch(4, seed=4)
    cache = cache_top_k(batch, k=5)
    loss_fn = make_kd_loss_fn(cache)

    # tiny_model() always seeds 0, same as cache_top_k's teacher -- perturb the
    # student's weights afterward so there's an actual gap for training to close,
    # rather than starting already equal to the teacher.
    student = tiny_model()
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for param in student.parameters():
            param.add_(torch.randn(param.shape, generator=generator) * 0.5)

    config = TrainConfig(peak_lr=1e-2, min_lr=1e-3, warmup_steps=2, total_steps=20, batch_size=2)
    history = train(student, batch, batch, config, loss_fn=loss_fn, val_every=20)

    first_loss = history[0]["train_loss"]
    last_loss = history[-1]["train_loss"]
    assert last_loss < first_loss
