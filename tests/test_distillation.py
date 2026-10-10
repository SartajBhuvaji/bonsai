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


def test_kd_loss_fn_is_the_teachers_own_coverage_floor_when_student_matches_teacher() -> None:
    """A student that *is* the teacher should match it exactly within the
    top-k support, leaving only the teacher's own "coverage floor": how much
    of its real probability mass that support doesn't capture.

    Earlier, this test asserted the loss was ~0 in this case. That only held
    because the pre-fix loss renormalized *both* sides over just the top-k
    support before comparing them, which silently discarded coverage
    entirely -- the real bug this file's other new regression test
    (`test_training_does_not_let_an_off_support_logit_dominate_the_full_vocab`)
    exists to catch. With the fix, matching the teacher exactly still drives
    the within-support term to ~0, but the loss is no longer 0 unless the
    teacher's own top-k support happens to capture ~all of its probability
    mass -- which a tiny, untrained, 64-token-vocab toy model's top-5 does
    not (a real teacher's top-20 over a ~150k vocab does, in practice). So
    this test asserts the loss lands at that coverage floor, computed
    independently via the same full-vocab-vs-support logsumexp gap, rather
    than asserting a fixed small constant.
    """
    teacher = tiny_model()  # identical weights to cache_top_k's own teacher (both seed 0)
    batch = tiny_batch(2, seed=3)
    k = 5
    cache = cache_top_k(batch, k=k)
    loss_fn = make_kd_loss_fn(cache)

    for input_ids in batch:
        loss = loss_fn(teacher, input_ids)

        with torch.no_grad():
            logits = teacher(input_ids=input_ids).logits[0, :-1]
            topk_logits, _ = torch.topk(logits, k, dim=-1)
            expected_coverage_floor = (
                torch.logsumexp(logits, dim=-1) - torch.logsumexp(topk_logits, dim=-1)
            ).mean()

        # Small fp slack: the teacher's cached logprobs and the student's own
        # top-k extraction each go through their own log_softmax/gather chain
        # in fp32, so an exact match isn't expected even for identical weights.
        assert abs(loss.item() - expected_coverage_floor.item()) < 1e-2


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


def test_training_does_not_let_an_off_support_logit_dominate_the_full_vocab() -> None:
    """Regression test for a real bug (plan.md, 2026-10-10): an earlier version
    of `make_kd_loss_fn` ran `log_softmax` *after* gathering the teacher's k
    ids, so the loss only ever normalized over those k logits. Every other
    vocab logit was free to grow without being penalized, and in practice a
    student trained that way converged to a healthy-looking KD loss while its
    real (full-vocab) argmax was dominated by a token the loss never looked
    at -- which is exactly the HumanEval-TS failure actually observed
    (repetition collapse on an off-support token from the first generated
    token on, across every KD arm tried: full-parameter, LoRA, and
    function-mode data alike).

    This drives enough real training steps that the bug reproduces reliably
    on CPU (confirmed against the pre-fix loss_fn before writing this test),
    then asserts the student's full-vocab argmax at every position lands
    inside the teacher's own top-k support -- i.e. training actually
    suppresses off-support logits, not just the on-support ranking.
    """
    batch = tiny_batch(4, seed=4)
    k = 5
    cache = cache_top_k(batch, k=k)
    loss_fn = make_kd_loss_fn(cache)

    student = tiny_model()
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for param in student.parameters():
            param.add_(torch.randn(param.shape, generator=generator) * 0.5)

    config = TrainConfig(peak_lr=1e-2, min_lr=1e-3, warmup_steps=5, total_steps=300, batch_size=4)
    train(student, batch, batch, config, loss_fn=loss_fn, val_every=300)

    student.eval()  # type: ignore[no-untyped-call]
    with torch.no_grad():
        for input_ids in batch:
            teacher_topk_ids, _ = cache[id(input_ids)]
            logits = student(input_ids=input_ids).logits[0, :-1]  # [seq_len-1, vocab]
            argmax_ids = logits.argmax(dim=-1)  # [seq_len-1]
            on_support = (argmax_ids.unsqueeze(-1) == teacher_topk_ids[:-1]).any(dim=-1)
            assert bool(on_support.all()), (
                "full-vocab argmax landed outside the teacher's top-k support at "
                f"{int((~on_support).sum())}/{on_support.numel()} positions -- "
                "an off-support token is dominating the real (non-truncated) "
                "softmax despite a converged KD loss"
            )
