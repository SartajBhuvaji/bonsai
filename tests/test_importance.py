"""Synthetic-tensor tests for `bonsai.importance` -- no model, no GPU.

Every activation shape here is hand-constructed to make the expected score
computable by hand, the same way `test_architecture.py` uses a tiny real
config instead of downloading Qwen3-30B-A3B-Base's actual weights. Wiring
these functions to real forward hooks against the real model is 1.3's
GPU-dependent validation step, not this file's job.
"""

import torch

from bonsai.importance import (
    aggregate,
    expert_importance,
    expert_max_output_norm,
    flag_outlier_experts,
    head_importance,
)


def test_aggregate_mean_abs() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert torch.allclose(aggregate(x, "mean_abs", dim=0), torch.tensor([2.5, 3.5, 4.5]))


def test_aggregate_l2() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    expected = torch.tensor([17.0, 29.0, 45.0]).sqrt()
    assert torch.allclose(aggregate(x, "l2", dim=0), expected)


def test_aggregate_variance() -> None:
    # Constant difference of 3 between the two rows in every column -> the
    # same biased variance (2.25) in every column, a simple hand-checkable case.
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert torch.allclose(aggregate(x, "variance", dim=0), torch.full((3,), 2.25))


def test_aggregate_unknown_fn_raises() -> None:
    try:
        aggregate(torch.zeros(2, 2), "bogus", dim=0)  # type: ignore[arg-type]
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown aggregation function")


def test_head_importance_ranks_bigger_output_higher() -> None:
    # batch=2, seq=3, num_heads=2, head_dim=4. Head 0's output is a constant
    # all-ones vector (norm 2), head 1's is all-threes (norm 6) -- head 1
    # should score higher under every aggregation function that isn't variance.
    attn_output = torch.zeros(2, 3, 2, 4)
    attn_output[..., 0, :] = 1.0
    attn_output[..., 1, :] = 3.0

    for fn in ("mean_abs", "l2"):
        scores = head_importance(attn_output, fn=fn)
        assert scores.shape == (2,)
        assert scores[1] > scores[0]

    mean_abs_scores = head_importance(attn_output, fn="mean_abs")
    assert torch.allclose(mean_abs_scores, torch.tensor([2.0, 6.0]))


def test_head_importance_rejects_wrong_ndim() -> None:
    try:
        head_importance(torch.zeros(2, 3, 4))
    except ValueError:
        return
    raise AssertionError("expected ValueError for a non-4D attn_output")


def test_expert_importance_weights_by_gate_and_norm() -> None:
    # 6 (token, selected-expert) rows across 3 experts. Expected means worked
    # out by hand from the norms and gate weights below.
    expert_indices = torch.tensor([0, 0, 1, 2, 2, 2])
    gate_weights = torch.tensor([1.0, 0.5, 2.0, 1.0, 1.0, 1.0])
    expert_outputs = torch.tensor(
        [
            [3.0, 4.0],  # expert 0, norm 5, weighted 5.0
            [3.0, 4.0],  # expert 0, norm 5, weighted 2.5
            [0.0, 0.0],  # expert 1, norm 0, weighted 0.0
            [3.0, 4.0],  # expert 2, norm 5, weighted 5.0
            [0.0, 0.0],  # expert 2, norm 0, weighted 0.0
            [6.0, 8.0],  # expert 2, norm 10, weighted 10.0
        ]
    )

    scores = expert_importance(expert_indices, gate_weights, expert_outputs, num_experts=3)

    assert torch.allclose(scores, torch.tensor([3.75, 0.0, 5.0]))


def test_expert_importance_unrouted_expert_scores_zero() -> None:
    expert_indices = torch.tensor([0, 0])
    gate_weights = torch.tensor([1.0, 1.0])
    expert_outputs = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    # 4 experts declared, only expert 0 ever routed to.
    scores = expert_importance(expert_indices, gate_weights, expert_outputs, num_experts=4)

    assert scores[0] > 0
    assert torch.allclose(scores[1:], torch.zeros(3))


def test_expert_importance_rejects_wrong_shapes() -> None:
    try:
        expert_importance(
            torch.zeros(3, dtype=torch.long),
            torch.zeros(3),
            torch.zeros(3, 2, 2),  # should be 2D, not 3D
            num_experts=2,
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for a non-2D expert_outputs")


def test_expert_max_output_norm() -> None:
    expert_indices = torch.tensor([0, 0, 1, 2, 2, 2])
    expert_outputs = torch.tensor(
        [
            [3.0, 4.0],  # expert 0, norm 5
            [3.0, 4.0],  # expert 0, norm 5
            [0.0, 0.0],  # expert 1, norm 0
            [3.0, 4.0],  # expert 2, norm 5
            [0.0, 0.0],  # expert 2, norm 0
            [6.0, 8.0],  # expert 2, norm 10
        ]
    )

    max_norms = expert_max_output_norm(expert_indices, expert_outputs, num_experts=3)

    assert torch.allclose(max_norms, torch.tensor([5.0, 0.0, 10.0]))


def test_flag_outlier_experts() -> None:
    # 9 unremarkable experts (max norm 1.0) plus one clear outlier (100.0).
    # quantile(0.9) over this 10-element array interpolates to 10.9, so only
    # the outlier clears the threshold -- worked out by hand, not assumed.
    max_norms = torch.tensor([1.0] * 9 + [100.0])

    mask = flag_outlier_experts(max_norms, percentile=90.0)

    assert mask.tolist() == [False] * 9 + [True]


def test_flag_outlier_experts_default_percentile_shape() -> None:
    max_norms = torch.rand(128)
    mask = flag_outlier_experts(max_norms)
    assert mask.shape == (128,)
    assert mask.dtype == torch.bool
