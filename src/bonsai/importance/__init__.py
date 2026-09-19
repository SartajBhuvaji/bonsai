"""Shared activation-based importance scoring for Phase 2's pruning axes.

Mirrors how 2407.14679 Section 2.2 actually structures this, not a guess at
it: one shared primitive -- aggregate an activation-derived quantity over a
set of tokens with a *pluggable* aggregation function (the paper ablates
mean(abs), L2 norm and variance in its Table 13 rather than hardcoding one,
since the best choice isn't the same for every axis) -- wrapped by thin,
axis-specific functions that each know which activations to feed it.

Two of Phase 2's three axes are thin wrappers around `aggregate`:

- `head_importance` (2.2): the paper's own `F_head` formula, unchanged.
- `expert_importance` (2.3): the gate-weighted expert-output-norm formula
  plan.md settled on (REAP, arXiv 2510.13999; independently, NVIDIA's own
  Model-Optimizer Minitron-MoE extension). Structurally different from
  `head_importance` despite sharing `aggregate`'s reduction step underneath:
  routing is sparse and per-token, so there's no dense (batch, sequence) grid
  to aggregate over -- only whichever tokens actually selected a given
  expert, a variable-size, per-expert set.

Depth (2.1) is deliberately NOT a third wrapper here. The papers score it via
PPL-delta or Block Importance (cosine distance between a layer's input and
output), never via activation magnitude, and bonsai keeps that split: 2.1
owns its own scoring code and doesn't call into this module.

This module is GPU-free by design: every function takes plain tensors
(whatever a forward hook on the real model would have captured) rather than
a model or a calibration dataset. Wiring these into actual hooks on
Qwen3-30B-A3B-Base and running them against real calibration data is 1.3's
GPU-dependent validation step, not this module's job.
"""

from __future__ import annotations

from typing import Literal, cast

import torch

AggregationFn = Literal["mean_abs", "l2", "variance"]


def aggregate(x: torch.Tensor, fn: AggregationFn, dim: int = 0) -> torch.Tensor:
    """Reduce `x` along `dim` with a pluggable aggregation function.

    The three functions 2407.14679's Table 13 ablates per axis: mean(abs),
    L2 norm, and (biased) variance. `dim` is the batch/sequence-like
    dimension being collapsed away; the axis(es) that identify the component
    being scored (head, expert, ...) are whatever's left.
    """
    if fn == "mean_abs":
        return x.abs().mean(dim=dim)
    if fn == "l2":
        # torch's stubs type vector_norm's return as Any.
        return cast(torch.Tensor, torch.linalg.vector_norm(x, ord=2, dim=dim))
    if fn == "variance":
        return x.var(dim=dim, unbiased=False)
    raise ValueError(f"unknown aggregation function: {fn!r}")


def head_importance(attn_output: torch.Tensor, fn: AggregationFn = "l2") -> torch.Tensor:
    """Score each attention head from its per-token output activations.

    `attn_output` is the per-head attention output *before* the `W^O` merge
    -- shape `[batch, seq, num_heads, head_dim]`. Implements 2407.14679's
    `F_head`: the per-token, per-head output vector's L2 norm (fixed --
    that's the `||...||` in the paper's formula), aggregated over every
    (batch, seq) position per head with `fn` (the part the paper's Table 13
    ablation varies). Returns one score per head, shape `[num_heads]`.
    """
    if attn_output.ndim != 4:
        raise ValueError(
            f"expected [batch, seq, num_heads, head_dim], got shape {tuple(attn_output.shape)}"
        )
    per_token_head_norm = torch.linalg.vector_norm(attn_output, ord=2, dim=-1)
    flattened = per_token_head_norm.reshape(-1, per_token_head_norm.shape[-1])
    return aggregate(flattened, fn, dim=0)


def expert_importance(
    expert_indices: torch.Tensor,
    gate_weights: torch.Tensor,
    expert_outputs: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Score each expert from its output on the tokens actually routed to it.

    `S_j = mean over routed tokens of (gate_weight * ||expert_output||_2)` --
    the REAP / NVIDIA Model-Optimizer Minitron-MoE formula plan.md settled
    on over the plan's original three candidates (activation-when-selected,
    selection frequency alone, router-logit magnitude alone): both cited
    sources found frequency- and logit-only metrics underperform this one.

    Unlike `head_importance`, there's no dense (batch, sequence) grid to
    hand `aggregate` -- routing is sparse and data-dependent per token, so
    the natural shape is a flat (token, expert) assignment list, mirroring
    how `Qwen3MoeExperts.forward` itself iterates per expert over whichever
    tokens it was actually given:

    - `expert_indices`: `[num_routed_tokens]`, int -- which expert each row
      went to (one row per (token, selected-expert) pair; a token selected
      by top-8 routing contributes 8 rows, one per expert it picked).
    - `gate_weights`: `[num_routed_tokens]`, float -- that pair's
      (renormalized) routing weight.
    - `expert_outputs`: `[num_routed_tokens, hidden_size]`, float -- the
      selected expert's `down_proj` output for that token.

    Experts with zero routed tokens in the calibration set score 0 and rank
    lowest -- correct for a calibration-set-specific score, and exactly the
    scenario the Super Experts regression test below guards against silently
    mis-scoring an expert that's rare but critical, not rare and irrelevant.
    """
    if expert_indices.ndim != 1 or gate_weights.ndim != 1 or expert_outputs.ndim != 2:
        raise ValueError(
            "expected expert_indices: [N], gate_weights: [N], expert_outputs: [N, hidden_size]; "
            f"got shapes {tuple(expert_indices.shape)}, {tuple(gate_weights.shape)}, "
            f"{tuple(expert_outputs.shape)}"
        )

    weighted_norm = gate_weights * torch.linalg.vector_norm(expert_outputs, ord=2, dim=-1)

    sums = torch.zeros(num_experts, dtype=weighted_norm.dtype, device=weighted_norm.device)
    counts = torch.zeros(num_experts, dtype=torch.long, device=weighted_norm.device)
    sums.index_add_(0, expert_indices, weighted_norm)
    counts.index_add_(0, expert_indices, torch.ones_like(expert_indices))

    scores = torch.zeros_like(sums)
    routed = counts > 0
    scores[routed] = sums[routed] / counts[routed]
    return scores


def expert_max_output_norm(
    expert_indices: torch.Tensor,
    expert_outputs: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Per-expert max output norm across its routed tokens.

    The outlier-detection signal behind 1.3's Super Experts guard: a
    mean-based score (`expert_importance`) can dilute a rare-but-massive
    output into a low average, which is exactly how Qwen3-30B-A3B's known
    Super Experts (arXiv 2507.23279) would get wrongly pruned by mean alone.
    Feed this into `flag_outlier_experts` before ranking by
    `expert_importance` for real.
    """
    if expert_indices.ndim != 1 or expert_outputs.ndim != 2:
        raise ValueError(
            f"expected expert_indices: [N], expert_outputs: [N, hidden_size]; got shapes "
            f"{tuple(expert_indices.shape)}, {tuple(expert_outputs.shape)}"
        )

    norms = torch.linalg.vector_norm(expert_outputs, ord=2, dim=-1)
    result = torch.zeros(num_experts, dtype=norms.dtype, device=norms.device)
    result.scatter_reduce_(0, expert_indices, norms, reduce="amax", include_self=True)
    return result


def flag_outlier_experts(max_norms: torch.Tensor, percentile: float = 99.5) -> torch.Tensor:
    """Bool mask of experts whose max routed-token output norm is an outlier.

    Threshold at the given percentile (99.5 by default, arXiv 2507.23279's
    Eq. 8) across whatever experts are passed in. Flagged experts should be
    exempted from pruning regardless of their `expert_importance` score --
    or, more simply, entire early layers can be exempted outright, per the
    same paper's finding that Super Experts cluster in the first few layers.
    """
    threshold = torch.quantile(max_norms.float(), percentile / 100.0)
    return max_norms >= threshold


__all__ = [
    "AggregationFn",
    "aggregate",
    "expert_importance",
    "expert_max_output_norm",
    "flag_outlier_experts",
    "head_importance",
]
