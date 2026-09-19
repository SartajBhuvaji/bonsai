"""Shared activation-based importance scoring for Phase 2's pruning axes.

Mirrors how 2407.14679 Section 2.2 actually structures this, not a guess at
it: one shared primitive -- aggregate an activation-derived quantity over a
set of tokens with a *pluggable* aggregation function (the paper ablates
mean(abs), L2 norm and variance in its Table 13 rather than hardcoding one,
since the best choice isn't the same for every axis) -- wrapped by thin,
axis-specific functions that each know which activations to feed it.

Two of Phase 2's three axes are thin wrappers around `aggregate`:

- `head_importance` (2.2): the paper's own `F_head` formula, unchanged.
- `expert_importance` (2.3, landing separately): the gate-weighted
  expert-output-norm formula plan.md settled on. Structurally different from
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


__all__ = [
    "AggregationFn",
    "aggregate",
    "head_importance",
]
