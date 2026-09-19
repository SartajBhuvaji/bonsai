"""Depth (layer) importance -- Block Importance (BI).

Deliberately a separate module from `bonsai.importance`'s `aggregate` /
`head_importance` / `expert_importance` trio, per plan.md 1.3's own design:
the papers score depth via PPL-delta or Block Importance, never via the
activation-magnitude formula heads/experts use, and bonsai keeps that split
in code, not just in the docstring.
"""

from __future__ import annotations

import torch


def block_importance(layer_inputs: torch.Tensor, layer_outputs: torch.Tensor) -> torch.Tensor:
    """Block Importance (BI) for one layer: `1 - mean cosine_similarity(input, output)`.

    2408.11796's fast depth-pruning criterion. Computable from the same
    forward pass that produces these activations -- unlike PPL-based
    scoring, which removes a layer and re-runs the model once per candidate.
    A BI near 0 means the layer is close to a no-op (its output is nearly a
    copy of its input, e.g. a near-identity residual pass-through, a
    redundancy signal); a BI near 1 means the layer transforms its input
    substantially.

    `layer_inputs`/`layer_outputs`: `[num_tokens, hidden_size]`, the same
    layer's input and output activations over a batch of tokens (already
    flattened across batch and sequence). No pluggable aggregation function
    here, unlike `aggregate` -- BI is a per-token cosine similarity, meaned
    directly over tokens, matching the paper's own `E_{X,t}[...]` definition.
    Returns a 0-dim scalar tensor; the caller loops over layers to build a
    `[num_layers]` tensor.
    """
    if layer_inputs.shape != layer_outputs.shape or layer_inputs.ndim != 2:
        raise ValueError(
            "expected layer_inputs and layer_outputs both [num_tokens, hidden_size]; "
            f"got shapes {tuple(layer_inputs.shape)}, {tuple(layer_outputs.shape)}"
        )
    cos_sim = torch.nn.functional.cosine_similarity(layer_inputs, layer_outputs, dim=-1)
    return 1.0 - cos_sim.mean()


__all__ = ["block_importance"]
