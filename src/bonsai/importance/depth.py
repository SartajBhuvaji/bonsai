"""Depth (layer) importance -- Block Importance (BI).

Deliberately a separate module from `bonsai.importance`'s `aggregate` /
`head_importance` / `expert_importance` trio, per plan.md 1.3's own design:
the papers score depth via PPL-delta or Block Importance, never via the
activation-magnitude formula heads/experts use, and bonsai keeps that split
in code, not just in the docstring.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM


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


@contextmanager
def skip_layer(model: Qwen3MoeForCausalLM, layer_idx: int) -> Iterator[None]:
    """Temporarily make one decoder layer act as identity: output = input.

    Simulates "this layer is pruned" for scoring purposes without touching
    any weights -- 2407.14679's PPL criterion needs exactly this: remove a
    single layer, measure the model's perplexity with it gone, put it back.
    Confirmed against the installed `Qwen3MoeModel.forward` source: every
    decoder layer is called as `decoder_layer(hidden_states, attention_mask=...,
    ...)`, `hidden_states` always positional, and returns a plain tensor (not
    a tuple) -- `identity_forward` below only needs to match that shape,
    accepting and ignoring everything else. Swaps the layer's bound `forward`
    for the duration of the `with` block and restores the original
    afterward, even if the block raises.
    """
    layers = model.model.layers
    original_forward: Callable[..., torch.Tensor] = layers[layer_idx].forward

    def identity_forward(
        hidden_states: torch.Tensor,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> torch.Tensor:
        return hidden_states

    layers[layer_idx].forward = identity_forward
    try:
        yield
    finally:
        layers[layer_idx].forward = original_forward


def _mean_loss(model: Qwen3MoeForCausalLM, input_ids_batches: list[torch.Tensor]) -> float:
    losses = [float(model(input_ids=ids, labels=ids).loss) for ids in input_ids_batches]
    return sum(losses) / len(losses)


def ppl_importance(
    model: Qwen3MoeForCausalLM, input_ids_batches: list[torch.Tensor]
) -> torch.Tensor:
    """Score each decoder layer by the perplexity cost of removing it.

    2407.14679's PPL criterion: for each layer, temporarily skip it
    (`skip_layer`) and measure the model's mean cross-entropy loss over
    `input_ids_batches` relative to the unmodified baseline. A bigger
    increase means the layer matters more -- unlike Block Importance, this
    genuinely measures the layer's effect on the model's actual output
    quality, at the cost of one full forward pass per layer instead of one
    forward pass total (`num_layers + 1` passes altogether).

    Each element of `input_ids_batches` is a `[batch, seq_len]` token ID
    tensor already on the model's device; loss is computed via the model's
    own `labels=` path (the standard HF shifted cross-entropy, confirmed
    against the installed `Qwen3MoeForCausalLM.forward` source -- no manual
    shifting needed here). The mean loss is unweighted across batches,
    matching this module's existing convention of leaving batching
    decisions to the caller rather than baking them into the scorer.
    """
    num_layers = len(model.model.layers)
    scores = torch.zeros(num_layers)
    with torch.no_grad():
        baseline = _mean_loss(model, input_ids_batches)
        for i in range(num_layers):
            with skip_layer(model, i):
                scores[i] = _mean_loss(model, input_ids_batches) - baseline
    return scores


__all__ = ["block_importance", "ppl_importance", "skip_layer"]
