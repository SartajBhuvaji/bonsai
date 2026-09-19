"""Depth pruning: permanently removing decoder layers.

The permanent counterpart to `bonsai.importance.depth.skip_layer`'s
temporary identity-swap used for scoring. Scoring only ever *simulates*
removing a layer, one at a time, to rank all of them; this module actually
removes the chosen set from the model.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM


def prune_layers(model: Qwen3MoeForCausalLM, layer_indices: Iterable[int]) -> None:
    """Permanently remove the given decoder layers from `model`, in place.

    Confirmed against the installed `Qwen3MoeModel.forward` source: it
    iterates `self.layers[: self.config.num_hidden_layers]`, so both the
    module list and the config's layer count have to change together, or
    the two go out of sync and the slice either silently drops surviving
    layers or reads past the (now shorter) list.

    Each surviving layer's `self_attn.layer_idx` is renumbered to its new,
    contiguous position (0-indexed) -- required, not optional, and found by
    running the pruned model rather than by reading the source alone: the
    default KV cache indexes into an internal list by `layer_idx`
    (`past_key_values.update(key_states, value_states, self.layer_idx)` in
    the installed `Qwen3MoeAttention.forward`), and without renumbering,
    surviving layers keep their original, now-non-contiguous indices (e.g.
    `0, 2, 4, 5` after pruning layers 1 and 3 from a 6-layer model), which
    raises `IndexError` the moment the cache is actually used, even though
    nothing about `Qwen3MoeAttention.forward`'s own logic depends on it.
    """
    indices = sorted({int(i) for i in layer_indices})
    num_layers = len(model.model.layers)
    if not indices:
        raise ValueError("layer_indices must be non-empty")
    if indices[0] < 0 or indices[-1] >= num_layers:
        raise ValueError(f"layer index out of range [0, {num_layers}): {indices}")
    if len(indices) >= num_layers:
        raise ValueError(f"cannot prune all {num_layers} layers (got {len(indices)} indices)")

    to_prune = set(indices)
    surviving = [layer for i, layer in enumerate(model.model.layers) if i not in to_prune]
    for new_idx, layer in enumerate(surviving):
        layer.self_attn.layer_idx = new_idx
    model.model.layers = torch.nn.ModuleList(surviving)
    model.config.num_hidden_layers = len(surviving)


__all__ = ["prune_layers"]
