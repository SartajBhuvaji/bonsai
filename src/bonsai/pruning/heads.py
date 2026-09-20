"""Attention-head pruning: shrinking each layer's query-head count.

Confirmed against the installed `Qwen3MoeAttention` source: `q_proj` is
`Linear(hidden_size, num_attention_heads * head_dim)`, `o_proj` is
`Linear(num_attention_heads * head_dim, hidden_size)` -- pruning a query
head means slicing `q_proj`'s output rows and `o_proj`'s input columns at
that head's `head_dim`-wide slice. `k_proj`/`v_proj`
(`num_key_value_heads * head_dim`) and `q_norm`/`k_norm` (a single
`RMSNorm` over `head_dim`, applied identically to every head, not a
per-head parameter) are untouched by pruning a query head.

Qwen3's GQA groups every `num_key_value_groups = num_attention_heads //
num_key_value_heads` *consecutive* query heads onto one shared key/value
head (`repeat_kv`, called every forward pass with
`module.num_key_value_groups` read fresh off the attention module --
real bug this module exists to avoid, same class of issue as depth
pruning's `self_attn.layer_idx`: get this wrong and every surviving layer
silently repeats key/value states the wrong number of times instead of
raising). Per plan.md 2.2's decision: prune the same count of query heads
from every key/value group, never an entire group to zero (which would
orphan its key/value head with nothing routing to it) -- a uniform
per-group count, not a free per-head selection, so `num_key_value_groups`
stays a single well-defined integer after pruning, exactly what the
standard GQA attention implementation assumes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM


def prune_heads(
    model: Qwen3MoeForCausalLM, head_indices_by_layer: Mapping[int, Sequence[int]]
) -> None:
    """Permanently remove the given query heads from each named layer, in place.

    `head_indices_by_layer`: `{layer_idx: [head_idx, ...]}` -- which query
    heads to drop, per layer, in the original (pre-pruning) `0..num_heads-1`
    indexing. Every layer's `q_proj`/`o_proj` weights (and biases, if
    present) are sliced together at head boundaries, and
    `self_attn.num_key_value_groups` is recomputed for the new head count
    -- `k_proj`/`v_proj`/`num_key_value_heads` are never touched. Requires
    the same number of pruned heads in every key/value group (see module
    docstring); a ragged per-group count raises, since `num_key_value_groups`
    would no longer be a single well-defined integer.

    If every layer in the model ends up with the same query-head count
    (true for every uniform-per-layer call this project actually makes,
    e.g. via `select_heads_to_prune`), `model.config.num_attention_heads`
    is updated to match -- only read at module *construction* time
    otherwise (every head-count-dependent computation inside
    `Qwen3MoeAttention.forward` infers the live count from `q_proj`'s
    actual tensor shape, not from config), so nothing breaks today by
    leaving it stale, but a later `save_pretrained` + reload would. Left
    as-is (not updated) if the resulting per-layer counts aren't actually
    uniform across the whole model, since a scalar config field can't
    describe that.
    """
    for layer_idx, prune_idx in head_indices_by_layer.items():
        attn = model.model.layers[layer_idx].self_attn
        head_dim = attn.head_dim
        num_heads = attn.q_proj.out_features // head_dim
        num_kv_heads = attn.k_proj.out_features // head_dim
        group_size = num_heads // num_kv_heads

        indices = sorted({int(i) for i in prune_idx})
        if not indices:
            continue
        if indices[0] < 0 or indices[-1] >= num_heads:
            raise ValueError(
                f"layer {layer_idx}: head index out of range [0, {num_heads}): {indices}"
            )
        if len(indices) >= num_heads:
            raise ValueError(
                f"layer {layer_idx}: cannot prune all {num_heads} heads (got {len(indices)} indices)"
            )

        per_group_counts = [
            sum(1 for i in indices if group * group_size <= i < (group + 1) * group_size)
            for group in range(num_kv_heads)
        ]
        if len(set(per_group_counts)) > 1:
            raise ValueError(
                f"layer {layer_idx}: expected the same number of pruned heads in every "
                f"key/value group (got {per_group_counts} across {num_kv_heads} groups of "
                f"{group_size}); a ragged per-group count leaves num_key_value_groups undefined"
            )
        pruned_per_group = per_group_counts[0]
        if pruned_per_group >= group_size:
            raise ValueError(
                f"layer {layer_idx}: cannot prune an entire key/value group "
                f"(would orphan its key/value head); got {pruned_per_group} of "
                f"{group_size} heads pruned in some group"
            )

        keep_heads = torch.ones(num_heads, dtype=torch.bool, device=attn.q_proj.weight.device)
        keep_heads[indices] = False
        keep_dims = keep_heads.repeat_interleave(head_dim)

        attn.q_proj.weight = torch.nn.Parameter(attn.q_proj.weight.data[keep_dims].clone())
        if attn.q_proj.bias is not None:
            attn.q_proj.bias = torch.nn.Parameter(attn.q_proj.bias.data[keep_dims].clone())
        attn.q_proj.out_features = int(keep_dims.sum())

        attn.o_proj.weight = torch.nn.Parameter(attn.o_proj.weight.data[:, keep_dims].clone())
        attn.o_proj.in_features = int(keep_dims.sum())

        attn.num_key_value_groups = group_size - pruned_per_group

    head_counts = {
        layer.self_attn.q_proj.out_features // layer.self_attn.head_dim
        for layer in model.model.layers
    }
    if len(head_counts) == 1:
        (uniform_count,) = head_counts
        model.config.num_attention_heads = uniform_count


def select_heads_to_prune(
    head_scores: torch.Tensor, k_per_group: int, *, num_key_value_heads: int
) -> dict[int, list[int]]:
    """Pick the `k_per_group` lowest-scoring query heads from every key/value group, per layer.

    `head_scores`: `[num_layers, num_heads]` (e.g.
    `bonsai.importance.head_importance`, from a calibration run). A
    uniform count per key/value group, not a free per-head search, per
    `prune_heads`'s requirement (see module docstring) -- ranking happens
    independently within each contiguous group of `num_heads //
    num_key_value_heads` query heads, not across the whole layer, so a
    group's own lowest-scoring heads are the ones dropped from it
    regardless of how it compares to other groups.
    """
    num_layers, num_heads = head_scores.shape
    if num_heads % num_key_value_heads != 0:
        raise ValueError(
            f"num_heads ({num_heads}) must be divisible by num_key_value_heads "
            f"({num_key_value_heads})"
        )
    group_size = num_heads // num_key_value_heads
    if k_per_group >= group_size:
        raise ValueError(
            f"k_per_group ({k_per_group}) must be < group size ({group_size}) -- pruning an "
            f"entire key/value group would orphan its key/value head"
        )

    result: dict[int, list[int]] = {}
    for layer_idx in range(num_layers):
        pruned: list[int] = []
        for group in range(num_key_value_heads):
            start = group * group_size
            group_scores = head_scores[layer_idx, start : start + group_size]
            lowest = torch.topk(group_scores, k_per_group, largest=False).indices + start
            pruned.extend(lowest.tolist())
        result[layer_idx] = sorted(pruned)
    return result


__all__ = ["prune_heads", "select_heads_to_prune"]
