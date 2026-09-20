"""Expert pruning: permanently shrinking each layer's expert grid.

Confirmed against the installed `Qwen3MoeExperts`/`Qwen3MoeTopKRouter`
source: experts are two batched parameter tensors (`gate_up_proj`,
`down_proj`, expert index as dimension 0) plus a router weight matrix
(`Qwen3MoeTopKRouter.weight`, shape `[num_experts, hidden_size]`, also
expert index as dimension 0) -- pruning an expert means slicing that
dimension out of all three tensors together, consistently, per plan.md
2.3's router-handling decision: slice the pruned rows, keep top-k
unchanged, rely on `Qwen3MoeTopKRouter`'s own `norm_topk_prob` to
renormalize survivors (already true in Qwen3-30B-A3B-Base's config), no
router retraining before a zero-shot eval.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM


def prune_experts(
    model: Qwen3MoeForCausalLM, expert_indices_by_layer: Mapping[int, Sequence[int]]
) -> None:
    """Permanently remove the given experts from each named layer, in place.

    `expert_indices_by_layer`: `{layer_idx: [expert_idx, ...]}` -- which
    experts to drop, per sparse layer. Layers not present in the mapping
    are left untouched. Every entry's `gate_up_proj`/`down_proj`
    (`Qwen3MoeExperts`) and the corresponding `Qwen3MoeTopKRouter.weight`
    row are sliced together, so a surviving expert's position in all three
    tensors stays consistent with itself, even though (unlike depth
    pruning) nothing here needs renumbering -- experts have no equivalent
    of a layer's `self_attn.layer_idx` cache dependency, since
    `Qwen3MoeExperts.forward` re-derives which experts were hit from
    `top_k_index` fresh on every call, not from a fixed per-expert
    identity.

    Refuses to prune a layer below `router.top_k` (`config.num_experts_per_tok`)
    surviving experts -- routing's own `torch.topk(..., self.top_k, ...)`
    would otherwise fail with an opaque `RuntimeError` deep in the router's
    forward instead of a clear error here.

    If every sparse layer in the model ends up with the same expert count
    (true for every uniform-per-layer call this project actually makes,
    e.g. via `select_experts_to_prune`), `model.config.num_experts` and
    `model.num_experts` (the top-level count `load_balancing_loss_func`
    reads when `output_router_logits=True`) are updated to match -- both
    only ever read at module *construction* time otherwise, so nothing
    breaks today by leaving them stale, but a later `save_pretrained` +
    reload, or turning on the router aux loss for retraining, would. A
    scalar config field can't describe a genuinely non-uniform per-layer
    result, so this sync is skipped (config left as-is) if the resulting
    counts aren't actually uniform across every sparse layer.
    """
    for layer_idx, prune_idx in expert_indices_by_layer.items():
        layer = model.model.layers[layer_idx]
        if not hasattr(layer.mlp, "experts"):
            raise ValueError(f"layer {layer_idx} is not a sparse (MoE) layer")

        experts = layer.mlp.experts
        router = layer.mlp.gate
        num_experts = experts.num_experts

        indices = sorted({int(i) for i in prune_idx})
        if not indices:
            continue
        if indices[0] < 0 or indices[-1] >= num_experts:
            raise ValueError(
                f"layer {layer_idx}: expert index out of range [0, {num_experts}): {indices}"
            )
        if len(indices) >= num_experts:
            raise ValueError(
                f"layer {layer_idx}: cannot prune all {num_experts} experts "
                f"(got {len(indices)} indices)"
            )
        remaining = num_experts - len(indices)
        if remaining < router.top_k:
            raise ValueError(
                f"layer {layer_idx}: cannot prune to {remaining} experts, below "
                f"router.top_k ({router.top_k}) -- routing would have nothing left "
                f"to select for some tokens"
            )

        keep_mask = torch.ones(num_experts, dtype=torch.bool, device=experts.gate_up_proj.device)
        keep_mask[indices] = False

        experts.gate_up_proj = torch.nn.Parameter(experts.gate_up_proj.data[keep_mask].clone())
        experts.down_proj = torch.nn.Parameter(experts.down_proj.data[keep_mask].clone())
        experts.num_experts = int(keep_mask.sum())

        router.weight = torch.nn.Parameter(router.weight.data[keep_mask].clone())
        router.num_experts = experts.num_experts

    sparse_counts = {
        layer.mlp.experts.num_experts
        for layer in model.model.layers
        if hasattr(layer.mlp, "experts")
    }
    if len(sparse_counts) == 1:
        (uniform_count,) = sparse_counts
        model.config.num_experts = uniform_count
        model.num_experts = uniform_count


def select_experts_to_prune(
    expert_scores: torch.Tensor,
    n_per_layer: int,
    *,
    protected: Mapping[int, Sequence[int]] | None = None,
) -> dict[int, list[int]]:
    """Pick the `n_per_layer` lowest-scoring experts per layer to prune.

    `expert_scores`: `[num_layers, num_experts]` (e.g.
    `bonsai.importance.expert_importance`, from a calibration run).
    Uniform count per layer, per plan.md 2.3's decision -- a simplifying
    starting choice for tractability, not one settled by confirmed
    evidence either way (scoping research surfaced conflicting claims on
    this exact question, re-checked and corrected in plan.md's 2.3
    writeup -- not a per-layer search yet).

    `protected` -- e.g. Super Experts flagged by
    `bonsai.importance.flag_outlier_experts` -- are excluded from the
    pruning candidates *before* ranking, not just checked afterward: a
    protected expert's score is treated as `+inf` for this selection only,
    so it can never appear in the returned mapping regardless of how low
    its real score is.
    """
    num_layers, num_experts = expert_scores.shape
    if n_per_layer >= num_experts:
        raise ValueError(f"n_per_layer ({n_per_layer}) must be < num_experts ({num_experts})")

    result: dict[int, list[int]] = {}
    for layer_idx in range(num_layers):
        scores = expert_scores[layer_idx].clone()
        for protected_idx in (protected or {}).get(layer_idx, []):
            scores[protected_idx] = float("inf")
        result[layer_idx] = torch.topk(scores, n_per_layer, largest=False).indices.tolist()
    return result


__all__ = ["prune_experts", "select_experts_to_prune"]
