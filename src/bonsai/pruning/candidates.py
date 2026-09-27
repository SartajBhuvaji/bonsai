"""Combine depth + expert selection into one candidate's prune indices.

Pure index arithmetic over already-computed importance scores -- no model,
no GPU. `bonsai.pruning.depth.prune_layers` and
`bonsai.pruning.experts.prune_experts` do the actual slicing once these
indices are known; this module only decides *which* indices, reproducibly,
from real committed score tensors
(`runs/20260919T150941Z-ff00be-importance-scores/importance-scores.pt`).

`08_combined_pruning_sweep.ipynb`'s own selection logic was never
committed (it ran in a scratch script, deleted after producing the run's
result artifact, same pattern as every other sweep in this project) --
this module reconstructs it so a specific combined candidate can actually
be built and saved, not just have its zero-shot loss read off a plot.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from bonsai.pruning.experts import select_experts_to_prune


def select_depth_experts_candidate(
    depth_scores: torch.Tensor,
    expert_scores: torch.Tensor,
    *,
    depth_n: int,
    expert_n: int,
    protected: Mapping[int, Sequence[int]] | None = None,
) -> tuple[list[int], dict[int, list[int]]]:
    """Pick the `depth_n` lowest-BI layers, then the `expert_n` lowest-scoring
    experts per *surviving* layer -- the depth+experts combination from
    `08_combined_pruning_sweep.ipynb`, generalized to any tier's counts.

    Returns `(layers_to_prune, expert_indices_by_new_layer)`. The second
    is keyed by each surviving layer's position *after* depth pruning
    (0-indexed among survivors), ready to pass straight into
    `prune_experts` once `prune_layers(model, layers_to_prune)` has
    already run -- matching the paper's own note (sec:combined-pruning-sweep)
    that expert/head selection has to be remapped to the model's new,
    post-depth-pruning layer numbering, not its original one.

    `protected`: experts keyed by *original* layer index (e.g. the three
    known Super Experts, arXiv:2507.23279) that must never be selected for
    pruning regardless of score, in whichever surviving layer they end up
    at after remapping.
    """
    num_layers = depth_scores.shape[0]
    layers_to_prune = (
        torch.topk(depth_scores, depth_n, largest=False).indices.sort().values.tolist()
    )
    pruned_set = set(layers_to_prune)
    surviving_layers = sorted(set(range(num_layers)) - pruned_set)
    old_to_new = {old: new for new, old in enumerate(surviving_layers)}

    expert_indices_by_new_layer: dict[int, list[int]] = {}
    for old_idx in surviving_layers:
        scores_row = expert_scores[old_idx : old_idx + 1]
        row_protected = {0: list((protected or {}).get(old_idx, []))}
        selected = select_experts_to_prune(scores_row, expert_n, protected=row_protected)
        expert_indices_by_new_layer[old_to_new[old_idx]] = selected[0]

    return layers_to_prune, expert_indices_by_new_layer


__all__ = ["select_depth_experts_candidate"]
