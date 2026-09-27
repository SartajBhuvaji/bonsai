"""Tests for `bonsai.pruning.candidates.select_depth_experts_candidate`.

Pure index arithmetic -- no model, no GPU -- so this is testable directly,
unlike `prune_layers`/`prune_experts` themselves (already covered by
`test_pruning_depth.py`/`test_pruning_experts.py`'s real-tiny-model tests).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from bonsai.pruning.candidates import select_depth_experts_candidate

IMPORTANCE_SCORES_PATH = Path("runs/20260919T150941Z-ff00be-importance-scores/importance-scores.pt")

# `runs/` is gitignored (real GPU-run artifacts, not committed) -- this one
# real-data regression test only runs where that specific file happens to
# exist (this machine, today), not in CI or a fresh clone. Every other test
# in this file uses synthetic tensors precisely so it doesn't have this
# limitation.
requires_committed_scores = pytest.mark.skipif(
    not IMPORTANCE_SCORES_PATH.exists(),
    reason=f"{IMPORTANCE_SCORES_PATH} is a local, gitignored run artifact, not committed",
)


def test_selects_exactly_depth_n_layers_and_expert_n_per_surviving_layer() -> None:
    torch.manual_seed(0)
    depth_scores = torch.randn(12)
    expert_scores = torch.randn(12, 16)

    layers_to_prune, experts_by_layer = select_depth_experts_candidate(
        depth_scores, expert_scores, depth_n=4, expert_n=6
    )

    assert len(layers_to_prune) == 4
    assert len(set(layers_to_prune)) == 4  # no duplicates
    assert len(experts_by_layer) == 12 - 4  # one entry per surviving layer
    assert all(len(experts) == 6 for experts in experts_by_layer.values())


def test_new_layer_keys_are_contiguous_from_zero() -> None:
    torch.manual_seed(1)
    depth_scores = torch.randn(10)
    expert_scores = torch.randn(10, 8)

    _, experts_by_layer = select_depth_experts_candidate(
        depth_scores, expert_scores, depth_n=3, expert_n=2
    )

    assert sorted(experts_by_layer.keys()) == list(range(10 - 3))


def test_protected_expert_is_never_selected_even_with_lowest_score() -> None:
    depth_scores = torch.tensor([5.0, 5.0, 5.0])  # nothing pruned by depth (depth_n=0)
    expert_scores = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],  # expert 0 has the lowest score in layer 0
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )

    _, experts_by_layer = select_depth_experts_candidate(
        depth_scores,
        expert_scores,
        depth_n=0,
        expert_n=1,
        protected={0: [0]},
    )

    # Layer 0's lowest-score expert (0) is protected, so expert 1 (next lowest) is
    # selected for pruning instead.
    assert experts_by_layer[0] == [1]


def test_protected_keys_are_original_not_new_layer_indices() -> None:
    # Depth-prune layer 0, so surviving layer 1 becomes new layer 0 -- protection
    # keyed at the *original* index 1 must still apply after remapping.
    depth_scores = torch.tensor([0.0, 5.0, 5.0])
    expert_scores = torch.tensor(
        [
            [1.0, 1.0],
            [0.1, 0.2],  # original layer 1 -> new layer 0; expert 0 is protected
            [1.0, 1.0],
        ]
    )

    layers_to_prune, experts_by_layer = select_depth_experts_candidate(
        depth_scores,
        expert_scores,
        depth_n=1,
        expert_n=1,
        protected={1: [0]},
    )

    assert layers_to_prune == [0]
    assert experts_by_layer[0] == [1]  # expert 0 protected -> expert 1 pruned instead


@requires_committed_scores
def test_reproduces_the_papers_aggressive_tier_depth_experts_selection() -> None:
    """Regression test against the real, committed importance scores this
    project's own combined-pruning-sweep (08_combined_pruning_sweep.ipynb,
    aggressive tier, depth+experts) measured a zero-shot loss of
    1.8479815144091845 for. `tools/build_candidate.py` re-measures that
    loss for real once a GPU is available; this test locks in that the
    *selection* itself is deterministic and reproducible from committed
    data alone, so a drift here is caught without needing a GPU at all.
    """
    scores = torch.load(IMPORTANCE_SCORES_PATH, map_location="cpu", weights_only=False)
    protected = {1: [68], 2: [92], 3: [82]}  # the three known Super Experts

    layers_to_prune, experts_by_layer = select_depth_experts_candidate(
        scores["depth_scores"],
        scores["expert_scores"],
        depth_n=16,
        expert_n=80,
        protected=protected,
    )

    assert layers_to_prune == [9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 22, 23, 24, 33, 34, 36]
    assert len(experts_by_layer) == 32
    assert all(len(experts) == 80 for experts in experts_by_layer.values())

    # Layers 1/2/3 all survive depth pruning and keep their original numbering
    # (nothing before layer 9 is pruned), so their Super Expert stays protected
    # at the same new-layer index.
    assert experts_by_layer[1] and 68 not in experts_by_layer[1]
    assert experts_by_layer[2] and 92 not in experts_by_layer[2]
    assert experts_by_layer[3] and 82 not in experts_by_layer[3]
