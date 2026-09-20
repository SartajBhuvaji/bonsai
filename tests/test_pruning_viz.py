"""Smoke tests for the pruning-sweep visualization.

Same style as bonsai.importance.viz's own tests: not pixel comparisons,
just that it draws without error and that shape validation rejects
mismatched inputs.
"""

import matplotlib

matplotlib.use("Agg")  # no display in CI

from bonsai.pruning.viz import plot_combined_pruning_sweep, plot_pruning_sweep


def test_plot_pruning_sweep_draws() -> None:
    counts = [1, 2, 4]
    named_losses = {
        "bi_greedy": [1.0, 1.1, 1.3],
        "ppl_greedy": [1.0, 1.05, 1.25],
        "contiguous_bi": [0.98, 1.0, 1.2],
    }
    random_losses = [[1.1, 1.2, 1.5], [1.3, 1.4, 1.6], [1.5, 2.0, 2.5]]

    fig = plot_pruning_sweep(counts, named_losses, random_losses, baseline_loss=0.97)

    assert len(fig.axes) == 1


def test_plot_pruning_sweep_linear_scale() -> None:
    counts = [1, 2]
    named_losses = {"bi_greedy": [1.0, 1.1]}
    random_losses = [[1.1, 1.2], [1.3, 1.4]]

    fig = plot_pruning_sweep(
        counts, named_losses, random_losses, baseline_loss=0.97, log_scale=False
    )

    assert len(fig.axes) == 1


def test_plot_pruning_sweep_rejects_too_many_named_strategies() -> None:
    counts = [1, 2]
    named_losses = {"a": [1.0, 1.1], "b": [1.0, 1.1], "c": [1.0, 1.1], "d": [1.0, 1.1]}
    random_losses = [[1.1], [1.2]]

    try:
        plot_pruning_sweep(counts, named_losses, random_losses, baseline_loss=0.97)
    except ValueError:
        return
    raise AssertionError("expected ValueError for more than 3 named strategies")


def test_plot_pruning_sweep_custom_labels_and_extra_hlines() -> None:
    counts = [8, 16]
    named_losses = {"greedy": [1.0, 1.2]}
    random_losses = [[1.1, 1.3], [1.2, 1.4]]

    fig = plot_pruning_sweep(
        counts,
        named_losses,
        random_losses,
        baseline_loss=0.97,
        title="Zero-shot loss vs. experts pruned per layer",
        xlabel="experts pruned per layer",
        extra_hlines={"super experts only": 1.47},
    )

    assert len(fig.axes) == 1
    assert fig.axes[0].get_xlabel() == "experts pruned per layer"


def test_plot_pruning_sweep_rejects_mismatched_lengths() -> None:
    counts = [1, 2, 4]
    named_losses = {"bi_greedy": [1.0, 1.1]}  # wrong length
    random_losses = [[1.1], [1.2], [1.3]]

    try:
        plot_pruning_sweep(counts, named_losses, random_losses, baseline_loss=0.97)
    except ValueError:
        return
    raise AssertionError("expected ValueError for a mismatched series length")


def test_plot_combined_pruning_sweep_draws() -> None:
    tiers = ["conservative", "moderate", "aggressive"]
    combo_losses = {
        "depth_experts": [1.25, 1.42, 1.85],
        "depth_heads": [1.19, 1.62, 2.80],
        "experts_heads": [1.17, 1.52, 2.27],
        "all_three": [1.37, 1.98, 3.76],
    }

    fig = plot_combined_pruning_sweep(tiers, combo_losses, baseline_loss=0.9668)

    assert len(fig.axes) == 1
    assert list(fig.axes[0].get_xticklabels()[i].get_text() for i in range(3)) == tiers


def test_plot_combined_pruning_sweep_linear_scale() -> None:
    tiers = ["conservative", "moderate"]
    combo_losses = {"depth_experts": [1.25, 1.42]}

    fig = plot_combined_pruning_sweep(tiers, combo_losses, baseline_loss=0.9668, log_scale=False)

    assert len(fig.axes) == 1


def test_plot_combined_pruning_sweep_rejects_too_many_combos() -> None:
    tiers = ["a", "b"]
    combo_losses = {
        "1": [1.0, 1.1],
        "2": [1.0, 1.1],
        "3": [1.0, 1.1],
        "4": [1.0, 1.1],
        "5": [1.0, 1.1],
    }

    try:
        plot_combined_pruning_sweep(tiers, combo_losses, baseline_loss=0.97)
    except ValueError:
        return
    raise AssertionError("expected ValueError for more than 4 combinations")


def test_plot_combined_pruning_sweep_rejects_mismatched_lengths() -> None:
    tiers = ["a", "b", "c"]
    combo_losses = {"depth_experts": [1.0, 1.1]}  # wrong length

    try:
        plot_combined_pruning_sweep(tiers, combo_losses, baseline_loss=0.97)
    except ValueError:
        return
    raise AssertionError("expected ValueError for a mismatched series length")
