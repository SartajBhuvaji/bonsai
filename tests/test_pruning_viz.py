"""Smoke tests for the pruning-sweep visualization.

Same style as bonsai.importance.viz's own tests: not pixel comparisons,
just that it draws without error and that shape validation rejects
mismatched inputs.
"""

import matplotlib

matplotlib.use("Agg")  # no display in CI

from bonsai.pruning.viz import plot_pruning_sweep


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


def test_plot_pruning_sweep_rejects_mismatched_lengths() -> None:
    counts = [1, 2, 4]
    named_losses = {"bi_greedy": [1.0, 1.1]}  # wrong length
    random_losses = [[1.1], [1.2], [1.3]]

    try:
        plot_pruning_sweep(counts, named_losses, random_losses, baseline_loss=0.97)
    except ValueError:
        return
    raise AssertionError("expected ValueError for a mismatched series length")
