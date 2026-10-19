"""Smoke tests for the pruning-sweep visualization.

Same style as bonsai.importance.viz's own tests: not pixel comparisons,
just that it draws without error and that shape validation rejects
mismatched inputs.
"""

import matplotlib

matplotlib.use("Agg")  # no display in CI

from bonsai.pruning.viz import (
    plot_axis_isolation,
    plot_combined_pruning_sweep,
    plot_n40_vs_full_suite,
    plot_pruning_sweep,
)


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


def test_plot_axis_isolation_draws() -> None:
    series = {
        "experts-only": ([5.15, 16.03, 23.28], [0.082, 0.579, 0.604]),
        "depth-only": ([15.58, 20.56], [0.107, 0.283]),
    }

    fig = plot_axis_isolation(series, baseline_pass_at_1=0.717)

    assert len(fig.axes) == 1
    assert fig.axes[0].get_ylim() == (0.0, 1.0)


def test_plot_axis_isolation_only_connects_three_or_more_points() -> None:
    series = {
        "experts-only": ([5.15, 16.03, 23.28], [0.082, 0.579, 0.604]),
        "depth-only": ([15.58, 20.56], [0.107, 0.283]),
    }

    fig = plot_axis_isolation(series, baseline_pass_at_1=0.717)

    lines = {line.get_label(): line for line in fig.axes[0].get_lines()}
    assert lines["experts-only"].get_linestyle() == "-"
    assert lines["depth-only"].get_linestyle() == "None"


def test_plot_axis_isolation_rejects_too_many_series() -> None:
    series = {
        "a": ([1.0], [0.5]),
        "b": ([1.0], [0.5]),
        "c": ([1.0], [0.5]),
        "d": ([1.0], [0.5]),
        "e": ([1.0], [0.5]),
    }

    try:
        plot_axis_isolation(series, baseline_pass_at_1=0.717)
    except ValueError:
        return
    raise AssertionError("expected ValueError for more than 4 series")


def test_plot_axis_isolation_rejects_mismatched_lengths() -> None:
    series = {"experts-only": ([5.15, 16.03], [0.082])}  # wrong length

    try:
        plot_axis_isolation(series, baseline_pass_at_1=0.717)
    except ValueError:
        return
    raise AssertionError("expected ValueError for a mismatched series length")


def test_plot_n40_vs_full_suite_draws() -> None:
    candidates = ["aggressive", "depth0/experts97"]
    n40 = [0.50, 0.40]
    full = [0.208, 0.283]

    fig = plot_n40_vs_full_suite(candidates, n40, full)

    assert len(fig.axes) == 1
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == candidates


def test_plot_n40_vs_full_suite_rejects_mismatched_lengths() -> None:
    candidates = ["a", "b", "c"]
    n40 = [0.5, 0.4]  # wrong length
    full = [0.2, 0.3, 0.1]

    try:
        plot_n40_vs_full_suite(candidates, n40, full)
    except ValueError:
        return
    raise AssertionError("expected ValueError for a mismatched series length")
