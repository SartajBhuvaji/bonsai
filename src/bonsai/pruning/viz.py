"""Visualizations for pruning-sweep results.

Kept separate from `bonsai.pruning` itself, same reasoning as
`bonsai.importance.viz`: matplotlib lives in the `notebook` extra, not the
package's real runtime dependencies -- code that prunes a model should
never have to pull in a plotting library to do it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from matplotlib.figure import Figure

# Same fixed categorical order as bonsai.importance.viz (blue, orange, aqua,
# yellow) -- kept as a separate literal here rather than a shared import,
# since bonsai.pruning has no other reason to depend on bonsai.importance.
_CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


def plot_pruning_sweep(
    counts: Sequence[int],
    named_losses: Mapping[str, Sequence[float]],
    random_losses: Sequence[Sequence[float]],
    *,
    baseline_loss: float,
    log_scale: bool = True,
    title: str = "Zero-shot loss vs. layers pruned",
    xlabel: str = "layers pruned",
    extra_hlines: Mapping[str, float] | None = None,
) -> Figure:
    """Zero-shot loss vs. how much of a pruning axis was removed, one line per strategy.

    Axis-agnostic despite the depth-pruning-flavored defaults (this is where
    the sweep was first built): `title`/`xlabel` let a caller relabel the
    x-axis for a different axis's count (e.g. experts pruned per layer)
    without duplicating this function.

    `named_losses[name][i]` is that strategy's loss at `counts[i]` -- e.g.
    `{"bi_greedy": [...], "ppl_greedy": [...], "contiguous_bi": [...]}`,
    colors assigned in insertion order from the project's fixed categorical
    order (at most 3 named strategies, leaving the 4th categorical slot for
    the random control below).

    `random_losses[i]` is every random seed's loss at `counts[i]` (shape
    `[len(counts)][num_seeds]`) -- shown as a shaded min-max band plus its
    mean, not a single line, since one random draw isn't representative on
    its own; this is the control every principled strategy above needs to
    actually beat, not just correlate well with each other.

    `baseline_loss` (nothing pruned) is drawn as a dashed horizontal
    reference. `extra_hlines` adds further named horizontal references (e.g.
    a single fixed-cost candidate outside the main count sweep, like pruning
    only a few known-critical units) in a distinct warning color, not the
    categorical series palette, since they're regression markers, not
    strategies being compared across the x-axis. Loss typically spans more
    than an order of magnitude once enough is pruned (a handful of unlucky
    random draws can blow up catastrophically), so the y-axis defaults to
    log scale.
    """
    if len(named_losses) > 3:
        raise ValueError(
            f"at most 3 named strategies supported (4th categorical slot is "
            f"reserved for the random control), got {len(named_losses)}"
        )
    lengths = {len(v) for v in named_losses.values()} | {len(random_losses)}
    if lengths != {len(counts)}:
        raise ValueError(
            f"expected every series to have one value per count ({len(counts)}); "
            f"got lengths {lengths}"
        )

    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 5))

    for color, (name, losses) in zip(_CATEGORICAL, named_losses.items(), strict=False):
        ax.plot(counts, losses, color=color, linewidth=1.8, marker="o", markersize=4, label=name)

    random_arr = np.array(random_losses, dtype=float)  # [len(counts), num_seeds]
    random_mean = random_arr.mean(axis=1)
    random_min = random_arr.min(axis=1)
    random_max = random_arr.max(axis=1)
    random_color = _CATEGORICAL[len(named_losses)]
    ax.plot(
        counts,
        random_mean,
        color=random_color,
        linewidth=1.8,
        marker="o",
        markersize=4,
        linestyle="--",
        label=f"random (n={random_arr.shape[1]} seeds, mean)",
    )
    ax.fill_between(counts, random_min, random_max, color=random_color, alpha=0.15, linewidth=0)

    ax.axhline(
        baseline_loss, color="#7a869a", linewidth=1.0, linestyle=":", label="baseline (0 pruned)"
    )
    for name, loss in (extra_hlines or {}).items():
        ax.axhline(loss, color="#c0392b", linewidth=1.2, linestyle="--", label=name)

    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("mean cross-entropy loss" + (" (log scale)" if log_scale else ""), fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def plot_combined_pruning_sweep(
    tiers: Sequence[str],
    combo_losses: Mapping[str, Sequence[float]],
    *,
    baseline_loss: float,
    log_scale: bool = True,
    title: str = "Zero-shot loss by combination and aggressiveness tier",
) -> Figure:
    """Grouped bar chart: one group per tier, one bar per axis-combination type.

    Unlike `plot_pruning_sweep` (one continuous count axis per strategy),
    combined-axis candidates don't share a single scalar "how much was
    pruned" -- each tier fixes a different count per axis, chosen so each
    axis's own zero-shot cost lands in a similar band (see plan.md 2.4),
    not a single sweepable number. A grouped bar chart compares them
    directly instead of forcing them onto a shared x-axis that doesn't
    really exist.

    `combo_losses[name][i]` is that combination's loss at `tiers[i]` --
    e.g. `{"depth_experts": [...], "depth_heads": [...],
    "experts_heads": [...], "all_three": [...]}`, colors assigned in
    insertion order from the project's fixed categorical order (at most 4
    combinations, one per slot).

    `baseline_loss` (nothing pruned) is drawn as a dashed horizontal
    reference. Loss can span more than an order of magnitude across tiers,
    so the y-axis defaults to log scale.
    """
    if len(combo_losses) > 4:
        raise ValueError(f"at most 4 combinations supported, got {len(combo_losses)}")
    lengths = {len(v) for v in combo_losses.values()}
    if lengths != {len(tiers)}:
        raise ValueError(
            f"expected every combination to have one value per tier ({len(tiers)}); "
            f"got lengths {lengths}"
        )

    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 5))

    n_combos = len(combo_losses)
    group_width = 0.8
    bar_width = group_width / n_combos
    x = np.arange(len(tiers))

    for i, (color, (name, losses)) in enumerate(
        zip(_CATEGORICAL, combo_losses.items(), strict=False)
    ):
        offset = (i - (n_combos - 1) / 2) * bar_width
        ax.bar(x + offset, losses, width=bar_width * 0.9, color=color, label=name)

    ax.axhline(
        baseline_loss, color="#7a869a", linewidth=1.0, linestyle=":", label="baseline (0 pruned)"
    )

    if log_scale:
        ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("aggressiveness tier", fontsize=9)
    ax.set_ylabel("mean cross-entropy loss" + (" (log scale)" if log_scale else ""), fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


__all__ = ["plot_combined_pruning_sweep", "plot_pruning_sweep"]
