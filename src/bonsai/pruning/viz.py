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
        baseline_loss,
        color="#c3c2b7",  # chart chrome: baseline / axis reference
        linewidth=1.0,
        linestyle=":",
        label="baseline (0 pruned)",
    )
    for name, loss in (extra_hlines or {}).items():
        ax.axhline(
            loss, color="#d03b3b", linewidth=1.2, linestyle="--", label=name
        )  # status: critical

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
        baseline_loss,
        color="#c3c2b7",  # chart chrome: baseline / axis reference
        linewidth=1.0,
        linestyle=":",
        label="baseline (0 pruned)",
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


def plot_axis_isolation(
    series: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    *,
    baseline_pass_at_1: float,
    title: str = "Full-suite pass@1 vs. parameters, by axis",
    xlabel: str = "approx. parameters (B)",
) -> Figure:
    """Downstream accuracy vs. parameter count, one line per pruning axis.

    Unlike `plot_pruning_sweep`/`plot_combined_pruning_sweep` (zero-shot
    loss on a shared pruned-count or tier axis), this compares real
    post-retraining accuracy across candidates that each pruned only one
    axis (depth-only, experts-only) at varying parameter budgets --
    there's no shared "count" between axes, only a shared parameter count
    and a shared accuracy metric, so parameters (not a pruned-count) is the
    x-axis.

    `series[name] = (params, pass_at_1)` -- e.g. `{"experts-only": ([...],
    [...]), "depth-only": ([...], [...])}`, each pair already sorted by the
    caller (not resorted here, so callers control draw order). A series
    with 3 or more points is drawn with a connecting line; a series with
    only 1 or 2 points is drawn as markers only, with no line -- a line
    through 2 points visually implies a trend that isn't actually
    established by only 2 measurements. Colors assigned in insertion order
    from the project's fixed categorical order (at most 4 series).

    `baseline_pass_at_1` (unpruned teacher) is drawn as a dashed horizontal
    reference -- the ceiling every candidate is compared against. Unlike
    the loss-based sweep plots, the $y$-axis here is always linear (pass@1
    is already bounded in $[0,1]$, and the interesting comparisons are
    differences of a few percentage points, not orders of magnitude).
    """
    if len(series) > 4:
        raise ValueError(f"at most 4 series supported, got {len(series)}")
    for name, (params, pass_at_1) in series.items():
        if len(params) != len(pass_at_1):
            raise ValueError(
                f"series {name!r}: params and pass_at_1 must have the same length, "
                f"got {len(params)} and {len(pass_at_1)}"
            )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    for color, (name, (params, pass_at_1)) in zip(_CATEGORICAL, series.items(), strict=False):
        ax.plot(
            params,
            pass_at_1,
            color=color,
            linewidth=1.8,
            linestyle="-" if len(params) >= 3 else "none",
            marker="o",
            markersize=5,
            label=name,
        )

    ax.axhline(
        baseline_pass_at_1,
        color="#c3c2b7",  # chart chrome: baseline / axis reference
        linewidth=1.0,
        linestyle=":",
        label="baseline (unpruned)",
    )

    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("full-suite HumanEval-TS pass@1", fontsize=9)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def plot_n40_vs_full_suite(
    candidates: Sequence[str],
    n40_pass_at_1: Sequence[float],
    full_pass_at_1: Sequence[float],
    *,
    title: str = "n=40 sample vs. full-suite pass@1",
) -> Figure:
    """Grouped bar chart: one group per candidate, two bars (sampled vs. full).

    Every candidate checked against both an $n{=}40$ sample and the full
    159-problem suite scored lower on the full suite -- this makes that
    gap, and how inconsistently large it is across candidates, visible at
    a glance. `candidates[i]`'s two bars are `n40_pass_at_1[i]` and
    `full_pass_at_1[i]`; the first two categorical colors are used for the
    two bars (sampled, full), not assigned per candidate, since the
    comparison that matters is sampled-vs-full within each candidate, not
    across candidates. `candidates` should be kept short (a handful of
    characters each, e.g. `"d0/e60"`) -- labels are drawn horizontal, not
    rotated, matching every other sweep figure in this project.
    """
    lengths = {len(candidates), len(n40_pass_at_1), len(full_pass_at_1)}
    if len(lengths) != 1:
        raise ValueError(
            f"candidates, n40_pass_at_1, and full_pass_at_1 must have the same "
            f"length, got {len(candidates)}, {len(n40_pass_at_1)}, {len(full_pass_at_1)}"
        )

    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(candidates))
    bar_width = 0.35

    ax.bar(
        x - bar_width / 2, n40_pass_at_1, width=bar_width * 0.9, color=_CATEGORICAL[0], label="n=40"
    )
    ax.bar(
        x + bar_width / 2,
        full_pass_at_1,
        width=bar_width * 0.9,
        color=_CATEGORICAL[1],
        label="full (159)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(candidates, fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("candidate", fontsize=9)
    ax.set_ylabel("HumanEval-TS pass@1", fontsize=9)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def plot_inference_throughput(
    candidates: Sequence[str],
    vllm_tok_s: Sequence[float],
    sglang_tok_s: Sequence[float],
    *,
    title: str = "Serving throughput by candidate and engine",
) -> Figure:
    """Grouped bar chart: one group per candidate, two bars (vLLM, SGLang throughput).

    Same shape as `plot_n40_vs_full_suite` (candidate on the x-axis, two
    bars per candidate), reused here because the comparison is the same
    kind -- two measurements per candidate that should be read side by
    side, not a continuous sweep. `candidates[i]`'s two bars are
    `vllm_tok_s[i]` and `sglang_tok_s[i]`; the first two categorical colors
    are the two engines, not assigned per candidate, since the comparison
    that matters is engine vs. engine within each candidate. `candidates`
    should be kept short (e.g. `"d0/e64"`), drawn horizontal, matching
    every other sweep figure in this project.
    """
    lengths = {len(candidates), len(vllm_tok_s), len(sglang_tok_s)}
    if len(lengths) != 1:
        raise ValueError(
            f"candidates, vllm_tok_s, and sglang_tok_s must have the same "
            f"length, got {len(candidates)}, {len(vllm_tok_s)}, {len(sglang_tok_s)}"
        )

    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(candidates))
    bar_width = 0.35

    ax.bar(
        x - bar_width / 2, vllm_tok_s, width=bar_width * 0.9, color=_CATEGORICAL[0], label="vLLM"
    )
    ax.bar(
        x + bar_width / 2,
        sglang_tok_s,
        width=bar_width * 0.9,
        color=_CATEGORICAL[1],
        label="SGLang",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(candidates, fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("candidate", fontsize=9)
    ax.set_ylabel("output throughput (tok/s)", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


__all__ = [
    "plot_axis_isolation",
    "plot_combined_pruning_sweep",
    "plot_inference_throughput",
    "plot_n40_vs_full_suite",
    "plot_pruning_sweep",
]
