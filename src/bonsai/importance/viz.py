"""Heatmap-style visualizations for `bonsai.importance` scores.

Kept separate from `bonsai.importance` on purpose, same reasoning as
`bonsai.architecture.viz`: this is the one place in the package that needs
matplotlib, which lives in the `notebook` extra, not the package's real
runtime dependencies. Pruning code scoring importance later should never
have to pull in a plotting library to get a number.

Sequential (magnitude) color throughout: one hue, light->dark, per the
project's data-viz convention -- the same validated blue ramp, and the same
light-diagram aesthetic `bonsai.architecture.viz` already established, so an
importance heatmap and the architecture diagram read as one visual system
rather than two unrelated styles.
"""

from __future__ import annotations

import torch
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from bonsai.architecture import ArchitectureSummary

# The dataviz skill's documented sequential blue ramp (steps 100/200/300/400/
# 500/600/700), light->dark. Not hand-picked -- lifted verbatim from the
# validated reference palette so this heatmap passes the same accessibility
# checks the palette instance already cleared.
_SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "bonsai_sequential_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
_OUTLINE_COLOR = "#eb6834"  # categorical slot 2 (orange) -- the outline, not a fill

# Fixed categorical order (slots 1/2/3: blue, orange, aqua) for the one chart
# here with more than one series -- assigned in this order, never cycled,
# per the project's color formula.
_CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a")


def _heatmap(
    ax: Axes,
    scores: torch.Tensor,
    *,
    title: str,
    xlabel: str,
    top_n_outline: int,
    vmax_percentile: float,
) -> None:
    data = scores.detach().cpu().float().numpy()

    # Importance scores here are extremely long-tailed by design -- a
    # handful of cells (Qwen3-30B-A3B's Super Experts, the last layer's
    # heads) can be 30-300x the median. A linear scale to the true max
    # saturates everything else into one indistinguishable pale blob,
    # hiding exactly the layer-to-layer/head-to-head structure this plot
    # exists to show. Clipping the color scale at a percentile (not the
    # max) spreads the full colormap across the typical cells instead;
    # `extend="max"` on the colorbar marks that clipping honestly rather
    # than silently, and the outline below still marks the true extremes.
    vmax = float(torch.quantile(scores.flatten().float(), vmax_percentile / 100.0))
    im = ax.imshow(data, cmap=_SEQUENTIAL_BLUE, aspect="auto", vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("layer", fontsize=9)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.02, extend="max")
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_ylabel(f"score (clipped at p{vmax_percentile:g})", fontsize=8)

    n = min(top_n_outline, scores.numel())
    if n > 0:
        top_flat = torch.topk(scores.flatten(), n).indices
        for idx in top_flat.tolist():
            row, col = divmod(idx, scores.shape[1])
            ax.add_patch(
                Rectangle(
                    (col - 0.5, row - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor=_OUTLINE_COLOR,
                    linewidth=1.4,
                )
            )


def plot_expert_heatmap(
    scores: torch.Tensor, *, top_n_outline: int = 5, vmax_percentile: float = 95.0
) -> Figure:
    """Layers x experts heatmap of `expert_importance` scores.

    `scores`: `[num_layers, num_experts]`. The `top_n_outline` highest-scoring
    cells across the whole model are outlined -- exactly where a reader's eye
    should land first (e.g. Qwen3-30B-A3B's Super Experts, arXiv 2507.23279,
    if this is real Qwen3-30B-A3B data). Generic on purpose: this outlines
    whatever the real scores say is highest, not a hardcoded set of known
    indices -- that comparison belongs in the notebook's narrative, not baked
    into reusable plotting code.

    `vmax_percentile` clips the color scale (not the outline, which still
    uses the true top `top_n_outline` cells) -- expert scores are extremely
    long-tailed (this project's own data: median ~0.6, p95 ~4.5, max ~192),
    so scaling color to the true max leaves almost the entire grid
    indistinguishable from zero.
    """
    if scores.ndim != 2:
        raise ValueError(
            f"expected scores: [num_layers, num_experts], got shape {tuple(scores.shape)}"
        )
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    _heatmap(
        ax,
        scores,
        title="Expert importance (gate-weighted output norm, per layer)",
        xlabel="expert",
        top_n_outline=top_n_outline,
        vmax_percentile=vmax_percentile,
    )
    fig.tight_layout()
    return fig


def plot_head_heatmap(
    scores: torch.Tensor, *, top_n_outline: int = 5, vmax_percentile: float = 95.0
) -> Figure:
    """Layers x heads heatmap of `head_importance` scores. `scores`: `[num_layers, num_heads]`.

    `vmax_percentile` clips the color scale the same way `plot_expert_heatmap`
    does, for the same reason -- head scores are long-tailed too, just less
    extremely (this project's own data: median ~32, p95 ~179, max ~968).
    """
    if scores.ndim != 2:
        raise ValueError(
            f"expected scores: [num_layers, num_heads], got shape {tuple(scores.shape)}"
        )
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    _heatmap(
        ax,
        scores,
        title="Attention head importance, per layer",
        xlabel="head",
        top_n_outline=top_n_outline,
        vmax_percentile=vmax_percentile,
    )
    fig.tight_layout()
    return fig


def plot_layer_importance(scores: torch.Tensor) -> Figure:
    """Bar chart of per-layer Block Importance (BI) scores. `scores`: `[num_layers]`.

    One series -- a flat slot-1 blue fill, no legend needed, per the
    project's color formula (a legend only earns its place at >= 2 series).
    A low BI score means a layer is close to a no-op (2408.11796's own
    naming for this metric, kept as-is rather than inverted for intuition);
    the lowest-scoring layer is called out directly since that's the one a
    reader most wants to find in a 48-bar chart at a glance.
    """
    if scores.ndim != 1:
        raise ValueError(f"expected scores: [num_layers], got shape {tuple(scores.shape)}")
    import matplotlib.pyplot as plt

    data = scores.detach().cpu().float().numpy()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(data)), data, color="#2a78d6")
    ax.set_title("Block Importance (BI) per layer", fontsize=11)
    ax.set_xlabel("layer", fontsize=9)
    ax.set_ylabel("BI score", fontsize=9)

    lowest = int(data.argmin())
    span = float(data.max() - data.min()) or 1.0
    ax.annotate(
        f"layer {lowest}\nlowest BI",
        xy=(lowest, data[lowest]),
        xytext=(lowest, data[lowest] + 0.08 * span),
        ha="center",
        fontsize=8,
        color="#4a4a4a",
        arrowprops={"arrowstyle": "-", "color": "#7a869a", "linewidth": 0.8},
    )
    fig.tight_layout()
    return fig


def plot_layer_summary(
    head_scores: torch.Tensor, expert_scores: torch.Tensor, depth_scores: torch.Tensor
) -> Figure:
    """Per-layer mean head/expert importance and BI, min-max normalized and overlaid.

    The three heatmap/bar plots above each show one axis in isolation; this
    is the one place all three come together, to answer a question none of
    them can alone: do independently-computed scoring methods (an
    activation-magnitude formula for heads, a gate-weighted one for experts,
    a cosine-distance one for depth) actually agree on which layers matter?
    Each series is normalized to its own [0, 1] range first -- the three
    metrics have unrelated units and scales, so only the *shape* across
    layers is comparable, not the absolute values between series.
    """
    if head_scores.ndim != 2 or expert_scores.ndim != 2 or depth_scores.ndim != 1:
        raise ValueError(
            "expected head_scores/expert_scores: [num_layers, *], depth_scores: [num_layers]; "
            f"got shapes {tuple(head_scores.shape)}, {tuple(expert_scores.shape)}, "
            f"{tuple(depth_scores.shape)}"
        )
    import matplotlib.pyplot as plt

    def normalize(x: torch.Tensor) -> torch.Tensor:
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-12)

    mean_head = normalize(head_scores.mean(dim=1))
    mean_expert = normalize(expert_scores.mean(dim=1))
    bi = normalize(depth_scores)

    fig, ax = plt.subplots(figsize=(10, 4))
    layers = range(head_scores.shape[0])
    ax.plot(
        layers, mean_head.numpy(), color=_CATEGORICAL[0], linewidth=1.6, label="mean head score"
    )
    ax.plot(
        layers, mean_expert.numpy(), color=_CATEGORICAL[1], linewidth=1.6, label="mean expert score"
    )
    ax.plot(layers, bi.numpy(), color=_CATEGORICAL[2], linewidth=1.6, label="block importance (BI)")
    ax.set_title("Per-layer summary: do the three scoring methods agree?", fontsize=11)
    ax.set_xlabel("layer", fontsize=9)
    ax.set_ylabel("normalized score (0-1, per-metric min-max)", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def plot_score_distribution(scores: torch.Tensor, *, title: str, xlabel: str) -> Figure:
    """Log-scale histogram of a flattened score tensor -- makes a long tail visible directly.

    Zero-valued entries (an expert never routed to in this calibration run)
    can't sit on a log axis; they're dropped from the histogram and counted
    in the title instead of silently vanishing.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    flat = scores.detach().cpu().float().flatten()
    zero_count = int((flat == 0).sum())
    positive = flat[flat > 0].numpy()

    fig, ax = plt.subplots(figsize=(8, 4))
    if positive.size == 0:
        ax.text(0.5, 0.5, "no positive scores", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=11)
        fig.tight_layout()
        return fig

    log_bins = np.logspace(np.log10(positive.min()), np.log10(positive.max()), 40).tolist()
    ax.hist(positive, bins=log_bins, color=_CATEGORICAL[0])
    ax.set_xscale("log")
    ax.set_title(
        title + (f"  ({zero_count} zero-valued, excluded)" if zero_count else ""), fontsize=11
    )
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("count", fontsize=9)
    fig.tight_layout()
    return fig


def plot_expert_concentration(expert_scores: torch.Tensor) -> Figure:
    """Per-layer cumulative-importance concentration curve.

    For each layer, sort experts by score descending and plot the cumulative
    share of that layer's total importance captured by the top N experts --
    a direct read on how aggressively experts could eventually be pruned
    (step 2.3) without giving up much of a layer's total importance mass.
    All 48 layers drawn thin and translucent (small-multiples-by-overplot,
    since 48 legible separate lines isn't realistic), with the mean curve
    bold on top; the diagonal is the "no concentration at all" reference --
    every real curve bowing above it is the concentration this plot exists
    to show.
    """
    if expert_scores.ndim != 2:
        raise ValueError(
            f"expected expert_scores: [num_layers, num_experts], got shape {tuple(expert_scores.shape)}"
        )
    import matplotlib.pyplot as plt

    num_layers, num_experts = expert_scores.shape
    x_pct = torch.linspace(0, 100, num_experts + 1)

    fig, ax = plt.subplots(figsize=(7, 7))
    curves = torch.zeros(num_layers, num_experts + 1)
    for i in range(num_layers):
        sorted_scores, _ = torch.sort(expert_scores[i], descending=True)
        total = sorted_scores.sum()
        cum = torch.cat([torch.zeros(1), sorted_scores.cumsum(0)])
        curves[i] = (cum / total * 100.0) if total > 0 else cum
        ax.plot(x_pct.numpy(), curves[i].numpy(), color=_CATEGORICAL[0], alpha=0.12, linewidth=0.8)

    ax.plot(
        x_pct.numpy(),
        curves.mean(dim=0).numpy(),
        color=_CATEGORICAL[0],
        linewidth=2.2,
        label="mean across layers",
    )
    ax.plot(
        [0, 100], [0, 100], color="#7a869a", linewidth=1.0, linestyle="--", label="no concentration"
    )
    ax.set_title("Expert importance concentration, per layer", fontsize=11)
    ax.set_xlabel("% of experts kept (highest-scoring first)", fontsize=9)
    ax.set_ylabel("% of layer's total importance captured", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def plot_expert_grid_for_layer(
    summary: ArchitectureSummary,
    layer_scores: torch.Tensor,
    *,
    layer_idx: int,
    vmax_percentile: float = 100.0,
) -> Figure:
    """One layer's real expert grid, colored by real importance scores.

    Same box-grid layout `bonsai.architecture.viz.plot_architecture` uses
    for its illustrative (binary active/inactive) expert grid, but colored
    by this layer's actual `expert_importance` scores instead -- the bridge
    between the abstract heatmap above and the real model diagram from
    `00_model_architecture.ipynb`.

    Defaults to the true max (`vmax_percentile=100`), unlike the two
    heatmaps above -- this view has only `num_experts` cells (128, not
    6,144), and its job is the opposite of the heatmap's: spotlighting a
    single standout expert, not revealing structure among the rest. A p95
    clip here saturates several ordinary experts to the same color as the
    real outlier (confirmed on this project's own layer 2: 7 of 128 experts
    tie for "max color" at p95, still 2 tied even at p99), which hides the
    exact thing this plot exists to show.
    """
    if layer_scores.ndim != 1 or layer_scores.shape[0] != summary.num_experts:
        raise ValueError(
            f"expected layer_scores: [{summary.num_experts}], got shape {tuple(layer_scores.shape)}"
        )
    import matplotlib.pyplot as plt

    vmax = float(torch.quantile(layer_scores.float(), vmax_percentile / 100.0)) or 1.0
    normalized = (layer_scores.float().clamp(max=vmax) / vmax).clamp(0, 1)

    n = summary.num_experts
    max_cols = 16
    cols = min(n, max_cols)
    rows = -(-n // cols)

    fig, ax = plt.subplots(figsize=(cols * 0.5, rows * 0.5 + 1))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_title(
        f"Layer {layer_idx} experts, colored by importance (clipped at p{vmax_percentile:g})",
        fontsize=11,
    )
    ax.axis("off")

    for i in range(n):
        row, col = divmod(i, cols)
        color = _SEQUENTIAL_BLUE(float(normalized[i]))
        ax.add_patch(
            Rectangle(
                (col + 0.08, rows - 1 - row + 0.08),
                0.84,
                0.84,
                facecolor=color,
                edgecolor="#3a4a63",
                linewidth=0.4,
            )
        )

    fig.tight_layout()
    return fig


__all__ = [
    "plot_expert_concentration",
    "plot_expert_grid_for_layer",
    "plot_expert_heatmap",
    "plot_head_heatmap",
    "plot_layer_importance",
    "plot_layer_summary",
    "plot_score_distribution",
]
