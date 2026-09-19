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

# Fixed categorical order (slots 1-4: blue, orange, aqua, yellow) for charts
# with more than one series -- assigned in this order, never cycled, per the
# project's color formula. Four slots, not the full eight: the documented
# default palette only clears the *adjacent*-pairs CVD/contrast gates this
# far in strict slot order (line charts only ever have neighbors touch);
# `plot_metric_agreement`'s scatter never uses more than one series, so the
# stricter all-pairs gate (which caps at three slots) doesn't apply here.
_CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


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


def plot_layer_importance(
    scores: torch.Tensor,
    *,
    title: str = "Block Importance (BI) per layer",
    ylabel: str = "BI score",
    annotation_label: str = "lowest BI",
) -> Figure:
    """Bar chart of a per-layer depth-scoring metric. `scores`: `[num_layers]`.

    One series -- a flat slot-1 blue fill, no legend needed, per the
    project's color formula (a legend only earns its place at >= 2 series).
    Works for either of plan.md 2.1's two depth criteria: Block Importance
    (BI, the default title/label) or PPL delta -- both share the same "low
    score = this layer is close to a no-op" interpretation, just at
    different absolute scales, so the lowest-scoring layer is called out
    directly either way.
    """
    if scores.ndim != 1:
        raise ValueError(f"expected scores: [num_layers], got shape {tuple(scores.shape)}")
    import matplotlib.pyplot as plt

    data = scores.detach().cpu().float().numpy()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(data)), data, color="#2a78d6")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("layer", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)

    lowest = int(data.argmin())
    span = float(data.max() - data.min()) or 1.0
    ax.annotate(
        f"layer {lowest}\n{annotation_label}",
        xy=(lowest, data[lowest]),
        xytext=(lowest, data[lowest] + 0.08 * span),
        ha="center",
        fontsize=8,
        color="#4a4a4a",
        arrowprops={"arrowstyle": "-", "color": "#7a869a", "linewidth": 0.8},
    )
    fig.tight_layout()
    return fig


def plot_layer_summary(scores: dict[str, torch.Tensor]) -> Figure:
    """Per-layer scores from independently-computed methods, normalized and overlaid.

    `scores`: an ordered mapping from series label to a `[num_layers]`
    tensor, already reduced to one value per layer by the caller (e.g.
    `head_scores.mean(dim=1)` for the head axis, not the raw
    `[num_layers, num_heads]` heatmap data -- this function doesn't know or
    care where a series came from). Colors are assigned in insertion order
    from the project's fixed categorical order, never cycled, so at most
    `len(_CATEGORICAL)` series are supported.

    The point: none of the heatmap/bar plots above can answer this alone --
    do independently-computed scoring methods (activation magnitude for
    heads/experts, cosine distance for BI, actual held-out perplexity for
    PPL) agree on which layers matter? Each series is normalized to its own
    [0, 1] range first, since the metrics have unrelated units and scales;
    only the *shape* across layers is comparable, not the absolute values
    between series.
    """
    if not scores:
        raise ValueError("expected at least one named score series")
    if len(scores) > len(_CATEGORICAL):
        raise ValueError(f"at most {len(_CATEGORICAL)} series supported, got {len(scores)}")
    shapes = {name: tuple(s.shape) for name, s in scores.items()}
    if any(s.ndim != 1 for s in scores.values()):
        raise ValueError(f"expected every series to be 1D [num_layers]; got shapes {shapes}")
    if len({s.shape[0] for s in scores.values()}) != 1:
        raise ValueError(f"expected every series to have the same length; got shapes {shapes}")

    import matplotlib.pyplot as plt

    def normalize(x: torch.Tensor) -> torch.Tensor:
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-12)

    num_layers = next(iter(scores.values())).shape[0]
    fig, ax = plt.subplots(figsize=(10, 4))
    for color, (name, series) in zip(_CATEGORICAL, scores.items(), strict=False):
        ax.plot(
            range(num_layers), normalize(series).numpy(), color=color, linewidth=1.6, label=name
        )
    ax.set_title("Per-layer summary: do independent scoring methods agree?", fontsize=11)
    ax.set_xlabel("layer", fontsize=9)
    ax.set_ylabel("normalized score (0-1, per-metric min-max)", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def plot_metric_agreement(
    x: torch.Tensor, y: torch.Tensor, *, x_label: str, y_label: str, title: str
) -> Figure:
    """Scatter one per-layer metric against another, with Pearson correlation annotated.

    Built for the question that matters once more than one depth-scoring
    criterion exists (plan.md 2.1): does a cheap proxy (BI, one forward
    pass) actually predict an expensive ground truth (PPL delta,
    `num_layers + 1` forward passes) well enough to trust in its place? Each
    point is one layer; the two layers with the highest `y` are labeled
    directly, since those are usually the ones a reader most wants to
    identify by index.
    """
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError(
            f"expected x and y to be 1D and the same shape; got {tuple(x.shape)}, {tuple(y.shape)}"
        )
    import matplotlib.pyplot as plt
    import numpy as np

    x_np = x.detach().cpu().float().numpy()
    y_np = y.detach().cpu().float().numpy()
    correlation = float(np.corrcoef(x_np, y_np)[0, 1])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x_np, y_np, color=_CATEGORICAL[0], s=30, alpha=0.8)

    for i in np.argsort(y_np)[-2:]:
        ax.annotate(
            f"layer {i}",
            xy=(x_np[i], y_np[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="#4a4a4a",
        )

    ax.set_title(f"{title}\nPearson r = {correlation:.2f}", fontsize=11)
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
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
    "plot_metric_agreement",
    "plot_score_distribution",
]
