"""Draw a Qwen3-MoE model's architecture from an `ArchitectureSummary`.

Kept separate from `bonsai.architecture` on purpose: this is the one module
in the package that needs matplotlib, which lives in the `notebook` extra,
not in the package's real runtime dependencies. Pruning code importing
`bonsai.architecture` later should never have to pull in a plotting library
to get a parameter count.

Every number drawn here comes from the `ArchitectureSummary` passed in --
nothing is a hardcoded Qwen3 constant, so the same function draws any
Qwen3-MoE config correctly, not just the one this project targets.
"""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from bonsai.architecture import ArchitectureSummary

_BOX_STYLE = "round,pad=0.02,rounding_size=0.08"


def _box(
    ax: Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    label: str,
    *,
    color: str = "#dbe4f0",
    fontsize: int = 9,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle=_BOX_STYLE,
            facecolor=color,
            edgecolor="#3a4a63",
            linewidth=1.1,
        )
    )
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True,
    )


def _arrow(ax: Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            color="#3a4a63",
            linewidth=1.1,
        )
    )


def _residual_arc(ax: Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    """A curved skip connection: bypasses the sub-block, merges at the '+'.

    Without this, the '+' boxes drawn for attention and the MoE block just
    look like an unexplained pass-through step -- this is what actually
    marks them as residual connections rather than decoration.
    """
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            color="#7a869a",
            linewidth=1.1,
            linestyle="--",
            connectionstyle="arc3,rad=0.55",
        )
    )


def _plot_model_stack(ax: Axes, summary: ArchitectureSummary) -> None:
    """Left panel: embedding -> N decoder layers -> norm -> lm head."""
    ax.set_xlim(0, 10)
    ax.set_ylim(3.5, 20)
    ax.set_title("Whole model", fontsize=11)
    ax.axis("off")

    box_w, box_h = 8, 1.6
    x = 1

    def emb_label() -> str:
        return f"Embedding\n({summary.hidden_size} x vocab)"

    _box(ax, (x, 18), box_w, box_h, emb_label())
    _arrow(ax, (5, 18), (5, 17.3))

    # Three offset rectangles behind the labeled one suggest "a repeated
    # stack", cheaper to read at a glance than drawing all num_layers boxes.
    stack_y = 13.4
    for offset in (0.3, 0.15):
        ax.add_patch(
            Rectangle(
                (x + offset, stack_y + offset),
                box_w,
                box_h,
                facecolor="#eef1f6",
                edgecolor="#9aa7bb",
                linewidth=0.8,
            )
        )
    _box(ax, (x, stack_y), box_w, box_h, "Decoder layer\n(see right)", color="#f6dfc4")
    ax.text(
        x + box_w + 0.5,
        stack_y + box_h / 2,
        f"x {summary.num_layers}",
        ha="left",
        va="center",
        fontsize=10,
        fontweight="bold",
    )
    _arrow(ax, (5, 18), (5, stack_y + box_h + 0.1))
    _arrow(ax, (5, stack_y), (5, 10.3))

    _box(ax, (x, 8.7), box_w, box_h, "Final RMSNorm")
    _arrow(ax, (5, 8.7), (5, 8.0))

    lm_head_label = "LM head\n(tied to embedding)" if summary.lm_head == 0 else "LM head\n(untied)"
    _box(ax, (x, 6.4), box_w, box_h, lm_head_label)

    active_fraction = summary.experts_per_token / summary.num_experts
    active_experts_params = summary.experts_total * active_fraction
    active_total = (
        (summary.attention_total + summary.router_total + active_experts_params)
        + summary.embedding
        + summary.lm_head
    )
    ax.text(
        5,
        4.6,
        f"{summary.total / 1e9:.1f}B total params\n"
        f"~{active_total / 1e9:.1f}B active per token\n"
        f"({summary.experts_per_token} of {summary.num_experts} experts routed)",
        ha="center",
        va="top",
        fontsize=9,
        color="#4a4a4a",
    )


def _plot_layer_detail(ax: Axes, summary: ArchitectureSummary) -> None:
    """Right panel: one decoder layer's internals -- attention and MoE block."""
    ax.set_xlim(0, 10)
    ax.set_ylim(3.5, 20)
    ax.set_title("One decoder layer, expanded", fontsize=11)
    ax.axis("off")

    box_w = 6
    x = 2

    _box(ax, (x, 18.4), box_w, 1.0, "Input")
    _arrow(ax, (5, 18.4), (5, 17.7))

    _box(ax, (x, 16.7), box_w, 1.0, "RMSNorm")
    _arrow(ax, (5, 16.7), (5, 16.0))

    gqa_note = (
        f"{summary.num_attention_heads} query heads, "
        f"{summary.num_key_value_heads} KV heads (GQA), "
        f"head_dim {summary.head_dim}"
    )
    _box(ax, (x, 14.4), box_w, 1.6, f"Self-Attention\n{gqa_note}", fontsize=8)
    _arrow(ax, (5, 14.4), (5, 13.5))

    ax.add_patch(Rectangle((4.6, 12.9), 0.8, 0.8, facecolor="none", edgecolor="#3a4a63"))
    ax.text(5, 13.3, "+", ha="center", va="center", fontsize=13)
    _arrow(ax, (5, 12.9), (5, 12.2))
    _residual_arc(ax, (x + box_w, 18.9), (x + box_w + 0.4, 13.3))
    ax.text(x + box_w + 0.5, 16.1, "skip", fontsize=7, color="#7a869a", rotation=90, va="center")

    _box(ax, (x, 10.6), box_w, 1.0, "RMSNorm")
    _arrow(ax, (5, 10.6), (5, 9.9))

    _plot_expert_grid(ax, summary, top=9.7)

    moe_bottom = 9.7 - 3.2
    ax.add_patch(
        Rectangle((4.6, moe_bottom - 1.0), 0.8, 0.8, facecolor="none", edgecolor="#3a4a63")
    )
    ax.text(5, moe_bottom - 0.6, "+", ha="center", va="center", fontsize=13)
    _arrow(ax, (5, moe_bottom), (5, moe_bottom - 0.2))
    _residual_arc(ax, (x + box_w, 12.9), (x + box_w + 0.4, moe_bottom - 0.6))
    ax.text(
        x + box_w + 0.5,
        (12.9 + moe_bottom) / 2,
        "skip",
        fontsize=7,
        color="#7a869a",
        rotation=90,
        va="center",
    )
    _arrow(ax, (5, moe_bottom - 1.0), (5, moe_bottom - 1.7))

    _box(ax, (x, moe_bottom - 2.7), box_w, 1.0, "Output")


def _plot_expert_grid(ax: Axes, summary: ArchitectureSummary, *, top: float) -> None:
    """The MoE block: router + a grid of experts, `experts_per_token` highlighted.

    One square per expert, wrapped at a fixed column count so 128 experts
    reads as a grid rather than a single very long row. The highlighted
    squares are the *first* `experts_per_token` only for a stable, readable
    picture -- routing is per-token and data-dependent at runtime, so this
    is illustrating the count, not claiming those specific experts are the
    ones any real token would route to.
    """
    box_w = 6
    x = 2
    _box(ax, (x, top - 0.9), box_w, 0.9, "Router (Top-K gate)", color="#e3d7f0")

    max_cols = 16
    n = summary.num_experts
    cols = min(n, max_cols)
    rows = -(-n // cols)  # ceil
    grid_w, grid_h = box_w, min(2.2, 0.22 * rows)
    cell_w, cell_h = grid_w / cols, grid_h / rows

    active = summary.experts_per_token
    grid_top = top - 1.1
    for i in range(n):
        row, col = divmod(i, cols)
        cx = x + col * cell_w
        cy = grid_top - grid_h - row * cell_h + (rows - 1) * cell_h
        color = "#e07b39" if i < active else "#cfd8e3"
        ax.add_patch(
            Rectangle(
                (cx + cell_w * 0.08, cy + cell_h * 0.08),
                cell_w * 0.84,
                cell_h * 0.84,
                facecolor=color,
                edgecolor="none",
            )
        )

    ax.text(
        x + box_w + 0.15,
        grid_top - grid_h / 2,
        f"{n} experts\n{active} active\nper token",
        ha="left",
        va="center",
        fontsize=8,
    )


def plot_architecture(summary: ArchitectureSummary) -> Figure:
    """Two-panel schematic: the whole model's stack, and one layer expanded.

    Every label is drawn from `summary` -- hidden size, layer count, head
    counts, expert count and the active-per-token count -- so this draws
    whatever model `bonsai.architecture.summarize` was pointed at, not just
    Qwen3-30B-A3B-Base specifically.
    """
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 8))
    _plot_model_stack(left, summary)
    _plot_layer_detail(right, summary)
    fig.suptitle(summary.repo_id, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


__all__ = ["plot_architecture"]
