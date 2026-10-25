"""Smoke tests for the importance-score heatmaps.

Same style as `test_viz.py`: not pixel comparisons, just that each plot
function draws without error, on tensors of the shapes it documents, and
that shape validation actually rejects the wrong ndim.
"""

from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")  # no display in CI

from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from bonsai.architecture import summarize
from bonsai.importance.viz import (
    plot_expert_concentration,
    plot_expert_grid_for_layer,
    plot_expert_heatmap,
    plot_head_heatmap,
    plot_layer_importance,
    plot_layer_summary,
    plot_metric_agreement,
    plot_score_distribution,
)


def test_plot_expert_heatmap_draws() -> None:
    scores = torch.rand(6, 12)
    fig = plot_expert_heatmap(scores)
    assert len(fig.axes) >= 1


def test_plot_expert_heatmap_rejects_wrong_ndim() -> None:
    try:
        plot_expert_heatmap(torch.rand(6))
    except ValueError:
        return
    raise AssertionError("expected ValueError for a non-2D scores tensor")


def test_plot_head_heatmap_draws() -> None:
    scores = torch.rand(6, 8)
    fig = plot_head_heatmap(scores)
    assert len(fig.axes) >= 1


def test_plot_head_heatmap_rejects_wrong_ndim() -> None:
    try:
        plot_head_heatmap(torch.rand(6, 8, 2))
    except ValueError:
        return
    raise AssertionError("expected ValueError for a non-2D scores tensor")


def test_plot_layer_importance_draws() -> None:
    scores = torch.rand(24)
    fig = plot_layer_importance(scores)
    assert len(fig.axes) == 1


def test_plot_layer_importance_rejects_wrong_ndim() -> None:
    try:
        plot_layer_importance(torch.rand(6, 8))
    except ValueError:
        return
    raise AssertionError("expected ValueError for a non-1D scores tensor")


def test_plot_expert_heatmap_handles_top_n_larger_than_grid() -> None:
    # A tiny 2x2 grid with top_n_outline's default (5) -- shouldn't crash
    # asking torch.topk for more elements than exist.
    fig = plot_expert_heatmap(torch.rand(2, 2))
    assert len(fig.axes) >= 1


def test_plot_layer_summary_draws() -> None:
    fig = plot_layer_summary(
        {
            "mean head score": torch.rand(6),
            "mean expert score": torch.rand(6),
            "block importance (BI)": torch.rand(6),
            "PPL delta": torch.rand(6),
        }
    )
    assert len(fig.axes) == 1


def test_plot_layer_summary_rejects_mismatched_lengths() -> None:
    try:
        plot_layer_summary({"a": torch.rand(6), "b": torch.rand(8)})
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched series lengths")


def test_plot_layer_summary_rejects_non_1d_series() -> None:
    try:
        plot_layer_summary({"a": torch.rand(6, 12)})
    except ValueError:
        return
    raise AssertionError("expected ValueError for a non-1D series")


def test_plot_layer_summary_rejects_empty_dict() -> None:
    try:
        plot_layer_summary({})
    except ValueError:
        return
    raise AssertionError("expected ValueError for no series")


def test_plot_layer_summary_rejects_too_many_series() -> None:
    try:
        plot_layer_summary({str(i): torch.rand(6) for i in range(10)})
    except ValueError:
        return
    raise AssertionError("expected ValueError for more series than categorical slots")


def test_plot_layer_importance_accepts_custom_labels() -> None:
    fig = plot_layer_importance(
        torch.rand(24),
        title="PPL delta per layer",
        ylabel="loss delta",
        annotation_label="lowest PPL delta",
    )
    assert len(fig.axes) == 1


def test_plot_metric_agreement_draws() -> None:
    fig = plot_metric_agreement(
        torch.rand(24), torch.rand(24), x_label="BI", y_label="PPL delta", title="BI vs PPL"
    )
    assert len(fig.axes) == 1


def test_plot_metric_agreement_rejects_mismatched_shapes() -> None:
    try:
        plot_metric_agreement(torch.rand(24), torch.rand(20), x_label="x", y_label="y", title="t")
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched x/y shapes")


def test_plot_metric_agreement_default_scale_is_linear() -> None:
    fig = plot_metric_agreement(
        torch.rand(24), torch.rand(24), x_label="BI", y_label="PPL delta", title="BI vs PPL"
    )
    ax = fig.axes[0]
    assert ax.get_xscale() == "linear"
    assert ax.get_yscale() == "linear"


def test_plot_metric_agreement_log_scale_draws() -> None:
    # x strictly positive (BI, as in real use); y includes a slightly
    # negative value, the real-world case (a near-no-op layer's PPL delta)
    # the symlog branch exists to handle without crashing or distorting
    # the rest of the axis.
    x = torch.rand(24) + 0.01
    y = torch.rand(24)
    y[0] = -0.0003
    fig = plot_metric_agreement(
        x, y, x_label="BI", y_label="PPL delta", title="BI vs PPL", log_scale=True
    )
    assert len(fig.axes) == 1


def test_plot_metric_agreement_log_scale_sets_axis_scales() -> None:
    x = torch.rand(24) + 0.01
    y = torch.rand(24)
    y[0] = -0.0003
    fig = plot_metric_agreement(
        x, y, x_label="BI", y_label="PPL delta", title="BI vs PPL", log_scale=True
    )
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "symlog"


def test_plot_score_distribution_draws() -> None:
    fig = plot_score_distribution(torch.rand(6, 12), title="test", xlabel="score")
    assert len(fig.axes) == 1


def test_plot_score_distribution_handles_all_zero() -> None:
    fig = plot_score_distribution(torch.zeros(4, 4), title="all zero", xlabel="score")
    assert len(fig.axes) == 1


def test_plot_expert_concentration_draws() -> None:
    fig = plot_expert_concentration(torch.rand(6, 24))
    assert len(fig.axes) == 1


def test_plot_expert_concentration_handles_all_zero_layer() -> None:
    scores = torch.rand(3, 8)
    scores[0] = 0.0
    fig = plot_expert_concentration(scores)
    assert len(fig.axes) == 1


def test_plot_expert_concentration_rejects_wrong_ndim() -> None:
    try:
        plot_expert_concentration(torch.rand(6))
    except ValueError:
        return
    raise AssertionError("expected ValueError for a non-2D expert_scores tensor")


def _tiny_summary() -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        moe_intermediate_size=8,
        num_experts=6,
        num_experts_per_tok=2,
    )


def test_plot_expert_grid_for_layer_draws(tmp_path: Path) -> None:
    _tiny_summary().save_pretrained(tmp_path)
    summary = summarize(str(tmp_path))
    fig = plot_expert_grid_for_layer(summary, torch.rand(summary.num_experts), layer_idx=0)
    assert len(fig.axes) == 1


def test_plot_expert_grid_for_layer_rejects_wrong_length(tmp_path: Path) -> None:
    _tiny_summary().save_pretrained(tmp_path)
    summary = summarize(str(tmp_path))
    try:
        plot_expert_grid_for_layer(summary, torch.rand(summary.num_experts + 1), layer_idx=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for a layer_scores length mismatch")
