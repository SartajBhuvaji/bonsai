"""Smoke tests for the architecture diagram.

Not pixel comparisons -- just that it draws without error for architectures
of very different shapes (few experts, many; few layers, many) and that the
figure has the two panels the docstring promises.
"""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no display in CI

from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from bonsai.architecture import ArchitectureSummary, summarize
from bonsai.architecture.viz import plot_architecture


def summary_for(tmp_path: Path, **overrides: Any) -> ArchitectureSummary:
    # Genuinely arbitrary: these pass straight through to Qwen3MoeConfig,
    # whose keyword parameters differ in type per-argument. ANN401 is already
    # off for tests/* (see pyproject.toml), so no noqa needed here.
    defaults: dict[str, Any] = {
        "vocab_size": 64,
        "hidden_size": 16,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "moe_intermediate_size": 8,
        "num_experts": 6,
        "num_experts_per_tok": 2,
    }
    config = Qwen3MoeConfig(**(defaults | overrides))
    config.save_pretrained(tmp_path)
    return summarize(str(tmp_path))


def test_plot_has_two_panels(tmp_path: Path) -> None:
    fig = plot_architecture(summary_for(tmp_path))

    assert len(fig.axes) == 2


def test_plot_handles_many_experts(tmp_path: Path) -> None:
    fig = plot_architecture(summary_for(tmp_path, num_experts=128, num_experts_per_tok=8))

    assert len(fig.axes) == 2


def test_plot_handles_single_expert_per_token(tmp_path: Path) -> None:
    fig = plot_architecture(summary_for(tmp_path, num_experts=8, num_experts_per_tok=1))

    assert len(fig.axes) == 2


def test_plot_handles_tied_embeddings(tmp_path: Path) -> None:
    fig = plot_architecture(summary_for(tmp_path, tie_word_embeddings=True))

    assert len(fig.axes) == 2
