"""Tests against a hand-built, tiny Qwen3-MoE config saved to a temp directory.

Deliberately not `AutoConfig.from_pretrained("Qwen/...")`: that would need
network access in CI for every test run, for a 30B-parameter config we don't
need in full to exercise this module's logic. A tiny config saved locally
exercises the exact same `AutoConfig`/`AutoModelForCausalLM` code path,
offline and in milliseconds.
"""

from pathlib import Path

from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from bonsai.architecture import summarize

TINY_CONFIG = Qwen3MoeConfig(
    vocab_size=64,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=2,
    moe_intermediate_size=8,
    num_experts=6,
    num_experts_per_tok=2,
    mlp_only_layers=[1],  # layer 1 is a dense MLP; the rest are sparse
    tie_word_embeddings=False,
)


def save_tiny_config(tmp_path: Path) -> str:
    TINY_CONFIG.save_pretrained(tmp_path)
    return str(tmp_path)


def test_layer_count_and_sparsity(tmp_path: Path) -> None:
    summary = summarize(save_tiny_config(tmp_path))

    assert summary.num_layers == 3
    assert [layer.is_sparse for layer in summary.layers] == [True, False, True]


def test_dense_layer_has_no_router_params(tmp_path: Path) -> None:
    summary = summarize(save_tiny_config(tmp_path))

    dense_layer = summary.layers[1]
    assert dense_layer.router == 0
    assert dense_layer.experts > 0  # the one dense MLP's params


def test_sparse_layer_has_router_and_all_experts(tmp_path: Path) -> None:
    summary = summarize(save_tiny_config(tmp_path))

    sparse_layer = summary.layers[0]
    assert sparse_layer.router > 0
    # Qwen3MoeExperts holds every expert's weights, not just the active top-k,
    # so this counts all 6, not num_experts_per_tok.
    per_expert = 2 * 8 * 16 + 16 * 8  # gate_up_proj + down_proj, from TINY_CONFIG
    assert sparse_layer.experts == 6 * per_expert


def test_total_is_sum_of_parts(tmp_path: Path) -> None:
    summary = summarize(save_tiny_config(tmp_path))

    parts = summary.embedding + summary.lm_head + sum(layer.total for layer in summary.layers)
    assert summary.total == parts
    assert summary.attention_total > 0
    assert summary.router_total > 0  # from the two sparse layers
    assert summary.experts_total > 0


def test_untied_embeddings_have_separate_lm_head(tmp_path: Path) -> None:
    summary = summarize(save_tiny_config(tmp_path))

    assert summary.lm_head > 0
    assert summary.lm_head == summary.embedding  # same shape, untied weights
