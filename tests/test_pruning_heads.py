"""Real-forward-pass tests for `bonsai.pruning.heads`.

Same TINY_CONFIG / tiny_model pattern as the other pruning test files: a
real, tiny, randomly-initialized model on CPU, not a meta-device shape
check. 6 query heads / 2 key-value heads (group size 3) so a group can
lose 1 or 2 heads while keeping at least one.
"""

from copy import deepcopy
from typing import cast

import torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

from bonsai.pruning.heads import prune_heads, select_heads_to_prune

TINY_CONFIG = Qwen3MoeConfig(
    vocab_size=64,
    hidden_size=24,
    intermediate_size=32,
    num_hidden_layers=3,
    num_attention_heads=6,
    num_key_value_heads=2,
    moe_intermediate_size=8,
    num_experts=6,
    num_experts_per_tok=2,
)


def tiny_model() -> Qwen3MoeForCausalLM:
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(deepcopy(TINY_CONFIG))  # type: ignore[no-untyped-call]
    return cast(Qwen3MoeForCausalLM, model)


def test_prune_heads_reduces_projection_sizes() -> None:
    model = tiny_model()
    head_dim = model.model.layers[0].self_attn.head_dim

    # groups are heads [0,1,2] and [3,4,5]; prune 1 per group.
    prune_heads(model, {0: [0, 3]})

    attn = model.model.layers[0].self_attn
    assert attn.q_proj.out_features == 4 * head_dim  # 6 heads - 2 pruned (1 per group)
    assert attn.q_proj.weight.shape[0] == 4 * head_dim
    assert attn.o_proj.in_features == 4 * head_dim
    assert attn.o_proj.weight.shape[1] == 4 * head_dim
    assert attn.num_key_value_groups == 2  # group size 3 - 1 pruned = 2

    other_attn = model.model.layers[1].self_attn
    assert other_attn.q_proj.out_features == 6 * head_dim  # untouched


def test_prune_heads_keeps_the_right_heads() -> None:
    model = tiny_model()
    head_dim = model.model.layers[0].self_attn.head_dim
    original_q = model.model.layers[0].self_attn.q_proj.weight.data.clone()

    prune_heads(model, {0: [0, 3]})

    kept = model.model.layers[0].self_attn.q_proj.weight.data
    keep_heads = [1, 2, 4, 5]
    expected_rows = torch.cat([original_q[h * head_dim : (h + 1) * head_dim] for h in keep_heads])
    assert torch.allclose(kept, expected_rows)


def test_prune_heads_pruned_model_still_runs() -> None:
    model = tiny_model()
    model.eval()  # type: ignore[no-untyped-call]
    input_ids = torch.randint(0, TINY_CONFIG.vocab_size, (1, 6))

    prune_heads(model, {0: [0, 3], 1: [1, 2, 4, 5]})

    with torch.no_grad():
        out = model(input_ids=input_ids, labels=input_ids)

    assert out.logits.shape == (1, 6, TINY_CONFIG.vocab_size)
    assert torch.isfinite(out.loss)


def test_prune_heads_changes_the_models_output() -> None:
    model = tiny_model()
    model.eval()  # type: ignore[no-untyped-call]
    input_ids = torch.randint(0, TINY_CONFIG.vocab_size, (1, 6))

    with torch.no_grad():
        baseline = model(input_ids=input_ids).logits.clone()

    prune_heads(model, {0: [0, 3]})
    with torch.no_grad():
        pruned = model(input_ids=input_ids).logits

    assert not torch.allclose(baseline, pruned)


def test_prune_heads_leaves_unlisted_layers_untouched() -> None:
    model = tiny_model()
    original = model.model.layers[1].self_attn.q_proj.weight.data.clone()

    prune_heads(model, {0: [0, 3]})

    assert torch.allclose(model.model.layers[1].self_attn.q_proj.weight.data, original)


def test_prune_heads_empty_list_is_a_noop_for_that_layer() -> None:
    model = tiny_model()
    original = model.model.layers[0].self_attn.q_proj.weight.data.clone()

    prune_heads(model, {0: []})

    assert model.model.layers[0].self_attn.q_proj.out_features == original.shape[0]
    assert torch.allclose(model.model.layers[0].self_attn.q_proj.weight.data, original)


def test_prune_heads_rejects_out_of_range_index() -> None:
    model = tiny_model()
    try:
        prune_heads(model, {0: [6]})
    except ValueError:
        return
    raise AssertionError("expected ValueError for an out-of-range head index")


def test_prune_heads_rejects_pruning_every_head() -> None:
    model = tiny_model()
    try:
        prune_heads(model, {0: list(range(TINY_CONFIG.num_attention_heads))})
    except ValueError:
        return
    raise AssertionError("expected ValueError for pruning every head in a layer")


def test_prune_heads_rejects_ragged_per_group_count() -> None:
    model = tiny_model()
    # group 0 (heads 0,1,2) loses 2 heads, group 1 (heads 3,4,5) loses 1 -- ragged.
    try:
        prune_heads(model, {0: [0, 1, 3]})
    except ValueError:
        return
    raise AssertionError("expected ValueError for a ragged per-key/value-group prune count")


def test_prune_heads_rejects_pruning_an_entire_group() -> None:
    model = tiny_model()
    # group 0 is heads [0,1,2]; pruning all 3 would orphan its key/value head.
    try:
        prune_heads(model, {0: [0, 1, 2]})
    except ValueError:
        return
    raise AssertionError("expected ValueError for pruning an entire key/value group")


def test_select_heads_to_prune_picks_lowest_per_group() -> None:
    # 1 layer, 6 heads, 2 kv groups of 3: [5,1,3] and [0.5,9,0.1]
    scores = torch.tensor([[5.0, 1.0, 3.0, 0.5, 9.0, 0.1]])

    selected = select_heads_to_prune(scores, k_per_group=1, num_key_value_heads=2)

    assert selected[0] == [1, 5]  # lowest in [0,1,2] is idx1 (1.0); lowest in [3,4,5] is idx5 (0.1)


def test_select_heads_to_prune_two_per_group() -> None:
    scores = torch.tensor([[5.0, 1.0, 3.0, 0.5, 9.0, 0.1]])

    selected = select_heads_to_prune(scores, k_per_group=2, num_key_value_heads=2)

    assert sorted(selected[0]) == [1, 2, 3, 5]  # 2 lowest per group


def test_select_heads_to_prune_rejects_k_too_large() -> None:
    scores = torch.rand(2, 6)
    try:
        select_heads_to_prune(scores, k_per_group=3, num_key_value_heads=2)
    except ValueError:
        return
    raise AssertionError("expected ValueError when k_per_group >= group size")


def test_select_heads_to_prune_rejects_indivisible_heads() -> None:
    scores = torch.rand(2, 7)
    try:
        select_heads_to_prune(scores, k_per_group=1, num_key_value_heads=2)
    except ValueError:
        return
    raise AssertionError(
        "expected ValueError when num_heads isn't divisible by num_key_value_heads"
    )
