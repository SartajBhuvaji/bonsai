"""Real-forward-pass tests for `bonsai.pruning.experts`.

Same TINY_CONFIG / tiny_model pattern as `test_pruning_depth.py`: a real,
tiny, randomly-initialized model on CPU, not a meta-device shape check.
"""

from copy import deepcopy
from typing import cast

import torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

from bonsai.pruning.experts import prune_experts, select_experts_to_prune

TINY_CONFIG = Qwen3MoeConfig(
    vocab_size=64,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=2,
    moe_intermediate_size=8,
    num_experts=8,
    num_experts_per_tok=2,
)


def tiny_model() -> Qwen3MoeForCausalLM:
    # See test_pruning_depth.py: from_config does not copy its config, so a
    # fresh deepcopy per model keeps prune_experts' in-place mutations from
    # leaking into TINY_CONFIG across tests.
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(deepcopy(TINY_CONFIG))  # type: ignore[no-untyped-call]
    return cast(Qwen3MoeForCausalLM, model)


def test_prune_experts_reduces_expert_count() -> None:
    model = tiny_model()

    prune_experts(model, {0: [1, 3], 1: [0]})

    layer0 = model.model.layers[0].mlp
    layer1 = model.model.layers[1].mlp
    layer2 = model.model.layers[2].mlp
    assert layer0.experts.num_experts == 6
    assert layer0.experts.gate_up_proj.shape[0] == 6
    assert layer0.experts.down_proj.shape[0] == 6
    assert layer0.gate.weight.shape[0] == 6
    assert layer0.gate.num_experts == 6
    assert layer1.experts.num_experts == 7
    assert layer2.experts.num_experts == 8  # untouched -- not in the mapping


def test_prune_experts_keeps_the_right_experts() -> None:
    model = tiny_model()
    original_gate_up = model.model.layers[0].mlp.experts.gate_up_proj.data.clone()

    prune_experts(model, {0: [1, 3]})

    kept = model.model.layers[0].mlp.experts.gate_up_proj.data
    expected = original_gate_up[[0, 2, 4, 5, 6, 7]]
    assert torch.allclose(kept, expected)


def test_prune_experts_pruned_model_still_runs() -> None:
    model = tiny_model()
    model.eval()  # type: ignore[no-untyped-call]
    input_ids = torch.randint(0, TINY_CONFIG.vocab_size, (1, 6))

    prune_experts(model, {0: [1, 3], 1: [0, 2], 2: [7]})

    with torch.no_grad():
        out = model(input_ids=input_ids, labels=input_ids)

    assert out.logits.shape == (1, 6, TINY_CONFIG.vocab_size)
    assert torch.isfinite(out.loss)


def test_prune_experts_changes_the_models_output() -> None:
    model = tiny_model()
    model.eval()  # type: ignore[no-untyped-call]
    input_ids = torch.randint(0, TINY_CONFIG.vocab_size, (1, 6))

    with torch.no_grad():
        baseline = model(input_ids=input_ids).logits.clone()

    prune_experts(model, {0: [1, 3], 1: [0, 2]})
    with torch.no_grad():
        pruned = model(input_ids=input_ids).logits

    assert not torch.allclose(baseline, pruned)


def test_prune_experts_leaves_unlisted_layers_untouched() -> None:
    model = tiny_model()
    original = model.model.layers[1].mlp.experts.gate_up_proj.data.clone()

    prune_experts(model, {0: [1, 3]})

    assert torch.allclose(model.model.layers[1].mlp.experts.gate_up_proj.data, original)


def test_prune_experts_empty_list_is_a_noop_for_that_layer() -> None:
    model = tiny_model()
    original = model.model.layers[0].mlp.experts.gate_up_proj.data.clone()

    prune_experts(model, {0: []})

    assert model.model.layers[0].mlp.experts.num_experts == 8
    assert torch.allclose(model.model.layers[0].mlp.experts.gate_up_proj.data, original)


def test_prune_experts_rejects_out_of_range_index() -> None:
    model = tiny_model()
    try:
        prune_experts(model, {0: [8]})
    except ValueError:
        return
    raise AssertionError("expected ValueError for an out-of-range expert index")


def test_prune_experts_rejects_pruning_every_expert() -> None:
    model = tiny_model()
    try:
        prune_experts(model, {0: list(range(TINY_CONFIG.num_experts))})
    except ValueError:
        return
    raise AssertionError("expected ValueError for pruning every expert in a layer")


def test_prune_experts_rejects_pruning_below_router_top_k() -> None:
    model = tiny_model()  # num_experts_per_tok (router.top_k) = 2
    try:
        prune_experts(model, {0: [1, 2, 3, 4, 5, 6, 7]})  # would leave 1 expert
    except ValueError:
        return
    raise AssertionError("expected ValueError for pruning below router.top_k surviving experts")


def test_prune_experts_syncs_config_when_uniform_across_layers() -> None:
    model = tiny_model()  # 3 layers, 8 experts each

    prune_experts(model, {0: [0, 1], 1: [2, 3], 2: [4, 5]})  # every layer -> 6 experts

    assert model.config.num_experts == 6
    assert model.num_experts == 6


def test_prune_experts_leaves_config_stale_when_not_uniform_across_layers() -> None:
    model = tiny_model()  # 3 layers, 8 experts each
    original_config_count = model.config.num_experts
    original_model_count = model.num_experts

    prune_experts(model, {0: [0, 1]})  # only layer 0 pruned -> 6, 8, 8

    assert model.model.layers[0].mlp.experts.num_experts == 6
    assert model.model.layers[1].mlp.experts.num_experts == 8
    assert model.config.num_experts == original_config_count
    assert model.num_experts == original_model_count


def test_prune_experts_rejects_non_moe_layer() -> None:
    config = deepcopy(TINY_CONFIG)
    config.mlp_only_layers = [0]
    torch.manual_seed(0)
    model = cast(Qwen3MoeForCausalLM, AutoModelForCausalLM.from_config(config))  # type: ignore[no-untyped-call]

    try:
        prune_experts(model, {0: [1]})
    except ValueError:
        return
    raise AssertionError("expected ValueError for a dense (non-MoE) layer")


def test_select_experts_to_prune_picks_lowest_scores() -> None:
    scores = torch.tensor(
        [
            [5.0, 1.0, 3.0, 2.0],
            [0.5, 0.5, 9.0, 0.1],
        ]
    )

    selected = select_experts_to_prune(scores, n_per_layer=2)

    assert sorted(selected[0]) == [1, 3]  # lowest two in layer 0: idx 1 (1.0), idx 3 (2.0)
    # lowest two in layer 1: idx 3 (0.1) plus whichever of the tied idx 0/1 (both 0.5) topk breaks to.
    assert 3 in selected[1]
    assert set(selected[1]) <= {0, 1, 3}
    assert len(selected[1]) == 2


def test_select_experts_to_prune_respects_protected() -> None:
    scores = torch.tensor([[5.0, 1.0, 3.0, 2.0]])

    selected = select_experts_to_prune(scores, n_per_layer=2, protected={0: [1]})

    # Without protection, expert 1 (score 1.0) would be the top pick.
    assert 1 not in selected[0]
    assert sorted(selected[0]) == [2, 3]


def test_select_experts_to_prune_rejects_n_too_large() -> None:
    scores = torch.rand(2, 4)
    try:
        select_experts_to_prune(scores, n_per_layer=4)
    except ValueError:
        return
    raise AssertionError("expected ValueError when n_per_layer >= num_experts")
