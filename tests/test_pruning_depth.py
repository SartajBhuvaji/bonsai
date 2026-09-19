"""Real-forward-pass tests for `bonsai.pruning.depth.prune_layers`.

Same TINY_CONFIG / tiny_model pattern as `test_importance_depth.py`: a
real, tiny, randomly-initialized model on CPU, not a meta-device shape
check, since correctness here means "the pruned model still runs and
actually has fewer layers," not just "the function returns."
"""

from copy import deepcopy
from typing import cast

import torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

from bonsai.pruning.depth import prune_layers

TINY_CONFIG = Qwen3MoeConfig(
    vocab_size=64,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=6,
    num_attention_heads=4,
    num_key_value_heads=2,
    moe_intermediate_size=8,
    num_experts=6,
    num_experts_per_tok=2,
)


def tiny_model() -> Qwen3MoeForCausalLM:
    # `from_config` does not copy the config it's given -- `model.config` is
    # the exact object passed in, so `prune_layers`' in-place
    # `num_hidden_layers` update would otherwise mutate the shared
    # module-level TINY_CONFIG and silently shrink every later test's model.
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(deepcopy(TINY_CONFIG))  # type: ignore[no-untyped-call]
    return cast(Qwen3MoeForCausalLM, model)


def test_prune_layers_reduces_layer_count_and_config() -> None:
    model = tiny_model()

    prune_layers(model, [1, 3])

    assert len(model.model.layers) == 4
    assert model.config.num_hidden_layers == 4


def test_prune_layers_keeps_the_right_layers() -> None:
    model = tiny_model()
    original_layers = list(model.model.layers)

    prune_layers(model, [1, 3])

    kept_ids = [id(layer) for layer in model.model.layers]
    expected_ids = [id(original_layers[i]) for i in (0, 2, 4, 5)]
    assert kept_ids == expected_ids


def test_prune_layers_pruned_model_still_runs() -> None:
    model = tiny_model()
    model.eval()  # type: ignore[no-untyped-call]
    input_ids = torch.randint(0, TINY_CONFIG.vocab_size, (1, 6))

    prune_layers(model, [1, 3])

    with torch.no_grad():
        out = model(input_ids=input_ids, labels=input_ids)

    assert out.logits.shape == (1, 6, TINY_CONFIG.vocab_size)
    assert torch.isfinite(out.loss)


def test_prune_layers_changes_the_models_output() -> None:
    model = tiny_model()
    model.eval()  # type: ignore[no-untyped-call]
    input_ids = torch.randint(0, TINY_CONFIG.vocab_size, (1, 6))

    with torch.no_grad():
        baseline = model(input_ids=input_ids).logits.clone()

    prune_layers(model, [1, 3])
    with torch.no_grad():
        pruned = model(input_ids=input_ids).logits

    assert baseline.shape != pruned.shape or not torch.allclose(baseline, pruned)


def test_prune_layers_accepts_duplicate_and_unsorted_indices() -> None:
    model = tiny_model()

    prune_layers(model, [3, 1, 1, 3])

    assert len(model.model.layers) == 4


def test_prune_layers_rejects_empty_indices() -> None:
    model = tiny_model()
    try:
        prune_layers(model, [])
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty layer_indices")


def test_prune_layers_rejects_out_of_range_index() -> None:
    model = tiny_model()
    try:
        prune_layers(model, [6])
    except ValueError:
        return
    raise AssertionError("expected ValueError for an out-of-range index")


def test_prune_layers_rejects_negative_index() -> None:
    model = tiny_model()
    try:
        prune_layers(model, [-1])
    except ValueError:
        return
    raise AssertionError("expected ValueError for a negative index")


def test_prune_layers_rejects_pruning_every_layer() -> None:
    model = tiny_model()
    try:
        prune_layers(model, range(TINY_CONFIG.num_hidden_layers))
    except ValueError:
        return
    raise AssertionError("expected ValueError for pruning every layer")
