"""Hand-checkable cases for `bonsai.importance.depth`.

`skip_layer`/`ppl_importance` need an actual forward pass (not just shapes,
unlike `bonsai.architecture`'s meta-device tests), so these build a real,
tiny, randomly-initialized model on CPU rather than downloading
Qwen3-30B-A3B-Base -- same TINY_CONFIG shape `test_architecture.py` and
`test_viz.py` already use, just instantiated with real weights this time.
"""

from typing import cast

import torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

from bonsai.importance.depth import block_importance, ppl_importance, skip_layer

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
)


def tiny_model() -> Qwen3MoeForCausalLM:
    torch.manual_seed(0)
    # `from_config` is untyped in transformers' stubs (`AutoModelForCausalLM`
    # is a factory, not a real return type); the cast is verified, not
    # assumed -- `AutoConfig`/`AutoModelForCausalLM` pick the model class
    # from `TINY_CONFIG`'s type, and this file's own tests exercise the
    # result as a real Qwen3MoeForCausalLM (`.model.layers`, `labels=`).
    model = AutoModelForCausalLM.from_config(TINY_CONFIG)  # type: ignore[no-untyped-call]
    return cast(Qwen3MoeForCausalLM, model)


def test_identical_input_output_scores_zero() -> None:
    # A layer that's a perfect identity pass-through -- cosine similarity 1,
    # BI = 1 - 1 = 0.
    x = torch.randn(8, 16)
    bi = block_importance(x, x.clone())
    assert torch.allclose(bi, torch.tensor(0.0), atol=1e-6)


def test_orthogonal_input_output_scores_one() -> None:
    # Two tokens, each output orthogonal to its own input -- cosine
    # similarity 0, BI = 1 - 0 = 1.
    layer_inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    layer_outputs = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    bi = block_importance(layer_inputs, layer_outputs)
    assert torch.allclose(bi, torch.tensor(1.0), atol=1e-6)


def test_opposite_direction_scores_two() -> None:
    # Cosine similarity -1 -- BI = 1 - (-1) = 2, the metric's max.
    layer_inputs = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    layer_outputs = torch.tensor([[-1.0, 0.0], [0.0, -2.0]])
    bi = block_importance(layer_inputs, layer_outputs)
    assert torch.allclose(bi, torch.tensor(2.0), atol=1e-6)


def test_rejects_mismatched_shapes() -> None:
    try:
        block_importance(torch.zeros(4, 8), torch.zeros(4, 16))
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched shapes")


def test_rejects_wrong_ndim() -> None:
    try:
        block_importance(torch.zeros(2, 4, 8), torch.zeros(2, 4, 8))
    except ValueError:
        return
    raise AssertionError("expected ValueError for a non-2D input")


def test_skip_layer_changes_the_models_output() -> None:
    model = tiny_model()
    model.eval()  # type: ignore[no-untyped-call]
    input_ids = torch.randint(0, TINY_CONFIG.vocab_size, (1, 6))

    with torch.no_grad():
        baseline = model(input_ids=input_ids).logits.clone()
        with skip_layer(model, 1):
            skipped = model(input_ids=input_ids).logits

    assert not torch.allclose(baseline, skipped)


def test_skip_layer_fully_restores_forward_afterward() -> None:
    model = tiny_model()
    model.eval()  # type: ignore[no-untyped-call]
    input_ids = torch.randint(0, TINY_CONFIG.vocab_size, (1, 6))

    with torch.no_grad():
        before = model(input_ids=input_ids).logits.clone()
        with skip_layer(model, 1):
            pass
        after = model(input_ids=input_ids).logits

    assert torch.allclose(before, after)


def test_skip_layer_restores_even_if_the_block_raises() -> None:
    model = tiny_model()
    original_forward = model.model.layers[1].forward

    class _Boom(Exception):
        pass

    try:
        with skip_layer(model, 1):
            raise _Boom
    except _Boom:
        pass

    # `is` doesn't work here: Python creates a fresh bound-method object on
    # every `.forward` attribute access, so two separately-obtained
    # references to the *same* underlying method are never `is`-identical.
    # `==` on bound methods compares `__self__`/`__func__`, which is the
    # actual "same method" check.
    assert model.model.layers[1].forward == original_forward


def test_ppl_importance_returns_one_score_per_layer() -> None:
    model = tiny_model()
    model.eval()  # type: ignore[no-untyped-call]
    input_ids_batches = [torch.randint(0, TINY_CONFIG.vocab_size, (1, 6))]

    scores = ppl_importance(model, input_ids_batches)

    assert scores.shape == (TINY_CONFIG.num_hidden_layers,)
    assert not torch.isnan(scores).any()
    assert torch.isfinite(scores).all()


def test_ppl_importance_restores_all_layers_afterward() -> None:
    model = tiny_model()
    model.eval()  # type: ignore[no-untyped-call]
    input_ids = torch.randint(0, TINY_CONFIG.vocab_size, (1, 6))

    with torch.no_grad():
        before = model(input_ids=input_ids).logits.clone()
        ppl_importance(model, [input_ids])
        after = model(input_ids=input_ids).logits

    assert torch.allclose(before, after)
