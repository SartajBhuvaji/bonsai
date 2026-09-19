"""Hand-checkable cases for `bonsai.importance.depth.block_importance`."""

import torch

from bonsai.importance.depth import block_importance


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
