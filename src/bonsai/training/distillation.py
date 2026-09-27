"""Forward-KL distillation loss against a cached top-k teacher distribution.

Phase 4.1 (plan.md): `bonsai.training.train`/`train_step`'s own docstring
already anticipates this -- `loss_fn` is the pluggable seam for
distillation, so training a pruned candidate against a teacher is just a
different `loss_fn` passed into the same loop, not a new one.

The teacher never needs to be resident on GPU during student training this
way: its top-k logits are computed once, up front, by
`tools/cache_teacher_logits.py`, and this module only ever reads that
cache back. That split is what lets a ~61GB teacher and a pruned student's
full training footprint (weights + gradients + optimizer state) each fit
on a single 80GB card without needing both at once -- see plan.md 4.1's
own reasoning for why caching beats holding the teacher resident.
"""

from __future__ import annotations

import torch
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

from bonsai.training import LossFn


def make_kd_loss_fn(cache: dict[int, tuple[torch.Tensor, torch.Tensor]]) -> LossFn:
    """Build a `loss_fn` compatible with `bonsai.training.train`/`train_step`.

    `cache` maps `id(input_ids)` -- Python object identity, not a hash of
    the tensor's content -- to that exact sample's cached
    `(teacher_topk_ids, teacher_topk_logprobs)`, both shape `[seq_len, k]`.
    This relies on `train`/`train_step` passing the very same tensor
    objects through unchanged from whatever list they were built from (confirmed
    by reading their source: `train_input_ids[i]` is indexed, never
    copied), so building `cache` from that same list before calling
    `train()` is enough -- no change to `train`/`train_step`'s signature or
    behavior is needed for this to work.

    `teacher_topk_logprobs` is expected already renormalized over just the
    top-k support (`tools/cache_teacher_logits.py`'s own output format,
    `exp(teacher_topk_logprobs).sum(-1) ≈ 1`), matching plan.md 4.1's
    "truncated-and-renormalized teacher distribution" description -- this
    function does not re-normalize it.
    """

    def loss_fn(model: Qwen3MoeForCausalLM, input_ids: torch.Tensor) -> torch.Tensor:
        key = id(input_ids)
        if key not in cache:
            raise KeyError(
                "no cached teacher logits for this sample -- pass the exact same "
                "tensor objects used to build `cache` into train()/train_step(), "
                "not copies or re-tokenized equivalents"
            )
        teacher_topk_ids, teacher_topk_logprobs = cache[key]

        # Predict token t+1 from position t, same convention as `labels=input_ids`'s
        # own shift inside the model's built-in loss -- made explicit here since KD
        # never goes through that built-in loss at all.
        student_logits = model(input_ids=input_ids).logits[0, :-1]  # [seq_len-1, vocab]
        dev = student_logits.device
        topk_ids = teacher_topk_ids[1:].to(dev)  # [seq_len-1, k]
        topk_logprobs = teacher_topk_logprobs[1:].to(dev)  # [seq_len-1, k]

        student_topk_logits = torch.gather(student_logits, -1, topk_ids)
        student_topk_logprobs = torch.log_softmax(student_topk_logits, dim=-1)

        teacher_probs = topk_logprobs.exp()
        # Forward KL, teacher || student, restricted to the teacher's own top-k
        # support -- the standard top-k KD approximation this project's caching
        # pass is built around (plan.md 4.1), not the full-vocab KL.
        kl_per_token = (teacher_probs * (topk_logprobs - student_topk_logprobs)).sum(-1)
        return kl_per_token.mean()

    return loss_fn


__all__ = ["make_kd_loss_fn"]
