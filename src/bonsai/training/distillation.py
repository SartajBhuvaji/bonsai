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

        # `teacher_topk_ids[t]`/`teacher_topk_logprobs[t]` is the top-k of the
        # teacher's logits at position t -- i.e. its prediction for token t+1,
        # the exact same position convention as `student_logits[t]` below
        # (both come from indexing `model(input_ids).logits[0]` directly, no
        # shift applied during caching -- see cache_teacher_logit_file). The
        # last position (predicting one token past the end of the sequence)
        # has nothing to compare against on either side, so it's dropped from
        # both with the same `[:-1]` slice -- not a `[1:]` shift on the
        # teacher side, which was a real bug (plan.md, 2026-10-10): it paired
        # the student's prediction for token t+1 (context 0..t) against the
        # teacher's prediction for token t+2 (context 0..t+1), a target one
        # position ahead of what the student could possibly produce from the
        # context it had. Confirmed via the regression test next to this file
        # (comparing the teacher against its own cache no longer leaves a
        # position-mismatch gap once this is `[:-1]` on both sides).
        student_logits = model(input_ids=input_ids).logits[0, :-1]  # [seq_len-1, vocab]
        dev = student_logits.device
        topk_ids = teacher_topk_ids[:-1].to(dev)  # [seq_len-1, k]
        topk_logprobs = teacher_topk_logprobs[:-1].to(dev)  # [seq_len-1, k]

        # `topk_logprobs` is the teacher's distribution *conditioned on* the
        # token being in its own top-k support (renormalized to sum to 1 over
        # just those k, per cache_teacher_logits.py) -- it carries no
        # information about how much of the teacher's real probability mass
        # that support actually covers. Matching it against the student's
        # conditional top-k distribution (gather, then softmax over just
        # those k logits) is the standard top-k-KD approximation and is
        # what this loss computes below (`kl_within_support`).
        #
        # That comparison alone is exactly the bug found in practice
        # (plan.md, 2026-10-10): it is blind by construction to what the
        # student does with every logit *outside* the support, since both
        # sides are renormalized away from the full vocab before comparing.
        # A student trained against it converged to a healthy-looking KD
        # loss in every arm tried (full-parameter, LoRA, function-mode
        # data) while its real (full-vocab) argmax was dominated by an
        # unconstrained off-support token from the first generated token on
        # -- HumanEval-TS repetition collapse in all three.
        #
        # The missing term is how much of the student's *real* probability
        # mass lands inside the support at all: coverage = P(token in
        # top-k) = topk_logsumexp - full_logsumexp (computed from the
        # correctly-normalized log-space -- not the student-wide KL this
        # project's caching intentionally avoids, just the one extra scalar
        # needed to constrain the complement). `-coverage` is 0 when all of
        # the student's mass is inside the support and grows the instant
        # an off-support logit inflates, independent of `kl_within_support`
        # which only ever compares ranking *inside* the support. Together
        # they are the standard decomposition of cross-entropy into
        # "which side of the support" and "where within it" (derived and
        # verified against a trained toy model before this fix landed; see
        # the regression test next to this file).
        student_topk_logits = torch.gather(student_logits, -1, topk_ids)
        student_topk_logprobs = torch.log_softmax(student_topk_logits, dim=-1)
        full_logsumexp = torch.logsumexp(student_logits, dim=-1)
        topk_logsumexp = torch.logsumexp(student_topk_logits, dim=-1)
        coverage_loss = full_logsumexp - topk_logsumexp  # -log P(token in top-k), >= 0

        teacher_probs = topk_logprobs.exp()
        kl_within_support = (teacher_probs * (topk_logprobs - student_topk_logprobs)).sum(-1)
        return (kl_within_support + coverage_loss).mean()

    return loss_fn


__all__ = ["make_kd_loss_fn"]
