"""Run the general-knowledge suite (MMLU, ARC-Challenge, Winogrande) via lm-evaluation-harness.

None of these three tasks' own harness configs set `num_fewshot` (checked
directly: `mmlu`'s default template, `arc_challenge.yaml`, `winogrande/default.yaml`
all omit it, which makes `lm_eval.simple_evaluate` fall back to 0-shot). Left at
that default, our numbers wouldn't be comparable to what plan.md actually wants
them for -- a rough check against Table 1/2 of both Minitron papers. `NUM_FEWSHOT`
pins the long-standing EleutherAI-harness / Open LLM Leaderboard convention
(MMLU 5-shot, ARC-Challenge 25-shot, Winogrande 5-shot) that era of paper
generally reports against, since neither paper's appendix is being read
programmatically here to confirm their exact number.
"""

from __future__ import annotations

from typing import Any

import lm_eval

NUM_FEWSHOT: dict[str, int] = {
    "mmlu": 5,
    "arc_challenge": 25,
    "winogrande": 5,
}


def run_general_knowledge(
    model_repo_id: str,
    *,
    revision: str = "main",
    dtype: str = "bfloat16",
    batch_size: str = "auto",
    limit: float | None = None,
) -> dict[str, Any]:
    """Run MMLU, ARC-Challenge and Winogrande against a HF model.

    One `simple_evaluate` call per task, not one call across all three --
    `simple_evaluate` takes a single `num_fewshot` for the whole call, and
    each task needs its own (see `NUM_FEWSHOT`). This also means one task's
    failure doesn't take the others down with it.

    `limit` caps the number of examples per task (an int count, or a float
    treated as a fraction) -- useful for a cheap smoke test before spending a
    full run on all three.
    """
    results: dict[str, Any] = {}
    for task, num_fewshot in NUM_FEWSHOT.items():
        results[task] = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={model_repo_id},revision={revision},dtype={dtype}",
            tasks=[task],
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            limit=limit,
        )
    return results


__all__ = ["NUM_FEWSHOT", "run_general_knowledge"]
