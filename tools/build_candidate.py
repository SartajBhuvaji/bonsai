"""Build and save a depth+experts pruned candidate.

Generalized (2026-10-08) from the aggressive-tier-only script Phase 4
started with. `--depth-n`/`--expert-n` default to 16/80 -- the aggressive
tier `08_combined_pruning_sweep.ipynb` already measured (zero-shot loss
1.8479815144091845, never saved as a real checkpoint, only the loss number
survived). Any other `(depth_n, expert_n)` pair builds a fresh, real
candidate from the same real importance scores, for the systematic
depth-vs-experts isolation study (plan.md, 2026-10-08): e.g. `--depth-n 0
--expert-n 97` isolates "same total parameter budget as the aggressive
candidate, but zero depth cut" to test whether depth removal specifically
-- not just aggressive pruning in general -- is what breaks downstream
code generation after distillation.

Selection always comes from the real, committed importance scores
(`runs/20260919T150941Z-ff00be-importance-scores/importance-scores.pt`
locally, or the private `sartajbhuvaji/bonsai-importance-scores` HF
dataset repo as a fallback) -- never arbitrary or random indices, for any
`(depth_n, expert_n)` pair.

Only the exact aggressive tier (16, 80) has a known-good zero-shot loss to
compare against and a locked layer-list regression check
(`AGGRESSIVE_DEPTH_EXPERTS_LAYERS_TO_PRUNE`, verified in
`tests/test_pruning_candidates.py`); every other pair has no prior
measurement, so this script reports the real zero-shot loss it measures
as new data instead of asserting it against anything.

`depth_n=0` is a real, supported case (the experts-only isolation point)
-- `bonsai.pruning.depth.prune_layers` refuses an empty index list by
design (calling it with nothing to prune is almost always a caller bug),
so this script skips that call entirely when there's nothing to prune
rather than passing it an empty list.

Needs a real GPU (or ~120GB system RAM) to load Qwen3-30B-A3B-Base in
bf16 -- the selection-index computation doesn't, but actually
instantiating and slicing the model does.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

from bonsai.config import DataConfig
from bonsai.data import list_shards, load_samples, make_split, tokenize_samples
from bonsai.importance.depth import mean_loss
from bonsai.pruning.candidates import (
    AGGRESSIVE_DEPTH_EXPERTS_LAYERS_TO_PRUNE,
    select_depth_experts_candidate,
)
from bonsai.pruning.depth import prune_layers
from bonsai.pruning.experts import prune_experts
from bonsai.tracking import log_metric, new_run, save_config_snapshot

BASE_MODEL = "Qwen/Qwen3-30B-A3B-Base"
IMPORTANCE_SCORES_PATH = Path("runs/20260919T150941Z-ff00be-importance-scores/importance-scores.pt")
IMPORTANCE_SCORES_HF_REPO = "sartajbhuvaji/bonsai-importance-scores"
AGGRESSIVE_DEPTH_N = 16
AGGRESSIVE_EXPERT_N = 80
# Layer 1/expert 68, layer 2/expert 92, layer 3/expert 82 -- the three Super
# Experts (arXiv:2507.23279) this project's own combined sweep exempted from
# every expert-pruning selection (sec:experts / sec:combined-pruning-sweep
# in the paper). Keyed by *original* (pre-depth-pruning) layer index. Applies
# regardless of depth_n/expert_n -- these three are never pruned.
PROTECTED_SUPER_EXPERTS = {1: [68], 2: [92], 3: [82]}
AGGRESSIVE_EXPECTED_ZERO_SHOT_LOSS = 1.8479815144091845
LOSS_TOLERANCE = 0.01  # absolute; catches drift, not fp32 noise between runs


def load_importance_scores(*, hf_token: str | None = None) -> dict[str, torch.Tensor]:
    """Load from the local run dir if present, else download the same file
    from the private `sartajbhuvaji/bonsai-importance-scores` HF dataset repo."""
    path = IMPORTANCE_SCORES_PATH
    if not path.exists():
        print(f"{path} not found locally, downloading from {IMPORTANCE_SCORES_HF_REPO} ...")
        path = Path(
            hf_hub_download(
                IMPORTANCE_SCORES_HF_REPO,
                "importance-scores.pt",
                repo_type="dataset",
                token=hf_token,
            )
        )
    return torch.load(path, map_location="cpu", weights_only=False)  # type: ignore[no-any-return]


def build_candidate(
    model: Qwen3MoeForCausalLM,
    scores: dict[str, torch.Tensor],
    *,
    depth_n: int,
    expert_n: int,
) -> None:
    """Apply the `(depth_n, expert_n)` selection to `model`, in place.

    Skips `prune_layers` entirely when `depth_n == 0` -- it refuses an
    empty index list by design, and there's genuinely nothing to prune on
    this axis for the experts-only isolation point.
    """
    layers_to_prune, expert_indices_by_new_layer = select_depth_experts_candidate(
        scores["depth_scores"],
        scores["expert_scores"],
        depth_n=depth_n,
        expert_n=expert_n,
        protected=PROTECTED_SUPER_EXPERTS,
    )
    if layers_to_prune:
        prune_layers(model, layers_to_prune)
    prune_experts(model, expert_indices_by_new_layer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--depth-n", type=int, default=AGGRESSIVE_DEPTH_N, help="layers to prune (of 48)"
    )
    parser.add_argument(
        "--expert-n",
        type=int,
        default=AGGRESSIVE_EXPERT_N,
        help="experts to prune per surviving layer (of 128)",
    )
    parser.add_argument("--push-to-hub", default=None, help="HF repo id to push the result to")
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()
    depth_n, expert_n = args.depth_n, args.expert_n
    is_aggressive_tier = depth_n == AGGRESSIVE_DEPTH_N and expert_n == AGGRESSIVE_EXPERT_N

    run_dir = new_run(f"build-candidate-depth{depth_n}-experts{expert_n}")
    scores = load_importance_scores(hf_token=args.hf_token)

    print(f"Loading {BASE_MODEL} ...")
    # transformers' AutoModelForCausalLM stubs are untyped across every
    # architecture it supports -- same cast bonsai.architecture.load_meta_model
    # uses, verified against this exact model family elsewhere in the project.
    model = cast(
        Qwen3MoeForCausalLM,
        AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map="auto"),
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    layers_to_prune, _expert_indices = select_depth_experts_candidate(
        scores["depth_scores"],
        scores["expert_scores"],
        depth_n=depth_n,
        expert_n=expert_n,
        protected=PROTECTED_SUPER_EXPERTS,
    )
    if is_aggressive_tier and layers_to_prune != AGGRESSIVE_DEPTH_EXPERTS_LAYERS_TO_PRUNE:
        raise RuntimeError(
            f"depth+experts layer selection drifted: got {layers_to_prune}, "
            f"expected {AGGRESSIVE_DEPTH_EXPERTS_LAYERS_TO_PRUNE}"
        )

    before_params = sum(p.numel() for p in model.parameters())
    build_candidate(model, scores, depth_n=depth_n, expert_n=expert_n)
    after_params = sum(p.numel() for p in model.parameters())
    print(f"params: {before_params:,} -> {after_params:,} ({after_params / before_params:.1%})")

    # Same disjoint-draw convention as every other calibration slice in this
    # paper: reserve shard 0 for calibration, fresh 32-sample seed=1 draw.
    shards = list_shards("bigcode/the-stack-dedup", "data/typescript", token=args.hf_token)
    base_data = DataConfig(dataset_id="bigcode/the-stack-dedup", data_dir="data/typescript")
    split = make_split(base_data, shards)
    calib_samples = load_samples(split, "calibration", n=32, seed=1, token=args.hf_token)
    tokenized = tokenize_samples(calib_samples, tokenizer)
    device = next(model.parameters()).device
    calib_batches = [torch.tensor([ids], device=device) for ids in tokenized.input_ids]

    model.eval()  # type: ignore[no-untyped-call]
    with torch.no_grad():
        loss = mean_loss(model, calib_batches)
    log_metric(run_dir, zero_shot_loss=loss, depth_n=depth_n, expert_n=expert_n)

    if is_aggressive_tier:
        print(f"zero-shot loss: {loss:.4f} (expected {AGGRESSIVE_EXPECTED_ZERO_SHOT_LOSS:.4f})")
        if abs(loss - AGGRESSIVE_EXPECTED_ZERO_SHOT_LOSS) > LOSS_TOLERANCE:
            print(
                "WARNING: zero-shot loss "
                f"{loss:.4f} != sweep's {AGGRESSIVE_EXPECTED_ZERO_SHOT_LOSS:.4f} "
                f"(tol {LOSS_TOLERANCE}), but layer selection matched the aggressive "
                "depth+experts regression -- continuing (re-measurement can drift with "
                "transformers/data revisions; selection is the hard gate)."
            )
            log_metric(run_dir, zero_shot_loss_warning=True)
    else:
        print(
            f"zero-shot loss: {loss:.4f} (no prior measurement for depth_n={depth_n}, "
            f"expert_n={expert_n} -- recording as new data, nothing to compare against)"
        )

    checkpoint_dir = run_dir / "checkpoints" / "candidate"
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    save_config_snapshot(run_dir, {"depth_n": depth_n, "expert_n": expert_n})
    print(f"saved to {checkpoint_dir}")

    if args.push_to_hub:
        # Private per plan.md 4.0's decided checkpoint-persistence policy --
        # only winners get saved at all, and never publicly by default.
        model.push_to_hub(args.push_to_hub, token=args.hf_token, private=True)
        tokenizer.push_to_hub(args.push_to_hub, token=args.hf_token, private=True)
        print(f"pushed to https://huggingface.co/{args.push_to_hub} (private)")


if __name__ == "__main__":
    main()
