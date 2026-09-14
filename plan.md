# bonsai: an MoE Minitron

Status: draft, exploratory. No customer, no deadline. We pick a public MoE
model, apply the Minitron pruning + distillation recipe to it, specialize the
result toward TypeScript, and write down what happens. Where the two Minitron
papers ([`docs/papers/2407.14679`](docs/papers/2407.14679-compact-lms-pruning-distillation.pdf),
[`docs/papers/2408.11796`](docs/papers/2408.11796-minitron-approach.pdf)) don't
cover a decision (they only ever prune dense transformers), we treat it as an
open question and design a step to answer it rather than guessing.

## What the papers actually say

Two papers, same authors, same recipe, applied twice.

**2407.14679** (the original Minitron, NeurIPS 2024) prunes Nemotron-4 15B down
to 8B and 4B. It has access to the original 8T-token pretraining blend. Core
findings, stated as ten best practices in the paper:

1. Train one large model, prune + distill to get the smaller sizes in the
   family; don't train each size from scratch.
2. Importance estimation is purely activation-based (no gradients): for width,
   aggregate with (batch=L2, sequence=mean); for depth, use PPL or Block
   Importance (BI). A 1024-sample calibration set is enough.
3. Single-shot importance estimation is as good as iterative. Skip iterative.
4. Width pruning beats depth pruning at these model scales (≤15B), but only
   after retraining — before retraining, depth looks fine.
5. Retrain with distillation loss (KL divergence against the teacher's logits),
   not conventional cross-entropy against ground truth.
6. Use logit + intermediate-state + embedding distillation when depth is cut
   significantly.
7. Use logit-only distillation when depth isn't cut significantly.
8. Prune to the closest feasible size rather than overshooting and retraining
   more.
9. Do a cheap retraining pass (~1.8B tokens) on every architecture candidate
   before picking one — rankings are unstable until ~300 steps in.
10. If the source model was trained in multiple phases, prune the final-phase
    checkpoint, not an earlier one.

Headline result: Nemotron-4 15B → 8B/4B using up to 40x fewer tokens than
training from scratch, with better MMLU than the from-scratch equivalents.

**2408.11796** (the "in practice" follow-up) reruns the recipe on Llama 3.1 8B
and Mistral NeMo 12B, where the pretraining data is NOT available (both are
proprietary). It adds two things on top of the ten best practices:

- **Teacher correction.** Before pruning, lightly fine-tune the teacher
  (~100B tokens) on whatever data you're actually going to distill on. Skipping
  this costs over 6% in LM validation loss — the teacher gives worse guidance
  when the distillation data doesn't match what it was pretrained on. Figure 1
  in that paper is the whole pipeline: pretrained model → teacher correction →
  pruning → distillation.
- **Downstream-task depth pruning.** PPL-based layer importance doesn't predict
  downstream accuracy well. They rank contiguous layer blocks by Winogrande
  accuracy after removal instead of by validation loss, and drop the worst
  contiguous block (not the worst individual layers — contiguous removal beats
  scattered removal, confirmed by Gromov et al.).

Both papers: forward-KL logit distillation only (no ground-truth CE term) once
depth isn't cut much; single-shot pruning, not iterative; width pruning
consistently beats depth for a fixed parameter budget.

## The part neither paper covers: MoE

Every model in both papers is dense. Nobody in either paper prunes an expert.
Applying Minitron to a Mixture-of-Experts model means adding a pruning axis
that doesn't exist in the source material, and that's the actual experiment
here, not an implementation detail:

- **Depth** (drop transformer blocks) and **attention width** (drop heads)
  carry over unchanged — MoE models still have ordinary attention layers.
- **MLP width pruning** as described in the papers doesn't apply directly: a
  dense model has one MLP per layer; an MoE model has one small MLP per
  *expert*, dozens of them, plus a router deciding which ones fire per token.
- **Expert pruning** is the new axis: drop the least-important experts per
  layer and shrink the router's output accordingly. This changes total
  parameters (memory) without necessarily changing active parameters per token
  (latency), which is the opposite tradeoff from depth/width pruning. Nothing
  in either paper tells us how to score expert importance, whether to prune
  experts uniformly across layers or let the search vary the count per layer,
  or how expert pruning interacts with attention/depth pruning when combined.

Both papers' importance-estimation machinery (Section 2.2 of 2407.14679) is
activation-based and forward-pass-only, which extends naturally to experts:
we can score each expert by how much its output activation magnitude
contributes when it's selected, aggregated over the calibration set, the same
way neuron/head/embedding importance is scored today. Whether that's the right
metric is exactly the kind of thing we're set up to test empirically rather
than assume.

## Target model

**Qwen3-30B-A3B-Base.** Confirmed from its `config.json`:

| | |
| --- | --- |
| Architecture | `Qwen3MoeForCausalLM` |
| Layers | 48 |
| Hidden size | 2048 |
| Attention heads / KV heads | 32 / 4 (GQA), head_dim 128 |
| Experts per layer | 128 routed, top-8 active per token |
| Per-expert MLP hidden size | 768 |
| Vocab | 151,936 |
| Context | 32,768 |
| Total / active params | ~30B / ~3B |

It's a base checkpoint (not instruction-tuned), openly licensed, and strong at
code, which matters for a coding-specialization project. Total-vs-active
params is the number that makes the expert-pruning axis interesting: at ~3B
active out of ~30B total, most of the model's memory footprint is experts that
any given token never touches. Step 0 includes reading the actual
`transformers` modeling code for this architecture to confirm router details
(whether there's a shared/always-on expert alongside the routed ones,
exactly how the router output is reshaped) before writing any pruning code
against assumptions from the config file alone.

## Data — one correction to what you gave me

For **distillation/retraining**, `bigcode/the-stack-dedup` with
`data_dir="data/typescript"` is real and gated-but-accessible (I confirmed the
account has access): 71 parquet shards, ~211MB each, ~15GB total.

For **calibration**, `nuprl/the-stack-ts` does not exist — I checked the
Hugging Face API directly (404, both authenticated and not) and searched
`nuprl`'s full dataset list; nothing by that name or an obvious near-miss is
there. `nuprl/stenotype` exists (a gated TypeScript type-inference dataset) but
it's a different thing — training data for predicting type annotations, not a
general TypeScript corpus.

Given calibration only needs 1024 samples (per both papers, Section 2.2 /
"Dataset" in 2408.11796) and its only job is to drive activation statistics for
importance scoring, I'm proposing we draw calibration samples from
`bigcode/the-stack-dedup/data/typescript` too — a disjoint shard from whatever
we hold out for retraining, so calibration and retraining data never overlap.
This is one dataset instead of two, no code license/access complexity, and
matches what the original Minitron paper did (calibration set = a random slice
of the same blend used for retraining). Flagging it here instead of silently
swapping it in: if you had a specific reason for a second, separate TypeScript
source, say so and I'll go find the right dataset name.

## Compute

Cloud on demand (Lambda). Practically, that means every step below should be
sized to a specific instance type and a specific wall-clock budget before we
rent anything, and the harness should assume ephemeral machines: nothing
lives only on the instance, checkpoints and logs sync to persistent storage
(the same instance store or an HF/S3 bucket) before teardown. Two things from
[[lambda-cloud-api-workflow]] apply directly: the API base is
`cloud.lambda.ai/api/v1` with basic auth, and SSH key registration fails
silently if skipped — the provisioning script needs to check for this rather
than assume it worked.

Rough sizing, refined as we go:

| Phase | What runs | GPU need |
| --- | --- | --- |
| Baseline eval, importance estimation | Forward passes only, ~30B model in bf16 | 1x 80GB |
| Pruning candidate ablations (Phase 2) | Lightweight retrain, small token budgets, ~3B active params | 1x 80GB, hours per candidate |
| Teacher correction | Fine-tune full 30B model, ~100B tokens per the paper | Multi-GPU node, likely the most expensive single phase |
| Final distillation retraining | ~3B active param student, teacher forward pass alongside | 1-8x 80GB depending on token budget chosen |

We start every new pipeline stage at a toy token/sample budget (thousands of
tokens, one candidate, a few calibration samples) purely to prove the code
path is correct cheaply, then scale the budget up once it's verified. This is
not a separate small model — it's the same Qwen3-30B-A3B-Base at a deliberately
tiny budget, so we're never debugging two different architectures.

## Evaluation

TypeScript first, general coding second, per your call. Concretely:

- **Primary, gates every decision:** MultiPL-E HumanEval-TS and MBPP-TS
  (pass@1, matching the papers' eval settings: temperature 0.2, nucleus
  sampling top-p=0.95).
- **Regression check, reported but doesn't gate:** HumanEval and MBPP in
  Python, plus one or two other MultiPL-E languages, to see what a
  TypeScript-slanted distillation costs elsewhere.
- **Tracked, not optimized for:** MMLU, HellaSwag, ARC-Challenge, Winogrande —
  the papers' general-knowledge suite, so we can compare our numbers to
  Table 1/2 of both papers even though we aren't chasing them.
- **Every pruning axis and combination gets the same eval suite run at the
  same checkpoint step**, so axis-vs-axis comparisons (expert-only vs.
  width-only vs. depth-only vs. combined) are apples to apples — this is the
  "report our findings" part of the project, and it only works if the
  comparison is controlled.

## Roadmap

Each step is meant to be small enough to review on its own, and most will land
as one PR the way the docs additions have so far. Steps are grouped into
phases; within a phase, order matters less.

### Phase 0 — Foundation (no GPU needed beyond a laptop)

**0.1 — Project skeleton for the experiment.** `src/bonsai/` gets real modules:
config loading (model/data/compute settings as versioned YAML or TOML, not
hardcoded), a tiny experiment-tracking convention (what gets logged, where
checkpoints and metrics land), and `.env.example` for `LAMBDA_API_KEY`,
`HF_TOKEN`. No pruning logic yet.

**0.2 — Read the model.** Load Qwen3-30B-A3B-Base's config and modeling code
(no weights download required for this, or download config-only). Confirm the
router implementation, whether a shared expert exists, and exactly which
weight tensors correspond to which axis (attention heads, per-expert MLP,
router). Write this up as a short architecture note — this is what the
pruning code in Phase 2 gets written against, so it needs to be right before
we write any of it.

**0.3 — Lambda provisioning script.** Spin up, verify SSH key registration
succeeded (don't trust silence), install the environment, confirm GPU
visibility, tear down. Tested against the cheapest available single-GPU
instance before we ever request a multi-GPU node.

### Phase 1 — Data and baseline

**1.1 — Calibration + retraining data pipeline.** Pull
`bigcode/the-stack-dedup/data/typescript`, split into a calibration shard
(1024 samples) and a retraining pool, disjoint. Tokenize, verify shapes and
that nothing leaks between the two splits.

**1.2 — Baseline eval of the unpruned teacher.** Run the full eval suite above
against Qwen3-30B-A3B-Base as-is. This is the number everything else in the
project gets compared to, so it happens once, early, and gets recorded
somewhere durable (not just a terminal scrollback).

### Phase 2 — Importance estimation and pruning, one axis at a time

Each of these produces a pruned-but-not-yet-retrained model plus its zero-shot
loss, deliberately kept separate so we can see each axis's effect in
isolation before combining anything (this mirrors Table 1 and Table 10 in
2407.14679, which is exactly this kind of ablation).

**2.1 — Depth pruning.** Implement both criteria from the papers: PPL/BI
(2407.14679) and downstream-task-based contiguous-block removal
(2408.11796). Since Winogrande doesn't exist in TypeScript, this step has to
pick a downstream proxy for a coding model — HumanEval-TS or MBPP-TS pass rate
on partially-pruned models is the obvious candidate, but it's expensive per
candidate block; a cheaper proxy (e.g. token-level accuracy on held-out
TypeScript) might be good enough. This is one of the open questions the step
is designed to answer, not a decision made now.

**2.2 — Attention width pruning.** Head importance via the papers' activation
formula (Section 2.2), (L2, mean) aggregation, single-shot.

**2.3 — Expert pruning (new).** Score each expert's importance from
calibration-set activations (candidate metrics: activation magnitude when
selected, selection frequency, router logit magnitude — try more than one,
since this axis has no prior art in either paper to anchor against). Prune to
a target expert count per layer, uniform first, then try letting a search vary
it per layer the way the papers' NAS varies layer/head/MLP counts. Adjust and
re-normalize the router after trimming.

**2.4 — Combining axes.** Depth+width, depth+experts, width+experts, all
three. 2407.14679's own finding was that depth+width rankings flip after
retraining (worse zero-shot, better post-retraining), so every combination
here gets the same lightweight retrain (~their 1.8B-token analog, scaled to
this model's active param count) before we rank it, never a zero-shot
comparison alone.

### Phase 3 — Teacher correction

**3.1 — Correct the teacher on TypeScript.** Fine-tune Qwen3-30B-A3B-Base on
the retraining pool from 1.1, following 2408.11796's recipe (low LR, short
warmup). This step exists for exactly the reason the second paper introduces
it: we don't have Qwen's original pretraining mix, so distilling from the
uncorrected teacher would hand the student worse guidance. Ablate corrected
vs. uncorrected teacher on a fixed pruned candidate to confirm this holds for
us the way it did for them (Figure 4 in 2408.11796), before committing to
using the corrected teacher for every subsequent step.

### Phase 4 — Distillation retraining and candidate selection

**4.1 — Lightweight retrain of every Phase 2 candidate**, using the corrected
teacher, logit-only forward-KL loss (both papers' best practice once depth
isn't cut drastically). Small token budget, purely to stabilize rankings per
best-practice #9.

**4.2 — Pick winners.** One winner per axis (best depth-only, best
width-only, best expert-only) and one overall winner among the combined
candidates, based on the Phase-4.1 rankings plus a fast slice of the eval
suite.

**4.3 — Full retraining of the winners.** Scale up the token budget for
however many candidates we decide are worth the compute (starting with the
overall winner; more if the axis-by-axis comparison is itself the interesting
result).

### Phase 5 — Evaluation and writeup

**5.1 — Full eval suite on every fully-retrained candidate.**

**5.2 — Report.** What each axis bought (params saved vs. accuracy vs. latency
proxy), whether expert pruning behaved like width or like something new,
whether teacher correction mattered as much for us as it did in 2408.11796,
and which combination we'd recommend for a TypeScript-focused deployment if
this stops being hypothetical. Lands in `docs/`.

## Explicitly not doing (yet)

- **Instruction tuning / alignment.** Both papers do this as a final stage
  (NeMo Aligner, RPO). Out of scope until the base pruned/distilled model
  exists and Phase 5 says it's worth aligning.
- **Changing the MoE routing algorithm itself.** We prune experts and adjust
  the router's output dimension; we're not redesigning top-k routing, load
  balancing losses, or training a new router from scratch.
- **Training a new base model.** The entire point of Minitron is not doing
  this. If a pruned/distilled candidate can't recover reasonable accuracy, the
  answer is a different pruning target or more retraining tokens, not
  pretraining from scratch.

## Open questions this plan doesn't resolve

These are deliberately left for the steps above to answer empirically, not
blocking the plan:

1. Best expert-importance metric (2.3).
2. Downstream proxy for depth-pruning criterion in a coding model, given no
   TypeScript Winogrande equivalent exists (2.1).
3. Whether uniform or per-layer-searched expert counts win (2.3, 2.4).
4. Right lightweight-retrain token budget for this model's active param count
   — the papers' 1.8B tokens was calibrated for a 15B dense model, not a 3B
   active-param MoE (2.2-2.4).
5. Whether teacher correction matters as much here as in 2408.11796, given
   we're adapting toward a much narrower domain (TypeScript) than their
   general-purpose retraining data (3.1).
