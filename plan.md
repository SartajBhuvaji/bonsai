# bonsai: an MoE Minitron

> **If you were pointed at this file to run something on a rented GPU:**
> search for `## MISSION BRIEF` below -- there are two, each a complete,
> standalone task. Use the one with the **latest date** in its heading
> (currently `## MISSION BRIEF (2026-09-27)`, just before Phase 4) unless
> you were told otherwise; the older, undated one (Phase 3, just after
> 3.3/3.4) is marked as history, already completed. You don't need to
> read the rest of this file first.

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
  (latency), which is the opposite tradeoff from depth/width pruning. How
  expert pruning interacts with attention/depth pruning when combined is still
  open (2.4).

Neither Minitron paper covers this, but NVIDIA has since extended Minitron
itself to MoE: their Model-Optimizer library implements an MoE variant of the
same activation-based scoring (`mcore_minitron.py`,
`_register_sequential_mlp_importance`), used on Nemotron-3-Nano-30B-A3B. That
formula, and the independently-published REAP paper (arXiv 2510.13999, ICLR
2026 — evaluated directly on Qwen3-30B-A3B), converge on the same metric:
**gate-weighted expert-output norm, averaged only over the tokens actually
routed to that expert** —
`S_j = mean_{x ∈ X_j}[ g_j(x) · ||f_j(x)||₂ ]`, where `X_j` is the set of
tokens for which expert j was in the router's top-8, `g_j(x)` is that token's
(renormalized) routing weight, `f_j(x)` is the expert's `down_proj` output.
Two independent groups landing on the same formula for this exact model is
about as strong an anchor as an unsolved axis gets, so 2.3 adopts it directly
rather than treating expert-importance metric choice as open (see the
research writeup this decision is based on, summarized in the open-questions
section below for the pieces still worth re-deriving ourselves rather than
taking on faith).

Researched via a background agent (Fable) tasked with a literature search
specifically for this axis, since neither Minitron paper nor any prior bonsai
step had anchor points for it. Full citation list: REAP (arXiv 2510.13999),
the unified MoE scoring writeup (arXiv 2606.15716), NAEE (arXiv 2402.14800),
AIMER (arXiv 2603.18492), EvoESAP (arXiv 2603.06003), the Super Experts paper
(arXiv 2507.23279), and NVIDIA's Model-Optimizer source
(`github.com/NVIDIA/Model-Optimizer`). Not independently re-verified
citation-by-citation; worth spot-checking before 2.3's implementation leans on
any one number from this list.

Spot-checked since (paper-review passes, 2026-09-19/20): EvoESAP's actual
finding was the opposite of what an earlier summary claimed (see the
uniform-vs-non-uniform correction above) -- fixed everywhere it was
cited. NAEE (2402.14800) does show selection-frequency-based pruning
performing worst in its own comparison, but that comparison is against
NAEE's own reconstruction-loss method, not an output-magnitude-based one
as `docs/latex/main.tex` originally implied -- fixed there (both mentions,
Related Work and 5.2). `src/bonsai/importance/__init__.py`'s
`expert_importance` docstring makes a different, narrower claim
(attributed to REAP/Model-Optimizer's own ablations, not NAEE) that
wasn't part of this correction and hasn't been separately re-verified.
REAP/NVIDIA Model-Optimizer/AIMER/Super Experts not yet independently
re-verified beyond what those paper-review passes happened to touch.

Confirmed by reading the actual `Qwen3MoeSparseMoeBlock` source in
`transformers` (this was step 0.2's job, done ahead of schedule while writing
this plan): no shared expert (unlike DeepSeek-V2) — just a router
(`Qwen3MoeTopKRouter`) and the experts. `mlp_only_layers` is empty and
`decoder_sparse_step` is 1 in Qwen3-30B-A3B-Base's config, so all 48 layers are
sparse; there's no dense-MLP layer to special-case. One detail that changes how
the pruning code has to be written: the 128 experts per layer are not 128
separate submodules — `Qwen3MoeExperts` holds them as two batched parameter
tensors, `gate_up_proj` shaped `[128, 2×768, 2048]` and `down_proj` shaped
`[128, 2048, 768]`. Pruning an expert means slicing along dimension 0 of these
two tensors (and shrinking the router's weight matrix to match), not deleting
a submodule from a list.

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

TypeScript first, general coding second, per your call. Trimmed deliberately
to keep the recurring cost down -- this suite reruns for every pruning
candidate in Phase 2/4, so what's in it multiplies. Concretely:

- **Primary, gates every decision:** MultiPL-E HumanEval-TS and MBPP-TS
  (pass@1, matching the papers' eval settings: temperature 0.2, nucleus
  sampling top-p=0.95).
- **Tracked, not optimized for:** MMLU (full, ~14k questions), Winogrande
  (full, ~1.2k) and ARC-Challenge (full, ~1.17k) -- three of the papers' own
  four-benchmark general-knowledge suite, so this gives a comparison point
  against Table 1/2 of both papers without chasing either number. HellaSwag
  and a Python/other-language regression check were considered and cut for
  cost (HellaSwag is ~10k questions, the same order as full MMLU, for the
  tier that gates nothing) -- revisit adding it back only for the Phase 5
  winner, not for every early candidate.
- **Every pruning axis and combination gets the same eval suite run at the
  same checkpoint step**, so axis-vs-axis comparisons (expert-only vs.
  width-only vs. depth-only vs. combined) are apples to apples — this is the
  "report our findings" part of the project, and it only works if the
  comparison is controlled.

## Notebooks

This is a learning project as much as a research one, so where a step
produces something worth *looking at* (an architecture, a distribution of
importance scores, a loss curve, an eval table), it gets a notebook alongside
the code, not instead of it.

The split: **logic lives in `src/bonsai/`, notebooks call it.** A notebook cell
should mostly read `from bonsai.architecture import summarize` plus a plot or
a table, not the implementation of `summarize` itself. This keeps the logic
testable and reusable (Phase 2's pruning code needs the same importance-scoring
functions the notebooks used to explain importance scoring in Phase 0/1 — one
implementation, not a copy living in a cell) and keeps the notebooks short
enough to actually read.

What a notebook is for, concretely: markdown cells that quote or paraphrase the
relevant part of whichever paper applies, connect it to the specific numbers
for our model (not the paper's Nemotron-4 or Llama numbers), and then code
cells that show it — render the architecture, plot the score distribution, print
the comparison table. One-time or visual code (plotting, rendering a table,
downloading something to poke at once) can live directly in the notebook; it
never needs a `src/bonsai/` home just for that.

Not every step gets one. Plumbing (the Lambda provisioning script, the config
loader itself) doesn't produce anything worth looking at. Steps below say
explicitly when a notebook is part of the deliverable.

Committed notebooks keep their execution output — that output is the evidence
the step actually ran and produced what the markdown claims, the same
convention `unfuse` already uses for its own model-surgery notebooks. A
notebook committed with its outputs stripped, or that no longer parses, fails
CI (`tools/check_notebooks.py`, adapted from unfuse's version of the same
script).

## Roadmap

Each step is meant to be small enough to review on its own, and most will land
as one PR the way the docs additions have so far. Steps are grouped into
phases; within a phase, order matters less.

### Phase 0 — Foundation (no GPU needed beyond a laptop)

**0.1 — Project skeleton for the experiment.** `src/bonsai/` gets real modules:
config loading (model/data/compute settings as versioned TOML via `tomllib`,
not hardcoded), a tiny experiment-tracking convention (what gets logged, where
checkpoints and metrics land), and `.env.example` for `LAMBDA_API_KEY`,
`HF_TOKEN`. Also the notebook convention itself: `notebooks/`, the
output-preserving CI check, the `notebook` extra (jupyter, nbconvert,
ipykernel, matplotlib). No pruning logic yet.

**0.2 — Read the model.** Load Qwen3-30B-A3B-Base's config and modeling code
(no weights download needed — config.json is public and unauthenticated, and a
meta-device model built from it needs no download either). Confirm the router
implementation, whether a shared expert exists, and exactly which weight
tensors correspond to which axis. Already done ahead of schedule while writing
this plan (see the MoE section above) — no shared expert, every layer is
sparse, experts are batched parameter tensors rather than submodules — but the
notebook is still the deliverable: it's what makes that finding checkable by
someone other than whoever ran the one-off script.
*Notebook:* `00_model_architecture.ipynb` — load the config, build the model on
`meta` device, walk every layer, table the per-axis parameter counts
(attention vs. router vs. experts vs. embedding), chart where the ~30B
parameters actually live, and state plainly which tensors are the pruning
targets for each axis in Phase 2.

**0.3 — Lambda provisioning script.** Spin up, verify SSH key registration
succeeded (don't trust silence), install the environment, confirm GPU
visibility, tear down. Tested against the cheapest available single-GPU
instance before we ever request a multi-GPU node. No notebook — nothing here
to look at, just plumbing that needs to work.

### Phase 1 — Data and baseline

**1.1 — Calibration + retraining data pipeline.** Pull
`bigcode/the-stack-dedup/data/typescript`, split into a calibration shard
(1024 samples) and a retraining pool, disjoint. Tokenize, verify shapes and
that nothing leaks between the two splits.
*Notebook:* `01_calibration_data.ipynb` — a few raw samples, the token-length
distribution, confirmation the calibration/retraining split is actually
disjoint.

**1.2 — Baseline eval of the unpruned teacher.** Run the full eval suite above
against Qwen3-30B-A3B-Base as-is. This is the number everything else in the
project gets compared to, so it happens once, early, and gets recorded
somewhere durable (not just a terminal scrollback).

Done. Primary/gating: HumanEval-TS pass@1 = 0.6708, MBPP-TS pass@1 = 0.7333
(both full test sets, temperature 0.2, top-p 0.95). Tracked, not
optimized-for: MMLU acc = 0.8101 (5-shot, full 14,042), ARC-Challenge
acc = 0.6536 / acc_norm = 0.6749 (25-shot, full 1,172), Winogrande
acc = 0.7798 (5-shot, full 1,267). Notebooks `02a_baseline_humaneval_ts.ipynb`,
`02b_baseline_mbpp_ts.ipynb`, `02c_baseline_general_knowledge.ipynb`.

**1.3 — Shared importance-scoring infrastructure.** Mirrors how the papers
actually structured this (checked directly against Section 2.2 of
2407.14679, not assumed): one shared low-level primitive — aggregate an
activation tensor over batch/sequence with a *pluggable* aggregation function
(mean-abs, L2, variance; the papers ablate which wins per axis rather than
hardcoding one) — wrapped by thin, axis-specific functions that each know
which tensor to feed it. Two of Phase 2's three axes are thin wrappers around
that shared primitive:
- **Attention width (2.2)**: wraps the papers' own `F_head` formula directly,
  no changes.
- **Experts (2.3)**: wraps the gate-weighted expert-output-norm formula
  decided above, aggregated only over each expert's routed tokens (not
  batch/sequence generally — MoE selection is sparse and per-token, so this
  wrapper's aggregation step is genuinely different from the dense case, not
  just a relabeling).
- **Depth (2.1) is deliberately NOT a third wrapper.** The papers keep it
  structurally separate (PPL-delta or Block-Importance cosine-distance, not
  activation-magnitude), and bonsai does the same — 2.1 owns its own scoring
  code, doesn't call into this shared primitive.

Validate at the toy budget (a few dozen calibration samples from 1.1) with a
sanity check before Phase 2 runs it for real: attention scores should rank an
obviously-redundant head below an obviously-load-bearing one; expert scores
should not prune Qwen3-30B-A3B's known Super Experts (arXiv 2507.23279, three
specific early-layer experts whose rare, large outputs anchor attention
sinks — deleting them alone previously measured a PPL jump from 8.70 to
59.86 in that paper, a concrete regression test this step can reuse directly).

Done. Real forward hooks against Qwen3-30B-A3B-Base (bf16, 1x H100), 24
calibration samples (shard 0, max_length 256) -- deliberately a toy slice,
not the full 1024. `head_importance` scores differentiate cleanly with no
NaNs at every layer checked (e.g. layer 47: min 47.9, max 696.4).
`expert_importance` + the outlier guard both independently confirmed on real
data: all three known Super Experts (layer 1/expert 68, layer 2/expert 92,
layer 3/expert 82) were caught by `flag_outlier_experts` (each a massive,
isolated spike in max output norm -- e.g. layer 2's expert 92 at 712 vs. the
next-highest expert at 39) AND ranked #1 of 128 in their layer by the mean
`expert_importance` score itself, not merely "not wrongly pruned." Sanity
check script not committed (GPU-dependent validation, not library code, per
this step's own scope) -- the result is what's durable, recorded here.
No notebook of its own — the score distributions are 2.3's notebook's job
once real pruning axes exist to plot; this step is the plumbing that feeds
it, per the `src/bonsai/`-first split in the Notebooks section above.

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

PPL/BI half done. `bonsai.importance.depth.skip_layer` (a context manager
that temporarily makes one decoder layer act as identity, restoring it
afterward even on exception) + `ppl_importance` (removes each layer in
turn, measures the real held-out perplexity cost against the unmodified
baseline) implement the PPL side; BI was already done in 1.3. Run for real
on Qwen3-30B-A3B-Base (16 calibration samples -- `num_layers + 1` = 49
forward passes per sample makes this far more expensive than BI's one):
Pearson r = 0.91 between BI and PPL delta across all 48 layers, and both
independently pick the exact same top 3 layers (0, 47, 1). BI can be
trusted as a cheap stand-in for PPL for iterating on candidates later in
this phase, reserving PPL's real cost for confirming a final choice.
*Notebook:* `04_ppl_depth_scoring.ipynb` -- the PPL bar chart and the
BI-vs-PPL agreement scatter, kept separate from `03_importance_scores.ipynb`
(which covers heads/experts/BI) since PPL's cost profile is fundamentally
different and the notebook's only real job is checking whether the cheap
proxy predicts the expensive one, not re-deriving the other three scores.

The downstream-task-proxy half (contiguous-block removal) is still open.

Pruning mechanism implemented: `bonsai.pruning.depth.prune_layers` actually
removes the given layers from `model.model.layers` (not `skip_layer`'s
temporary identity-swap) and updates `config.num_hidden_layers` to match.
Real bug caught by testing against an actual forward pass, not just by
reading source: surviving layers keep their original `self_attn.layer_idx`
after removal, which is non-contiguous (e.g. `0, 2, 4, 5` after pruning
layers 1 and 3) and breaks the default KV cache's internal list indexing
with `IndexError` the moment it's used, even though nothing in
`Qwen3MoeAttention.forward` itself depends on it -- fixed by renumbering
each surviving layer's `layer_idx` to its new position. `mean_loss`
(promoted from a private helper already used internally by `ppl_importance`)
supplies the "zero-shot loss" half of this step's deliverable.

Sweep run for real against Qwen3-30B-A3B-Base (32 fresh calibration
samples, seed 1): zero-shot loss at nine layer counts (1, 2, 4, 6, 8, 12,
16, 20, 24 of 48), under four selection strategies -- BI-greedy,
PPL-greedy, best-contiguous-block-by-BI, and a 3-seed random control.
Principled selection clearly beats random at every count, by close to an
order of magnitude once more than a handful of layers are pruned (n=24:
BI/PPL-greedy loss ~2.3-2.5 vs. random seeds 4.8-14.6). BI-greedy and
PPL-greedy track each other closely at every count -- not just in ranking
correlation (already r=0.91) but in the actual pruning outcome, confirming
BI as the working criterion for iterating on candidates. Contiguous-block
removal (2408.11796's own preferred approach) is a genuinely mixed result
on this model: competitive or marginally best at small counts, but falls
behind both greedy strategies by n=24 -- reported honestly rather than
picking whichever framing looks cleaner.
*Notebook:* `05_depth_pruning_sweep.ipynb` -- the sweep figure (log-scale
loss vs. layers pruned, one line per strategy plus random's min-max band)
and the relative-loss table behind it.

Still open: the actual target layer count for a real candidate to carry
into 2.4, and the downstream-task-proxy criterion noted above.

**2.2 — Attention width pruning.** Head importance via the papers' activation
formula (Section 2.2), (L2, mean) aggregation, single-shot.

Pruning mechanism decision (not in the papers -- Qwen3-30B-A3B-Base's GQA
has no equivalent in a dense-MHA model either paper prunes): 32 query heads
share 4 key/value heads in groups of 8 (`num_key_value_groups`), and
`Qwen3MoeAttention.forward` reads `self.num_key_value_groups` fresh off
the module every call via `repeat_kv` -- pruning individual query heads
freely, without regard to group boundaries, would leave that a single
scalar can no longer describe how many times each surviving key/value
head's output needs repeating. Decided: prune the same count of query
heads from every key/value group (never a whole group, which would orphan
its key/value head with nothing to route to it), keeping
`num_key_value_groups` a single well-defined integer after pruning, same
as the real GQA implementation assumes. This is a coarser search than
scoring gives (uniform per-group count, not a free ranking across all 32
heads) -- same tradeoff 2.3 already made for uniform per-layer expert
counts, for the same tractability reason, not because within-group
head-importance differences don't matter.

Pruning mechanism implemented: `bonsai.pruning.heads.prune_heads` slices
`q_proj`'s output rows and `o_proj`'s input columns at head boundaries,
and recomputes `self_attn.num_key_value_groups` for the new head count.
`k_proj`/`v_proj` are never touched. `select_heads_to_prune` picks the
lowest-`k_per_group` heads independently within each contiguous key/value
group from a `head_importance` score tensor. 14 real-forward-pass CPU
tests pass, including the ragged-per-group and whole-group-pruned error
cases.

Done. Real sweep against Qwen3-30B-A3B-Base (32 fresh calibration samples,
seed=1, run on a rented 4x A6000 instance via `device_map="auto"`
sharding -- no H100/A100-80GB capacity available at launch time, unlike
the single-GPU depth/expert sweeps): all 7 valid per-key/value-group
counts (k=1..7 of 8 -- GQA's uniform-per-group constraint means this is
the entire valid search space, not a chosen subset like the other two
sweeps). Greedy beats random at every count, but by a much narrower
margin than depth (up to ~9x) or experts (2-5x): 1.2-2.9x here, actually
*shrinking* as k grows (2.7x at k=1, 1.2x at k=7). Absolute cost, compared
properly at every pruned-fraction the three sweeps share (not a single
point): experts is by far the most redundant axis -- e.g. at 50% pruned,
greedy costs +25.5% for experts vs. +129.6% for heads and +153.8% for
depth. Heads and depth track each other closely at all 3 fractions their
counts coincide on (12.5%, 25%, 50%), with depth's greedy loss slightly
higher at every one -- a small, single-seed-run difference, not a result
to lean on, so heads and depth read as comparably (un)forgiving to prune
rather than one clearly beating the other. *Notebook:*
`07_head_pruning_sweep.ipynb` -- the sweep figure, the relative-loss table
with the greedy/random ratio column, and the cross-axis comparison against
depth/experts. Written up in `docs/latex/main.tex` -- all three axes are
now characterized in the paper (Sections 6-8), none yet combined.

Still open: the actual target per-group head count for a real candidate
to carry into 2.4, and whether a free (non-uniform-per-group) head
search would meaningfully beat this sweep's constrained one -- untested,
since `prune_heads` doesn't support it by design (see the mechanism
decision above).

**2.3 — Expert pruning (new).** Score each expert as
`S_j = mean over its routed tokens of (gate_weight × ||expert_output||₂)`
(REAP / NVIDIA Model-Optimizer's Minitron-MoE formula, decided above — not
selection frequency or raw router-logit magnitude, both of which underperform
this metric in every published head-to-head found). Exempt or percentile-guard
outlier "Super Experts" before ranking, per 1.3's regression test. Prune a
uniform expert count per layer to start, as a simplifying assumption rather
than one settled by confirmed evidence — CORRECTED (paper-review pass,
2026-09-19): re-checked EvoESAP (arXiv 2603.06003) directly against the
primary source, and it reports the *opposite* of what this line originally
claimed — non-uniform allocation beating uniform by up to 19.6% on MATH-500
at matched sparsity, not underperforming it. AIMER (arXiv 2603.18492) does
not address the uniform-vs-non-uniform question at all. The original
"underperformed uniform" claim traces to the initial Fable literature-search
summary and was never independently re-verified before landing here — this
is exactly the failure mode that summary's own caveat warned about. Starting
uniform is kept as a tractability choice for 2.3's first pass, not because
of evidence either way; revisit in 2.4 once single-axis results exist, and
treat a per-layer search as a real candidate rather than a low-priority one.
Router handling:
slice the pruned rows out of `Qwen3MoeTopKRouter`'s weight matrix, keep
top-8, no other changes — Qwen3-30B-A3B-Base's `norm_topk_prob: true` already
renormalizes surviving experts' weights to sum to 1, which is exactly what
every cited one-shot paper's protocol relies on; no router retraining before
the zero-shot eval, since the router adapts during Phase 3/4's retraining
regardless.
*Notebook:* `03_importance_scores.ipynb` (renumbered from an earlier draft's
`02_importance_scores.ipynb` -- `02` was already claimed by the baseline-eval
notebooks, 02a/02b/02c) — for whichever axes have run by this point (depth,
width, experts), plot the score distribution per layer and call out which
layers/heads/experts rank lowest and why that is or isn't surprising given
what 0.2's architecture notebook showed.

Pruning mechanism implemented: `bonsai.pruning.experts.prune_experts`
slices the pruned experts out of `Qwen3MoeExperts.gate_up_proj`/`down_proj`
and the matching `Qwen3MoeTopKRouter.weight` rows together, per named layer.
No renumbering needed (unlike depth's `self_attn.layer_idx` bug) --
`Qwen3MoeExperts.forward` re-derives which experts were hit fresh from
`top_k_index` every call, not from a fixed per-expert identity.
`select_experts_to_prune` picks the lowest-`n_per_layer` by score, with a
`protected` argument that excludes given indices from the ranking entirely
(treated as `+inf` for selection purposes), not just filtered afterward --
used to exempt the three known Super Experts (layer 1/68, layer 2/92, layer
3/82).

CORRECTED (systematic review pass, 2026-09-20): `prune_experts` originally
left `model.config.num_experts` stale after pruning (only the per-module
`experts.num_experts`/`router.num_experts` were updated), and had no
guard against pruning a layer below `router.top_k` surviving experts --
both real bugs, not yet hit by anything the project has actually run
(the real sweep above never saves/reloads a pruned model, and
`output_router_logits` is never turned on), but exactly the kind of
silent landmine the `layer_idx`/`num_key_value_groups` fixes above were
about avoiding. Fixed: `prune_experts` now also raises if pruning would
leave fewer than `router.top_k` experts in a layer, and syncs
`model.config.num_experts`/`model.num_experts` (the latter is what
`load_balancing_loss_func` reads for the router aux loss) whenever every
sparse layer ends up with the same expert count -- which is every call
this project actually makes, via `select_experts_to_prune`'s uniform
per-layer design. Left un-synced (not silently wrong) if a caller ever
prunes a genuinely non-uniform count across layers, since a scalar config
field can't describe that. Same review pass also found and fixed the
identical staleness gap in `prune_heads` for `num_attention_heads`.

Done. Real sweep against Qwen3-30B-A3B-Base (32 fresh calibration samples,
seed=1): eight uniform per-layer counts (8, 16, 32, 48, 64, 80, 96, 112 of
128), greedy (protected-guard) vs. 3-seed random control, plus one candidate
outside the count sweep pruning only the three known Super Experts. Greedy
beats random at every count, though by a smaller multiple than depth
pruning's (roughly 2-5x in relative-loss terms here vs. close to an order of
magnitude for layers) -- 128 experts/layer offers far more substitutable
choices than 48 layers do, so principled selection matters less
dramatically, if still clearly. Pruning is close to free up to a point:
n=8/layer (384 experts total) lands at the same loss as the unpruned
baseline; the curve only bends noticeably past n=48-64. The standout
finding is the Super Experts regression test: pruning just those 3 experts
(of 6,144 in the model) costs more loss (+51.8%) than greedy-pruning up to
1,536 ordinary experts (n=32/layer, +9.6%) -- greedy doesn't cross that line
until somewhere between n=80 and n=96/layer (3,840-4,608 experts, roughly
1,300-1,500x as many as the three Super Experts alone). This reproduces
arXiv 2507.23279's own claim directly in this pipeline's own loss metric
and on this project's own TypeScript calibration data, and is the concrete
justification for the `protected` guard rather than trusting the score
alone. *Notebook:* `06_expert_pruning_sweep.ipynb` -- the sweep figure, the
relative-loss table, and the Super Experts regression-test writeup.

Still open: the actual target per-layer expert count for a real candidate
to carry into 2.4.

**2.4 — Combining axes.** Depth+width, depth+experts, width+experts, all
three. 2407.14679's own finding was that depth+width rankings flip after
retraining (worse zero-shot, better post-retraining), so every combination
here gets the same lightweight retrain (~their 1.8B-token analog, scaled to
this model's active param count) before we rank it, never a zero-shot
comparison alone.

Target counts decided (2026-09-20), from the three real sweeps' own cost
curves rather than a uniform fraction across axes -- experts are
dramatically more redundant than depth or heads on this model (matched-
fraction comparison, already in the paper's §8), so a uniform percentage
would leave experts badly under-exploited. Picked per-axis fractions that
land at a *similar zero-shot cost* instead, across three aggressiveness
tiers:

| tier | depth (of 48) | experts (of 128/layer) | heads (of 8/group) |
|---|---|---|---|
| conservative | n=8 (+12.0%) | n=48 (+13.7%) | k=1 (+5.2%) |
| moderate | n=12 (+21.9%) | n=64 (+25.5%) | k=2 (+21.4%) |
| aggressive | n=16 (+41.6%) | n=80 (+46.3%) | k=3 (+58.5%) |

(% = each axis's own already-measured single-axis zero-shot loss increase
at that count, from 05/06/07's sweeps -- reused directly, not recomputed.)

Done. Real combined-pruning zero-shot sweep (not yet a lightweight
retrain -- that's still 4.1's job), 3 tiers x 4 combination types
(depth+experts, depth+heads, experts+heads, all three) = 12 candidates,
run on a rented 4x A6000 (`scratch_combined_pruning_sweep.py`, deleted
after, same convention as the single-axis sweeps). Real correctness
detail this script had to get right: `expert_scores`/`head_scores`/
`depth_scores` are indexed by the *original* 48-layer model's layer
index, so combining depth pruning with experts/heads requires selecting
on only the surviving layers' scores, in their new (post-depth-pruning)
order -- not the original one -- including remapping the Super Experts
`protected` dict to whichever of layers 1/2/3 actually survive.

| tier | depth+experts | depth+heads | experts+heads | all three |
|---|---|---|---|---|
| conservative | +29.4% | +22.9% | +20.8% | +41.6% |
| moderate | +46.9% | +67.8% | +57.0% | +105.2% |
| aggressive | +91.1% | +189.3% | +135.2% | +288.7% |

(baseline loss 0.9668; % = zero-shot loss increase over baseline)

Real finding, not just "more pruning costs more": comparing each combo's
actual loss to what *independent* compounding of its two-or-three single-
axis costs would predict (naive multiplicative:
$(1+a)(1+b)\cdots - 1$) shows depth+heads interacts genuinely badly --
worse than independent at every tier (+5.1pp conservative, +19.8pp
moderate, +64.9pp aggressive excess over the naive prediction), while
depth+experts is actually *subadditive* at moderate/aggressive
(-6.1pp, -16.0pp -- cheaper than independent compounding would predict).
Plausible reading: depth pruning removes whole layers, including their
attention heads; combining it with separate head pruning on the
*surviving* layers may be compounding damage to the same mechanism
(attention) rather than spreading the cut across more orthogonal
components the way depth+experts does (removing layers vs. removing
MLP/expert capacity hits more independent parts of the model). Not
proven, just the most plausible account of a real, measured effect.
Since experts are also the most redundant single axis by a wide margin
(§8 of the paper), depth+experts is the cheapest pairwise combo at
moderate/aggressive tiers despite combining two full axes, while
depth+heads is the most expensive pairwise combo at every tier.
all\_three is the worst zero-shot candidate at every tier, as expected --
but per 2407.14679's own finding (cited throughout this plan), zero-shot
ranking isn't the final word once retraining happens, so this isn't
grounds to drop it before 4.1's lightweight retrain actually tests that.
*Notebook:* `08_combined_pruning_sweep.ipynb` -- the grouped-bar figure,
the relative-loss table, and the naive-multiplicative interaction
analysis. Written up in `docs/latex/main.tex` (new §9, "Combining Axes")
-- all three single axes plus their zero-shot combinations are now in the
paper; only retraining and final candidate selection remain unwritten.

**Scope expansion (2026-09-20, user request):** the goal isn't just
characterizing pruning tradeoffs academically -- it's finding a pruned +
distilled candidate that *beats* the unpruned baseline on TypeScript
specifically (HumanEval-TS/MBPP-TS pass@1), so it's servable to a
TypeScript-heavy-inference customer. That reframes what "carry into 4.1"
means: not just one winner per axis plus one combined winner (4.2's
original scope), but a broader roster --
a few candidates from *each* single axis (the 9 already-measured points
above, 3 tiers x 3 axes) *and* the combined-sweep candidates above (12,
pending) -- all lightly retrained/distilled and actually benchmarked, not
just filtered by zero-shot loss alone. See 4.0/4.2 below for how this
roster gets evaluated and which of it gets saved.

### Phase 3 — Teacher correction

**3.1 — Correct the teacher on TypeScript.** Fine-tune Qwen3-30B-A3B-Base on
the retraining pool from 1.1, following 2408.11796's recipe (low LR, short
warmup). This step exists for exactly the reason the second paper introduces
it: we don't have Qwen's original pretraining mix, so distilling from the
uncorrected teacher would hand the student worse guidance. Ablate corrected
vs. uncorrected teacher on a fixed pruned candidate to confirm this holds for
us the way it did for them (Figure 4 in 2408.11796), before committing to
using the corrected teacher for every subsequent step.

Training infrastructure implemented (2026-09-20): `bonsai.training` --
first real backward-pass code in this project (everything before this,
`bonsai.importance`/`bonsai.pruning`, is forward-pass-only). `train_step`
does gradient-accumulated single-sample forward/backward passes
(sequences aren't padded to a common length, matching every other
forward-pass call site in this codebase); `lr_schedule` implements linear
warmup + cosine decay to a floor, matching 2408.11796's Table 4 recipe's
*shape*, not its literal numbers (94-380B tokens across 32 DGX H100 nodes
is wildly disproportionate to what this project can run -- the actual
token budget here is an open empirical question, not assumed from the
paper). `loss_fn` is a pluggable seam so 4.1's distillation (forward-KL
against teacher logits, not plain LM loss) reuses this same loop later.
13 real forward+backward-pass CPU tests pass (tiny model, confirms
gradients actually flow and reduce loss, confirms the LR schedule's
exact values at warmup/peak/floor).

Hyperparameter sweep run (2026-09-20): 6 configs (3 peak LRs x 2 warmup
fractions, 150 steps each, 8x A100, `device_map="auto"` + capped
`max_memory` + `bitsandbytes.optim.AdamW8bit` -- plain AdamW's ~4x
per-parameter memory footprint OOMs full-parameter fine-tuning even on
8x40GB). Winner: `peak_lr=2e-5, min_lr=2e-6, warmup_fraction=0.15`, best
on both validation-loss delta and final train loss.

Multi-GPU utilization investigated in parallel: `device_map="auto"`
shards *layers* across GPUs, not parameters, so forward/backward only
ever has one GPU active at a time (confirmed via `nvidia-smi`: 42% on one
GPU, 0-8% on the rest). Built `bonsai.training.distributed` (FSDP via
`accelerate`, shards parameters so every GPU computes every step) to fix
this properly -- validated on a real 8-GPU run after fixing two real OOM
bugs (the `sync_module_states` broadcast path OOMing during FSDP's own
init, and a missing `use_orig_params=True` breaking 8-bit AdamW's
per-parameter quantization). Utilization confirmed high (55-89% across
all 8 GPUs) but wall-clock was ~11 sec/step vs. the naive approach's ~4
sec/step -- FSDP's per-layer all-gather/reduce-scatter overhead outweighs
its parallel-compute benefit for this workload (small batch, MoE, most
layers barely active per token). Genuinely higher utilization, genuinely
slower and costlier; kept the module (tested, PR'd, merged) for workloads
where it will actually win, but used the naive approach for this run.

Real correction run (2026-09-20): 1500 steps, winning hyperparameters,
6,000 fresh non-repeating TypeScript samples (one epoch, no repeats,
disjoint from the calibration pool), 8x A100, same naive
`device_map="auto"` + 8-bit AdamW setup as the sweep. ~4.2 sec/step,
~105 min total. Validation loss 0.9331 -> 0.8842. Checkpoint (57GB)
pushed to
[`sartajbhuvaji/bonsai-qwen3-30b-a3b-teacher-corrected`](https://huggingface.co/sartajbhuvaji/bonsai-qwen3-30b-a3b-teacher-corrected).
This is the corrected teacher for Phase 4's distillation step -- not
pruned, not benchmarked on downstream tasks yet.

Still open: the Figure-4-style ablation (corrected vs. uncorrected
teacher on a fixed pruned candidate) hasn't been run yet -- needs a
pruned candidate selected first (Phase 3's combined-axis pruning work),
which is tomorrow's task.
*Notebook:* `09_teacher_correction.ipynb` (renumbered 2026-09-21 -- `03` was
already taken by `03_importance_scores.ipynb` by the time this was written;
notebook numbers follow build order, not phase number) — reproduce that Figure 4 comparison
for us: LM validation loss, corrected vs. uncorrected teacher, over training
tokens. Loss curves from both the sweep and the real run are logged
(`metrics.jsonl` per run dir) and ready to plot.

**3.2 — Benchmark the corrected teacher.** In progress (2026-09-21). Same
eval suite as 1.2 (HumanEval-TS pass@1, MBPP-TS pass@1, MMLU,
ARC-Challenge, Winogrande), same harness (`bonsai.eval`,
`tools/bigcode_eval_main.py`), just `model_repo_id` pointed at
`sartajbhuvaji/bonsai-qwen3-30b-a3b-teacher-corrected` instead of
`Qwen/Qwen3-30B-A3B-Base` -- no code changes needed, inference only, no
training. Purpose: 3.1's validation-loss improvement (0.9331 -> 0.8842)
is a proxy for "correction helped," not proof -- better perplexity on
TypeScript text doesn't guarantee better pass@1, and this is the only
direct check against the metrics the project actually gates on. Also
establishes the accuracy ceiling every Phase 4 distilled candidate
inherits from, before spending GPU time distilling 21 of them from a
checkpoint that might not have actually improved on what matters.

Running on a rented `gpu_1x_h100_pcie` (80GB, $3.29/hr) -- inference-only
sizing, comfortably holds the ~61GB corrected-teacher weights with
headroom for generation activations/KV-cache, no need for 1.2's/3.1's
multi-GPU sharding tricks. Reuses 1.2's already-validated
`batch_size=8` for the generation tasks (HumanEval-TS/MBPP-TS) rather
than re-tuning: `batch_size=32` was tried during 1.2 and hung for over
an hour, a real finding specific to this MoE model (batched decode
doesn't amortize across sequences that route to different experts the
way it would for a dense model, and batching also waits on each batch's
slowest sequence to hit a stop token). `batch_size="auto"` for the
MMLU/ARC/Winogrande suite (lm-eval's own probing, no generation-length
tail to worry about since it's log-likelihood scoring, not decoding).
*Notebooks:* `10a_teacher_corrected_humaneval_ts.ipynb`,
`10b_teacher_corrected_mbpp_ts.ipynb`,
`10c_teacher_corrected_general_knowledge.ipynb` -- mirror 1.2's
02a/02b/02c structure exactly, model pointer swapped.

**Two real findings from a `--limit 10` diagnostic pass (2026-09-21),
before committing to the full 159-problem run:**

1. **Export-statement habit (TS-specific, cosmetic, fixable).** The
   corrected teacher adds a module `export` statement to every completion
   (10/10 in the diagnostic sample) -- the base model never does (0/10).
   This tanked the raw diagnostic pass@1 to 0.0, but every completion's
   underlying logic was correct: stripping the export line and
   re-scoring the same completions (no regeneration) jumped pass@1 to
   0.9. Root cause: correction fine-tuned on raw TypeScript *files*
   (which naturally contain their own exports), not the bare
   function-completion format MultiPL-E's scaffold expects. This also
   explains the slow, uneven batch timing seen in the full-159 attempt
   (killed early to run this diagnostic instead) -- MultiPL-E's
   stop-sequence heuristics don't recognize a trailing export line as a
   stopping boundary. Fix (agreed, not yet built): a
   `postprocess_generation` override in `bonsai.eval`'s task
   registration to strip leading/trailing export statements before
   scoring -- a no-op for the base model (never emits them), so the
   already-recorded 1.2 baseline stays comparable.

2. **Non-TypeScript generation instability (real, not cosmetic, but
   TS-isolated).** A follow-up `--limit 10` diagnostic on plain
   (Python) HumanEval -- not part of this project's gating suite, a
   one-off cross-check -- scored 0.3 raw, with real pathologies in the
   actual completions: two empty function bodies, three cases where the
   docstring's closing `"""` was glued onto the next statement on the
   same line (a genuine Python `SyntaxError`, not a style mismatch), and
   one completion that degenerated into a 9,227-character repeated-comment
   loop that never produced code. Re-inspecting all 10 of the TS
   diagnostic's completions found none of these pathologies -- all 10
   are clean, syntactically valid TypeScript. So this isn't a
   general instability in the checkpoint; it's isolated to generation
   *outside* the language correction targeted.

   Checked against 2408.11796 directly (`docs/papers/2408.11796-minitron-approach.pdf`,
   Training Details section): the original authors report the same
   shape of effect from their own teacher correction -- "the correction
   process has a minor effect on the teacher model's accuracy on
   downstream tasks, with some tasks improving and some degrading,"
   hypothesized as "an artifact of the dataset used for fine-tuning,"
   and they explicitly flag using fewer than their ~100B-token budget as
   an untested "interesting topic for future work," not something they
   validated as safe. Our correction used ~1.9M tokens (6,000 samples x
   ~316 tokens avg) -- roughly 5 orders of magnitude below their tested
   regime -- which plausibly explains why our off-target degradation
   (Python) is sharper than their "minor effect": a much smaller, more
   concentrated update is less averaged-out than a 100B-token one. Not
   evidence of a broken run; the same acknowledged trade-off, just
   sharper at this project's much smaller compute budget. Worth writing
   up in the paper as a direct extension of 2408.11796's own "future
   work" note. Not yet acted on beyond this note -- general coding
   capability outside TypeScript isn't part of this project's gating
   criteria, but is now a candidate addition to the "tracked, not
   optimized-for" tier alongside MMLU/ARC/Winogrande, pending a
   larger-than-n=10 sample if this gets followed up on.

Export-stripping fix implemented and merged (PR #46, `bonsai.eval`'s
`strip_module_exports`/`ExportStrippingMultiPLE`, 11/11 tests passing
against a real `bigcode_eval` install, verified end-to-end: the full
generate+postprocess+execute pipeline scores pass@1=0.9 directly on the
same 10-problem sample that scored 0.0 raw).

General-knowledge suite run for real (2026-09-21, `10c_teacher_corrected_general_knowledge.ipynb`):

| task | base (1.2) | corrected teacher | delta |
|---|---|---|---|
| MMLU (5-shot) | 0.8101 | 0.8082 | -0.0019 |
| ARC-Challenge acc (25-shot) | 0.6536 | 0.6433 | -0.0103 |
| ARC-Challenge acc_norm | 0.6749 | 0.6689 | -0.0060 |
| Winogrande (5-shot) | 0.7798 | 0.7806 | +0.0008 |

All deltas are within noise (largest ~1pp, on ARC-Challenge). General
knowledge/reasoning is essentially preserved by correction.

**Full 159-problem HumanEval-TS run (2026-09-22): pass@1=0.5597 raw,
0.5660 after fixing a real gap in the export-stripping fix -- both well
below the base model's 0.6708.** The 10-problem diagnostic's 0.9 was not
representative; reading too much into a sample that small was a real
mistake, corrected here rather than left standing. Two things found and
fixed/accounted for before trusting this number:

1. **Regex gap, fixed.** `strip_module_exports`'s original pattern
   required the export statement to start a line (`^export...`). The
   model sometimes emits an entire completion as one line with no
   newline before the trailing export -- confirmed directly on 2 of 159
   completions, missed by the anchored pattern. Fixed by dropping the
   `^` anchor (`export` is a reserved word in TS/JS, never a legitimate
   identifier, so the broader match can't accidentally eat real code).
   Regression test added (`test_strip_module_exports_removes_same_line_export_no_newline`).
2. **Not a regex problem, real generation failures.** Re-scoring the
   same 159 completions with the fixed regex (`--load_generations_path`,
   no regeneration) moved pass@1 by only +0.0063 (89 -> 90 of 159) --
   confirming the regex gap explained almost none of the shortfall.
   Manually inspecting the remaining failures found genuine, distinct
   pathologies: **duplicate function declarations** (the model writing
   `function anti_shuffle` twice in the same completion -- a real
   TypeScript compile error, unrelated to exports) and **degenerate
   repetition loops** (the same fragment, e.g. `export function eat(...)
   {`, repeated 50+ times until hitting the length cap). This corrects
   an earlier claim in this log: the 10-problem diagnostic's inspection
   found the repetition-loop pathology only in the *Python* HumanEval
   cross-check and concluded TypeScript generation was clean at full
   scale -- confirmed wrong here; the same instability shows up in TS
   too, just less often (roughly 2-4 of 159, not 1 of 10).

Bottom line, stated plainly: teacher correction, as configured (1500
steps, ~1.9M tokens, TypeScript-only), makes the teacher genuinely
*worse* at the primary, gating benchmark it was supposed to help with.
The validation-loss improvement (0.9331 -> 0.8842, 3.1) did not
translate to better HumanEval-TS pass@1 -- this is the actual answer
3.2 was built to get, and it does not match the paper's own framing of
teacher correction's expected effect ("minor," "some tasks improving
and some degrading"). Plausible reading, not yet confirmed: the
~5-orders-of-magnitude-smaller token budget (1.9M vs. 2408.11796's
~100B) may not just amplify off-target degradation (as the Python
cross-check suggested) but destabilize in-domain generation too --
fewer, more concentrated gradient updates leaving the model likelier to
loop or duplicate a declaration than a properly-averaged 100B-token
correction would. Not proven; MBPP-TS (390 problems, next) is another
real data point before drawing a firm conclusion, and the still-open
Figure-4-style ablation (corrected vs. uncorrected teacher on a fixed
*pruned* candidate, not the unpruned teacher itself) remains the
decisive test for whether this teacher should actually be used for
Phase 4 distillation as-is, or whether the correction recipe (token
budget, learning rate, warmup) needs revisiting first.

**3.3 — LoRA correction, testing method (not just token budget) as the
cause.** In progress (2026-09-22). The original plan for today was a
larger-token-budget full-parameter correction run to test the
token-budget hypothesis directly -- blocked: no capacity for
`gpu_4x_a6000`/`gpu_8x_a100` (the multi-GPU types full-parameter
correction needs) at the time. `gpu_1x_h100_pcie` (80GB) had capacity,
which can't fit full-parameter correction (~152GB needed: weights +
gradients + 8-bit-AdamW states) but comfortably fits LoRA correction
(frozen ~61GB base + a small trainable adapter, no `device_map` sharding
or 8-bit optimizer needed) -- and 2408.11796 itself flags LoRA as
untested future work for this exact step, so this became the
experiment instead of waiting on capacity.

Matched to the original full-parameter run (`scratch_teacher_correction_real.py`,
deleted per convention) on everything except method: same 1500 steps,
same retraining-pool source (fresh, non-overlapping, seed=5 this time),
same warmup fraction, same batch size. LoRA adapters (r=16, alpha=32,
dropout=0.05) on attention projections only (\texttt{q\_proj},
\texttt{k\_proj}, \texttt{v\_proj}, \texttt{o\_proj}) -- not the experts,
which dominate the model's parameters; a real scope choice for
tractability, not an oversight, and worth revisiting if this run shows a
clear effect. LR is $1\times10^{-4}$ (vs.\ the full-parameter run's
$2\times10^{-5}$) since LoRA updates a much smaller subspace and
conventionally needs a higher LR to move as far in the same step count
-- using the same LR here would confound "does LoRA behave differently"
with "was this LoRA run just undertrained."

A real correctness gotcha caught before launching, not after: gradient
checkpointing combined with a fully-frozen base model needs
`model.enable_input_require_grads()`, or checkpointed segments backprop
through an input that doesn't require grad and the LoRA adapters
silently get zero gradient instead of an error -- fixed in the script
before the real run, not discovered from a flat loss curve after.

Once trained, the LoRA adapter gets merged into the base weights
(`merge_and_unload()`) so the resulting checkpoint is a plain dense
model, loadable and deployable identically to the full-parameter
corrected teacher -- no peft-specific handling needed downstream.
Benchmarked the same way as 3.2, but on an n=40 sample of HumanEval-TS
rather than the full 159 for faster iteration: at n=40, pass@1's 95% CI
is roughly $\pm$16pp (vs.\ $\pm$8pp at n=159) -- too noisy for a final
number, but enough to catch a large directional difference from the
full-parameter corrected teacher's 0.566, which is all this pass needs
to answer "is this worth a full run."
*Notebook:* `11a_teacher_corrected_lora_humaneval_ts_sample.ipynb`.

**3.4 — Retraining-data quality audit, in parallel with 3.3's training
run.** Prompted by a real gap: nothing before this checked the actual
*content* quality of the retraining pool -- `01_calibration_data.ipynb`
verifies token-length distribution and calibration/retraining
disjointness, not whether individual files are well-formed. Pulled 300
representative samples (same source/pipeline the real runs use) and
checked for patterns that could mechanistically explain the corrected
teacher's repetition-loop and duplicate-declaration pathologies (3.2):

- **Repetition within a training sample: 0/300.** Rules this out as the
  explanation for the model's repetition-loop behavior -- the data
  itself isn't repetitive.
- **Duplicate declarations within a sample: 0/300 genuine** (an initial
  regex flagged 7, all false positives on manual check -- e.g.\ the
  English word ``class'' inside a comment, coincidentally followed by an
  unrelated `if`). Rules this out too.
- **Truncation at 512 tokens, real and substantial: 92/300 (30.7\%) of
  samples exceed `max_length=512` and get cut off mid-statement, with
  standard unmasked next-token loss applied right up to the cutoff.**
  Concrete example: a sample truncated mid-array-literal, ending
  `...Resistor.BLACK, Resistor.BROWN, Resistor.RED, ... Res` --
  mid-identifier. At this rate, roughly 1{,}800 of the real run's 6{,}000
  samples teach the model that stopping abruptly mid-expression is a
  normal, complete training target. A mechanistically direct candidate
  for the model's observed difficulty cleanly terminating declarations
  -- arguably more concrete than the token-budget-scale hypothesis in
  3.2's closing paragraph, and not mutually exclusive with it.
- Smaller, secondary finding: $\sim$21\% of the pool is `node_modules`
  vendored bundles (12/300, auto-generated) or `.d.ts` type-stub files
  (26/300, often auto-generated), diluting the training signal away from
  the hand-written, function-implementation style HumanEval-TS actually
  tests.

Doesn't invalidate 3.3's in-progress LoRA run -- both runs share this
same data pipeline and 512-token cutoff, so the method comparison (LoRA
vs.\ full-parameter) stays valid regardless. Gives a concrete, testable
follow-up independent of both the method and token-budget questions:
filter the retraining pool to samples that fit within `max_length`
(or raise `max_length`, or mask loss on the truncated tail), and
consider excluding `node_modules`/`.d.ts` paths, then re-run correction
to see whether this alone closes some of the HumanEval-TS gap.

Concrete recipe decided for that follow-up (2026-09-22), from the actual
length distribution of the same 300-sample pull (checked across several
caps, not guessed): 512 (current) truncates 30.7%; 768 -> 20.0%; 1024 ->
13.3%; 1536 -> 8.3%; **2048 -> 3.7%**; 4096 -> 1.3%. `max_length=2048` is
not an arbitrary bump -- it's already `tokenize_samples`'s own default,
the same value `01_calibration_data.ipynb` uses; the correction scripts
used 512 instead without a documented reason (plausibly carried over
from the *eval* harness's unrelated `max_length_generation=512`, a
different parameter capping generation length, not training-input
length). Plan: raise the correction scripts' `MAX_LENGTH` to 2048, and
drop (redraw a replacement sample for) the residual $\sim$3.7\% still
over that cap rather than truncating them, so zero training samples end
mid-statement. Mean sample length is only 556 tokens, so this mainly
lengthens the current long tail, not the typical step -- a bounded
compute-cost increase, not a proportional one.

**Implemented and applied to 3.3, same day, before trusting any result
from it.** 3.3's LoRA run was already $\sim$40\% through (step
$\sim$600/1500) on the old `MAX_LENGTH=512` pipeline when this audit's
findings landed -- cancelled rather than let finish, since any result
from data with a known, fixable defect wouldn't be trustworthy enough to
act on. `scratch_teacher_correction_lora.py` updated: `MAX_LENGTH=2048`;
pulls `1.3x` the needed sample count as a buffer, filters out
`node_modules/` paths, `.d.ts` paths, and anything still over
`MAX_LENGTH` (checked by tokenizing without truncation first), then
takes exactly `num_train_samples` from what survives -- raising loudly
rather than silently training on fewer samples if the buffer isn't
enough. Verified on a 5-step smoke test (26 pulled -> 20 kept, matching
the expected ~16% combined drop rate) before relaunching the real
1500-step run. This means 3.3's LoRA-vs-full-parameter method comparison
is now confounded with the data fix relative to 3.2's full-parameter
result (two things changed, not one) -- accepted deliberately: training
on 1,800+ known-truncated samples first and comparing methods on top of
that would have been the worse experiment. Isolating method from data
quality, if it turns out to matter, is a future follow-up, not blocking
this run.

**Real run's actual drop rate: 7,800 pulled -> 6,000 kept -> 1,800
dropped (23.1%)**, in the same ballpark as the audit's ~16% additive
estimate (some variance expected -- this run draws seed=5, not the
audit's seed=3, and BUFFER_FACTOR was deliberately sized to overshoot
rather than land exactly). Framed against the old pipeline for what
this actually buys: the old pipeline didn't drop anything, it silently
corrupted 30.7% of samples via mid-statement truncation instead -- same
6,000-sample training target either way, but now 100% of it is
well-formed instead of ~69%. Two honesty caveats, not yet resolved:
the audit itself was a 300-sample subset from only 5 of 70 shards (an
estimate, not a full-pool guarantee), and it checked seed=3's draw, not
this run's seed=5 -- same pipeline and corpus, so no real reason to
expect different characteristics, but not a direct re-verification
either. Plan: re-check data quality (repetition, duplicate
declarations, minified/auto-generated content) against this run's actual
checkpoint once benchmarked, rather than assume the fix generalizes
without looking.

**3.3 training complete (2026-09-22).** 1500 steps, ~1h58min on 1x H100.
Validation loss: 0.8927 (first recorded checkpoint, step 25) -> 0.8346
(final, step 1500), steady decline then a stable plateau from roughly
step 400 onward -- no instability, no divergence. Checkpoint merged
(`merge_and_unload`) and pushed to
[`sartajbhuvaji/bonsai-qwen3-30b-a3b-teacher-corrected-lora`](https://huggingface.co/sartajbhuvaji/bonsai-qwen3-30b-a3b-teacher-corrected-lora).
Note on comparability: this run's "first recorded" val loss isn't a true
pre-training baseline (it's the training loop's first periodic check, 25
steps in), unlike the full-parameter run's separately-measured "before
correction" figure (0.9331) -- so the two runs' *deltas* aren't a fair
comparison, but the *final* values are (same calibration set, same loss
function): 0.8346 here vs.\ 0.8842 for the full-parameter run. Lower
final validation loss, on top of the data-quality fix 3.3 also carries
that the full-parameter run didn't have. Downstream HumanEval-TS
benchmarking not yet run -- deferred, next task.

**HumanEval-TS 5-problem sanity check (2026-09-22): 4/5 pass@1=0.8,
all clean.** Before committing to the full 159-problem run: all 5
completions are syntactically valid TypeScript, no repetition, no
truncation artifacts; the one failure is a clean stub (`// Your code
here`, returns `[]]`) -- a genuine logic gap, not a pathological
generation. Real, notable qualitative finding: **none of the 5
completions added a trailing `export` statement** -- unlike the
full-parameter corrected teacher, which added one to every single
completion in its own diagnostic. Plausible reading: LoRA's more
restricted update (attention projections only) didn't pick up that
file-level stylistic habit the way full-parameter fine-tuning, which
touches every weight including the output layer, did. Not proven at
n=5, but worth watching for at full scale.
*Notebook:* `11b_teacher_corrected_lora_humaneval_ts.ipynb`.

**Full 159-problem run (2026-09-22): pass@1=0.5723 raw -- confirms the
n=5 export finding at scale, but does not close the gap to the base
model.** Only 2/159 completions contain "export" at all (vs.\ the
full-parameter teacher's 10/10 in its own diagnostic) -- LoRA's
attention-only update genuinely doesn't pick up the file-level export
habit the way full-parameter fine-tuning did. Real improvement over the
full-parameter teacher's 0.5660 (+0.0063), and this run also carries the
3.4 data-quality fix the full-parameter run didn't have -- but still
8.6pp below the base model's 0.6708.

A different, new failure mode found on inspection, not present in the
full-parameter teacher's completions: **runaway trailing test-case
generation**. 4 completions run to 3,000-6,000 characters (vs.\ a 731-char
mean) -- not verbatim repetition or duplicate declarations this time,
but a complete, plausible-looking function implementation followed by
an ever-growing sequence of `console.log(fn(...))` example calls with
incrementing arguments, never stopping until the length cap. Checked
directly: stripping text from the first `\nconsole.log(` onward (15
completions affected) and re-scoring the same completions, no
regeneration, raised pass@1 to **0.5849** (93/159, +0.0126). Smaller
effect than the export fix, and less clean to generalize -- unlike a
trailing `export` statement, a single legitimate `console.log` isn't
inherently wrong, so this fix is coarser (strips from the *first*
occurrence, which could theoretically cut a correct completion that
happens to end with one intentional log call) and hasn't been added to
`bonsai.eval` as a permanent postprocessing step, just verified as a
diagnostic. The two short completions initially flagged as possible
empty bodies (`strlen`, `add`) turned out to be complete, correct,
naturally short implementations -- not a pathology.

| | pass@1 |
|---|---|
| base model (unpruned, uncorrected) | 0.6708 |
| full-parameter corrected teacher | 0.5660 |
| LoRA-corrected teacher, raw | 0.5723 |
| LoRA-corrected teacher, console.log-stripped | 0.5849 |

Two independent interventions (LoRA method + 3.4's data-quality fix;
then this postprocessing check) each moved pass@1 in the right
direction, by small amounts. Neither closes the gap to the base model.
Still open: whether the remaining ~8.6pp gap is the token-budget-scale
issue from 3.2's original hypothesis, something else about the
correction recipe, or a combination -- not yet isolated.

**General knowledge, run for the LoRA-corrected teacher (2026-09-25):
preserved, same as the full-parameter teacher.**
*Notebook:* `11c_teacher_corrected_lora_general_knowledge.ipynb`.

| task | base (1.2) | full-parameter corrected | LoRA-corrected |
|---|---|---|---|
| MMLU (5-shot) | 0.8101 | 0.8082 | 0.8070 |
| ARC-Challenge acc (25-shot) | 0.6536 | 0.6433 | 0.6485 |
| ARC-Challenge acc\_norm | 0.6749 | 0.6689 | 0.6715 |
| Winogrande (5-shot) | 0.7798 | 0.7806 | 0.7845 |

All deltas within noise for both correction methods (largest ~1pp).
Consistent picture across both: neither correction method measurably
hurts general knowledge/reasoning -- the regression is specific to
HumanEval-TS, the metric the project actually gates on.

**MBPP-TS (390 problems), done (2026-09-27).** Run on a DigitalOcean 1x
H100 by a different Claude session (handoff worked -- see MISSION BRIEF
below for how it was briefed). **pass@1 = 0.6077** (237/390) vs.\ the
base model's 0.7333 -- **12.6pp below base, a wider gap than
HumanEval-TS's 9.85pp** (17.1% relative drop vs.\ 14.7%) -- correcting
an inaccurate claim from the handoff session's own commit message that
MBPP-TS was "closer to base": checked directly, it isn't, on both an
absolute-pp and a relative basis. Same conclusion as 3.3, now confirmed
on a second gating benchmark: LoRA correction + the 3.4 data-quality fix
does not restore primary-benchmark accuracy vs.\ the uncorrected base,
and the gap isn't closing as more of the correction recipe gets fixed --
if anything this result is worse, proportionally, than the first one.
Run dir `runs/20260926T143533Z-572b8e-teacher-corrected-lora-eval` lives
on the GPU box that ran it, not this machine.
*Notebook:* `11d_teacher_corrected_lora_mbpp_ts.ipynb`, committed with
its real execution output.

**Real, reusable side effect of this handoff: resumable bigcode-eval
runs.** The handoff session found that `bigcode-eval` already has an
unused intermediate-checkpoint mechanism (`save_every_k_tasks`, always
left at its disabled default of `-1` in every run this project has done
so far) and wired a `--resume` flag through `tools/bigcode_eval_main.py`
to use it (`src/bonsai/eval/checkpoint.py` -- loads whatever checkpoint
exists, resumes generation from there). This directly fixes the exact
failure mode that lost real progress twice earlier in this project: a
killed HumanEval-TS run at 97% completion (1-hour timeout), and this
same MBPP-TS task's own first two attempts on Lambda (one timeout
mistake, one provider switch) -- both restarted from problem 0 because
nothing survived a kill. Every future long eval run should use
`--resume` from the start.

**Full results table, both gating benchmarks, all three checkpoints:**

| | HumanEval-TS pass@1 | MBPP-TS pass@1 |
|---|---|---|
| baseline (uncorrected) | 0.6708 | 0.7333 |
| full-parameter corrected | 0.5660 | -- (never run, predates 3.4's fix) |
| LoRA-corrected, raw | 0.5723 | 0.6077 |
| LoRA-corrected, export/example-stripped | 0.5849 | -- (not checked yet) |

MBPP-TS's gap (12.6pp, 17.1% relative) is wider than HumanEval-TS's
(9.85pp, 14.7% relative) -- both benchmarks now agree that this
project's teacher-correction recipe, in every configuration tested,
regresses the model's own TypeScript code-generation accuracy.

**What the broader literature says about this (checked 2026-09-27,
not just our own two papers).** 2408.11796 itself only acknowledges the
effect ("minor," "some tasks improving and some degrading," "an
artifact of the dataset") without investigating it -- 2407.14679 never
needs teacher correction at all, so says nothing. The wider knowledge
distillation literature has a real, established answer to the question
this raises (does the teacher's own regressed accuracy mean it's a
worse *distillation* teacher?): not necessarily.

- **Cho \& Hariharan, "On the Efficacy of Knowledge Distillation"
  (ICCV 2019, arXiv:1910.01348).** Found that more accurate/larger
  teachers often make *worse* distillation teachers -- a "capacity gap"
  effect where a student can't represent a mapping too different from
  what it can express. They found a less-converged, lower-accuracy
  teacher sometimes distills into a *better* student than a
  fully-trained one, and note that stopping a teacher's training early
  is one way to exploit this.
- **Kim, Park, Lee \& Kwak, "The Role of Teacher Calibration in
  Knowledge Distillation" (2025, arXiv:2508.20224).** A teacher's
  *calibration error* -- how well its confidence matches its actual
  correctness, not its raw accuracy -- predicts distillation success
  better than accuracy does. A less accurate but better-calibrated
  teacher can outdistill a more accurate but overconfident one.
- Domain-adaptation-specific work (found via search, not read in full)
  surfaces a two-sided nuance: a domain-adapted teacher's narrower,
  more target-focused behavior can help distillation into that domain,
  but unconstrained on-policy distillation from a narrowly fine-tuned
  teacher can also act as an over-conservative anchor limiting how much
  the student adapts. Which effect wins seems to depend on specifics
  this project hasn't tested.

Neither paper tells us which way *our* corrected teacher will go once
distilled into an actual pruned student -- our accuracy measurements
aren't a calibration study, and we haven't run one. But they're a real,
citable reason not to treat teacher accuracy as a stand-in for teacher
usefulness, and they directly support the plan already in place: the
Figure-4-style ablation against a pruned candidate is the right next
experiment, not an optional extra step, because it's the only way to
test the thing that actually matters for Phase 4 instead of continuing
to relitigate the corrected teacher's own benchmark scores.

## MISSION BRIEF

**Written 2026-09-26 for whichever Claude session picks this up next;
completed 2026-09-27 -- kept below for history, not because there's
still a task here.** This section was self-contained by design: stated
the task, model, benchmark, and GPU situation up front so a fresh
session with no prior conversation context could act on it directly
without needing the rest of this file. It worked -- the result and a
real infrastructure improvement (above) came back from it.

**The task, in one sentence:** run the MBPP-TS benchmark against the
LoRA-corrected teacher checkpoint, using the GPU you already have
access to (DigitalOcean, not Lambda -- no provisioning needed, you're
not launching anything, just SSH'd into or running directly on a box
that already has a GPU).

**Which model, exactly:** `sartajbhuvaji/bonsai-qwen3-30b-a3b-teacher-corrected-lora`
on Hugging Face. Not the full-parameter corrected teacher
(`sartajbhuvaji/bonsai-qwen3-30b-a3b-teacher-corrected`, a different,
older checkpoint) -- that one predates a data-quality fix (3.4) and its
MBPP-TS score wouldn't be a fair comparison point.

**Which benchmark, exactly:** MBPP-TS (`multiple-mbpp-ts` in the
`bigcode-evaluation-harness` task registry this project patches in via
`bonsai.eval.register()`), full 390-problem set. Not HumanEval-TS
(already done for this model, 0.5723/0.5849, see 3.3 above) and not the
general-knowledge suite (already done, see 3.3 above) -- those are
finished, this is the one gap left.

**Prerequisites on the GPU box:** clone this repo, `pip install -e
".[eval]"` (needs Node.js + `tsc` too -- MultiPL-E's TS task shells out
to compile/run the generated code; see `tools/setup_instance.sh` with
`WITH_EVAL=1` for the exact steps, adapted as needed since that script
assumes a Lambda Stack image). Needs ~80GB of GPU memory (the model is
~61GB in bf16 plus generation activations/KV-cache) -- confirm the
DigitalOcean GPU has this much before starting, or it'll OOM partway in.

**The exact command:**

```bash
accelerate launch tools/bigcode_eval_main.py \
    --model sartajbhuvaji/bonsai-qwen3-30b-a3b-teacher-corrected-lora \
    --revision main --precision bf16 \
    --tasks multiple-mbpp-ts \
    --max_length_generation 512 --batch_size 8 \
    --allow_code_execution --save_generations \
    --metric_output_path <run_dir>/mbpp-ts.json \
    --save_generations_path <run_dir>/mbpp-ts-generations.json
```

Run this from `tools/bigcode_eval_main.py` specifically, not
`bigcode_eval`'s own upstream entrypoint -- this vendored copy calls
`bonsai.eval.register()` first, which patches in an export-statement
postprocessing fix this project's own eval harness needs (see 3.2/3.3
above for why). Skipping that step would score real, correct
completions as failures.

**Two hard-learned lessons, both costly to relearn -- read before
running:**

1. **`batch_size=8`, not higher.** `batch_size=32` was tried during this
   project's very first baseline eval and hung for over an hour -- this
   MoE model's batched decode doesn't amortize across sequences routing
   to different experts the way a dense model's would, and batching
   also waits on each batch's slowest sequence to hit a stop token.
   Don't re-tune this; it's already been found the hard way.
2. **This is slow -- budget 10-11 hours, and set your timeout
   accordingly.** Real observed pace on this exact benchmark/model
   combination was ~100s/problem average across 390 problems on the
   attempts already made (see below) -- much slower than this same
   model's HumanEval-TS pace (~54s/problem). If you're running this
   inside a Jupyter notebook via `nbconvert`, set
   `--ExecutePreprocessor.timeout` to at least 16 hours (57600 seconds)
   -- a 1-hour timeout killed an earlier HumanEval-TS attempt on this
   same model at 97% completion, all progress lost, because
   `bigcode-eval` only writes its output files once the *entire* task
   finishes, not incrementally. If you're running this as a plain
   script/CLI with no notebook wrapper, this specific pitfall doesn't
   apply, but budget the same 10-11 hours regardless and don't kill it
   early on an impatient read of the first few percent.

**Status before this landed, kept for history:** two attempts on a
different GPU provider (Lambda Cloud) were killed before completion by
the previous session -- once from the 1-hour-timeout mistake described
in lesson 2 above (on the HumanEval-TS run, not this one), once
deliberately to switch providers. Nothing was salvageable from either
at the time, for exactly the reason lesson 2 warns about -- which is
also what motivated the resumable-eval fix described above. The
DigitalOcean run that actually finished was a clean restart from
problem 0, not a resume; `--resume` didn't exist yet when it started.
The result and comparison numbers are above, not repeated here.

## MISSION BRIEF (2026-09-27) — corrected-vs-uncorrected teacher ablation

**Written for whichever Claude session picks this up next, possibly on a
different account/GPU than the one that wrote it** (same handoff pattern
as the MISSION BRIEF above, which worked -- see 2026-09-27's MBPP-TS
result). Self-contained: states the task, exact scripts, exact commands,
and a real blocker to resolve first, so a fresh session with no prior
conversation context can act on it directly.

**The question, in one sentence:** does the teacher-correction regression
found in secs. teacher-correction/teacher-correction-lora (both correction
methods score *worse* than the uncorrected baseline on HumanEval-TS/MBPP-TS,
despite improved validation loss) actually carry through to a real
distilled student, or does distillation wash it out? Nobody knows yet --
this is the first real test, not a zero-shot loss proxy.

**What's already built and tested (2026-09-27, this session, no GPU used):**
- `src/bonsai/pruning/candidates.py` -- `select_depth_experts_candidate`,
  pure index arithmetic, reproduces `08_combined_pruning_sweep.ipynb`'s
  aggressive-tier depth+experts selection from real importance scores.
  Tested against synthetic tensors (`tests/test_pruning_candidates.py`,
  4 tests) plus one regression test against the real committed scores
  (5th test, skipped automatically if that file's missing -- see blocker
  below).
- `src/bonsai/training/distillation.py` -- `make_kd_loss_fn`, a forward-KL
  loss against cached top-k teacher logits, built to drop straight into
  `bonsai.training.train`/`train_step` unchanged (their own docstring
  already anticipated this exact seam). Tested with real tiny-model
  forward/backward passes on CPU (`tests/test_distillation.py`, 4 tests,
  including a real gradient-descent-reduces-loss check).
- `tools/build_candidate.py` -- loads `Qwen/Qwen3-30B-A3B-Base`, applies
  the aggressive depth+experts selection, verifies the result's zero-shot
  loss lands at the sweep's own 1.8479815144091845 (tolerance 0.01) before
  saving anything, so a selection/pruning drift is caught before any GPU
  time goes into distillation. **Never run for real -- needs a GPU.**
- `tools/cache_teacher_logits.py` -- one inference-only pass per teacher
  (`--teacher baseline` or `--teacher lora-corrected`), same fresh
  retraining-pool sample set (seed=11, disjoint from every prior draw's
  seed) for both, top-20 logits saved per token. **Never run for real.**
- `tools/train_distilled_candidate.py` -- trains the candidate against one
  cached teacher's logits, 8-bit AdamW, 500 steps. Run this twice, once
  per teacher's cache, identical everything else. **Never run for real.**

All four new/changed files pass `ruff check .`, `ruff format --check .`,
`mypy` (note: `tools/` isn't in mypy's checked path per `pyproject.toml`'s
`[tool.mypy] files`, but was checked manually anyway), and the full
`pytest` suite (158+ tests, all passing) as of this writing -- verified
*before* handoff, not left for whoever runs this next to discover broken.

**BLOCKER -- resolved (2026-09-27).** `runs/` is entirely gitignored, so
the real forward-hook importance scores `build_candidate.py` needs
(`runs/20260919T150941Z-ff00be-importance-scores/importance-scores.pt`)
only ever existed on the machine that originally computed them, not in
this repo, not on a fresh clone. Fixed: uploaded to a **private** HF
dataset repo, `sartajbhuvaji/bonsai-importance-scores`
(`importance-scores.pt`, same content, verified byte-for-byte via the
same tensor shapes/dtypes after download), with explicit human approval
before creating it. `build_candidate.py`'s `load_importance_scores()` now
tries the local path first and falls back to `hf_hub_download` from that
repo automatically -- confirmed working end-to-end (downloaded, loaded,
`depth_scores`/`expert_scores` shapes matched) before this note was
written. No action needed on the GPU side beyond having HF Hub read
access to that repo, which the project's usual token already has.

**Once unblocked, the exact commands, in order:**
```bash
# 1. Build and save the candidate (needs the importance-scores.pt above)
python tools/build_candidate.py --push-to-hub sartajbhuvaji/bonsai-candidate-aggressive-depth-experts

# 2. Cache both teachers' logits against the same sample set (run both)
python tools/cache_teacher_logits.py --teacher baseline
python tools/cache_teacher_logits.py --teacher lora-corrected

# 3. Train the candidate against each cache (run both, same candidate both times)
python tools/train_distilled_candidate.py \
    --candidate <checkpoint dir from step 1, or the pushed HF repo> \
    --teacher-cache <cache dir from step 2's baseline run>/checkpoints/baseline-topk-logits.pt \
    --run-name distill-aggressive-vs-baseline \
    --push-to-hub sartajbhuvaji/bonsai-distilled-aggressive-vs-baseline

python tools/train_distilled_candidate.py \
    --candidate <same checkpoint as above> \
    --teacher-cache <cache dir from step 2's lora-corrected run>/checkpoints/lora-corrected-topk-logits.pt \
    --run-name distill-aggressive-vs-corrected \
    --push-to-hub sartajbhuvaji/bonsai-distilled-aggressive-vs-corrected
```

**Hard-learned lessons from every prior run in this project -- read before
running:**
1. **Smoke-test with a handful of steps before the real run.** Every real
   training run in this project (3.1, 3.3) found a real bug this way
   (frozen-base+LoRA gradient checkpointing needing
   `enable_input_require_grads`, GPU-side LoRA merge OOMing). Run
   `train_distilled_candidate.py` against a tiny slice first if practical,
   or at minimum watch the first ~10 steps' loss before trusting an
   unattended multi-hour run.
2. **Time the first ~50 samples of `cache_teacher_logits.py` and
   extrapolate before committing to the full 6,000-sample run** -- this
   project has twice discovered a run was going to take far longer than
   assumed only after wasting real GPU-hours on it (MBPP-TS's two failed
   Lambda attempts, 2026-09-24/25). No real pace estimate exists yet for
   this specific inference-only caching workload.
3. **`bitsandbytes` isn't a declared project dependency** (checked
   `pyproject.toml` -- it isn't there; every prior 8-bit-AdamW run in this
   project installed it ad hoc on the GPU instance). `pip install
   bitsandbytes` before running `train_distilled_candidate.py`.
4. Push everything with `--push-to-hub` -- **private by default** (this
   script sets `private=True`), per plan.md 4.0's decided policy: only
   candidates that survive get saved at all, never publicly by default.

**Execution progress (2026-09-27, DO H100 `107.170.17.174`, tmux `phase4`):**
- Step 1 **done** — candidate built (zero-shot loss 1.5891, layer list matched;
  warned-not-failed on sweep loss drift), pushed private
  `sartajbhuvaji/bonsai-candidate-aggressive-depth-experts`;
  `runs/20260927T161454Z-e94ad0-build-candidate-aggressive-depth-experts/`.
- Step 3a smoke **done** — 10-step KD vs 50-sample baseline cache OK
  (`train_loss` ~1.40, `val_loss` ~1.59).
- Step 2b **done** — baseline 6000-sample cache in **1229s** (~0.20 s/sample);
  `runs/20260927T163225Z-91fcb3-…/baseline-topk-logits.pt` (**563 MB**).
- Step 2c **done** — LoRA-corrected 6000-sample cache;
  `runs/20260927T165332Z-4b256d-…/lora-corrected-topk-logits.pt`.
- Step 3b **done** — distill vs baseline teacher: final **train KD 0.5573**,
  **val KD 0.5350** (`20260927T171637Z-3ddb90-distill-aggressive-vs-baseline`);
  pushed private `sartajbhuvaji/bonsai-distilled-aggressive-vs-baseline`.
- Step 3c **done** — distill vs LoRA teacher: final **train KD 0.5810**,
  **val KD 0.5339** (`20260927T175820Z-2e6846-distill-aggressive-vs-corrected`);
  pushed private `sartajbhuvaji/bonsai-distilled-aggressive-vs-corrected`.
  Pipeline finished **2026-09-27T18:39:22Z**.
- Orchestration: `tools/run_phase4_ablation_from_cache.sh` (resume after smoke);
  log `/root/phase4_pipeline.log`; monitor via `12a` (pipeline log), `12b` +
  `tools/refresh_phase4_kd_artifacts.py` (training curves → `docs/latex/figures/phase4_kd/`).
- Fixes landed during run: KD device mismatch (`load_teacher_cache` +
  `make_kd_loss_fn`); `cache_teacher_logits.py` progress logging + data-before-model;
  `PYTHONUNBUFFERED=1` on pipeline scripts.

**Ablation answer (2026-09-27, complete):** KD val loss was a **tie** (baseline
teacher **0.5350** vs LoRA teacher **0.5339**). HumanEval-TS **n=40** on both
distilled students (same first-40 problems, temp 0.2 / top-p 0.95,
`notebooks/12c_phase4_distilled_humaneval_ts_sample.ipynb`):
- vs baseline teacher: **pass@1 = 0.0**
  (`runs/20260927T191055Z-263aa6-distill-vs-baseline-humaneval-ts-n40`)
- vs LoRA teacher: **pass@1 = 0.0**
  (`runs/20260927T204350Z-02fcbc-distill-vs-corrected-humaneval-ts-n40`)

**Interpretation:** teacher correction neither helps nor hurts this distilled
student on the project's primary coding metric at this sample size — both arms
are equally bad (0/40), and KD couldn't separate them either. That is **not**
evidence that distillation "washed out" the teachers' HumanEval gap (base ~0.67
vs corrected ~0.57); it likely reflects **500-step KD on one aggressive
candidate** being insufficient for downstream pass@1, so this ablation
**fails to reject** "correction carries through" but also **doesn't show a
directional win for correction." n=40 CI is ~±16pp per plan.md 3.3 — treat
as directional only. Full 159-problem eval optional if a tighter estimate is
needed before writing up. Raw n=40 generations showed **empty bodies and
repetition loops** (see `docs/debug/phase4_distill_humaneval_generations_problems_0-2.txt`), not a harness artifact.

**Incremental rerun — baseline teacher only (2026-09-27, in progress):** 500-step
KD was too short (pass@1 0/40, degenerate code). Re-distill **same** aggressive
candidate vs **same** baseline logit cache with **3000 steps** (else unchanged:
8-bit AdamW, peak_lr=2e-4, warmup 75, batch 4). Reuse local cache
`runs/20260927T163225Z-91fcb3-…/baseline-topk-logits.pt` when the GPU box still
has it; on re-cache, push to private dataset
`sartajbhuvaji/bonsai-teacher-logits-baseline` (`--push-to-hub` on
`cache_teacher_logits.py`). Student Hub target:
`sartajbhuvaji/bonsai-distilled-aggressive-vs-baseline-3k`. Orchestration:
`tools/run_phase4_baseline_kd_3k.sh`, log `/root/phase4_baseline_kd_3k.log`,
tmux + `metrics.jsonl` per step. Then n=40 HumanEval-TS on the 3k checkpoint.
**Do not** run the LoRA teacher arm until this result is reviewed.

**Backlog (when there is time, post-ablation): persist teacher logit caches on
Hub and teach the trainer to fetch them.** The cache is reusable across every
future candidate (keyed by teacher + sample set, not by which student is being
trained), so paying the ~1h×2 inference cost once per teacher and reusing beats
re-caching on every ablation.
1. ~~Add `--push-to-hub` to `tools/cache_teacher_logits.py`~~ **done (2026-09-27)** —
   uploads the `.pt` to a private HF dataset repo (`private=True`). Target for
   baseline re-cache: `sartajbhuvaji/bonsai-teacher-logits-baseline`. Still TODO:
   optional one-time backfill upload of the existing Phase 4 baseline cache file
   without re-running inference.
2. Extend `tools/train_distilled_candidate.py`'s `--teacher-cache` to accept
   either a local path **or** that HF dataset repo id — local path first, then
   `hf_hub_download` fallback the same way `tools/build_candidate.py` already
   does for `sartajbhuvaji/bonsai-importance-scores` / `importance-scores.pt`.

**General KD CLI (2026-09-27):** `tools/run_kd_distillation.py` — any `--teacher`
/ `--student` repo (no pruning), `--teacher-cache`, `--cache-only` / `--train-only`,
and hyperparameter flags; implementation in `bonsai.training.kd_run`.

### Phase 4 — Distillation retraining and candidate selection

**4.0 — Checkpoint persistence.** Decided (2026-09-20): save selected pruned/
distilled candidates to private repos on Hugging Face Hub under the
project account (`sartajbhuvaji`, write-scope token already in `.env`),
not local disk only -- needed for Phase 4.3's later vLLM/SGLang inference
benchmarking regardless of which machine that runs on. NOT saving every
candidate tested: a full bf16 checkpoint is ~60GB, and between the 9
single-axis + 12 combined-axis zero-shot candidates (2.1/2.2/2.3/2.4)
that's well over 1TB if saved indiscriminately. Only candidates that
survive to Phase 4.2's lightweight-retrain stage get saved, and only the
winners from 4.2 get saved post-full-retrain. Naming convention and exact
repo-per-candidate vs. one-repo-many-branches structure not yet decided --
pick it when the first real save happens, not speculatively now. This is
also exactly why the config-staleness fix from the systematic review pass
(2026-09-20, `prune_experts`/`prune_heads` now sync `model.config`) matters
in practice, not just in theory: `save_pretrained` writes `model.config` to
disk, and vLLM/SGLang (like `transformers`) trust that config to describe
the actual tensor shapes -- a stale config would have produced a checkpoint
that fails to load correctly in exactly the inference engines 4.3 targets.

**4.1 — Lightweight retrain of every Phase 2/2.4 candidate carried
forward**, using the corrected teacher, logit-only forward-KL loss (both
papers' best practice once depth isn't cut drastically). Small token
budget, purely to stabilize rankings per best-practice #9. Candidate
roster expanded per the 2.4 scope-expansion note above: a few points per
single axis (not just one pre-picked winner) plus every combined-sweep
candidate worth carrying forward by zero-shot loss.

Sizing worked out for real (2026-09-21), from the target model's actual
architecture (`hidden=2048, layers=48, experts=128/layer,
moe_intermediate=768, heads=32/4kv, head_dim=128`), not estimates --
computed and cross-checked against the paper's stated 30.5B/3.35B
figures (exact match). The 21-candidate roster spans 8.3B to 30.4B total
params; the three "heads-only" candidates (one per tier) are barely
smaller than the unpruned model (30.2-30.4B) since attention is a small
fraction of an expert-dominated MoE -- experts being the most redundant
axis (§8 of the paper) cuts both ways: removing them saves the most
size, but head pruning alone barely moves the needle on total params
even though it costs real zero-shot loss.

This matters for how 4.1 is built: holding the corrected teacher
(~61GB bf16 weights) resident on GPU *simultaneously* with a
worst-case ~30B student's training footprint (weights + gradients +
8-bit-AdamW states, ~183GB) would need ~244GB+ of GPU memory, repeated
across all 21 candidates -- doesn't fit the cheaper 4x A6000 (192GB
total), only the 8x A100 (320GB, $15.92/hr, same instance type 3.1's
real run used).

**Decided (2026-09-21): cache the teacher's output distribution once,
instead of recomputing it live for every candidate.** Forward-KL loss
needs the corrected teacher's logits for the same training tokens every
candidate is retrained on -- identical across all 21 candidates since
they share one retraining sample set, so there's no reason to hold the
teacher resident 21 separate times. One inference-only pass (teacher
weights only, no gradients/optimizer, so no need for the 9GiB/GPU
placement-cap trick 3.1's training needed -- device_map="auto"'s naive
sharding is fine here) computes and caches, per training token, the
top-$k$ logits ($k{=}20$ starting point) as (token-id, log-prob) pairs
against a truncated-and-renormalized teacher distribution -- a standard
KD approximation, not the full 151,936-vocab distribution (which would
need ~560GB across the retraining pool at fp16; top-20 needs ~230MB).
After that one pass, every candidate's lightweight retrain only needs
its *own* training footprint -- worst case ~183GB, comfortably inside
the 4x A6000's 192GB with headroom for activations, most candidates far
less. Net effect: the whole 21-candidate roster should fit on the
cheaper, already-available 4x A6000 ($4.36/hr) instead of needing the
8x A100 ($15.92/hr) at all, and the teacher only needs to run once
instead of 21 times.

Retraining sample set for this pass: a fresh slice of the retraining
pool, disjoint from 3.1's 6,000 already-consumed samples (different
seed, verified non-overlapping the same way 1.1 verified the
calibration/retraining split -- the pool is ~69 shards' worth after
calibration and correction together consumed a small fraction of it, so
collision risk is low but checked, not assumed). Exact token budget is
still the open question logged in Discussion (§11) -- no Nemotron-scale
precedent to scale from -- resolved empirically per candidate by
watching for rank stability (best-practice #9's ~300-step heuristic is
Nemotron-4-15B-specific, not assumed to transfer directly).

Not yet built or run -- this section documents the design decided
before implementation, so the reasoning is recorded before code exists,
not reconstructed after the fact.

**4.2 — Pick winners, on more than zero-shot loss.** The goal (refined
2026-09-20): a TypeScript-specialized model that's smaller, faster, and
cheaper to serve than the unpruned baseline, *and* high-accuracy on
TypeScript specifically -- both are real, weighted criteria, not one
subordinate to the other. Not "smallest model that barely clears a bar,"
and not "highest accuracy regardless of size" either; a genuine
multi-objective comparison across candidates, reported honestly rather
than collapsed into one composite score that hides the tradeoff. Ranking
needs more than Phase-4.1's loss numbers:
- HumanEval-TS/MBPP-TS pass@1 (the project's own primary metrics, §4 of
  the paper) -- the accuracy side of the tradeoff, weighted as heavily as
  the efficiency side below, not a gate the efficiency numbers override.
- The tracked-but-not-optimized-for general-knowledge suite (MMLU,
  ARC-Challenge, Winogrande), to catch a candidate that scores well on
  TypeScript while regressing badly and uninterestingly elsewhere.
- Params saved / active-param reduction per candidate -- the "smaller"
  side of the tradeoff.
- Real serving cost once 4.3's vLLM/SGLang numbers exist (throughput,
  latency) -- the "faster and cheaper to run" side, a serving-time claim
  that param count alone doesn't settle.
Save the resulting shortlist's checkpoints per 4.0.

**4.3 — Full retraining of the winners**, then real inference-engine
benchmarking once GPU capacity allows (stated goal, 2026-09-20: "when we
get a free H100," so this is explicitly deferred, not blocking anything
before it) -- load each saved winner in vLLM and SGLang, measure real
serving throughput/latency, not just the accuracy metrics above. This is
the actual point of saving real, correctly-configured checkpoints in 4.0
rather than only recording loss numbers in a `.pt` file: an eval-suite
winner that doesn't serve efficiently isn't actually the right answer for
"servable to a TypeScript-heavy-inference customer."

### Phase 5 — Evaluation and writeup

**5.1 — Full eval suite on every fully-retrained candidate**, including
the vLLM/SGLang throughput/latency numbers from 4.3 once available.

**5.2 — Report.** What each axis bought (params saved vs. accuracy vs. real
inference throughput/latency, not just a proxy), whether expert pruning
behaved like width or like something new, whether teacher correction
mattered as much for us as it did in 2408.11796, whether any candidate
actually beat the unpruned baseline on TypeScript specifically (the actual
question this project is trying to answer, per the 2026-09-20 goal), and
which combination we'd recommend for a TypeScript-focused deployment if
this stops being hypothetical. Lands in `docs/`.
*Notebook:* `04_results.ipynb` — the final comparison table and charts across
every candidate (params vs. TypeScript pass@1 vs. the tracked
MMLU/Winogrande/ARC-Challenge numbers vs. vLLM/SGLang throughput once
available), the same one the writeup's numbers come from. Whether the
fuller suite (Python regression, HellaSwag) is worth running for just the
overall winner is a call for this step, not decided
now.

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

1. ~~Best expert-importance metric (2.3).~~ Resolved (1.3): gate-weighted
   expert-output norm over routed tokens, per REAP / NVIDIA Model-Optimizer's
   Minitron-MoE extension. Not independently re-verified against the primary
   sources yet — worth a spot-check before 2.3 leans on it for a real pruning
   decision, but no longer a blank slate.
2. Downstream proxy for depth-pruning criterion in a coding model, given no
   TypeScript Winogrande equivalent exists (2.1).
3. Whether uniform or per-layer-searched expert counts win (2.3, 2.4).
   Genuinely open, not resolved — see the correction in 2.3's writeup above.
   EvoESAP (arXiv 2603.06003), re-checked directly, actually favors
   non-uniform allocation; this project's own real per-layer-search
   experiment is still outstanding.
4. Right lightweight-retrain token budget for this model's active param count
   — the papers' 1.8B tokens was calibrated for a 15B dense model, not a 3B
   active-param MoE (2.2-2.4).
5. Whether teacher correction matters as much here as in 2408.11796, given
   we're adapting toward a much narrower domain (TypeScript) than their
   general-purpose retraining data (3.1).
6. Whether the expert-pruning axis's calibration-domain sensitivity (flagged
   in the MoE section above — expert rankings reportedly shift more with
   calibration-set choice than head/neuron rankings do) shows up for us the
   way the cited literature suggests, or whether TypeScript-only calibration
   sidesteps it (2.3).
