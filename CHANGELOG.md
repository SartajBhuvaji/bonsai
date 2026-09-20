# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `docs/latex/main.tex`: new "Combining Axes: Depth, Experts, and Heads
  Together" section covering the real multi-axis sweep from
  `08_combined_pruning_sweep.ipynb` -- per-tier target-count table,
  figure, relative-loss table, and the naive-multiplicative interaction
  analysis (depth+heads compounds worse than independent pruning would
  predict; depth+experts is actually subadditive). All three single
  pruning axes and their zero-shot combinations are now in the paper.
  Abstract, introduction, and Discussion/Open-Questions updated
  accordingly, including two new open questions (whether the zero-shot
  interaction ordering survives retraining, and which tier/combination
  to carry forward).

## [0.21.0] - 2026-09-20

### Added

- `bonsai.pruning.viz.plot_combined_pruning_sweep`: grouped bar chart
  (one group per aggressiveness tier, one bar per axis-combination type)
  for comparing multi-axis pruned candidates, which don't share a single
  sweepable count the way `plot_pruning_sweep`'s line-chart layout
  assumes.
- `notebooks/08_combined_pruning_sweep.ipynb`: the first real multi-axis
  pruning experiment (plan.md 2.4) -- actual combined depth+expert+head
  pruning (not each axis swept alone) against Qwen3-30B-A3B-Base, 3
  aggressiveness tiers x 4 combination types (depth+experts, depth+heads,
  experts+heads, all three). Comparing each combo's actual zero-shot loss
  to what independent (naive multiplicative) compounding of its
  single-axis costs would predict shows real interaction effects, not
  just "more pruning costs more": depth+heads compounds genuinely worse
  than independent at every tier (excess grows to +64.9pp at the
  aggressive tier); depth+experts is actually *subadditive* at the
  moderate/aggressive tiers (cheaper than independent compounding
  predicts); experts+heads tracks the independent prediction closely
  throughout.

## [0.20.0] - 2026-09-20

### Added

- `bonsai.training`: the first real backward-pass code in this project
  (`bonsai.importance`/`bonsai.pruning` are forward-pass-only). `train_step`
  does gradient-accumulated single-sample forward/backward passes
  (sequences aren't padded, matching every other forward-pass call site in
  this codebase); `lr_schedule` implements linear warmup + cosine decay to
  a floor, matching arXiv 2408.11796's Table 4 recipe's shape, not its
  literal numbers (94-380B tokens across 32 DGX H100 nodes doesn't scale
  down to this project's compute); `train` runs the full loop, logging via
  `bonsai.tracking`. `loss_fn` is a pluggable seam so Phase 4's
  distillation (forward-KL against teacher logits) reuses this same loop
  later. Built for plan.md 3.1 (teacher correction).

## [0.19.2] - 2026-09-20

### Added

- `docs/latex/main.tex`: new "Head Pruning: Zero-Shot Loss vs. Query
  Heads Pruned" section covering the real sweep from
  `07_head_pruning_sweep.ipynb` -- figure, relative-loss table, and the
  cross-axis matched-fraction comparison (experts are by far the most
  redundant axis; heads and depth read as comparably prunable). All
  three pruning axes are now characterized in the paper. Abstract,
  introduction, and Discussion/Open-Questions updated accordingly,
  including two new open questions (target per-group head count, and
  whether a free non-uniform-per-group head search would beat the
  constrained sweep).

## [0.19.1] - 2026-09-20

### Fixed

- `bonsai.pruning.experts.prune_experts`: previously left
  `model.config.num_experts` stale after pruning (only the per-module
  `experts.num_experts`/`router.num_experts` were updated) -- fixed to
  sync `model.config.num_experts` and `model.num_experts` (the latter
  read by `load_balancing_loss_func` for the router aux loss) whenever
  every sparse layer ends up with the same expert count. Also now raises
  if pruning would leave a layer with fewer experts than
  `router.top_k`, instead of failing later with an opaque `RuntimeError`
  inside routing.
- `bonsai.pruning.heads.prune_heads`: the same `model.config` staleness
  gap for `num_attention_heads`, fixed the same way.
- `docs/latex/main.tex`: two mentions of a NAEE (arXiv 2402.14800)
  finding overstated what the paper actually shows -- it reports
  selection-frequency-based pruning performing worst against its own
  reconstruction-loss method, not against an output-magnitude-based
  criterion as the text implied. Reworded both, and simplified the
  second mention to cross-reference the first instead of repeating it.
  `select_experts_to_prune`'s docstring also had the same backwards
  EvoESAP claim already fixed elsewhere in the paper and in plan.md --
  fixed here too.

Found by a systematic Fable/Sonnet review pass across the pruning code,
tests, visualizations, the paper, and plan.md; everything else it
checked (formulas verified against the actual Minitron PDF text, pruning
mechanisms verified against the real `transformers` source, unit tests
confirmed to exercise real forward-pass correctness, every notebook
number cross-checked against plan.md and the paper) came back clean.

## [0.19.0] - 2026-09-20

### Added

- `notebooks/07_head_pruning_sweep.ipynb`: the first real head-pruning
  experiment (plan.md 2.2) -- actual query-head removal (not scoring)
  against Qwen3-30B-A3B-Base, all 7 valid per-key/value-group counts
  (GQA's uniform-per-group constraint means this sweep already covers its
  entire valid search space, unlike depth/experts' chosen subsets).
  Greedy beats random at every count, but by a much narrower margin than
  depth (up to ~9x) or experts (2-5x): 1.2-2.9x here, shrinking as more
  heads are pruned. Compared properly across all three sweeps at shared
  pruned-fractions: experts is by far the most redundant axis (e.g. +25.5%
  vs. +129.6%/+153.8% for heads/depth at 50% pruned); heads and depth read
  as comparably (un)forgiving to prune, not clearly ranked against each
  other. Run on a rented 4x A6000 instance via `device_map="auto"`
  sharding -- no H100/A100-80GB capacity was available at launch time.

## [0.18.0] - 2026-09-20

### Added

- `bonsai.pruning.heads`: `prune_heads` (permanently slices the pruned
  query heads out of `Qwen3MoeAttention.q_proj`/`o_proj` at head
  boundaries, in place, and recomputes `self_attn.num_key_value_groups`
  for the new head count) and `select_heads_to_prune` (lowest-`k_per_group`
  by score, ranked independently within each contiguous key/value group).
  Decided the pruning mechanism has to respect GQA group boundaries --
  `Qwen3MoeAttention.forward` reads `self.num_key_value_groups` fresh off
  the module every call, so a free per-head selection across group
  boundaries would leave that scalar undefined; pruning the same count of
  query heads from every key/value group (never a whole group) keeps it
  well-defined, mirroring how depth/expert pruning each needed their own
  real bookkeeping fix (`self_attn.layer_idx` renumbering,
  `Qwen3MoeTopKRouter` row slicing).

## [0.17.1] - 2026-09-20

### Added

- `docs/latex/main.tex`: new "Expert Pruning: Zero-Shot Loss vs. Experts
  Pruned" section covering the real sweep from
  `06_expert_pruning_sweep.ipynb` -- figure, relative-loss table, and the
  Super Experts regression-test writeup. Abstract, introduction, and
  discussion/open-questions updated to reflect that expert pruning (not
  just depth) now has real results.

### Fixed

- `docs/latex/references.bib`: `superexperts2025` had a placeholder
  `{Anonymous}` author, and `reap2025`/`naee2024` used an `and others`
  truncation -- replaced with the real, complete author lists (verified
  against each paper's arXiv listing). `qwen3moe` pointed only at the HF
  model card despite claiming to be "Qwen3 Technical Report" -- added the
  real arXiv citation (2505.09388). Added a missing bibliography entry for
  NVIDIA Model-Optimizer, cited three times in the paper with no entry.
- `docs/latex/main.tex`: found via an independent review pass (a
  background agent tasked with fact-checking the paper against its own
  sources) --
  Figure 2's caption claimed the three outlined cells were the Super
  Experts; the figure actually outlines the model-wide top-5 by score,
  which catches only one of the three (the other two rank highest only
  within their own layer). Also: a Discussion bullet claimed published
  evidence that non-uniform expert-count allocation underperforms uniform
  allocation, sourced to EvoESAP (arXiv 2603.06003) -- re-checked directly
  against the primary source and found EvoESAP reports the opposite
  (non-uniform beats uniform by up to 19.6% on MATH-500); reworded to state
  this honestly as an untested, simplifying assumption rather than settled
  evidence, and fixed the same error in plan.md. Smaller fixes: an
  "order-of-magnitude" claim didn't match the metric it cited alongside;
  a "statistically indistinguishable from zero" claim had no variance/test
  behind it; the active-parameter count (3.4B) didn't match the project's
  own measured value (3.35B); a dangling "unlike depth pruning" comparison
  that Section 6 never set up. Added a missing Discussion bullet for the
  retrain-token-budget open question (in plan.md but not the paper). Also
  tightened the abstract and Sections 6-7's prose, which had drifted into
  a more notebook-register voice than the rest of the paper.

## [0.17.0] - 2026-09-19

### Added

- `bonsai.pruning.viz.plot_pruning_sweep`: generalized with `title`/`xlabel`
  parameters and an `extra_hlines` argument for fixed-cost reference
  candidates outside the main count sweep, so the same function serves both
  depth and expert pruning's sweep plots instead of duplicating it.
- `notebooks/06_expert_pruning_sweep.ipynb`: the first real expert-pruning
  experiment (plan.md 2.3) -- actual expert removal (not scoring) against
  Qwen3-30B-A3B-Base, eight uniform per-layer counts, greedy
  (protected-guard) vs. 3-seed random control. Greedy beats random at every
  count, by a smaller multiple than depth pruning's (roughly 2-5x in
  relative-loss terms vs. close to an order of magnitude for layers).
  Standout finding: pruning just the three known Super Experts costs more
  loss (+51.8%) than greedy-pruning up to 1,536 ordinary experts (+9.6%),
  reproducing arXiv 2507.23279's own claim directly in this pipeline's own
  loss metric -- the concrete justification for `select_experts_to_prune`'s
  `protected` guard.

## [0.16.0] - 2026-09-19

### Added

- `bonsai.pruning.experts`: `prune_experts` (permanently slices the pruned
  experts out of `Qwen3MoeExperts.gate_up_proj`/`down_proj` and the matching
  `Qwen3MoeTopKRouter.weight` rows, in place, per named MoE layer) and
  `select_experts_to_prune` (lowest-`n_per_layer` by score, with a
  `protected` guard -- e.g. Super Experts -- excluded from the ranking
  entirely rather than just filtered after). Mirrors
  `bonsai.pruning.depth.prune_layers`'s mechanism-implementation pattern;
  unlike depth pruning, no index renumbering is needed since
  `Qwen3MoeExperts.forward` re-derives hit experts fresh from `top_k_index`
  each call rather than depending on fixed per-expert identity.

## [0.15.1] - 2026-09-19

### Added

- `docs/latex/main.tex`: new "Depth Pruning: Zero-Shot Loss vs. Layers
  Pruned" section covering the real sweep from `05_depth_pruning_sweep.ipynb`
  -- figure, relative-loss table, and the honest mixed-result writeup on
  contiguous-block removal. Abstract and introduction updated to stop
  claiming "no pruning has occurred," since depth pruning now has real
  results.

## [0.15.0] - 2026-09-19

### Added

- `bonsai.pruning.viz.plot_pruning_sweep`: zero-shot loss vs. layers pruned,
  one line per named strategy plus a shaded min-max band for a random
  control (the check every principled pruning criterion needs to actually
  clear, not just correlate well with each other).
- `notebooks/05_depth_pruning_sweep.ipynb`: the first real pruning
  experiment (plan.md 2.1) -- actual layer removal (not scoring) against
  Qwen3-30B-A3B-Base, nine layer counts, four strategies (BI-greedy,
  PPL-greedy, best-contiguous-block, 3-seed random control). Principled
  selection beats random by close to an order of magnitude past a handful
  of pruned layers; BI-greedy and PPL-greedy track each other closely in
  actual pruning outcome, not just ranking correlation; contiguous-block
  removal (2408.11796's own preferred approach) is a genuinely mixed
  result on this model -- competitive at small counts, behind both greedy
  strategies by n=24.

## [0.14.1] - 2026-09-19

### Fixed

- `accelerate` is now a core runtime dependency, not something every fresh
  GPU instance has to `pip install` by hand after `AutoModelForCausalLM.from_pretrained(...,
  device_map=...)` fails with `requires accelerate`. Hit this exact gap on
  two separate fresh instances now -- worth fixing permanently rather than
  patching around it a third time.

## [0.14.0] - 2026-09-19

### Added

- `bonsai.pruning.depth.prune_layers`: the first actual pruning
  implementation (plan.md 2.1) -- permanently removes decoder layers from a
  model, the counterpart to `bonsai.importance.depth.skip_layer`'s temporary
  identity-swap used for scoring. Caught a real bug via testing against an
  actual forward pass rather than by reading source alone: surviving layers
  keep their original `self_attn.layer_idx` after removal (non-contiguous,
  e.g. `0, 2, 4, 5` after pruning layers 1 and 3), which breaks the default
  KV cache's internal list indexing with `IndexError` the moment it's
  used -- fixed by renumbering each surviving layer's `layer_idx` to its
  new position.
- `bonsai.importance.depth.mean_loss`: promoted from a private helper
  already used internally by `ppl_importance` to a public function --
  Phase 2's pruning steps need it to measure a pruned candidate's zero-shot
  loss, not just to score which layers to prune in the first place.

## [0.13.2] - 2026-09-19

### Added

- `docs/latex/main.tex`: explicit numbered equations for every scoring
  formula in use (head importance, Block Importance, perplexity delta --
  expert importance already had one), two algorithm blocks (expert
  importance from routing data; perplexity-delta depth scoring via layer
  skipping), and six figures copied in as vector PDFs from the project's
  own real notebook outputs (head/expert heatmaps, expert concentration
  curve, PPL bar chart, BI-vs-PPL scatter, the three-method per-layer
  summary). All real results already reported in prose, now also shown.

## [0.13.1] - 2026-09-19

## [0.13.0] - 2026-09-19

### Added

- `docs/latex/`: first draft of the project's research paper, using the
  ICML 2025 template. Covers everything real as of this writing -- target
  model/data, baseline evaluation, the full importance-scoring methodology
  and results (heads, experts including the Super Experts confirmation,
  BI-vs-PPL agreement at r=0.91), and an explicit open-questions section.
  No fabricated results; no pruning claimed, since none has run yet.
  Vendored `icml2025.sty`/`.bst` and supporting style files from the
  official ICML 2025 author package. Compiled with Tectonic, a
  self-contained LaTeX engine, no system LaTeX install needed; `main.pdf`
  is committed alongside the source. Citation author lists for REAP, NAEE,
  and the Super Experts paper are incomplete placeholders, to be filled in
  before this goes anywhere public.

## [0.12.0] - 2026-09-19

### Added

- `bonsai.importance.depth.skip_layer` + `ppl_importance`: the PPL half of
  plan.md 2.1's depth-pruning criteria, alongside BI (1.3). `skip_layer` is
  a context manager that temporarily makes one decoder layer act as
  identity, restoring it afterward even on exception; `ppl_importance`
  removes each layer in turn and measures the real held-out perplexity
  cost against the unmodified baseline (`num_layers + 1` forward passes,
  versus BI's one). Unit-tested against a real, tiny, randomly-initialized
  CPU model, not just shapes -- confirms the model's actual output changes
  when a layer is skipped and is bit-for-bit restored afterward.
- `bonsai.importance.viz.plot_metric_agreement`: scatter one per-layer
  metric against another with Pearson correlation annotated, for checking
  whether a cheap importance proxy actually predicts an expensive one.

### Changed

- `bonsai.importance.viz.plot_layer_summary` now takes a `dict[str, Tensor]`
  of named per-layer series instead of three fixed positional arguments --
  supports any number of scoring methods (up to the categorical palette's
  four adjacent-pairs-safe slots), not just the original three, so adding
  PPL as a fourth signal didn't need a new function.
- `bonsai.importance.viz.plot_layer_importance` takes optional
  `title`/`ylabel`/`annotation_label` overrides, so both of 2.1's depth
  criteria (BI, PPL) reuse the same bar-chart function instead of two
  near-identical ones.
- Every figure in `notebooks/03_importance_scores.ipynb` now also saves a
  `.pdf` (vector) alongside its `.png` -- PNGs pixelate at print/journal
  resolution.

## [0.11.0] - 2026-09-19

### Added

- `bonsai.importance.depth.block_importance`: Block Importance (BI),
  2408.11796's fast depth-pruning signal -- `1 - cosine_similarity(layer_input,
  layer_output)`, computable in the same forward pass that produces those
  activations rather than needing a separate pass per candidate layer (unlike
  PPL-based scoring). Kept in its own module rather than folded into
  `bonsai.importance`'s aggregation-primitive trio, matching plan.md 1.3's own
  design: depth is scored structurally differently from heads/experts in both
  papers, and bonsai keeps that split in code, not just in the docstring.
- `bonsai.importance.viz`: heatmaps and supporting charts for every importance
  axis -- `plot_head_heatmap`/`plot_expert_heatmap` (layers x heads/experts,
  percentile-clipped color scale so a handful of extreme outliers don't wash
  out the rest of the grid into one indistinguishable pale blob),
  `plot_layer_importance` (BI bar chart), `plot_layer_summary` (normalized
  head/expert/BI overlay, to check whether independently-computed scoring
  methods agree on which layers matter), `plot_score_distribution` (log-scale
  histograms exposing the long-tailed skew directly), `plot_expert_concentration`
  (per-layer cumulative-importance curves, a real signal for how aggressively
  2.3 could eventually prune), and `plot_expert_grid_for_layer` (one layer's
  real expert grid, same schematic layout as `bonsai.architecture.viz`'s
  diagram, colored by real scores -- defaults to true-max color scaling rather
  than a percentile clip, since a 128-cell single-layer view needs to spotlight
  one standout expert, not reveal structure the way the full 6,144-cell
  heatmap does). Sequential blue throughout, lifted from the project's
  validated color-formula palette.
- `notebooks/03_importance_scores.ipynb`: plan.md 1.3's GPU-dependent
  validation, extended from a spot-check into a full multi-axis view -- real
  forward hooks against `Qwen3-30B-A3B-Base` (bf16), 48 samples from the
  pinned TypeScript calibration shard, all 48 layers. Purely observational,
  no pruning. Confirms Qwen3-30B-A3B's three known Super Experts (arXiv
  2507.23279) rank #1 in their layer on real data, not just a smaller
  earlier spot-check; finds the first and last layers dominate every scoring
  method while the middle third looks comparatively redundant; finds a
  layer's top ~16% of experts already capture about a third of its total
  importance. Closes with a plain-language summary, each point paired with a
  cropped visual (the real architecture diagram, or a zoomed version of the
  chart above it) pointing at exactly what to look at.

## [0.10.0] - 2026-09-19

### Added

- `bonsai.importance`: shared activation-based importance scoring (plan.md
  1.3), structured the way 2407.14679 Section 2.2 actually does it -- one
  aggregation primitive (`aggregate`, pluggable mean(abs)/L2/variance, not
  hardcoded) wrapped by thin axis-specific functions. First wrapper:
  `head_importance`, the paper's own `F_head` formula unchanged. Depth stays
  out of this module by design, matching the papers' own split (PPL/BI, not
  activation magnitude). GPU-free and unit-tested against synthetic tensors;
  wiring this to real forward hooks against Qwen3-30B-A3B-Base is still
  pending.
- `bonsai.importance.expert_importance`: the second wrapper -- gate-weighted
  expert-output norm, averaged only over each expert's routed tokens (REAP,
  arXiv 2510.13999, and independently NVIDIA Model-Optimizer's Minitron-MoE
  extension converge on this formula; it beat the plan's other two
  candidates, selection frequency and router-logit magnitude, in every
  published comparison found). Plus `expert_max_output_norm` /
  `flag_outlier_experts`, a percentile-based guard against pruning
  Qwen3-30B-A3B's known Super Experts (arXiv 2507.23279) by diluting their
  rare-but-massive output into a low mean score.

## [0.9.0] - 2026-09-18

### Added

- `notebooks/02b_baseline_mbpp_ts.ipynb`: the second primary/gating tier of
  plan.md step 1.2's baseline eval -- MBPP-TS pass@1 = 0.7333 against
  `Qwen3-30B-A3B-Base`, unpruned, full 390-problem test set. Same
  `batch_size=8`/`max_length_generation=512` config as HumanEval-TS. Run on
  GPU 0 of a 2x H100 instance, concurrently with the general-knowledge suite
  on GPU 1.

## [0.8.0] - 2026-09-18

### Added

- `notebooks/02c_baseline_general_knowledge.ipynb`: the tracked-not-gating
  tier of plan.md step 1.2's baseline eval, run against `Qwen3-30B-A3B-Base`
  on the full test sets -- MMLU (5-shot): acc 0.8101; ARC-Challenge
  (25-shot): acc 0.6536, acc_norm 0.6749; Winogrande (5-shot): acc 0.7798.
  Run on GPU 1 of a 2x H100 instance, concurrently with MBPP-TS on GPU 0 --
  `accelerate launch` without multi-GPU flags pins to one GPU by default, so
  two independent single-GPU jobs coexist on one box with no cross-talk
  (confirmed: each process shows up against only its own GPU's UUID in
  `nvidia-smi --query-compute-apps`).

## [0.7.0] - 2026-09-15

### Added

- `bonsai.eval.general_knowledge`: runs MMLU (5-shot), ARC-Challenge
  (25-shot) and Winogrande (5-shot) against a HF model via
  `lm_eval.simple_evaluate`. Shot counts are pinned explicitly -- none of the
  three tasks' own harness configs set one, which would otherwise silently
  fall back to 0-shot and make the numbers incomparable to the papers'
  tables plan.md wants them checked against.
- `tools/bigcode_eval_main.py`: a vendored copy of
  `bigcode-evaluation-harness`'s own `main.py`. `pip install`ing the `eval`
  extra's git dependency only packages the `bigcode_eval` library -- `main.py`
  is a top-level script in that repo, not part of the installed package, so
  there's no `python -m bigcode_eval.main` to run. Calls
  `bonsai.eval.register()` itself before parsing `--tasks`, since
  `accelerate launch` runs it as its own subprocess.

### Changed

- `tools/check_notebooks.py` now tolerates IPython shell/line magic (`!cmd`,
  `%magic`) in code cells, blanking a magic line (and its `\`-continuations)
  before `ast.parse` rather than pulling in IPython's own cell transformer
  for one narrow need. The first notebook using `!` shell cells hit this.
- `tools/setup_instance.sh` gains `WITH_EVAL=1`, installing the `eval` extra
  plus Node.js and the TypeScript compiler -- MultiPL-E's TS task shells out
  to `tsc` and `node` to actually compile and run generated code, and neither
  ships with the Python env this script otherwise builds. Off by default; a
  plain calibration-data or architecture-notebook run doesn't need it.
- `notebooks/02a_baseline_humaneval_ts.ipynb`: the first real baseline-eval
  result (plan.md step 1.2) -- HumanEval-TS pass@1 = 0.6708 against
  `Qwen3-30B-A3B-Base`, unpruned. Run as its own notebook rather than folded
  into a single all-five-tasks one, so a slow or failed task doesn't block
  or hide the others; MBPP-TS and the general-knowledge suite get their own
  notebooks in the same pattern. `batch_size=8` (not 32, and
  `max_length_generation=512` not 1024): batched generation waits for the
  slowest sequence in a batch to stop, and this MoE model's decode cost
  doesn't amortize across a batch as cleanly as a dense model's would -- the
  first attempt at batch_size=32 sat for over an hour without finishing.

## [0.6.0] - 2026-09-15

### Added

- `bonsai.eval`: registers MBPP's MultiPL-E TypeScript translation
  (`mbpp-ts`) as a `bigcode-evaluation-harness` task (`multiple-mbpp-ts`).
  The harness's own `multiple.py` only wraps the HumanEval translations;
  `nuprl/MultiPL-E`'s `mbpp-ts` config carries the same schema, so
  `MbppMultiPLE` reuses `GeneralMultiPLE`'s prompt-building, postprocessing,
  execution and pass@k scoring wholesale, pointed at a different dataset
  config. New `eval` extra (`bigcode-eval`, git-installed -- not on PyPI)
  kept separate from `dev` since it's only needed on the GPU instance
  actually running an eval.

### Changed

- `bonsai.data` moves into its own package (`src/bonsai/data/`), matching
  `bonsai.architecture`'s existing layout. Import path unaffected.

## [0.5.0] - 2026-09-14

### Added

- `bonsai.data`: pulls, splits, and tokenizes the TypeScript
  calibration/retraining data from `bigcode/the-stack-dedup` (plan.md step
  1.1). `make_split` reserves whole parquet shards for calibration so the two
  pools are disjoint by construction; `load_samples`, `tokenize_samples`, and
  `verify_disjoint` (a `hexsha`-keyed leak check) round out the pipeline.
  `configs/qwen3-30b-a3b.toml` now pins the real split: shard 0 of 71 for
  calibration, the rest for retraining.
- `notebooks/01_calibration_data.ipynb`: pulls real calibration and
  retraining samples through the pinned split, plots the token-length
  distribution, and confirms disjointness empirically.

## [0.4.0] - 2026-09-14

### Added

- `tools/launch_lambda.sh`, `tools/setup_instance.sh`, `tools/gpu_env.sh`:
  Lambda Cloud provisioning, adapted from `unfuse`'s own tooling -- launch a
  GPU instance with SSH-key-registration verification (Lambda silently issues
  its own keypair if this is skipped), prepare the environment and confirm
  GPU visibility plus a `Qwen3MoeForCausalLM` import, and capture the
  machine's provenance for whatever ran there.

## [0.3.1] - 2026-09-14

### Changed

- `bonsai.architecture` and `bonsai.viz` are now `bonsai.architecture` (a
  package) and `bonsai.architecture.viz`, grouping the model-inspection code
  together and out of `src/bonsai/`'s top level. `from bonsai.architecture
  import summarize` is unaffected.

## [0.3.0] - 2026-09-14

### Added

- `bonsai.config`: TOML-loaded experiment configuration (model / data /
  compute), plus `configs/qwen3-30b-a3b.toml`.
- `bonsai.tracking`: local run tracking under `runs/` -- metrics as
  JSON-lines, config snapshots, no external service.
- `bonsai.architecture`: inspects a Qwen3-MoE model's parameter breakdown by
  pruning axis (attention / router / experts / embedding) from `config.json`
  alone, via a `meta`-device model build. No weights downloaded.
- `bonsai.viz`: a two-panel architecture diagram (whole-model stack, one
  decoder layer expanded with its residual connections and expert grid),
  drawn entirely from an `ArchitectureSummary` -- no hardcoded model numbers.
- `notebooks/`: explainer notebooks alongside the code they call, with the
  convention documented in `notebooks/README.md`. First one,
  `00_model_architecture.ipynb`, walks Qwen3-30B-A3B-Base's real architecture,
  diagrams it, and states the pruning targets for each axis, including the
  expert axis neither Minitron paper covers.
- `tools/check_notebooks.py` and a `notebooks` tox/CI job: committed notebooks
  must keep their execution outputs.
- Runtime dependencies: `transformers`, `huggingface-hub`, `torch`. A
  `notebook` extra (`jupyter`, `nbconvert`, `ipykernel`, `matplotlib`).

## [0.2.13] - 2026-09-14

## [0.2.12] - 2026-09-13

### Added

- The original Minitron paper (arXiv 2407.14679 v2) in `docs/papers/`, alongside
  the follow-up already there.

## [0.2.11] - 2026-09-13

### Changed

- Dependabot now watches GitHub Actions only, quarterly, grouped into one PR.
  The pip ecosystem is gone: the dependencies here are `>=` floors with no
  lockfile, so bumping them changed nothing about what actually gets installed.

## [0.2.10] - 2026-09-13

## [0.2.9] - 2026-09-13

## [0.2.8] - 2026-09-13

## [0.2.7] - 2026-09-13

## [0.2.6] - 2026-09-13

## [0.2.5] - 2026-09-13

## [0.2.4] - 2026-09-13

## [0.2.3] - 2026-09-13

## [0.2.2] - 2026-09-13

### Fixed

- Dependabot no longer applies its own `major`/`minor`/`patch` labels, which
  collided with the release labels and would have versioned bonsai off a
  dependency's semver bump. Its PRs are now labelled `dependencies` and `patch`.

## [0.2.1] - 2026-09-13

### Added

- `docs/papers/`, with the Minitron paper (arXiv 2408.11796 v3) and an index
  recording the version, retrieval date and checksum of each PDF.

## [0.2.0] - 2026-09-13

### Added

- Project skeleton: hatchling packaging with a `src` layout, ruff, mypy in strict
  mode, pytest, and tox as the single definition of every check.
- CI on Python 3.11, 3.12 and 3.13, plus a distribution build check.
- Label-driven releases: `patch`, `minor` or `major` on a PR bumps the version,
  tags and publishes on merge.
