# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
